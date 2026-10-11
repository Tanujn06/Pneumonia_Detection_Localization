"""
gradcam.py - Grad-CAM heatmaps for the DenseNet121 pneumonia classifier
Loads Member 2's checkpoint and explains the single pneumonia logit using a
DenseNet121 feature map: norm5 (last, 7x7 at 224px) or denseblock3 (14x14).
Author: Member 3
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure imports resolve whether executed from root or rc/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from model import get_model
except ImportError:
    from src.model import get_model

try:
    from dataset import read_xray, IMAGE_SIZE, MEAN, STD
except ImportError:
    from src.dataset import read_xray, IMAGE_SIZE, MEAN, STD


DEFAULT_CHECKPOINT = "/content/drive/MyDrive/Pneumonia_Project/checkpoint1/best_model.pth"


def get_eval_transform():
    """Same deterministic preprocessing as validation/test in train.py and evaluate.py."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_model(checkpoint_path, device=None):
    """Load the DenseNet121 classifier from a checkpoint saved by train.py."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # train.py stores numpy scalars (val_auc) alongside the weights, which the
    # PyTorch >= 2.6 weights_only loader rejects. The checkpoint is our own file.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if (
        isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    ) else checkpoint

    model = get_model(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    """Load a DICOM / PNG / JPG X-ray as an RGB PIL image (DICOMs use Member 1's read_xray)."""
    image_path = Path(image_path)
    if image_path.suffix.lower() == ".dcm":
        return Image.fromarray(read_xray(image_path)).convert("RGB")
    return Image.open(image_path).convert("RGB")


# Layers Grad-CAM can target. Resolution is for a 224x224 input.
#   norm5       : last feature map (after ReLU), 7x7   -> one cell ~146 px of a 1024 px X-ray
#   denseblock3 : output of the third dense block, 14x14 -> one cell ~73 px
TARGET_LAYERS = ("norm5", "denseblock3")


class GradCAM:
    """
    Grad-CAM (Selvaraju et al., 2017) for PneumoniaDenseNet121.

    Instead of forward/backward hooks, the DenseNet forward pass is replayed
    explicitly so the activations can be captured without clashing with the
    in-place ReLU that torchvision applies after the feature extractor:
        features -> ReLU -> global average pool -> classifier
    """

    def __init__(self, model, target_layers=("norm5",)):
        unknown = set(target_layers) - set(TARGET_LAYERS)
        if unknown:
            raise ValueError(f"Unsupported target layer(s) {unknown}; choose from {TARGET_LAYERS}")
        self.model = model
        self.densenet = model.densenet
        self.target_layers = tuple(target_layers)

    def compute(self, input_tensor):
        """
        Args:
            input_tensor: normalized image batch of shape (1, 3, H, W).
        Returns:
            cams: {layer: np.ndarray (H, W) in [0, 1], upsampled to the input size}
            prob: predicted pneumonia probability (float).
        """
        self.model.eval()
        x = input_tensor

        with torch.enable_grad():
            captured = {}
            h = x
            for name, module in self.densenet.features.named_children():
                h = module(h)
                if name == "norm5":
                    h = F.relu(h)
                if name in self.target_layers:
                    captured[name] = h
            pooled = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
            logit = self.densenet.classifier(pooled)
            names = list(captured)
            grads = torch.autograd.grad(logit.sum(), [captured[n] for n in names])

        cams = {}
        for name, grad in zip(names, grads):
            activations = captured[name]
            # Channel weights = global-average-pooled gradients of the pneumonia logit.
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()
            cams[name] = cam.astype(np.float32)

        prob = torch.sigmoid(logit).item()
        return cams, prob

    def __call__(self, input_tensor):
        """Heatmap for the first target layer: returns (cam, prob)."""
        cams, prob = self.compute(input_tensor)
        return cams[self.target_layers[0]], prob


def explain_multi(model, pil_image, device=None, target_layers=("norm5",), flip_tta=False):
    """
    Grad-CAM for several layers in one pass.
    flip_tta also explains the horizontally mirrored image (train.py trains with
    RandomHorizontalFlip), mirrors that heatmap back and averages the two.

    Returns ({(layer, flip_tta): cam at original size}, probability). When flip_tta
    is True both the plain and the averaged heatmaps are returned. The probability
    is always for the unflipped image, matching evaluate.py.
    """
    device = device or next(model.parameters()).device
    tensor = get_eval_transform()(pil_image).unsqueeze(0).to(device)
    gradcam = GradCAM(model, target_layers)
    cams, prob = gradcam.compute(tensor)
    out = {(layer, False): cam for layer, cam in cams.items()}

    if flip_tta:
        flipped, _ = gradcam.compute(torch.flip(tensor, dims=[3]))
        for layer, cam in cams.items():
            avg = (cam + flipped[layer][:, ::-1]) / 2
            out[(layer, True)] = avg / avg.max() if avg.max() > 0 else avg

    return {k: resize_cam(v, pil_image.size) for k, v in out.items()}, prob


def explain(model, pil_image, device=None, target_layer="norm5", flip_tta=False):
    """Run preprocessing + Grad-CAM on a PIL image; returns (cam at original size, probability)."""
    cams, prob = explain_multi(model, pil_image, device, (target_layer,), flip_tta)
    return cams[(target_layer, flip_tta)], prob


def resize_cam(cam, size):
    """Resize a [0, 1] heatmap to (width, height) of the original image."""
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(cam_img, dtype=np.float32) / 255.0


def overlay_heatmap(pil_image, cam, alpha=0.4, colormap="jet"):
    """Blend a [0, 1] heatmap (same size as image) over the X-ray. Returns an RGB uint8 array."""
    import matplotlib
    cmap = matplotlib.colormaps[colormap]
    heat = (cmap(cam)[..., :3] * 255).astype(np.float32)
    base = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    blended = (1 - alpha) * base + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for one chest X-ray")
    parser.add_argument("--image", type=str, required=True, help="Path to a .dcm/.png/.jpg X-ray")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to best_model.pth")
    parser.add_argument("--output", type=str, default="gradcam_output.png",
                        help="Where to save the overlay figure")
    parser.add_argument("--target_layer", type=str, default="norm5", choices=TARGET_LAYERS)
    parser.add_argument("--flip_tta", action="store_true", help="Average with the mirrored image's heatmap")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    image = load_image(args.image)
    cam, prob = explain(model, image, device, args.target_layer, args.flip_tta)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input X-ray")
    axes[1].imshow(overlay_heatmap(image, cam))
    axes[1].set_title(f"Grad-CAM (P(pneumonia) = {prob:.3f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"P(pneumonia) = {prob:.4f}")
    print(f"Saved Grad-CAM overlay to: {args.output}")


if __name__ == "__main__":
    main()
