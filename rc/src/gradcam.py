"""
gradcam.py - Grad-CAM heatmaps for a DenseNet121 pneumonia classifier.
Assumes model.py exposes get_model() and the wrapper has model.densenet.
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for path in (CURRENT_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from model import get_model
except ImportError:
    from src.model import get_model

try:
    from dataset import read_xray, IMAGE_SIZE, MEAN, STD
except ImportError:
    from src.dataset import read_xray, IMAGE_SIZE, MEAN, STD

DEFAULT_CHECKPOINT = "/content/drive/MyDrive/Pneumonia_Project/checkpoints/best_model.pth"
TARGET_LAYERS = ("norm5", "denseblock3")


def get_eval_transform():
    """Preprocessing must match the preprocessing used during evaluation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_model(checkpoint_path, device=None):
    """Load a trusted training checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # weights_only=False supports checkpoints containing NumPy metadata.
    # Only load checkpoints that you trust.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model = get_model(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    """Load a DICOM, PNG, or JPG X-ray as an RGB PIL image."""
    image_path = Path(image_path)
    if image_path.suffix.lower() == ".dcm":
        return Image.fromarray(read_xray(image_path)).convert("RGB")
    with Image.open(image_path) as source:
        return source.convert("RGB")


class GradCAM:
    """Grad-CAM for a single-logit DenseNet pneumonia classifier."""

    def __init__(self, model, target_layers=("norm5",)):
        unknown = set(target_layers) - set(TARGET_LAYERS)
        if unknown:
            raise ValueError(
                f"Unsupported target layer(s): {unknown}; choose from {TARGET_LAYERS}"
            )
        if not target_layers:
            raise ValueError("At least one target layer must be provided.")
        if not hasattr(model, "densenet"):
            raise AttributeError(
                "Expected model.densenet; check the model.py implementation."
            )
        self.model = model
        self.densenet = model.densenet
        self.target_layers = tuple(target_layers)

    def compute(self, input_tensor):
        """
        input_tensor: normalized batch of one image, shape (1, 3, H, W).
        Returns {layer: normalized HxW CAM}, pneumonia probability.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Expected input shape (1, 3, H, W).")

        self.model.eval()
        captured = {}

        # Explicitly replay DenseNet so the desired feature maps retain gradients.
        with torch.enable_grad():
            h = input_tensor
            for name, module in self.densenet.features.named_children():
                h = module(h)
                if name == "norm5":
                    h = F.relu(h)
                if name in self.target_layers:
                    captured[name] = h

            missing = set(self.target_layers) - set(captured)
            if missing:
                raise RuntimeError(f"Could not capture target layers: {missing}")

            pooled = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
            logits = self.densenet.classifier(pooled)
            if logits.numel() != 1:
                raise ValueError(
                    f"Expected one logit for one image; got {tuple(logits.shape)}"
                )

            grads = torch.autograd.grad(
                logits.sum(),
                [captured[name] for name in self.target_layers],
            )

        cams = {}
        for name, grad in zip(self.target_layers, grads):
            activation = captured[name]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam, size=input_tensor.shape[-2:],
                mode="bilinear", align_corners=False
            )[0, 0].detach().cpu().numpy()
            cam = np.asarray(cam, dtype=np.float32)
            cam -= float(cam.min())
            maximum = float(cam.max())
            if maximum > 0:
                cam /= maximum
            cams[name] = cam

        return cams, torch.sigmoid(logits).item()

    def __call__(self, input_tensor):
        cams, probability = self.compute(input_tensor)
        return cams[self.target_layers[0]], probability


def resize_cam(cam, size):
    """Resize a 2D CAM to (width, height), retaining values in [0, 1]."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("CAM must be a finite 2D array.")
    cam = np.clip(cam, 0.0, 1.0)
    image = Image.fromarray(np.round(cam * 255).astype(np.uint8))
    image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def explain_multi(model, pil_image, device=None, target_layers=("norm5",), flip_tta=False):
    """
    Generate CAMs for multiple layers. If flip_tta=True, also return a CAM
    averaged with the horizontally flipped image's mirrored-back CAM.
    Probability always corresponds to the original image.
    """
    device = device or next(model.parameters()).device
    tensor = get_eval_transform()(pil_image).unsqueeze(0).to(device)
    gradcam = GradCAM(model, target_layers)
    cams, probability = gradcam.compute(tensor)
    output = {(layer, False): cam for layer, cam in cams.items()}

    if flip_tta:
        flipped_cams, _ = gradcam.compute(torch.flip(tensor, dims=[3]))
        for layer, cam in cams.items():
            flipped_back = flipped_cams[layer][:, ::-1].copy()
            average = (cam + flipped_back) / 2.0
            maximum = float(average.max())
            if maximum > 0:
                average = average / maximum
            output[(layer, True)] = average.astype(np.float32)

    return {
        key: resize_cam(cam, pil_image.size) for key, cam in output.items()
    }, probability


def explain(model, pil_image, device=None, target_layer="norm5", flip_tta=False):
    """Return (CAM at original image size, pneumonia probability)."""
    if target_layer not in TARGET_LAYERS:
        raise ValueError(f"target_layer must be one of {TARGET_LAYERS}")
    cams, probability = explain_multi(
        model, pil_image, device, (target_layer,), flip_tta
    )
    return cams[(target_layer, flip_tta)], probability


def overlay_heatmap(pil_image, cam, alpha=0.4, colormap="jet"):
    """
    Blend a positive-evidence CAM over the X-ray.
    An all-zero CAM returns the original image unchanged; callers should label
    this case 'No positive Grad-CAM evidence' rather than showing a false map.
    """
    import matplotlib

    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("CAM must be a finite 2D array.")
    if cam.shape != (pil_image.height, pil_image.width):
        raise ValueError("CAM dimensions must match the image dimensions.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    base = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    if float(cam.max()) <= 0:
        return base.astype(np.uint8)

    cam = np.clip(cam, 0.0, 1.0)
    heat = (matplotlib.colormaps[colormap](cam)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for one chest X-ray."
    )
    parser.add_argument("--image", required=True, help="Path to .dcm, .png, or .jpg")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default="gradcam_output.png")
    parser.add_argument("--target_layer", choices=TARGET_LAYERS, default="norm5")
    parser.add_argument("--flip_tta", action="store_true")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    image = load_image(args.image)
    cam, probability = explain(
        model, image, device, args.target_layer, args.flip_tta
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input X-ray")
    axes[1].imshow(overlay_heatmap(image, cam))
    if float(cam.max()) <= 0:
        axes[1].set_title(
            f"No positive Grad-CAM evidence\nP(pneumonia) = {probability:.4f}"
        )
    else:
        axes[1].set_title(
            f"Grad-CAM ({args.target_layer})\nP(pneumonia) = {probability:.4f}"
        )
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"P(pneumonia) = {probability:.4f}")
    print(f"Target layer = {args.target_layer}")
    print(f"CAM range = [{float(cam.min()):.4f}, {float(cam.max()):.4f}]")
    if float(cam.max()) <= 0:
        print("Warning: no positive Grad-CAM evidence for this image.")
    print(f"Saved Grad-CAM overlay to: {args.output}")


if __name__ == "__main__":
    main()
