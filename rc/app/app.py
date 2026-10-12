"""
app.py - Thin Google Colab runner for pneumonia classification and Grad-CAM.

Responsibilities:
- Accept one X-ray upload.
- Call model/image/Grad-CAM utilities from gradcam.py.
- Display prediction and visualization.
- Save the visualization.

This file intentionally does not duplicate model inference or Grad-CAM logic.
"""

from pathlib import Path
import sys
import importlib

import torch
import matplotlib.pyplot as plt
from google.colab import files

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT = Path(
    "/content/Pneumonia_Detection_Localization/"
    "Pneumonia_Detection_Localization"
)
SRC = PROJECT / "rc" / "src"
CHECKPOINT = Path(
    "/content/drive/MyDrive/Pneumonia_Project/"
    "checkpoint1/best_model.pth"
)

# denseblock3 gave a nonzero CAM in the diagnostic test.
# This is a visualization setting, not a guarantee of accurate localization.
TARGET_LAYER = "denseblock3"
FLIP_TTA = True
DECISION_THRESHOLD = 0.50

if not SRC.is_dir():
    raise FileNotFoundError(f"Source directory not found: {SRC}")
if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}\n"
        "Make sure Google Drive is mounted and the path is correct."
    )

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the project's existing implementation. Reload it so Colab uses the
# current file after it has been updated during this session.
import gradcam
importlib.reload(gradcam)

load_model = gradcam.load_model
load_image = gradcam.load_image
explain = gradcam.explain
overlay_heatmap = gradcam.overlay_heatmap


# ============================================================
# 2. MAIN APPLICATION
# ============================================================

def main():
    print("=" * 58)
    print("PNEUMONIA DETECTION AND LOCALIZATION")
    print("=" * 58)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading trained model...")

    model = load_model(str(CHECKPOINT), device)
    print("Model loaded successfully.")

    print("\nUpload one chest X-ray (PNG, JPG, JPEG, or DCM).")
    uploaded = files.upload()

    if not uploaded:
        print("No image uploaded. Run the cell again to try another image.")
        return

    filename = next(iter(uploaded))

    # Colab writes uploaded files to the current working directory.
    image_path = Path(filename).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Uploaded image was not found at: {image_path}"
        )

    print(f"\nUploaded image: {image_path.name}")
    print("Loading X-ray and generating prediction/Grad-CAM...")

    image = load_image(str(image_path))
    cam, probability = explain(
        model,
        image,
        device=device,
        target_layer=TARGET_LAYER,
        flip_tta=FLIP_TTA,
    )
    overlay = overlay_heatmap(image, cam, alpha=0.4)

    probability = float(probability)
    prediction = (
        "Pneumonia-positive"
        if probability >= DECISION_THRESHOLD
        else "Pneumonia-negative"
    )
    no_pneumonia_probability = 1.0 - probability
    has_positive_cam = float(cam.max()) > 0.0

    # ========================================================
    # 3. DISPLAY CLASSIFICATION RESULT
    # ========================================================

    print("\n" + "=" * 58)
    print("CLASSIFICATION RESULT")
    print("=" * 58)
    print(f"Pneumonia probability:      {probability * 100:.2f}%")
    print(f"No-pneumonia probability:   {no_pneumonia_probability * 100:.2f}%")
    print(f"Predicted class:            {prediction}")
    print(f"Decision threshold:         {DECISION_THRESHOLD:.2f}")
    print(f"Grad-CAM target layer:      {TARGET_LAYER}")
    print(f"Grad-CAM range:             [{float(cam.min()):.4f}, {float(cam.max()):.4f}]")

    if not has_positive_cam:
        print(
            "Grad-CAM note: No positive evidence was produced for this "
            "image; the original image is shown without a heatmap overlay."
        )

    print("=" * 58)

    # ========================================================
    # 4. DISPLAY AND SAVE VISUALIZATION
    # ========================================================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Chest X-ray")

    if has_positive_cam:
        axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"Grad-CAM ({TARGET_LAYER})")
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay")
    else:
        # Do not render an all-zero CAM as a colored heatmap.
        axes[1].imshow(image, cmap="gray")
        axes[1].text(
            0.5, 0.5,
            "No positive Grad-CAM\nevidence for this image",
            transform=axes[1].transAxes,
            ha="center", va="center",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.75, "pad": 8},
        )
        axes[1].set_title("Grad-CAM unavailable")
        axes[2].imshow(image, cmap="gray")
        axes[2].set_title("Original X-ray (no overlay)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path("/content") / f"{image_path.stem}_gradcam.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Visualization saved to: {output_path}")
    print(
        "\nNote: This is an experimental model output. Grad-CAM is not a "
        "clinically validated pneumonia localization method, and predictions "
        "must not be used as a medical diagnosis."
    )


if __name__ == "__main__":
    main()
