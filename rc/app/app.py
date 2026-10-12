
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

from google.colab import files
from IPython.display import display

# ============================================================
# 1. PROJECT PATHS
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

if not SRC.exists():
    raise FileNotFoundError(f"Source directory not found: {SRC}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}\n"
        "Check that Google Drive is mounted and the path is correct."
    )

sys.path.insert(0, str(SRC))

from gradcam import (
    load_model,
    load_image,
    explain,
    overlay_heatmap,
)


# ============================================================
# 2. MAIN APPLICATION
# ============================================================

def main():
    print("=" * 55)
    print("PNEUMONIA DETECTION AND LOCALIZATION")
    print("=" * 55)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    print("Loading trained model...")

    model = load_model(str(CHECKPOINT), device)

    print("Model loaded successfully!")

    # --------------------------------------------------------
    # 3. UPLOAD CHEST X-RAY
    # --------------------------------------------------------

    print("\nUpload a chest X-ray image.")
    print("Supported formats: PNG, JPG, JPEG, DCM")

    uploaded = files.upload()

    if not uploaded:
        print("No image uploaded. Please run the cell again.")
        return

    filename = next(iter(uploaded))
    image_path = Path("/content") / filename

    print(f"\nUploaded image: {filename}")

    # --------------------------------------------------------
    # 4. PREDICTION AND GRAD-CAM
    # --------------------------------------------------------

    print("\nProcessing X-ray...")

    image = load_image(str(image_path))

    cam, probability = explain(
        model,
        image,
        device=device,
        target_layer="norm5",
        flip_tta=True,
    )

    overlay = overlay_heatmap(
        image,
        cam,
        alpha=0.4,
    )

    probability = float(probability)

    # --------------------------------------------------------
    # 5. DISPLAY CLASSIFICATION RESULT
    # --------------------------------------------------------

    prediction = (
        "Pneumonia-positive"
        if probability >= 0.5
        else "Pneumonia-negative"
    )

    print("\n" + "=" * 55)
    print("PNEUMONIA DETECTION RESULT")
    print("=" * 55)

    print(f"Pneumonia probability:     {probability * 100:.2f}%")
    print(f"No-pneumonia probability: {(1 - probability) * 100:.2f}%")
    print(f"Predicted class: {prediction}")
    print("Decision threshold: 0.50")

    print("=" * 55)

    # --------------------------------------------------------
    # 6. DISPLAY GRAD-CAM VISUALIZATION
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Chest X-ray")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 7. SAVE VISUALIZATION
    # --------------------------------------------------------

    output_path = Path("/content") / (
        Path(filename).stem + "_gradcam.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nVisualization saved to: {output_path}")

    print(
        "\nNote: Grad-CAM highlights regions influencing the "
        "model's prediction. It is not a clinically validated "
        "pneumonia localization method."
    )


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
