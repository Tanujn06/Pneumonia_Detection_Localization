
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import torch
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from gradcam import load_model, load_image, explain, overlay_heatmap

# Change this to the location of your downloaded checkpoint.
CHECKPOINT = Path(
    "drive\MyDrive\Pneumonia_Project\checkpoint1\best_model.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 55)
    print("PNEUMONIA DETECTION AND LOCALIZATION")
    print("=" * 55)

    if not CHECKPOINT.exists():
        print(f"\nCheckpoint not found:\n{CHECKPOINT}")
        print("Download best_model.pth from Google Drive and update CHECKPOINT.")
        return

    # Upload/select a chest X-ray
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        title="Select a chest X-ray",
        filetypes=[
            ("Chest X-ray images", "*.png *.jpg *.jpeg *.dcm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not image_path:
        print("No image selected.")
        return

    try:
        print("\nLoading trained model...")
        model = load_model(str(CHECKPOINT), DEVICE)

        print("Loading X-ray...")
        image = load_image(image_path)

        print("Generating prediction and Grad-CAM...")
        cam, probability = explain(
            model,
            image,
            device=DEVICE,
            target_layer="norm5",
            flip_tta=True,
        )

        overlay = overlay_heatmap(image, cam, alpha=0.4)

        # Classification result
        print("\n" + "=" * 55)
        print("CLASSIFICATION RESULT")
        print("=" * 55)
        print(f"Pneumonia probability:    {probability * 100:.2f}%")
        print(f"No-pneumonia probability: {(1-probability) * 100:.2f}%")
        print(
            "Prediction:",
            "Pneumonia-positive" if probability >= 0.5
            else "Pneumonia-negative",
        )
        print("=" * 55)

        # Show original X-ray and heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input X-ray")

        axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")

        axes[2].imshow(overlay)
        axes[2].set_title("Heatmap Overlay")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        # Save output alongside the selected image
        output_path = Path(image_path).with_name(
            Path(image_path).stem + "_gradcam.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nVisualization saved to: {output_path}")

    except Exception as exc:
        print(f"\nAnalysis failed: {exc}")
        raise


if __name__ == "__main__":
    main()
