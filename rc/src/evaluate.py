"""
evaluate.py - Final Evaluation Pipeline on held-out test.csv
Loads best_model.pth checkpoint, evaluates once on test.csv,
and exports all required metrics, confusion matrix, and ROC curve to Google Drive.
Supports Validation-tuned optimal threshold and Test-Time Augmentation (TTA).
Author: Member 2
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Ensure imports resolve
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
    from dataset import PneumoniaDataset
except ImportError:
    from src.dataset import PneumoniaDataset


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def plot_and_save_confusion_matrix(cm, output_path):
    """Generates and saves high-resolution confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Normal (0)', 'Pneumonia (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix on Held-Out Test Set',
           ylabel='True Label',
           xlabel='Predicted Label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_and_save_roc_curve(fpr, tpr, roc_auc_val, output_path):
    """Generates and saves ROC curve plot."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='crimson', lw=2.5, label=f'DenseNet121 (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def predict_loader(model, loader, device, tta=False):
    """Runs inference across a dataloader with optional Test-Time Augmentation (TTA)."""
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images, targets = batch['image'], batch['label']

            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            if tta:
                # Horizontal flip TTA
                flipped_images = torch.flip(images, dims=[3])
                logits_flip = model(flipped_images)
                probs_flip = torch.sigmoid(logits_flip)
                probs = (probs + probs_flip) / 2.0

            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    return np.array(all_targets).astype(int), np.array(all_probs)


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate DenseNet121 Model on Held-out test.csv")
    parser.add_argument("--test_csv", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/metadata/test.csv",
                        help="Path to test.csv metadata")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Optional val.csv path to calibrate optimal classification threshold")
    parser.add_argument("--img_dir", type=str,
                        default="/content/rsna_train_images/stage_2_train_images/",
                        help="Directory containing test images")
    parser.add_argument("--checkpoint", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/checkpoints/best_model.pth",
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--results_dir", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/results",
                        help="Directory on Drive to save evaluation figures and JSON")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default probability threshold")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (horizontal flip)")
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Member 2 Evaluation] Device: {device} | TTA: {args.tta}")

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load Model & Weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = get_model(pretrained=False).to(device)

    state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    test_transform = get_test_transforms()

    # 2. Threshold Calibration on Validation Set (if provided)
    chosen_threshold = args.threshold
    if args.val_csv and os.path.exists(args.val_csv):
        print(f"Calibrating decision threshold on validation set: {args.val_csv} ...")
        val_dataset = PneumoniaDataset(args.val_csv, args.img_dir, transform=test_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        val_targets, val_probs = predict_loader(model, val_loader, device, tta=args.tta)

        best_val_acc = 0.0
        best_th = 0.5
        # Search thresholds to maximize validation accuracy
        for th in np.linspace(0.25, 0.75, 101):
            preds = (val_probs >= th).astype(int)
            acc_th = (preds == val_targets).mean()
            if acc_th > best_val_acc:
                best_val_acc = acc_th
                best_th = float(th)

        chosen_threshold = best_th
        print(f"  >>> Optimal Threshold found on validation set: {chosen_threshold:.4f} (Val Accuracy: {best_val_acc * 100:.2f}%)")

    # 3. Test Dataset & DataLoader
    test_dataset = PneumoniaDataset(args.test_csv, args.img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Running inference across held-out test set ({len(test_dataset)} cases)...")

    all_targets, all_probs = predict_loader(model, test_loader, device, tta=args.tta)
    all_preds = (all_probs >= chosen_threshold).astype(int)

    # 4. Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    roc_auc_val = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_val),
        "threshold_used": float(chosen_threshold),
        "tta_used": bool(args.tta),
        "confusion_matrix": cm.tolist()
    }

    # 5. Save Artifacts to Google Drive
    metrics_file = os.path.join(args.results_dir, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    cm_file = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, cm_file)

    roc_file = os.path.join(args.results_dir, "roc_curve.png")
    plot_and_save_roc_curve(fpr, tpr, roc_auc_val, roc_file)

    # 6. Output Summary Table
    print("\n" + "="*50)
    print("       FINAL HELD-OUT TEST EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy:         {acc * 100:.2f}% ({acc:.4f})")
    print(f"  ROC-AUC:          {roc_auc_val:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  Decision Thresh:  {chosen_threshold:.4f}")
    print(f"  TTA Enabled:      {args.tta}")
    print("-" * 50)
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("="*50)
    print(f"Artifacts exported to Google Drive ({args.results_dir}):")
    print(f"  -> {metrics_file}")
    print(f"  -> {cm_file}")
    print(f"  -> {roc_file}\n")


if __name__ == '__main__':
    evaluate()
