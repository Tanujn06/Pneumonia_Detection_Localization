"""
evaluate.py - Final Evaluation Pipeline on held-out test.csv
Loads best_model.pth checkpoint, evaluates once on test.csv,
and exports all required metrics, confusion matrix, and ROC curve to Google Drive.
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


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate DenseNet121 Model on Held-out test.csv")
    parser.add_argument("--test_csv", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/metadata/test.csv",
                        help="Path to test.csv metadata")
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
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability classification threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Member 2 Evaluation] Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load Model & Weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = get_model(pretrained=False).to(device)

    state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Test Dataset & DataLoader
    test_transform = get_test_transforms()
    test_dataset = PneumoniaDataset(args.test_csv, args.img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test set contains {len(test_dataset)} patient cases.")

    all_targets = []
    all_probs = []

    print("Running inference across held-out test set...")
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images, targets = batch['image'], batch['label']

            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets).astype(int)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= args.threshold).astype(int)

    # 3. Compute Metrics
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
        "confusion_matrix": cm.tolist()
    }

    # 4. Save Artifacts to Google Drive
    metrics_file = os.path.join(args.results_dir, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    cm_file = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, cm_file)

    roc_file = os.path.join(args.results_dir, "roc_curve.png")
    plot_and_save_roc_curve(fpr, tpr, roc_auc_val, roc_file)

    # 5. Output Summary Table
    print("\n" + "="*48)
    print("         FINAL HELD-OUT TEST METRICS")
    print("="*48)
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  ROC-AUC:          {roc_auc_val:.4f}")
    print("-" * 48)
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("="*48)
    print(f"Artifacts exported to Google Drive ({args.results_dir}):")
    print(f"  -> {metrics_file}")
    print(f"  -> {cm_file}")
    print(f"  -> {roc_file}\n")


if __name__ == '__main__':
    evaluate()
