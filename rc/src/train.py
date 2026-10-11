"""
train.py - Model Training & Validation Pipeline
Tracks validation ROC-AUC, computes training/validation loss and accuracy,
saves the best checkpoint directly to Google Drive whenever validation AUC improves,
and exports complete training logs.
Author: Member 2
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score

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
    from dataset import PneumoniaDataset
except ImportError:
    from src.dataset import PneumoniaDataset


def get_transforms():
    """Standard data augmentations for training and deterministic resizing for validation."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full training epoch and returns (train_loss, train_accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images, targets = batch[0], batch[1]
        else:
            images, targets = batch['image'], batch['label']

        images = images.to(device)
        targets = targets.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    train_loss = running_loss / len(loader.dataset)
    train_acc = correct / total if total > 0 else 0.0
    return train_loss, train_acc


def validate(model, loader, criterion, device):
    """Runs validation loop and returns (val_loss, val_acc, val_auc)."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images, targets = batch['image'], batch['label']

            images = images.to(device)
            targets = targets.to(device).float().view(-1, 1)

            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    val_loss = running_loss / len(loader.dataset)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    val_acc = float(accuracy_score(all_targets.astype(int), all_preds))

    try:
        val_auc = float(roc_auc_score(all_targets, all_probs))
    except ValueError:
        val_auc = 0.5

    return val_loss, val_acc, val_auc


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 Pneumonia Classifier for 15 Epochs")
    parser.add_argument("--train_csv", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/metadata/train.csv",
                        help="Path to train.csv metadata")
    parser.add_argument("--val_csv", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/metadata/val.csv",
                        help="Path to val.csv metadata")
    parser.add_argument("--img_dir", type=str,
                        default="/content/rsna_train_images/stage_2_train_images/",
                        help="Path to directory containing images/DICOMs")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/checkpoints",
                        help="Directory to save best_model.pth and logs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Optional positive class weight for BCEWithLogitsLoss (e.g. 3.0 for class imbalance)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    args = parser.parse_args()

    # Device selection: CUDA -> Apple MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Member 2 Training] Active Device: {device}")

    # Step 1: Create checkpoint directory if not existing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    log_text_path = os.path.join(args.checkpoint_dir, "training_logs.txt")
    log_json_path = os.path.join(args.checkpoint_dir, "training_history.json")

    # Step 2: Prepare datasets and dataloaders
    train_tf, val_tf = get_transforms()
    print(f"Loading train dataset from: {args.train_csv}")
    print(f"Loading validation dataset from: {args.val_csv}")

    train_dataset = PneumoniaDataset(args.train_csv, args.img_dir, transform=train_tf)
    val_dataset = PneumoniaDataset(args.val_csv, args.img_dir, transform=val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    print(f"Dataset size: Train = {len(train_dataset)}, Val = {len(val_dataset)}")

    # Step 3: Initialize Model, Loss, Optimizer
    model = get_model(pretrained=True).to(device)

    if args.pos_weight is not None:
        pw_tensor = torch.tensor([args.pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"Using BCEWithLogitsLoss with pos_weight={args.pos_weight:.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_auc = 0.0
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = 0.0
    history = []

    header = (
        "=" * 88 + "\n" +
        f"TRAINING PIPELINE: DenseNet121 Binary Classifier ({args.epochs} Epochs)\n" +
        f"Optimizer: Adam (lr={args.lr}) | Model Selection: Validation ROC-AUC\n" +
        f"Checkpoint Target: {best_model_path}\n" +
        "=" * 88
    )
    print(header)
    with open(log_text_path, "w") as f_log:
        f_log.write(header + "\n")

    # Step 4: Run Training Loop for all epochs
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': float(val_auc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
            }
            torch.save(checkpoint, best_model_path)

        epoch_summary = (
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc * 100:.2f}% "
            f"| Val ROC-AUC: {val_auc:.4f}"
        )
        if is_best:
            epoch_summary += f"  >>> [SAVED NEW BEST: Epoch {epoch:02d}, Val AUC: {best_val_auc:.4f}]"

        print(epoch_summary)
        with open(log_text_path, "a") as f_log:
            f_log.write(epoch_summary + "\n")

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_auc': float(val_auc),
            'is_best': bool(is_best)
        })

        with open(log_json_path, "w") as f_json:
            json.dump({
                'total_epochs': args.epochs,
                'best_epoch': best_epoch,
                'best_val_auc': float(best_val_auc),
                'best_val_acc': float(best_val_acc),
                'best_val_loss': float(best_val_loss),
                'checkpoint_path': best_model_path,
                'history': history
            }, f_json, indent=2)

    # Step 5: Final Summary
    footer = (
        "\n" + "=" * 88 + "\n" +
        "TRAINING PIPELINE COMPLETE\n" +
        "=" * 88 + "\n" +
        f"Total Epochs Executed:       {args.epochs}\n" +
        f"Best Model Saved at Epoch:   {best_epoch}\n" +
        f"Peak Validation ROC-AUC:     {best_val_auc:.4f}\n" +
        f"Validation Accuracy at Best: {best_val_acc * 100:.2f}%\n" +
        f"Validation Loss at Best:     {best_val_loss:.4f}\n" +
        f"Exact Checkpoint Path:       {best_model_path}\n" +
        f"Training Log File:           {log_text_path}\n" +
        f"Structured History JSON:     {log_json_path}\n" +
        "=" * 88 + "\n" +
        "NOTE: Model selection was performed strictly on the validation set.\n" +
        "Held-out test set (test.csv) was NOT touched during model selection.\n" +
        "Run evaluate.py once to obtain final held-out test metrics.\n" +
        "=" * 88
    )
    print(footer)
    with open(log_text_path, "a") as f_log:
        f_log.write(footer + "\n")


if __name__ == '__main__':
    main()
