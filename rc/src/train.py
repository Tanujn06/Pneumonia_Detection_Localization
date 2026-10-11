"""
train.py - Model Training & Validation Pipeline
Tracks validation ROC-AUC and saves the best checkpoint directly to Google Drive.
Author: Member 2
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

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
    model.train()
    running_loss = 0.0

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

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
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

    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 Pneumonia Classifier")
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
                        help="Directory on Google Drive to save best_model.pth")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Member 2 Training] Active Device: {device}")

    # Step 1: Create checkpoint directory if not existing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")

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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_auc = 0.0

    print(f"\n=== Starting Training ({args.epochs} Epochs) ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val ROC-AUC: {val_auc:.4f}")

        # Track validation ROC-AUC and save best checkpoint based on Val AUC (NOT test data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  >>> New best checkpoint saved to: {best_model_path} (Val AUC: {best_val_auc:.4f})")

    print(f"\nTraining complete! Peak Validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Saved Best Model Checkpoint: {best_model_path}")


if __name__ == '__main__':
    main()
