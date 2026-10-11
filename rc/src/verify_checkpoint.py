"""
verify_checkpoint.py - Inspect and verify best_model.pth checkpoint
Confirms the saved epoch, validation metrics, model architecture weights, and checkpoint path.
Author: Member 2
"""

import os
import sys
import argparse
import torch

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


def verify(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return False

    print("="*70)
    print("           CHECKPOINT VERIFICATION REPORT")
    print("="*70)
    print(f"File Path: {checkpoint_path}")
    print(f"File Size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        epoch = ckpt.get('epoch', 'Unknown')
        val_auc = ckpt.get('val_auc', None)
        val_acc = ckpt.get('val_acc', None)
        val_loss = ckpt.get('val_loss', None)
        state_dict = ckpt['model_state_dict']

        print(f"Saved Best Epoch:      {epoch}")
        if val_auc is not None:
            print(f"Validation ROC-AUC:    {val_auc:.4f}")
        if val_acc is not None:
            print(f"Validation Accuracy:   {val_acc * 100:.2f}%")
        if val_loss is not None:
            print(f"Validation Loss:       {val_loss:.4f}")
    else:
        state_dict = ckpt
        print("Format: Raw state_dict")

    # Verify model loading
    model = get_model(pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Architecture:          DenseNet121 Binary Classifier")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
        prob = torch.sigmoid(out).item()

    print(f"Dummy Forward Pass:    Success (Output Logit: {out.item():.4f}, Sigmoid: {prob:.4f})")
    print("="*70)
    print("Status: CHECKPOINT VALID AND READY FOR EVALUATION")
    print("="*70)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify best_model.pth checkpoint")
    parser.add_argument("--checkpoint", type=str,
                        default="/content/drive/MyDrive/Pneumonia_Project/checkpoints/best_model.pth",
                        help="Path to best_model.pth checkpoint")
    args = parser.parse_args()
    verify(args.checkpoint)
