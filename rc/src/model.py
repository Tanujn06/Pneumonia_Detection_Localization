"""
model.py - DenseNet121 Binary Classifier for Pneumonia Detection
Author: Member 2
"""

import torch
import torch.nn as nn
from torchvision import models


class PneumoniaDenseNet121(nn.Module):
    """
    DenseNet121 architecture adapted for binary classification.
    Replaces the default 1000-class classifier with a single linear unit
    outputting 1 raw logit per image (to be paired with BCEWithLogitsLoss).
    """
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.2):
        super(PneumoniaDenseNet121, self).__init__()

        try:
            from torchvision.models import DenseNet121_Weights
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            self.densenet = models.densenet121(weights=weights)
        except (ImportError, AttributeError):
            self.densenet = models.densenet121(pretrained=pretrained)

        # DenseNet121 classifier input dimension is 1024
        in_features = self.densenet.classifier.in_features

        # Replace final classifier with single logit output
        if drop_rate > 0:
            self.densenet.classifier = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, 1)
            )
        else:
            self.densenet.classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape: (batch_size, 1)
        return self.densenet(x)


def get_model(pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    """Helper factory to instantiate the DenseNet121 binary classifier."""
    return PneumoniaDenseNet121(pretrained=pretrained, drop_rate=drop_rate)


if __name__ == '__main__':
    model = get_model(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f'Model loaded successfully! Output logit shape: {out.shape}')
