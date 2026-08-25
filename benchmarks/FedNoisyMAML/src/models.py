"""
Simple MLP classifier for 2D Gaussian mixture classification.
"""

import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    """2-layer MLP for multi-class classification."""

    def __init__(self, input_dim=8, hidden_size=64, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ConvNet(nn.Module):
    """CNN for image classification, compatible with MAML functional forward."""

    def __init__(self, in_channels=1, n_classes=10, img_size=28):
        super().__init__()
        feat_size = 64 * (img_size // 4) ** 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)
