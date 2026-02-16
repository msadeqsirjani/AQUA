"""
SimpleCNN5 — 5-layer CNN (3 conv + 2 fc).

MNIST  (1-ch, 28×28): spatial after 3 pools → 3×3, flatten=1152
CIFAR-10 (3-ch, 32×32): spatial after 3 pools → 4×4, flatten=2048
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN5(nn.Module):
    """
    5-layer CNN: Conv→Conv→Conv→FC→FC.
    MaxPool after each conv layer.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # 3 convolutional blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Flatten size: 128 × (3×3 for MNIST, 4×4 for CIFAR-10)
        flat = 128 * (3 * 3 if in_channels == 1 else 4 * 4)

        # 2 fully-connected layers
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    # ── Energy tracking helpers ──────────────────────────────────────────
    def get_compute_layers(self) -> list:
        """Returns all conv/linear layers for energy computation."""
        return [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    def get_spatial_sizes(self, in_channels: int) -> list:
        """Returns spatial sizes (H, W) for each layer's input."""
        if in_channels == 1:  # MNIST
            return [(28, 28), (14, 14), (7, 7), (), ()]
        else:  # CIFAR-10
            return [(32, 32), (16, 16), (8, 8), (), ()]
