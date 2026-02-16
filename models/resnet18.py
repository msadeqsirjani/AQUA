"""
ResNet-18 — Standard ResNet-18 for CIFAR-100.

Architecture matches torchvision ResNet-18:
  stem  → layer1 (64)  → layer2 (128, stride=2)
        → layer3 (256, stride=2) → layer4 (512, stride=2) → fc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    Standard ResNet-18 for CIFAR-100.

    Architecture:
      stem  → layer1 (64)  → layer2 (128, stride=2)
            → layer3 (256, stride=2) → layer4 (512, stride=2) → fc
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()

        # Stem: Conv7×7 + BN + ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # ── Energy tracking helpers ──────────────────────────────────────────
    def get_compute_layers(self) -> list:
        """Returns all conv/linear layers for energy computation."""
        layers = [self.conv1]  # Stem
        # Collect all conv layers from residual blocks
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                layers.append(block.conv1)
                layers.append(block.conv2)
                if len(block.shortcut) > 0:  # Downsample conv
                    layers.append(block.shortcut[0])
        layers.append(self.fc)  # Final classifier
        return layers

    def get_spatial_sizes(self) -> list:
        """Returns spatial sizes (H, W) for each layer's input (CIFAR-100)."""
        # CIFAR-100: 32×32 input
        # After stem (stride=2) + maxpool (stride=2): 8×8
        sizes = [(32, 32)]  # stem conv

        # layer1 (2 blocks, no downsample)
        for _ in range(4):  # 2 blocks × 2 convs
            sizes.append((8, 8))

        # layer2 (2 blocks, first has downsample)
        sizes.append((8, 8))  # block1.conv1 (input is 8×8)
        sizes.append((4, 4))  # block1.conv2 (after stride=2)
        sizes.append((8, 8))  # downsample shortcut
        for _ in range(2):  # block2
            sizes.append((4, 4))

        # layer3 (2 blocks, first has downsample)
        sizes.append((4, 4))  # block1.conv1
        sizes.append((2, 2))  # block1.conv2 (after stride=2)
        sizes.append((4, 4))  # downsample shortcut
        for _ in range(2):  # block2
            sizes.append((2, 2))

        # layer4 (2 blocks, first has downsample)
        sizes.append((2, 2))  # block1.conv1
        sizes.append((1, 1))  # block1.conv2 (after stride=2)
        sizes.append((2, 2))  # downsample shortcut
        for _ in range(2):  # block2
            sizes.append((1, 1))

        sizes.append(())  # fc layer
        return sizes
