"""
Model Architectures

Standard neural network architectures:
- SimpleCNN5: 5-layer CNN (3 conv + 2 fc) for MNIST and CIFAR-10
- ResNet18: Standard ResNet-18 for CIFAR-100
"""

from models.simplecnn5 import SimpleCNN5
from models.resnet18 import ResNet18, BasicBlock

__all__ = [
    'SimpleCNN5',
    'BasicBlock',
    'ResNet18',
]
