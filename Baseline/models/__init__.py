"""
Baseline Model Architectures (Full-Precision)

Standard neural network architectures without quantization:
- SimpleCNN5: 5-layer CNN (3 conv + 2 fc) for MNIST and CIFAR-10
- ResNet18: Standard ResNet-18 for CIFAR-100
"""

from Baseline.models.models import SimpleCNN5, ResNet18

__all__ = [
    'SimpleCNN5',
    'ResNet18',
]
