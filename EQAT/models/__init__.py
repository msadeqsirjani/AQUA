"""
EQAT Model Architectures

Quantization-aware neural network architectures with learnable per-layer bit-widths:
- SimpleCNN5: 5-layer CNN (3 conv + 2 fc) for MNIST and CIFAR-10
- ResNet18EQAT: ResNet-18 with EQAT quantization for CIFAR-100
"""

from EQAT.models.models import SimpleCNN5, ResNet18EQAT

__all__ = [
    'SimpleCNN5',
    'ResNet18EQAT',
]
