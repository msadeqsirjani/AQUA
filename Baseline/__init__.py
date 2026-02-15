"""
Baseline: Full-Precision Training (No Quantization)

Standard training implementation for comparison with EQAT and AQUA.
All models use full-precision (FP32) weights and activations.

Key Features:
- SimpleCNN5 for MNIST and CIFAR-10
- ResNet18 for CIFAR-100
- Standard cross-entropy loss
- Energy measurement with fixed bit-widths (FP32 or 8-bit)

Directory Structure:
- core/      : Energy measurement (BaselineEnergyModel)
- models/    : Model architectures (SimpleCNN5, ResNet18)
- scripts/   : Training scripts for MNIST, CIFAR-10, CIFAR-100
"""

# Import core components
from Baseline.core import BaselineEnergyModel

# Import models
from Baseline.models import SimpleCNN5, ResNet18

__version__ = '1.0.0'

__all__ = [
    # Core
    'BaselineEnergyModel',
    # Models
    'SimpleCNN5',
    'ResNet18',
]
