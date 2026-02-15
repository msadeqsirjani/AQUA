"""
EQAT: Energy-Efficient Quantization-Aware Training

Implementation of "Energy-Efficient Quantization-Aware Training with Dynamic Bit-Width Optimization"
by Ali Karkehabadi and Avesta Sasan, UC Davis — GLSVLSI '25.

Key Features:
- Learnable per-layer bit-widths (2-8 bits)
- 3-term loss: cross-entropy + KL divergence + energy penalty
- Hardware-aware energy model
- Differentiable quantization with straight-through estimators

Directory Structure:
- core/      : Core components (quantizer, energy, loss, trainer)
- models/    : Model architectures (SimpleCNN5, ResNet18EQAT)
- scripts/   : Training scripts for MNIST, CIFAR-10, CIFAR-100
"""

# Import core components
from EQAT.core import (
    EQATConvBlock,
    EQATLinear,
    EnergyModel,
    EQATLoss,
    EQATTrainer,
)

# Import models
from EQAT.models import (
    SimpleCNN5,
    ResNet18EQAT,
)

__version__ = '1.0.0'

__all__ = [
    # Core components
    'EQATConvBlock',
    'EQATLinear',
    'EnergyModel',
    'EQATLoss',
    'EQATTrainer',
    # Models
    'SimpleCNN5',
    'ResNet18EQAT',
]
