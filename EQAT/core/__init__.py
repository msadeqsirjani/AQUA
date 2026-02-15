"""
EQAT Core Components

Provides the fundamental building blocks for Energy-Efficient Quantization-Aware Training:
- Quantization operations (fake quantization, PACT activations)
- Energy modeling (hardware-aware energy estimation)
- Loss functions (3-term loss: CE + KL + energy)
- Training loop (EQAT trainer with bit-width optimization)
"""

from EQAT.core.quantizer import (
    EQATConvBlock,
    EQATLinear,
    fake_quant_weight,
    _FakeQuantAct,
)
from EQAT.core.energy import EnergyModel
from EQAT.core.loss import EQATLoss
from EQAT.core.trainer import EQATTrainer

__all__ = [
    'EQATConvBlock',
    'EQATLinear',
    'fake_quant_weight',
    'EnergyModel',
    'EQATLoss',
    'EQATTrainer',
]
