"""
EQAT Energy Model — Eq. 10-12, 19-20 from the paper.

E_layer = E_comp + E_data
E_comp  = MACs × E_MAC × q²        [Eq. 11 — quadratic in bits]
E_data  = DataSize × E_access × q  [Eq. 12 — linear in bits]

Energy is normalised by the 8-bit baseline so it stays in [0, 1],
keeping β = 0.001→0.01 interpretable alongside L_CE ~ 2.3.

Gradient (Eq. 20, via autograd):
  ∂E_i/∂q_i = 2·MACs_i·E_MAC·q_i + DataSize_i·E_access
"""

import torch
import torch.nn as nn

# Ratio from Yang et al. 2017: DRAM access ≈ 200× MAC energy
E_MAC    = 1.0
E_ACCESS = 200.0


def _macs(layer, in_hw: tuple) -> int:
    """MACs for a Conv2d or Linear layer given spatial input (H, W) or ()."""
    if isinstance(layer, nn.Conv2d):
        H, W   = in_hw
        kH, kW = layer.kernel_size if isinstance(layer.kernel_size, tuple) \
                 else (layer.kernel_size, layer.kernel_size)
        pH, pW = layer.padding if isinstance(layer.padding, tuple) \
                 else (layer.padding, layer.padding)
        sH, sW = layer.stride if isinstance(layer.stride, tuple) \
                 else (layer.stride, layer.stride)
        H_out  = (H + 2*pH - kH) // sH + 1
        W_out  = (W + 2*pW - kW) // sW + 1
        return layer.in_channels * layer.out_channels * kH * kW * H_out * W_out
    if isinstance(layer, nn.Linear):
        return layer.in_features * layer.out_features
    return 0


def _datasize(layer, in_hw: tuple) -> int:
    """Total data moved through memory: weights + input acts + output acts (Eq. 12)."""
    if isinstance(layer, nn.Conv2d):
        H, W   = in_hw
        kH, kW = layer.kernel_size if isinstance(layer.kernel_size, tuple) \
                 else (layer.kernel_size, layer.kernel_size)
        pH, pW = layer.padding if isinstance(layer.padding, tuple) \
                 else (layer.padding, layer.padding)
        sH, sW = layer.stride if isinstance(layer.stride, tuple) \
                 else (layer.stride, layer.stride)
        H_out  = (H + 2*pH - kH) // sH + 1
        W_out  = (W + 2*pW - kW) // sW + 1
        weights   = layer.in_channels * layer.out_channels * kH * kW
        in_acts   = layer.in_channels * H * W        # input activation reads
        out_acts  = layer.out_channels * H_out * W_out
        return weights + in_acts + out_acts
    if isinstance(layer, nn.Linear):
        # weights + input activations + output activations
        return layer.in_features * layer.out_features + layer.in_features + layer.out_features
    return 0


class EnergyModel(nn.Module):
    """
    Differentiable normalised energy for a list of EQAT blocks.

    Args:
        blocks    : list of EQATConvBlock / EQATLinear
        in_hws    : list of (H, W) for conv blocks, () for linear blocks
    """

    def __init__(self, blocks: list, in_hws: list):
        super().__init__()
        self.blocks = blocks

        self._macs_list = []
        self._data_list = []
        for blk, hw in zip(blocks, in_hws):
            inner = blk.conv if hasattr(blk, 'conv') else blk.fc
            self._macs_list.append(_macs(inner, hw))
            self._data_list.append(_datasize(inner, hw))

        # Baseline energy at q = 8 (constant, not a parameter)
        q0 = 8.0
        self._baseline = sum(
            m * E_MAC * q0**2 + d * E_ACCESS * q0
            for m, d in zip(self._macs_list, self._data_list)
        )

    def forward(self) -> torch.Tensor:
        """Returns normalised energy in [0, 1] (differentiable w.r.t. q̃)."""
        # Initialise on the same device as q̃ parameters (fixes CUDA crash)
        device = self.blocks[0].q_tilde.device
        total  = torch.zeros(1, device=device).squeeze()
        for blk, m, d in zip(self.blocks, self._macs_list, self._data_list):
            q = blk.get_q()
            total = total + m * E_MAC * q**2 + d * E_ACCESS * q
        return total / self._baseline

    def normalised(self) -> float:
        """Returns current normalised energy as Python float (for logging)."""
        return self.forward().item()
