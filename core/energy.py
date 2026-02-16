"""
Energy Model — Fixed bit-width energy computation.

E_layer = E_comp + E_data
E_comp  = MACs × E_MAC × q²        [quadratic in bits]
E_data  = DataSize × E_access × q  [linear in bits]

For Baseline, q is fixed (either 32 for FP32 or 8 for 8-bit uniform quantization).
Energy is normalized by the 8-bit baseline for comparison.
"""

import torch
import torch.nn as nn

# Energy ratios from Yang et al. 2017: DRAM access ≈ 200× MAC energy
E_MAC = 1.0
E_ACCESS = 200.0


def _macs(layer, in_hw: tuple) -> int:
    """MACs for a Conv2d or Linear layer given spatial input (H, W) or ()."""
    if isinstance(layer, nn.Conv2d):
        H, W = in_hw
        kH, kW = layer.kernel_size if isinstance(layer.kernel_size, tuple) \
                 else (layer.kernel_size, layer.kernel_size)
        pH, pW = layer.padding if isinstance(layer.padding, tuple) \
                 else (layer.padding, layer.padding)
        sH, sW = layer.stride if isinstance(layer.stride, tuple) \
                 else (layer.stride, layer.stride)
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        return layer.in_channels * layer.out_channels * kH * kW * H_out * W_out
    if isinstance(layer, nn.Linear):
        return layer.in_features * layer.out_features
    return 0


def _datasize(layer, in_hw: tuple) -> int:
    """Total data moved through memory: weights + input acts + output acts."""
    if isinstance(layer, nn.Conv2d):
        H, W = in_hw
        kH, kW = layer.kernel_size if isinstance(layer.kernel_size, tuple) \
                 else (layer.kernel_size, layer.kernel_size)
        pH, pW = layer.padding if isinstance(layer.padding, tuple) \
                 else (layer.padding, layer.padding)
        sH, sW = layer.stride if isinstance(layer.stride, tuple) \
                 else (layer.stride, layer.stride)
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        weights = layer.in_channels * layer.out_channels * kH * kW
        in_acts = layer.in_channels * H * W
        out_acts = layer.out_channels * H_out * W_out
        return weights + in_acts + out_acts
    if isinstance(layer, nn.Linear):
        return layer.in_features * layer.out_features + layer.in_features + layer.out_features
    return 0


class BaselineEnergyModel:
    """
    Energy model for Baseline with fixed bit-widths.

    Args:
        layers     : list of nn.Conv2d / nn.Linear layers
        in_hws     : list of (H, W) for conv layers, () for linear layers
        bitwidth   : fixed bit-width (32 for FP32, 8 for 8-bit quantization)
    """

    def __init__(self, layers: list, in_hws: list, bitwidth: float = 32.0):
        self.layers = layers
        self.bitwidth = bitwidth

        self._macs_list = []
        self._data_list = []
        for layer, hw in zip(layers, in_hws):
            self._macs_list.append(_macs(layer, hw))
            self._data_list.append(_datasize(layer, hw))

        # Baseline energy at q = 8 (for normalization)
        q0 = 8.0
        self._baseline = sum(
            m * E_MAC * q0**2 + d * E_ACCESS * q0
            for m, d in zip(self._macs_list, self._data_list)
        )

    def compute_energy(self) -> float:
        """Compute total energy with fixed bit-width."""
        q = self.bitwidth
        total = sum(
            m * E_MAC * q**2 + d * E_ACCESS * q
            for m, d in zip(self._macs_list, self._data_list)
        )
        return total

    def normalized(self) -> float:
        """Returns normalized energy (relative to 8-bit baseline)."""
        return self.compute_energy() / self._baseline

    def get_bitwidths(self) -> list:
        """Returns list of bit-widths (all the same for baseline)."""
        return [self.bitwidth] * len(self.layers)

    def get_stats(self) -> dict:
        """Returns energy statistics for logging."""
        return {
            'energy_normalized': self.normalized(),
            'energy_absolute': self.compute_energy(),
            'bitwidth': self.bitwidth,
            'avg_bitwidth': self.bitwidth,
            'num_layers': len(self.layers),
        }
