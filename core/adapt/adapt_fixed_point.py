"""
Fixed-point quantizer for Adaptive Precision Training (AdaPT).
Kummer et al., arXiv:2107.13490, 2021.

AdaPT uses fixed-point quantization where each b-bit number is
represented as an integer scaled by 2^(-f), with f fractional bits.
The optimal f is chosen per-tensor to minimize MSE.

The bit-width is mutable: it starts high and decreases during training
when the precision controller determines it is safe to do so.
"""

import torch
import torch.nn as nn


class FixedPointQuantizer(nn.Module):
    """Fixed-point fake quantizer with mutable bit-width.

    Quantizes to b-bit fixed-point with automatically chosen fractional
    bits f that minimize quantization MSE. Bit-width can be reduced
    during training by the AdaPTPrecisionController.

    Args:
        total_bits: Initial bit-width (default: 8).
        min_bits: Minimum allowed bit-width (default: 2).
        is_weight: If True, quantizes weights (symmetric);
                   if False, quantizes activations (asymmetric, EMA range).
        ema_momentum: EMA momentum for activation range tracking.
    """

    def __init__(self, total_bits=8, min_bits=2, is_weight=False,
                 ema_momentum=0.99):
        super().__init__()
        assert total_bits >= min_bits
        self.current_bits = total_bits
        self.min_bits = min_bits
        self.is_weight = is_weight
        self.ema_momentum = ema_momentum

        self.register_buffer("_fractional_bits", torch.tensor(0))
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self.register_buffer("observer_enabled", torch.tensor(1, dtype=torch.int32))

    # ------------------------------------------------------------------
    # Fractional-bit search
    # ------------------------------------------------------------------

    def find_best_fractional(self, w):
        """Grid search over f in [0, current_bits-1] for minimum MSE.

        For each candidate f the fixed-point grid is:
          step = 2^(-f)
          integer_bits = current_bits - f - 1  (sign bit)
          range = [-2^integer_bits, 2^integer_bits - step]

        Returns:
            int: Optimal number of fractional bits.
        """
        b = self.current_bits
        w_det = w.detach()
        best_f = 0
        best_mse = float("inf")

        for f in range(b):  # f in [0, b-1]
            w_q = self._quantize(w_det, b, f)
            mse = (w_det - w_q).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_f = f

        return best_f

    # ------------------------------------------------------------------
    # Fixed-point quantization
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize(w, bits, frac_bits):
        """Apply fixed-point quantization.

        Args:
            w: Input tensor (float).
            bits: Total bit-width b (including sign).
            frac_bits: Number of fractional bits f.

        Returns:
            Fake-quantized tensor (float, same shape as w).
        """
        int_bits = bits - frac_bits - 1  # sign bit
        step = 2.0 ** (-frac_bits)
        q_min = -(2.0 ** int_bits)
        q_max = 2.0 ** int_bits - step

        # Clamp, round to grid, reconstruct
        w_clamped = w.clamp(q_min, q_max)
        w_q = torch.round(w_clamped / step) * step
        return w_q

    def quantize_fixed_point(self, w, bits, frac_bits):
        """Public API for fixed-point quantization."""
        return self._quantize(w, bits, frac_bits)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Fake-quantize x using current_bits with optimal fractional bits.

        Uses STE: forward uses quantized values, backward is identity.
        """
        if self.current_bits >= 16:
            return x  # effectively full-precision, skip

        # Find optimal fractional bits
        f = self.find_best_fractional(x)
        self._fractional_bits.fill_(f)

        # Quantize
        x_q = self._quantize(x, self.current_bits, f)

        # STE: forward = quantized, backward = identity
        return x + (x_q - x).detach()

    # ------------------------------------------------------------------
    # Bit-width control (called by AdaPTPrecisionController)
    # ------------------------------------------------------------------

    def reduce_bits(self):
        """Reduce bit-width by 1, down to min_bits.

        Returns:
            True if bits were actually reduced.
        """
        if self.current_bits > self.min_bits:
            self.current_bits -= 1
            return True
        return False

    def disable_observer(self):
        """Freeze observer (compatibility with trainer patterns)."""
        self.observer_enabled.fill_(0)

    def extra_repr(self):
        return (f"bits={self.current_bits}, min_bits={self.min_bits}, "
                f"frac={self._fractional_bits.item()}, "
                f"is_weight={self.is_weight}")
