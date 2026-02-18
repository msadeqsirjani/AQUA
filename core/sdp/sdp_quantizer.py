"""
Structured Dynamic Precision (SDP) quantizer.
ACM Transactions on Design Automation of Electronic Systems, 2022.
DOI: https://dl.acm.org/doi/10.1145/3549535

Core idea: represent each N-bit value as HIGH (top M bits, always computed)
+ LOW (bottom L bits, computed only for important elements in each group).
Importance is determined by magnitude-based top-K within structured groups.

Effective bits: eff = M + (1 - sparsity) * L
Example: N=8, M=4, sparsity=0.5 => eff = 4 + 0.5*4 = 6 bits average.
"""

import torch
import torch.nn as nn


class SDPQuantizer(nn.Module):
    """Structured Dynamic Precision fake quantizer.

    Quantizes to N total bits, but zeros out the low-order L = N - M bits
    for elements whose magnitude falls below the top-K threshold in each
    group of size G. This simulates the hardware behaviour where only
    important elements use full precision.

    Args:
        total_bits: Total quantization bit-width N (default: 8).
        high_bits: High-order bits M always computed (default: 4).
        group_size: Number of elements per importance group G (default: 8).
        sparsity: Fraction of elements that keep full precision (default: 0.5).
        is_weight: If True use symmetric quant; if False use asymmetric + EMA.
        ema_momentum: EMA momentum for activation range tracking (default: 0.99).
    """

    def __init__(self, total_bits=8, high_bits=4, group_size=8, sparsity=0.5,
                 is_weight=False, ema_momentum=0.99):
        super().__init__()
        assert high_bits < total_bits, "high_bits must be < total_bits"
        assert 0.0 < sparsity <= 1.0, "sparsity must be in (0, 1]"

        self.total_bits = total_bits      # N
        self.high_bits = high_bits        # M
        self.low_bits = total_bits - high_bits  # L
        self.group_size = group_size      # G
        self.sparsity = sparsity          # fraction kept at full precision
        self.is_weight = is_weight
        self.ema_momentum = ema_momentum

        # Symmetric quantization range for N-bit
        self.q_min = -(2 ** (total_bits - 1))      # -128 for 8-bit
        self.q_max = 2 ** (total_bits - 1) - 1      # 127  for 8-bit

        # Scale/zero-point for the N-bit quantizer
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self.register_buffer("observer_enabled", torch.tensor(1, dtype=torch.int32))

        # Tracking: fraction of elements that actually kept low bits
        self.register_buffer("_last_mask_ratio", torch.tensor(0.0))

    @property
    def effective_bits(self):
        """Average effective bits per element."""
        return self.high_bits + self.sparsity * self.low_bits

    # ------------------------------------------------------------------
    # Scale / observer logic (same as Jacob fake-quant)
    # ------------------------------------------------------------------

    def _update_scale(self, x):
        if self.is_weight:
            r_min = x.detach().min().item()
            r_max = x.detach().max().item()
        else:
            batch_min = x.detach().min().item()
            batch_max = x.detach().max().item()
            if self.running_min.item() == float("inf"):
                self.running_min.fill_(batch_min)
                self.running_max.fill_(batch_max)
            else:
                self.running_min.mul_(self.ema_momentum).add_(
                    batch_min * (1 - self.ema_momentum))
                self.running_max.mul_(self.ema_momentum).add_(
                    batch_max * (1 - self.ema_momentum))
            r_min = self.running_min.item()
            r_max = self.running_max.item()

        r_min = min(r_min, 0.0)
        r_max = max(r_max, 0.0)
        if r_max == r_min:
            return
        scale = (r_max - r_min) / (self.q_max - self.q_min)
        self.scale.fill_(max(scale, 1e-8))
        self.zero_point.fill_(0)  # symmetric

    def disable_observer(self):
        self.observer_enabled.fill_(0)

    # ------------------------------------------------------------------
    # Core SDP operations
    # ------------------------------------------------------------------

    @staticmethod
    def split_high_low(x_int, low_bits):
        """Split integer tensor into high and low parts.

        x_high = x_int >> L  (arithmetic right-shift, preserves sign for high bits)
        x_low  = x_int & ((1 << L) - 1)  (mask bottom L bits — unsigned magnitude)

        For signed integers we operate on the absolute value for the mask
        and reconstruct with sign.
        """
        L = low_bits
        # Work on absolute values for clean bit manipulation
        sign = x_int.sign()
        mag = x_int.abs()
        # Use integer division/modulo to simulate bit ops on float tensors
        divisor = 2 ** L
        x_high = (mag / divisor).floor()   # top M bits of magnitude
        x_low = mag - x_high * divisor     # bottom L bits of magnitude
        return sign, x_high, x_low

    @staticmethod
    def compute_importance_mask(x_int, group_size, sparsity):
        """Compute binary importance mask via per-group magnitude top-K.

        Divides the flattened tensor into groups of size G.
        Within each group, the top round(G * sparsity) elements by
        absolute value get mask=1 (keep low bits); the rest get mask=0.

        Args:
            x_int: Integer quantized tensor (any shape).
            group_size: Group size G.
            sparsity: Fraction of elements kept at full precision.

        Returns:
            Binary mask tensor, same shape as x_int.
        """
        orig_shape = x_int.shape
        flat = x_int.detach().abs().reshape(-1).float()
        n = flat.numel()

        # Pad to multiple of group_size
        remainder = n % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            flat = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)
        else:
            pad_size = 0

        # Reshape into groups
        n_padded = flat.numel()
        groups = flat.reshape(-1, group_size)  # (n_groups, G)

        # Top-K per group
        K = max(1, round(group_size * sparsity))
        # Get the K-th largest value as threshold per group
        topk_vals, _ = groups.topk(K, dim=1)          # (n_groups, K)
        thresholds = topk_vals[:, -1].unsqueeze(1)     # (n_groups, 1)

        # Mask: 1 where |x| >= threshold
        mask = (groups >= thresholds).float()

        # Flatten back and remove padding
        mask = mask.reshape(-1)
        if pad_size > 0:
            mask = mask[:n]
        mask = mask.reshape(orig_shape)

        return mask

    def forward(self, x):
        """Full SDP forward pass with STE.

        Steps:
          1. Quantize x to N-bit integer representation
          2. Split into HIGH (M bits) and LOW (L bits)
          3. Compute importance mask via per-group top-K
          4. Zero out LOW bits for unimportant elements
          5. Reconstruct and dequantize
          6. STE: gradient passes through as identity
        """
        # Update scale if observer is active
        if self.observer_enabled.item():
            self._update_scale(x)

        S = self.scale
        Z = self.zero_point.float()
        L = self.low_bits
        divisor = 2 ** L

        # Step 1: Quantize to N-bit integer
        x_int = torch.clamp(torch.round(x / S) + Z, self.q_min, self.q_max)

        # Step 2: Split into HIGH and LOW
        sign, high, low = self.split_high_low(x_int, L)

        # Step 3: Compute importance mask (no grad)
        mask = self.compute_importance_mask(x_int, self.group_size, self.sparsity)

        # Track mask statistics
        self._last_mask_ratio.fill_(mask.mean().item())

        # Step 4: Zero out LOW bits for unimportant elements
        low_masked = low * mask

        # Step 5: Reconstruct integer and dequantize
        x_int_sdp = sign * (high * divisor + low_masked)
        x_sdp = S * (x_int_sdp - Z)

        # Step 6: STE — forward uses SDP result, backward sees identity
        return x + (x_sdp - x).detach()

    @property
    def last_mask_ratio(self):
        """Fraction of elements that kept full precision in last forward."""
        return self._last_mask_ratio.item()

    def extra_repr(self):
        return (f"N={self.total_bits}, M={self.high_bits}, L={self.low_bits}, "
                f"G={self.group_size}, sparsity={self.sparsity}, "
                f"eff_bits={self.effective_bits:.1f}, "
                f"is_weight={self.is_weight}")
