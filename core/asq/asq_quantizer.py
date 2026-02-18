"""
ASQ and POST quantizers from Zhou et al. 2025
"Precision Neural Network Quantization via Learnable Adaptive Modules"
(arXiv:2504.17263).

Two core contributions:
  1. ASQActivationQuantizer — input-dependent adaptive step size
  2. POSTWeightQuantizer — Power Of Square root of Two for weights
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ASQ: Adaptive Step Size Quantization (Section 3)
# ---------------------------------------------------------------------------

class ASQActivationQuantizer(nn.Module):
    """Adaptive Step Size Quantization for activations.

    A tiny MLP takes per-channel statistics (mean, std, max) of the input
    and outputs a scale adjustment delta_s. The effective quantization
    scale is  s_eff = s_base * (1 + delta_s),  making the step size
    input-dependent while keeping overhead minimal.

    Architecture (Option B from paper):
        Linear(3*C, 16) -> ReLU -> Linear(16, 1) -> Tanh * bound

    Args:
        num_channels: Number of input channels C.
        bits: Quantization bit-width (default: 4).
        delta_scale_bound: Max relative adjustment ±bound (default: 0.5).
    """

    def __init__(self, num_channels, bits=4, delta_scale_bound=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.bits = bits
        self.delta_scale_bound = delta_scale_bound
        self.enabled = True  # set False to bypass quantization (e.g. calibration)

        # Unsigned activation range: [0, 2^b - 1]
        self.q_min = 0
        self.q_max = 2 ** bits - 1

        # Learnable base scale (LSQ-style, scalar shared across channels)
        self.s_base = nn.Parameter(torch.tensor(1.0))

        # MLP: 3C features -> 16 hidden -> 1 output
        self.mlp = nn.Sequential(
            nn.Linear(3 * num_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        # Track whether MLP is frozen (for warmup phase)
        self._mlp_frozen = False

    def init_scale(self, x):
        """Initialize s_base from calibration data.

        s_base = 2 * mean(|x|) / sqrt(q_max)  (LSQ heuristic)
        """
        with torch.no_grad():
            s_init = 2.0 * x.detach().abs().mean() / math.sqrt(self.q_max)
            self.s_base.data.fill_(max(s_init.item(), 1e-8))

    def freeze_mlp(self):
        """Freeze MLP parameters (for warmup: train s_base only)."""
        self._mlp_frozen = True
        for p in self.mlp.parameters():
            p.requires_grad_(False)

    def unfreeze_mlp(self):
        """Unfreeze MLP parameters after warmup."""
        self._mlp_frozen = False
        for p in self.mlp.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        if not self.enabled:
            return x
        # Handle both 4D (conv) and 2D (linear) inputs
        is_4d = x.dim() == 4
        B = x.size(0)
        C = x.size(1)

        if is_4d:
            # Step 1: Compute per-channel statistics over spatial dims
            # x shape: (B, C, H, W)
            x_mean = x.mean(dim=[2, 3])          # (B, C)
            x_std = x.std(dim=[2, 3])             # (B, C)
            x_max = x.amax(dim=[2, 3])            # (B, C)
        else:
            # For 2D tensors (B, C), treat as (B, C) directly
            x_mean = x                            # (B, C)
            x_std = torch.zeros_like(x_mean)      # no spatial dim
            x_max = x.abs()                       # (B, C)

        # Step 1: Concatenate features -> (B, 3C)
        features = torch.cat([x_mean, x_std, x_max], dim=1)  # (B, 3C)

        # Step 2: MLP produces delta_s
        # delta_s shape: (B, 1) — same adjustment for all channels
        delta_s = self.mlp(features)                          # (B, 1)
        delta_s = torch.tanh(delta_s) * self.delta_scale_bound  # (B, 1)

        # Step 3: Effective per-input scale
        s_eff = self.s_base.abs() * (1.0 + delta_s)          # (B, 1)
        s_eff = s_eff.clamp(min=1e-8)

        # Step 4: Quantize using LSQ formula with s_eff
        if is_4d:
            s = s_eff.unsqueeze(-1).unsqueeze(-1)             # (B, 1, 1, 1)
        else:
            s = s_eff                                          # (B, 1)

        x_scaled = x / s
        x_clipped = x_scaled.clamp(self.q_min, self.q_max)
        # STE round: forward rounds, backward is identity
        x_rounded = x_clipped + (x_clipped.round() - x_clipped).detach()
        x_dq = x_rounded * s

        return x_dq

    def extra_repr(self):
        return (f"channels={self.num_channels}, bits={self.bits}, "
                f"bound={self.delta_scale_bound}, "
                f"s_base={self.s_base.item():.6f}")


# ---------------------------------------------------------------------------
# POST: Power Of Square root of Two weight quantizer (Section 4)
# ---------------------------------------------------------------------------

class POSTWeightQuantizer(nn.Module):
    """POST (Power Of Square root of Two) weight quantizer.

    Quantizes weights to grid points (sqrt(2))^k for integer k.
    This gives 2x denser grid near zero compared to standard POT (2^k),
    improving representational efficiency at low bit-widths.

    At training: differentiable fake-quantization in log domain with STE.
    At inference: LUT of precomputed grid values.

    Args:
        bits: Quantization bit-width (default: 4).
    """

    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits

        # Signed quantization: 1 sign bit + (bits-1) magnitude bits
        # Number of positive levels = 2^(bits-1) - 1  (exclude zero)
        # k ranges from k_min to k_max with step 1
        n_positive = 2 ** (bits - 1) - 1
        # Center the grid: k goes from -(n_positive//2) to +(n_positive - n_positive//2 - 1)
        self.k_min = -(n_positive // 2)
        self.k_max = self.k_min + n_positive - 1

        # Precompute LUT: all grid values (sqrt(2))^k
        self.log_base = math.log(2.0) / 2.0  # log(sqrt(2))
        lut_k = torch.arange(self.k_min, self.k_max + 1, dtype=torch.float32)
        lut_values = torch.pow(math.sqrt(2.0), lut_k)
        self.register_buffer("lut_values", lut_values)    # (n_positive,)
        self.register_buffer("lut_k", lut_k)              # (n_positive,)

        # Learnable scale for the overall weight magnitude
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.enabled = True  # set False to bypass quantization (e.g. calibration)

    def forward(self, w):
        """Quantize weight tensor w using POST grid.

        1. Scale weights: w_s = w / scale
        2. Extract sign
        3. Map |w_s| to nearest POST grid point in log domain
        4. Reconstruct: w_q = sign * grid_value * scale

        Gradient strategy: full STE on the normalised weight (w/scale).
        The log-exp chain is computed without gradient (detached) to avoid
        the extreme mag_q/mag gain ratio that causes NaN for near-zero
        weights.  After the STE, scale is multiplied back so it receives
        a gradient proportional to the quantization error (same principle
        as LSQ).
        """
        if not self.enabled:
            return w
        s = self.scale.abs().clamp(min=1e-8)
        w_s = w / s

        # Compute quantized normalised weight entirely without gradient
        with torch.no_grad():
            sign = w_s.sign()
            mag = w_s.abs().clamp(min=1e-10)  # avoid log(0)

            # Quantize in log domain: k = round(log(mag) / log(sqrt(2)))
            log_mag = torch.log(mag) / self.log_base
            k = log_mag.clamp(float(self.k_min), float(self.k_max)).round()

            # Reconstruct magnitude from POST grid: exp(k * log(sqrt(2)))
            mag_q = torch.exp(k * self.log_base)
            w_q_s = sign * mag_q  # quantised w/s (detached)

        # STE in the normalised domain: forward = w_q_s, backward = w_s
        # Gradient for w: ∂output/∂w = s * 1 * (1/s) = 1 (identity STE)
        # Gradient for scale: ∂output/∂s = w_q_s − w/s  (quantisation error)
        w_s_out = w_s + (w_q_s - w_s).detach()

        # Multiply back by scale (scale gets gradient naturally)
        w_q = w_s_out * s

        # Handle near-zero weights
        zero_mask = (w.detach().abs() < 1e-10)
        w_q = w_q.masked_fill(zero_mask, 0.0)

        return w_q

    def init_scale(self, w):
        """Initialize scale from weight tensor statistics."""
        with torch.no_grad():
            # Set scale so the weight range maps to the middle of the LUT
            mid_val = self.lut_values[len(self.lut_values) // 2].item()
            w_std = w.std().item()
            if w_std > 0 and mid_val > 0:
                self.scale.data.fill_(w_std / mid_val)
            else:
                self.scale.data.fill_(1.0)

    def extra_repr(self):
        return (f"bits={self.bits}, k_range=[{self.k_min}, {self.k_max}], "
                f"n_levels={len(self.lut_values)}, scale={self.scale.item():.6f}")


# ---------------------------------------------------------------------------
# LSQ baseline quantizer (for comparison)
# ---------------------------------------------------------------------------

class LSQQuantizer(nn.Module):
    """Learned Step Size Quantization (LSQ) baseline.

    Standard fixed-scale LSQ with a single learnable step size s.
    No input-dependent adaptation (unlike ASQ).
    """

    def __init__(self, bits=4, is_weight=False):
        super().__init__()
        self.bits = bits
        self.is_weight = is_weight

        if is_weight:
            self.q_min = -(2 ** (bits - 1))
            self.q_max = 2 ** (bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** bits - 1

        self.s = nn.Parameter(torch.tensor(1.0))
        self.enabled = True  # set False to bypass quantization (e.g. calibration)

    def init_scale(self, x):
        with torch.no_grad():
            if self.is_weight:
                s_init = 2.0 * x.abs().mean() / math.sqrt(self.q_max)
            else:
                s_init = 2.0 * x.abs().mean() / math.sqrt(self.q_max)
            self.s.data.fill_(max(s_init.item(), 1e-8))

    def forward(self, x):
        if not self.enabled:
            return x
        s = self.s.abs().clamp(min=1e-8)
        x_scaled = x / s
        x_clipped = x_scaled.clamp(self.q_min, self.q_max)
        x_rounded = x_clipped + (x_clipped.round() - x_clipped).detach()
        return x_rounded * s

    def extra_repr(self):
        return f"bits={self.bits}, is_weight={self.is_weight}, s={self.s.item():.6f}"
