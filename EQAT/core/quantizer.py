"""
EQAT Quantizer — exactly following the paper (Eq. 13, 16-18).

Quantization range: symmetric [-1, 1] for weights (Eq. 18 basis).
Step size:  Δ = 2 / (2^q - 1)
STE grad w.r.t. x: identity (pass-through).
STE grad w.r.t. q: analytical via ∂Δ/∂q = -2·2^q·ln2 / (2^q-1)²  [Eq. 18]

Both weights AND activations are quantized (Algorithm 1: ŵ and ẑ).
Activations use PACT clipping [0, alpha] after ReLU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Weight quantization ────────────────────────────────────────────────────

class _FakeQuantW(torch.autograd.Function):
    """
    Symmetric weight quantization in [-1, 1].
    Δ = 2 / (2^q - 1)  →  ∂Δ/∂q = -2·2^q·ln2 / (2^q-1)²   [Eq. 18]
    """
    @staticmethod
    def forward(ctx, w, q):
        q_r = max(2.0, q.item())
        n   = 2.0 ** q_r - 1.0          # number of positive levels
        scale = 2.0 / n                  # step size Δ
        w_c   = w.clamp(-1.0, 1.0)
        w_q   = (w_c / scale).round() * scale
        ctx.save_for_backward(w, q, torch.tensor(q_r))
        return w_q

    @staticmethod
    def backward(ctx, g):
        w, q, q_r_t = ctx.saved_tensors
        q_r = q_r_t.item()

        # STE for weights: identity inside [-1,1], zero outside
        gw = g.clone()
        gw[w.abs() > 1.0] = 0.0

        # Gradient for q  (Eq. 17-18)
        pow2q     = 2.0 ** q_r
        n         = pow2q - 1.0
        scale     = 2.0 / n
        d_scale_q = -(2.0 * pow2q * math.log(2.0)) / (n ** 2)
        levels    = (w.clamp(-1.0, 1.0) / scale).round()
        gq        = (g * levels * d_scale_q).sum()    # sum over all elements (Eq. 17-18)

        return gw, gq


def fake_quant_weight(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Quantize weight tensor w to q bits, range [-1, 1]."""
    return _FakeQuantW.apply(w, q)


# ── Activation quantization (PACT) ────────────────────────────────────────

class _FakeQuantAct(torch.autograd.Function):
    """
    PACT activation quantization for [0, alpha] range (post-ReLU outputs).
    Δ_act = alpha / (2^q - 1)  →  ∂Δ_act/∂q = -alpha·2^q·ln2/(2^q-1)²
    """
    @staticmethod
    def forward(ctx, x, q, alpha):
        q_r  = max(2.0, q.item())
        a    = alpha.item()
        n    = 2.0 ** q_r - 1.0
        s    = a / n if a > 1e-8 else 1.0
        x_c  = x.clamp(0.0, a)
        x_q  = (x_c / s).round() * s
        ctx.save_for_backward(x, q, alpha, torch.tensor(q_r))
        return x_q

    @staticmethod
    def backward(ctx, g):
        x, q, alpha, q_r_t = ctx.saved_tensors
        q_r = q_r_t.item()
        a   = alpha.item()

        # STE for x: pass-through in [0, alpha]
        gx = g.clone()
        gx[x < 0]  = 0.0
        gx[x > a]  = 0.0

        # PACT alpha gradient: gradient at clipped region
        g_alpha = g[x > a].sum().reshape_as(alpha) if (x > a).any() \
                  else torch.zeros_like(alpha)

        # Gradient for q
        if a < 1e-8:
            return gx, torch.zeros_like(q), g_alpha
        pow2q     = 2.0 ** q_r
        n         = pow2q - 1.0
        s         = a / n
        d_scale_q = -(a * pow2q * math.log(2.0)) / (n ** 2)
        mask      = (x >= 0) & (x <= a)
        levels    = (x.clamp(0.0, a) / s).round()
        gq = (g[mask] * levels[mask] * d_scale_q).sum() if mask.any() \
             else torch.zeros_like(q)

        return gx, gq.reshape_as(q), g_alpha


# ── EQAT layer wrappers ───────────────────────────────────────────────────

class EQATConvBlock(nn.Module):
    """
    Conv2d + BN + ReLU with:
      - per-layer learnable bit-width q̃  (Eq. 13)
      - weight quantization (Eq. 16)
      - PACT activation quantization (ẑ = Quantize(z, q))
    """

    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1,
                 q_min=2.0, q_max=8.0):
        super().__init__()
        self.conv   = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn     = nn.BatchNorm2d(out_c)
        self.q_min  = q_min
        self.q_max  = q_max
        # q̃ initialised at 1.5  →  q = 2 + 6·σ(1.5) ≈ 6.7 bits
        self.q_tilde = nn.Parameter(torch.tensor(1.5))
        # PACT clipping parameter for activations
        self.alpha   = nn.Parameter(torch.tensor(6.0))

    def get_q(self) -> torch.Tensor:
        """Returns effective bit-width q_i ∈ [q_min, q_max].  [Eq. 13]"""
        return self.q_min + (self.q_max - self.q_min) * torch.sigmoid(self.q_tilde)

    def get_bitwidth(self) -> float:
        return self.get_q().item()

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        q = self.get_q()

        if quantize:
            # Normalise weights to [-1,1] then fake-quantize
            w_norm = self.conv.weight / (self.conv.weight.abs().max() + 1e-8)
            w_q    = fake_quant_weight(w_norm, q)
            # Rescale back
            w_q    = w_q * (self.conv.weight.abs().max() + 1e-8)
        else:
            w_q = self.conv.weight

        out = F.conv2d(x, w_q, None,
                       self.conv.stride, self.conv.padding)
        out = self.bn(out)
        out = F.relu(out)

        if quantize:
            out = _FakeQuantAct.apply(out, q, self.alpha.abs())

        return out

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size


class EQATLinear(nn.Module):
    """
    Linear layer with per-layer learnable bit-width.
    Optionally applies ReLU + PACT activation quantization.
    """

    def __init__(self, in_f, out_f, activate=True,
                 q_min=2.0, q_max=8.0):
        super().__init__()
        self.fc      = nn.Linear(in_f, out_f)
        self.activate = activate
        self.q_min   = q_min
        self.q_max   = q_max
        self.q_tilde = nn.Parameter(torch.tensor(1.5))
        self.alpha   = nn.Parameter(torch.tensor(6.0))

    def get_q(self) -> torch.Tensor:
        return self.q_min + (self.q_max - self.q_min) * torch.sigmoid(self.q_tilde)

    def get_bitwidth(self) -> float:
        return self.get_q().item()

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        q = self.get_q()

        if quantize:
            w_norm = self.fc.weight / (self.fc.weight.abs().max() + 1e-8)
            w_q    = fake_quant_weight(w_norm, q)
            w_q    = w_q * (self.fc.weight.abs().max() + 1e-8)
        else:
            w_q = self.fc.weight

        out = F.linear(x, w_q, self.fc.bias)

        if self.activate:
            out = F.relu(out)
            if quantize:
                out = _FakeQuantAct.apply(out, q, self.alpha.abs())

        return out
