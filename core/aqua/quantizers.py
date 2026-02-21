"""
AQUA quantizers and dtype catalog.

Provides fixed-bit-width quantizers for each supported dtype in the
AQUA categorical selection framework:

    fp32  -- identity (no quantization)
    fp16  -- IEEE half-precision rounding via STE
    int8  -- symmetric uniform 8-bit
    fp8   -- E3M4 floating-point 8-bit
    int4  -- symmetric uniform 4-bit
    fp4   -- E3M0 floating-point 4-bit
    int2  -- symmetric uniform 2-bit
    int1  -- binary sign quantization (XNOR-Net style)

Each quantizer has a ``forward(x)`` that fake-quantizes a weight tensor
and an ``init_scale(w)`` for calibration.

``DTYPE_CATALOG`` maps dtype name -> metadata (bits, class, energy type).
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Legacy continuous-bit-width quantizers (kept for reference / testing)
# ---------------------------------------------------------------------------

class IntQuantizer(nn.Module):
    """Uniform (integer) fake quantizer with learned step size.

    Quantizes using a symmetric grid: [-2^(b-1), 2^(b-1)-1] with a
    learnable scale *s* (LSQ-style).  STE is applied to the rounding
    operation so gradients flow through to both *s* and the bit-width.
    """

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(1.0))

    def init_scale(self, w):
        with torch.no_grad():
            self.s.data.fill_(max(2.0 * w.abs().mean().item() / math.sqrt(127), 1e-8))

    @staticmethod
    def _quant_fixed(x_s, bits_int):
        n = 2 ** bits_int
        q_min = -(n // 2)
        q_max = n // 2 - 1
        x_c = x_s.clamp(q_min, q_max)
        return x_c + (x_c.round() - x_c).detach()

    def forward(self, x, bits):
        s = self.s.abs().clamp(min=1e-8)
        x_s = x / s
        b_lo = max(int(bits.detach().floor().item()), 2)
        b_hi = b_lo + 1
        frac = bits - b_lo
        q_lo = self._quant_fixed(x_s, b_lo)
        q_hi = self._quant_fixed(x_s, b_hi)
        return ((1.0 - frac) * q_lo + frac * q_hi) * s


class FpQuantizer(nn.Module):
    """Floating-point fake quantizer with E3M{b-4} format."""

    E_BITS = 3

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(1.0))

    def init_scale(self, w):
        with torch.no_grad():
            self.s.data.fill_(max(w.abs().max().item() / 4.0, 1e-8))

    def _quant_fixed(self, x_s, bits_int):
        bits_int = max(bits_int, 4)
        e_bits = self.E_BITS
        m_bits = bits_int - 1 - e_bits

        bias = 2 ** (e_bits - 1) - 1
        emax = 2 ** e_bits - 2 - bias
        emin = 1 - bias

        sign = x_s.sign()
        mag = x_s.abs()
        eps = 1e-10
        mag_safe = mag.clamp(min=eps)

        log2_val = torch.log2(mag_safe)
        e = torch.floor(log2_val).clamp(emin, emax)
        pow2e = torch.pow(2.0, e)

        mantissa = (mag_safe / pow2e - 1.0).clamp(0.0, 1.0 - eps)

        if m_bits > 0:
            n_m = 2 ** m_bits
            mq = (mantissa * n_m).round().clamp(0, n_m - 1) / n_m
        else:
            mq = torch.zeros_like(mantissa)

        mag_q = pow2e * (1.0 + mq)
        max_rep = 2.0 ** emax * (2.0 - 2.0 ** (-max(m_bits, 0)))
        mag_q = mag_q.clamp(0, max_rep)

        x_q = sign * mag_q
        x_q = x_q.masked_fill(mag < eps, 0.0)
        return x_s + (x_q - x_s).detach()

    def forward(self, x, bits):
        s = self.s.abs().clamp(min=1e-8)
        x_s = x / s
        bits_fp = bits.clamp(min=4.0)
        b_lo = int(bits_fp.detach().floor().item())
        b_hi = b_lo + 1
        frac = bits_fp - b_lo
        q_lo = self._quant_fixed(x_s, b_lo)
        q_hi = self._quant_fixed(x_s, b_hi)
        return ((1.0 - frac) * q_lo + frac * q_hi) * s


# ---------------------------------------------------------------------------
# Fixed-bit-width quantizers for the categorical dtype selection
# ---------------------------------------------------------------------------

class FP32Quantizer(nn.Module):
    """Identity quantizer -- keeps weights in full precision."""

    def init_scale(self, w):
        pass

    def forward(self, x):
        return x


class FP16Quantizer(nn.Module):
    """IEEE half-precision rounding with STE.

    Simulates fp16 dynamic range and precision limits while keeping
    the computation in fp32.  Gradient flows through via STE.
    """

    FP16_MAX = 65504.0
    # Smallest positive normal in fp16: 2^-14
    FP16_MIN_NORMAL = 2 ** -14
    # fp16 has 10 mantissa bits -> unit-in-last-place at magnitude 1 is 2^-10
    FP16_EPS = 2 ** -10

    def init_scale(self, w):
        pass

    def forward(self, x):
        x_clamped = x.clamp(-self.FP16_MAX, self.FP16_MAX)
        x_fp16 = x_clamped.half().float()
        # STE: forward = fp16-rounded, backward = identity
        return x + (x_fp16 - x).detach()


class FixedIntQuantizer(nn.Module):
    """Symmetric uniform fake quantizer at a fixed integer bit-width."""

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.s = nn.Parameter(torch.tensor(1.0))

    def init_scale(self, w):
        with torch.no_grad():
            n = 2 ** self.bits
            self.s.data.fill_(max(2.0 * w.abs().mean().item() / math.sqrt(n - 1), 1e-8))

    def forward(self, x):
        s = self.s.abs().clamp(min=1e-8)
        x_s = x / s
        n = 2 ** self.bits
        q_min = -(n // 2)
        q_max = n // 2 - 1
        x_c = x_s.clamp(q_min, q_max)
        x_q = x_c + (x_c.round() - x_c).detach()
        return x_q * s


class FixedFpQuantizer(nn.Module):
    """E3M{b-4} floating-point fake quantizer at a fixed bit-width."""

    E_BITS = 3

    def __init__(self, bits):
        super().__init__()
        self.bits = max(bits, 4)
        self.s = nn.Parameter(torch.tensor(1.0))

    def init_scale(self, w):
        with torch.no_grad():
            self.s.data.fill_(max(w.abs().max().item() / 4.0, 1e-8))

    def forward(self, x):
        s = self.s.abs().clamp(min=1e-8)
        x_s = x / s

        e_bits = self.E_BITS
        m_bits = self.bits - 1 - e_bits

        bias = 2 ** (e_bits - 1) - 1
        emax = 2 ** e_bits - 2 - bias
        emin = 1 - bias

        sign = x_s.sign()
        mag = x_s.abs()
        eps = 1e-10
        mag_safe = mag.clamp(min=eps)

        log2_val = torch.log2(mag_safe)
        e = torch.floor(log2_val).clamp(emin, emax)
        pow2e = torch.pow(2.0, e)

        mantissa = (mag_safe / pow2e - 1.0).clamp(0.0, 1.0 - eps)

        if m_bits > 0:
            n_m = 2 ** m_bits
            mq = (mantissa * n_m).round().clamp(0, n_m - 1) / n_m
        else:
            mq = torch.zeros_like(mantissa)

        mag_q = pow2e * (1.0 + mq)
        max_rep = 2.0 ** emax * (2.0 - 2.0 ** (-max(m_bits, 0)))
        mag_q = mag_q.clamp(0, max_rep)

        x_q = sign * mag_q
        x_q = x_q.masked_fill(mag < eps, 0.0)
        # STE
        return (x_s + (x_q - x_s).detach()) * s


class BinaryQuantizer(nn.Module):
    """1-bit sign quantization with learned scale (XNOR-Net style).

    Forward: ``sign(x) * alpha`` where alpha is a learned positive scalar
    initialized from the mean absolute weight value.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def init_scale(self, w):
        with torch.no_grad():
            self.alpha.data.fill_(max(w.abs().mean().item(), 1e-8))

    def forward(self, x):
        alpha = self.alpha.abs().clamp(min=1e-8)
        # STE sign: forward = sign, backward = identity (clipped)
        x_sign = x.sign()
        x_sign = x + (x_sign - x).detach()
        return x_sign * alpha


# ---------------------------------------------------------------------------
# Dtype catalog
# ---------------------------------------------------------------------------

DTYPE_CATALOG = OrderedDict([
    ("fp32", {"bits": 32, "quantizer_cls": FP32Quantizer,    "energy_type": "fp",  "args": {}}),
    ("fp16", {"bits": 16, "quantizer_cls": FP16Quantizer,    "energy_type": "fp",  "args": {}}),
    ("int8", {"bits": 8,  "quantizer_cls": FixedIntQuantizer, "energy_type": "int", "args": {"bits": 8}}),
    ("fp8",  {"bits": 8,  "quantizer_cls": FixedFpQuantizer,  "energy_type": "fp",  "args": {"bits": 8}}),
    ("int4", {"bits": 4,  "quantizer_cls": FixedIntQuantizer, "energy_type": "int", "args": {"bits": 4}}),
    ("fp4",  {"bits": 4,  "quantizer_cls": FixedFpQuantizer,  "energy_type": "fp",  "args": {"bits": 4}}),
    ("int2", {"bits": 2,  "quantizer_cls": FixedIntQuantizer, "energy_type": "int", "args": {"bits": 2}}),
    ("int1", {"bits": 1,  "quantizer_cls": BinaryQuantizer,   "energy_type": "int", "args": {}}),
])

ALL_DTYPE_NAMES = list(DTYPE_CATALOG.keys())


def resolve_dtypes(allowed=None, blocked=None):
    """Return an ordered list of dtype names after applying allow/block filters.

    Args:
        allowed: list of dtype names to allow (None = all).
        blocked: list of dtype names to exclude (takes priority over allowed).

    Returns:
        List of dtype name strings in catalog order.
    """
    if allowed is not None and len(allowed) > 0:
        names = [n for n in ALL_DTYPE_NAMES if n in allowed]
    else:
        names = list(ALL_DTYPE_NAMES)

    if blocked:
        names = [n for n in names if n not in blocked]

    if len(names) == 0:
        raise ValueError("No dtypes remaining after applying allowed/blocked filters")
    return names


def make_quantizer_bank(dtype_names, weight_tensor=None):
    """Create an nn.ModuleList of quantizers for the given dtype names.

    Args:
        dtype_names: ordered list of dtype name strings.
        weight_tensor: optional weight tensor for scale initialization.

    Returns:
        (nn.ModuleList, list of bit-widths, list of energy_type strings)
    """
    quantizers = nn.ModuleList()
    bits_list = []
    energy_types = []
    for name in dtype_names:
        entry = DTYPE_CATALOG[name]
        q = entry["quantizer_cls"](**entry["args"])
        if weight_tensor is not None:
            q.init_scale(weight_tensor)
        quantizers.append(q)
        bits_list.append(entry["bits"])
        energy_types.append(entry["energy_type"])
    return quantizers, bits_list, energy_types
