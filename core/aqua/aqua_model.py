"""
AQUA quantized layers — fixed analytical dtype assignment + outlier protection.

Each ``AQUAConv2d`` / ``AQUALinear`` wraps a standard layer and has:

* A **fixed quantizer** (assigned analytically, not learned)
* An **outlier mask** — top sensitive weights are stored in fp16
* A learnable quantizer **scale** parameter for QAT fine-tuning

During forward::

    w = weight.clone()
    w[outlier_mask] = 0               # zero outliers before quantization
    w_q = quantizer(w)                # quantize the "normal" weights
    w_q[outlier_mask] = outlier_fp16  # restore outlier values in fp16
    output = conv2d / linear(x, w_q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import DTYPE_CATALOG


def _make_single_quantizer(dtype_name, weight_tensor=None):
    """Instantiate a single quantizer from the catalog.

    Returns the quantizer module with scale initialized from weight_tensor.
    """
    entry = DTYPE_CATALOG[dtype_name]
    q = entry["quantizer_cls"](**entry["args"])
    if weight_tensor is not None:
        q.init_scale(weight_tensor)
    return q


class AQUAConv2d(nn.Conv2d):
    """Conv2d with fixed AQUA quantization + outlier protection."""

    def __init__(self, original, dtype_name, outlier_mask=None):
        super().__init__(
            original.in_channels, original.out_channels, original.kernel_size,
            stride=original.stride, padding=original.padding,
            dilation=original.dilation, groups=original.groups,
            bias=original.bias is not None,
        )
        self.weight.data.copy_(original.weight.data)
        if original.bias is not None:
            self.bias.data.copy_(original.bias.data)

        self.dtype_name = dtype_name
        self.dtype_bits = DTYPE_CATALOG[dtype_name]["bits"]

        w_for_init = original.weight.data
        if outlier_mask is not None:
            w_for_init = w_for_init.clone()
            w_for_init[outlier_mask] = 0.0
        self.quantizer = _make_single_quantizer(dtype_name, w_for_init)

        if outlier_mask is not None:
            self.register_buffer("outlier_mask", outlier_mask.bool())
            self.register_buffer(
                "outlier_values",
                original.weight.data[outlier_mask].half(),
            )
        else:
            self.register_buffer("outlier_mask", None)
            self.register_buffer("outlier_values", None)

    def forward(self, x):
        w = self.weight

        if self.outlier_mask is not None:
            w = w.clone()
            w[self.outlier_mask] = 0.0

        w_q = self.quantizer(w)

        if self.outlier_mask is not None:
            w_q = w_q.clone()
            w_q[self.outlier_mask] = self.outlier_values.float()

        return F.conv2d(x, w_q, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class AQUALinear(nn.Linear):
    """Linear with fixed AQUA quantization + outlier protection."""

    def __init__(self, original, dtype_name, outlier_mask=None):
        super().__init__(
            original.in_features, original.out_features,
            bias=original.bias is not None,
        )
        self.weight.data.copy_(original.weight.data)
        if original.bias is not None:
            self.bias.data.copy_(original.bias.data)

        self.dtype_name = dtype_name
        self.dtype_bits = DTYPE_CATALOG[dtype_name]["bits"]

        w_for_init = original.weight.data
        if outlier_mask is not None:
            w_for_init = w_for_init.clone()
            w_for_init[outlier_mask] = 0.0
        self.quantizer = _make_single_quantizer(dtype_name, w_for_init)

        if outlier_mask is not None:
            self.register_buffer("outlier_mask", outlier_mask.bool())
            self.register_buffer(
                "outlier_values",
                original.weight.data[outlier_mask].half(),
            )
        else:
            self.register_buffer("outlier_mask", None)
            self.register_buffer("outlier_values", None)

    def forward(self, x):
        w = self.weight

        if self.outlier_mask is not None:
            w = w.clone()
            w[self.outlier_mask] = 0.0

        w_q = self.quantizer(w)

        if self.outlier_mask is not None:
            w_q = w_q.clone()
            w_q[self.outlier_mask] = self.outlier_values.float()

        return F.linear(x, w_q, self.bias)


# -----------------------------------------------------------------------
# Model conversion utilities
# -----------------------------------------------------------------------

def replace_with_aqua(model, dtype_assignment, outlier_masks=None):
    """Replace all Conv2d/Linear layers with AQUA quantized versions.

    Args:
        model: Pretrained FP32 model.
        dtype_assignment: Dict[layer_name -> dtype_name] from the solver.
        outlier_masks: Optional Dict[layer_name -> bool tensor].

    Returns:
        Modified model (in-place).
    """
    replacements = []
    for name, module in model.named_modules():
        if name not in dtype_assignment:
            continue
        if isinstance(module, nn.Conv2d) and type(module) is nn.Conv2d:
            replacements.append((name, module, "conv"))
        elif isinstance(module, nn.Linear) and type(module) is nn.Linear:
            replacements.append((name, module, "linear"))

    for name, module, kind in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = dict(model.named_modules())[parts[0]]
            attr = parts[1]
        else:
            parent = model
            attr = name

        dt = dtype_assignment[name]
        mask = outlier_masks.get(name) if outlier_masks else None

        if kind == "conv":
            new_layer = AQUAConv2d(module, dt, outlier_mask=mask)
        else:
            new_layer = AQUALinear(module, dt, outlier_mask=mask)
        setattr(parent, attr, new_layer)

    return model


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def get_aqua_layer_stats(model):
    """Per-layer statistics for logging."""
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (AQUAConv2d, AQUALinear)):
            has_outliers = m.outlier_mask is not None
            n_outlier = int(m.outlier_mask.sum().item()) if has_outliers else 0
            stats.append({
                "name": name,
                "dtype": m.dtype_name,
                "bits": m.dtype_bits,
                "numel": m.weight.numel(),
                "n_outlier": n_outlier,
                "type": "conv" if isinstance(m, AQUAConv2d) else "linear",
            })
    return stats


@torch.no_grad()
def compute_per_layer_quant_error(model):
    """Compute MSE between FP32 weights and quantized weights per layer."""
    errors = []
    for name, m in model.named_modules():
        if isinstance(m, (AQUAConv2d, AQUALinear)):
            w = m.weight.data
            w_test = w.clone()
            if m.outlier_mask is not None:
                w_test[m.outlier_mask] = 0.0
            w_q = m.quantizer(w_test)
            if m.outlier_mask is not None:
                w_q[m.outlier_mask] = m.outlier_values.float()
            mse = ((w - w_q) ** 2).mean().item()
            errors.append({
                "name": name,
                "mse": mse,
                "dtype": m.dtype_name,
                "bits": m.dtype_bits,
                "w_norm": w.norm().item(),
            })
    return errors


def collect_quantizer_params(model):
    """Collect learnable quantizer scale parameters for the optimizer."""
    params = []
    for m in model.modules():
        if isinstance(m, (AQUAConv2d, AQUALinear)):
            for p in m.quantizer.parameters():
                if p.requires_grad:
                    params.append(p)
    return params


def count_aqua_layers(model):
    """Count the number of AQUA layers in the model."""
    return sum(1 for m in model.modules()
               if isinstance(m, (AQUAConv2d, AQUALinear)))


def get_avg_bits(model):
    """Compute parameter-weighted average bit-width."""
    total_params = 0
    weighted_bits = 0.0
    for m in model.modules():
        if isinstance(m, (AQUAConv2d, AQUALinear)):
            n = m.weight.numel()
            total_params += n
            weighted_bits += m.dtype_bits * n
    return weighted_bits / max(total_params, 1)


def get_dtype_counts(model):
    """Count how many layers use each dtype."""
    counts = {}
    for m in model.modules():
        if isinstance(m, (AQUAConv2d, AQUALinear)):
            counts[m.dtype_name] = counts.get(m.dtype_name, 0) + 1
    return counts
