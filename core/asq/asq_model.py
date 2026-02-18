"""
ASQ model preparation (Zhou et al. 2025).

Replaces Conv2d and Linear layers with wrapped versions using:
  - ASQActivationQuantizer for input activations (input-dependent scale)
  - POSTWeightQuantizer for weights (sqrt(2) grid)

Also provides an LSQ baseline variant for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asq_quantizer import ASQActivationQuantizer, POSTWeightQuantizer, LSQQuantizer


class ASQConv2d(nn.Conv2d):
    """Conv2d with ASQ activation quantizer + POST weight quantizer."""

    def __init__(self, *args, bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_quantizer = ASQActivationQuantizer(self.in_channels, bits=bits)
        self.weight_quantizer = POSTWeightQuantizer(bits=bits)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ASQLinear(nn.Linear):
    """Linear with ASQ activation quantizer + POST weight quantizer."""

    def __init__(self, *args, bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_quantizer = ASQActivationQuantizer(self.in_features, bits=bits)
        self.weight_quantizer = POSTWeightQuantizer(bits=bits)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)


class LSQConv2d(nn.Conv2d):
    """Conv2d with standard LSQ quantizers (baseline for comparison)."""

    def __init__(self, *args, bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_quantizer = LSQQuantizer(bits=bits, is_weight=False)
        self.weight_quantizer = LSQQuantizer(bits=bits, is_weight=True)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class LSQLinear(nn.Linear):
    """Linear with standard LSQ quantizers (baseline for comparison)."""

    def __init__(self, *args, bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_quantizer = LSQQuantizer(bits=bits, is_weight=False)
        self.weight_quantizer = LSQQuantizer(bits=bits, is_weight=True)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)


def _replace_layers(model, conv_cls, linear_cls, bits):
    """Generic layer replacement: swap Conv2d->conv_cls, Linear->linear_cls."""
    replacements = []
    for name, module in model.named_modules():
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

        if kind == "conv":
            new = conv_cls(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=module.bias is not None, bits=bits,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
        else:
            new = linear_cls(
                module.in_features, module.out_features,
                bias=module.bias is not None, bits=bits,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)

        setattr(parent, attr, new)

    return model


def replace_with_asq(model, bits=4):
    """Replace all Conv2d/Linear with ASQ+POST quantized versions.

    Args:
        model: Pretrained FP32 model (modified in-place).
        bits: Quantization bit-width (default: 4).

    Returns:
        Model with ASQConv2d/ASQLinear layers.
    """
    return _replace_layers(model, ASQConv2d, ASQLinear, bits)


def replace_with_lsq(model, bits=4):
    """Replace all Conv2d/Linear with standard LSQ quantized versions.

    Args:
        model: Pretrained FP32 model (modified in-place).
        bits: Quantization bit-width (default: 4).

    Returns:
        Model with LSQConv2d/LSQLinear layers (baseline).
    """
    return _replace_layers(model, LSQConv2d, LSQLinear, bits)


def _set_quantizers_enabled(model, enabled):
    """Enable or disable all quantizers in the model."""
    from .asq_quantizer import ASQActivationQuantizer, POSTWeightQuantizer, LSQQuantizer
    for m in model.modules():
        if isinstance(m, (ASQActivationQuantizer, POSTWeightQuantizer, LSQQuantizer)):
            m.enabled = enabled


def init_quantizer_scales(model, data_loader, device, n_batches=200):
    """Initialize all quantizer scales by running calibration data.

    Quantizers are **disabled** during the calibration forward passes so
    that every layer sees clean FP32-quality activations (no cascading
    distortion from uninitialized quantizers).  After capturing the
    activations, scales are initialized and quantizers are re-enabled.

    Args:
        model: Model with ASQ or LSQ layers.
        data_loader: Calibration DataLoader.
        device: Torch device.
        n_batches: Number of batches for calibration (default: 200).
    """
    model.to(device).eval()

    # Disable all quantizers so calibration sees FP32 activations
    _set_quantizers_enabled(model, False)

    target_types = (ASQConv2d, ASQLinear, LSQConv2d, LSQLinear)

    # Capture full activation tensors for scale init (one batch is enough
    # when quantizers are off because the activations are clean)
    init_hooks = []
    init_tensors = {}

    def make_init_hook(name):
        def hook_fn(module, inp, out):
            init_tensors[name] = inp[0].detach()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, target_types):
            init_hooks.append(module.register_forward_hook(make_init_hook(name)))

    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)

    with torch.no_grad():
        model(inputs)

    for h in init_hooks:
        h.remove()

    # Initialize scales from clean activations
    for name, module in model.named_modules():
        if isinstance(module, target_types) and name in init_tensors:
            x = init_tensors[name]
            module.act_quantizer.init_scale(x)
            module.weight_quantizer.init_scale(module.weight)

    # Re-enable all quantizers
    _set_quantizers_enabled(model, True)

    n_inited = len(init_tensors)
    print(f"  Initialized scales for {n_inited} quantized layers "
          f"({n_batches} calibration batches)")
