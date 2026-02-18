"""
SDP model preparation.
Wraps Conv2d and Linear layers with SDP weight and activation quantizers.
"""

import torch.nn as nn
import torch.nn.functional as F

from .sdp_quantizer import SDPQuantizer
from ..jacob_fake_quant import JacobFakeQuantize


class SDPConv2d(nn.Conv2d):
    """Conv2d with SDP weight quantizer and standard activation quantizer.

    Weight: SDPQuantizer (dynamic precision — important elements get full N bits,
            unimportant get only M high bits).
    Activation: JacobFakeQuantize at N bits (uniform, no dynamic skipping — activations
                are streamed and not amenable to structured skipping).
    """

    def __init__(self, *args, total_bits=8, high_bits=4, group_size=8,
                 sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = SDPQuantizer(
            total_bits=total_bits, high_bits=high_bits,
            group_size=group_size, sparsity=sparsity, is_weight=True,
        )
        self.act_quantizer = JacobFakeQuantize(
            num_bits=total_bits, mode="asymmetric", is_weight=False,
        )

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class SDPLinear(nn.Linear):
    """Linear with SDP weight quantizer and standard activation quantizer."""

    def __init__(self, *args, total_bits=8, high_bits=4, group_size=8,
                 sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = SDPQuantizer(
            total_bits=total_bits, high_bits=high_bits,
            group_size=group_size, sparsity=sparsity, is_weight=True,
        )
        self.act_quantizer = JacobFakeQuantize(
            num_bits=total_bits, mode="asymmetric", is_weight=False,
        )

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)


def replace_with_sdp(model, total_bits=8, high_bits=4, group_size=8, sparsity=0.5):
    """Replace all Conv2d/Linear with SDP-quantized versions.

    Copies pretrained weights. BN layers are left intact.

    Args:
        model: Pretrained FP32 model (modified in-place).
        total_bits: Total bit-width N.
        high_bits: High-order bits M (always computed).
        group_size: Group size G for importance mask.
        sparsity: Fraction of elements kept at full N bits.

    Returns:
        Modified model with SDPConv2d/SDPLinear layers.
    """
    replacements = []
    for name, module in model.named_modules():
        if type(module) is nn.Conv2d:
            replacements.append((name, module, "conv"))
        elif type(module) is nn.Linear:
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
            new = SDPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=module.bias is not None,
                total_bits=total_bits, high_bits=high_bits,
                group_size=group_size, sparsity=sparsity,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
        else:
            new = SDPLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                total_bits=total_bits, high_bits=high_bits,
                group_size=group_size, sparsity=sparsity,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)

        setattr(parent, attr, new)

    return model
