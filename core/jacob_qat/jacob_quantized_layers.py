"""
Quantized Conv2d and Linear layers with fake quantization
on both weights and activations (Jacob et al. 2018).
"""

import torch.nn as nn
import torch.nn.functional as F

from ..jacob_fake_quant import JacobFakeQuantize


class QConv2d(nn.Conv2d):
    """Conv2d with INT8 fake quantization on weights and activations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = JacobFakeQuantize(8, "symmetric", is_weight=True)
        self.act_quantizer = JacobFakeQuantize(8, "asymmetric", is_weight=False)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QLinear(nn.Linear):
    """Linear with INT8 fake quantization on weights and activations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = JacobFakeQuantize(8, "symmetric", is_weight=True)
        self.act_quantizer = JacobFakeQuantize(8, "asymmetric", is_weight=False)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)
