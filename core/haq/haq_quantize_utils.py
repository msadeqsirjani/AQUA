"""
Modernized quantization utilities from HAQ (Wang et al., CVPR 2019).
Original: lib/utils/quantize_utils.py from https://github.com/mit-han-lab/haq

Changes from original:
  - Removed torch.autograd.Variable (deprecated in PyTorch 2.x)
  - Removed volatile=True, use torch.no_grad() instead
  - Replaced .data access with .detach() where appropriate
  - Proper nn.init usage
  - Removed sklearn/k-means dependency — use linear quantization only
  - Clean Python 3.10+ style
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class STE(torch.autograd.Function):
    """Straight-Through Estimator: forward uses quantized value,
    backward passes gradients through as identity."""

    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class QModule(nn.Module):
    """Base class for quantized modules with per-layer bit-width control.

    Supports mixed-precision: each layer can have independent w_bit and a_bit.
    Uses linear (uniform) quantization with STE for training.
    """

    def __init__(self, w_bit=-1, a_bit=-1, half_wave=True):
        super().__init__()

        if half_wave:
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1
        self._w_bit = w_bit
        self._b_bit = 32
        self._half_wave = half_wave

        self.init_range = 6.0
        self.activation_range = nn.Parameter(torch.tensor([self.init_range]))
        self.weight_range = nn.Parameter(torch.tensor([-1.0]), requires_grad=False)

        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = True
        self._calibrate = False

    @property
    def w_bit(self):
        return self._w_bit

    @w_bit.setter
    def w_bit(self, w_bit):
        self._w_bit = w_bit

    @property
    def a_bit(self):
        return self._a_bit if self._half_wave else self._a_bit + 1

    @a_bit.setter
    def a_bit(self, a_bit):
        self._a_bit = a_bit if self._half_wave else a_bit - 1

    @property
    def b_bit(self):
        return self._b_bit

    @property
    def half_wave(self):
        return self._half_wave

    @property
    def quantized(self):
        return self._quantized

    def set_quantize(self, quantized):
        self._quantized = quantized

    def set_tanh_weight(self, tanh_weight):
        self._tanh_weight = tanh_weight
        if self._tanh_weight:
            self.weight_range.data[0] = 1.0

    def set_fix_weight(self, fix_weight):
        self._fix_weight = fix_weight

    def set_activation_range(self, activation_range):
        self.activation_range.data[0] = activation_range

    def set_weight_range(self, weight_range):
        self.weight_range.data[0] = weight_range

    def set_trainable_activation_range(self, trainable=True):
        self._trainable_activation_range = trainable
        self.activation_range.requires_grad_(trainable)

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def _quantize_activation(self, inputs):
        if not (self._quantized and self._a_bit > 0):
            return inputs

        if self._calibrate:
            estimate_range = min(self.init_range, inputs.abs().max().item())
            self.activation_range.data = torch.tensor(
                [estimate_range], device=inputs.device
            )
            return inputs

        if self._trainable_activation_range:
            if self._half_wave:
                ori_x = 0.5 * (
                    inputs.abs()
                    - (inputs - self.activation_range).abs()
                    + self.activation_range
                )
            else:
                ori_x = 0.5 * (
                    (-inputs - self.activation_range).abs()
                    - (inputs - self.activation_range).abs()
                )
        else:
            if self._half_wave:
                ori_x = inputs.clamp(0.0, self.activation_range.item())
            else:
                ori_x = inputs.clamp(
                    -self.activation_range.item(), self.activation_range.item()
                )

        scaling_factor = self.activation_range.item() / (2.0**self._a_bit - 1.0)
        x = ori_x.detach().clone()
        x.div_(scaling_factor).round_().mul_(scaling_factor)

        return STE.apply(ori_x, x)

    def _quantize_weight(self, weight):
        if self._tanh_weight:
            weight = weight.tanh()
            weight = weight / weight.abs().max()

        if not (self._quantized and self._w_bit > 0):
            return weight

        threshold = self.weight_range.item()
        if threshold <= 0:
            threshold = weight.abs().max().item()
            self.weight_range.data[0] = threshold

        if self._calibrate:
            threshold = weight.abs().max().item()
            self.weight_range.data[0] = threshold
            return weight

        ori_w = weight
        scaling_factor = threshold / (2.0 ** (self._w_bit - 1) - 1.0)
        w = ori_w.clamp(-threshold, threshold)
        w.div_(scaling_factor).round_().mul_(scaling_factor)

        if self._fix_weight:
            return w.detach()
        else:
            return STE.apply(ori_w, w)

    def _quantize(self, inputs, weight, bias):
        inputs = self._quantize_activation(inputs=inputs)
        weight = self._quantize_weight(weight=weight)
        return inputs, weight, bias

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        return "w_bit={}, a_bit={}, half_wave={}".format(
            self.w_bit if self.w_bit > 0 else -1,
            self.a_bit if self.a_bit > 0 else -1,
            self.half_wave,
        )


class QConv2d(QModule):
    """Quantized Conv2d with per-layer mixed-precision support."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        w_bit=-1,
        a_bit=-1,
        half_wave=True,
    ):
        super().__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(
            inputs=inputs, weight=self.weight, bias=self.bias
        )
        return F.conv2d(
            inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.w_bit > 0 or self.a_bit > 0:
            s += ", w_bit={}, a_bit={}".format(self.w_bit, self.a_bit)
        return s.format(**self.__dict__)


class QLinear(QModule):
    """Quantized Linear with per-layer mixed-precision support."""

    def __init__(
        self, in_features, out_features, bias=True, w_bit=-1, a_bit=-1, half_wave=True
    ):
        super().__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(
            inputs=inputs, weight=self.weight, bias=self.bias
        )
        return F.linear(inputs, weight=weight, bias=bias)

    def extra_repr(self):
        s = "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
        if self.w_bit > 0 or self.a_bit > 0:
            s += ", w_bit={}, a_bit={}".format(self.w_bit, self.a_bit)
        return s


def calibrate(model, loader, device="cuda"):
    """Calibrate activation ranges by running one batch through the model."""
    was_parallel = hasattr(model, "module")
    if was_parallel:
        model = model.module

    print("\n==> Start calibrate")
    for module in model.modules():
        if isinstance(module, QModule):
            module.set_calibrate(True)

    inputs, _ = next(iter(loader))
    inputs = inputs.to(device, non_blocking=True)
    with torch.no_grad():
        model(inputs)

    for module in model.modules():
        if isinstance(module, QModule):
            module.set_calibrate(False)

    print("==> End calibrate")
    if was_parallel:
        model = nn.DataParallel(model)
    return model


def set_quantize_bits(model, quantizable_idx, strategy):
    """Apply a mixed-precision bit-width strategy to the model.

    Args:
        model: Model with QConv2d/QLinear layers
        quantizable_idx: List of module indices that are quantizable
        strategy: List of (w_bit, a_bit) tuples or single int for uniform w_bit
    """
    for idx, layer_idx in enumerate(quantizable_idx):
        for i, m in enumerate(model.modules()):
            if i == layer_idx:
                if isinstance(strategy[idx], (list, tuple)):
                    m.w_bit = strategy[idx][0]
                    m.a_bit = strategy[idx][1]
                else:
                    m.w_bit = strategy[idx]
                    m.a_bit = strategy[idx]
                # Reset weight range so it recalibrates
                m.weight_range.data[0] = -1.0
                break
