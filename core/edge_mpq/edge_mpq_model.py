"""
Edge-MPQ model preparation (Zhao et al., IEEE Trans. Computers, 2024).

Applies a mixed-precision bit-width assignment to a model by
replacing Conv2d/Linear layers with quantized versions using
JacobFakeQuantize at the specified per-layer bit-width.
"""

import torch.nn as nn
import torch.nn.functional as F

from ..jacob_fake_quant import JacobFakeQuantize


class MPQConv2d(nn.Conv2d):
    """Conv2d with configurable-bit fake quantization."""

    def __init__(self, *args, w_bits=8, a_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = JacobFakeQuantize(w_bits, "symmetric", is_weight=True)
        self.act_quantizer = JacobFakeQuantize(a_bits, "asymmetric", is_weight=False)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class MPQLinear(nn.Linear):
    """Linear with configurable-bit fake quantization."""

    def __init__(self, *args, w_bits=8, a_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = JacobFakeQuantize(w_bits, "symmetric", is_weight=True)
        self.act_quantizer = JacobFakeQuantize(a_bits, "asymmetric", is_weight=False)

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)


def apply_mixed_precision(model, bit_assignment):
    """Replace Conv2d/Linear layers with mixed-precision quantized versions.

    Walks the module tree and replaces each layer whose name appears
    in bit_assignment with an MPQConv2d/MPQLinear at the assigned bit-width.
    Copies pretrained weights. BN layers are left intact (caller should
    fold them beforehand if desired).

    Args:
        model: Pretrained FP32 model (modified in-place).
        bit_assignment: Dict {layer_name: bits} from ILP solver.
            Uses same bits for both weights and activations.

    Returns:
        Modified model with mixed-precision fake quantization.
    """
    # Build a flat lookup: name -> (parent_module, attr_name, child_module)
    replacements = []
    for name, module in model.named_modules():
        if name in bit_assignment:
            bits = bit_assignment[name]
            if isinstance(module, nn.Conv2d) and not isinstance(module, MPQConv2d):
                replacements.append((name, module, bits, "conv"))
            elif isinstance(module, nn.Linear) and not isinstance(module, MPQLinear):
                replacements.append((name, module, bits, "linear"))

    for name, module, bits, layer_type in replacements:
        # Navigate to parent
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr = name

        if layer_type == "conv":
            new_layer = MPQConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=module.bias is not None,
                w_bits=bits, a_bits=bits,
            )
            new_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)
        else:
            new_layer = MPQLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                w_bits=bits, a_bits=bits,
            )
            new_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)

        setattr(parent, attr, new_layer)

    return model


def get_layer_macs(model, input_h=32, input_w=32):
    """Compute per-layer MAC counts by running a dummy forward with hooks.

    Returns:
        Dict {layer_name: num_macs} for Conv2d and Linear layers.
    """
    import torch

    macs_dict = {}
    hooks = []

    def make_hook(name, module):
        def hook_fn(mod, inp, out):
            x = inp[0]
            if isinstance(mod, nn.Conv2d):
                _, _, h_in, w_in = x.shape
                k_h, k_w = mod.kernel_size
                s_h, s_w = mod.stride
                p_h, p_w = mod.padding
                d_h, d_w = mod.dilation
                h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
                w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
                macs = (mod.in_channels * mod.out_channels *
                        k_h * k_w * h_out * w_out // mod.groups)
                macs_dict[name] = int(macs)
            elif isinstance(mod, nn.Linear):
                macs_dict[name] = mod.in_features * mod.out_features
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, input_h, input_w, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return macs_dict
