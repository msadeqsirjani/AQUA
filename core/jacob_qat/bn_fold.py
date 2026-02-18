"""
BatchNorm folding into Conv2d and model preparation for QAT.

Implements the BN folding equations from Jacob et al. 2018:
  W_fold = W * gamma / sqrt(var + eps)
  b_fold = beta - gamma * mean / sqrt(var + eps)
"""

import torch
import torch.nn as nn

from .jacob_quantized_layers import QConv2d, QLinear


def fold_bn_into_conv(conv, bn):
    """Fold BatchNorm parameters into a Conv2d, returning a QConv2d.

    Args:
        conv: nn.Conv2d with trained weights
        bn: nn.BatchNorm2d with running_mean, running_var, weight (gamma), bias (beta)

    Returns:
        QConv2d with folded weights and bias, ready for QAT
    """
    # Extract BN parameters
    gamma = bn.weight.data                # scale
    beta = bn.bias.data                   # shift
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps

    inv_std = gamma / torch.sqrt(var + eps)  # gamma / sqrt(var + eps)

    # Create QConv2d with same configuration
    qconv = QConv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups,
        bias=True,  # folded conv always has bias
    )

    # W_fold = W * (gamma / sqrt(var + eps))  — broadcast over out_channels
    # inv_std shape: [out_channels] -> reshape to [out_channels, 1, 1, 1]
    qconv.weight.data = conv.weight.data * inv_std.view(-1, 1, 1, 1)

    # b_fold = beta - gamma * mean / sqrt(var + eps) + existing_bias_folded
    if conv.bias is not None:
        qconv.bias.data = (conv.bias.data - mean) * inv_std + beta
    else:
        qconv.bias.data = beta - mean * inv_std

    return qconv


def _replace_layers(module, parent_name=""):
    """Recursively replace Conv2d+BN pairs and standalone layers."""
    children = list(module.named_children())
    i = 0
    while i < len(children):
        name, child = children[i]

        # Check for Conv2d followed by BatchNorm2d in sequential-like containers
        if isinstance(child, nn.Conv2d) and not isinstance(child, QConv2d):
            # Look for BN as next sibling
            bn = None
            bn_name = None
            if i + 1 < len(children):
                next_name, next_child = children[i + 1]
                if isinstance(next_child, nn.BatchNorm2d):
                    bn = next_child
                    bn_name = next_name

            if bn is not None:
                qconv = fold_bn_into_conv(child, bn)
                setattr(module, name, qconv)
                setattr(module, bn_name, nn.Identity())
                i += 2
                continue
            else:
                # Standalone Conv2d -> QConv2d (copy weights)
                qconv = QConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups,
                    bias=child.bias is not None,
                )
                qconv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qconv.bias.data.copy_(child.bias.data)
                setattr(module, name, qconv)
                i += 1
                continue

        elif isinstance(child, nn.Linear):
            qlinear = QLinear(child.in_features, child.out_features,
                              bias=child.bias is not None)
            qlinear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                qlinear.bias.data.copy_(child.bias.data)
            setattr(module, name, qlinear)
            i += 1
            continue

        else:
            # Recurse into submodules
            _replace_layers(child, f"{parent_name}.{name}" if parent_name else name)
            i += 1


def prepare_model_for_qat(model):
    """Prepare a pretrained FP32 model for QAT.

    1. Fold all (Conv2d, BatchNorm2d) pairs
    2. Replace standalone Conv2d -> QConv2d
    3. Replace Linear -> QLinear
    4. Remove BN layers (replace with nn.Identity)

    Args:
        model: Pretrained FP32 model (modified in-place)

    Returns:
        The modified model with fake quantization layers
    """
    model.eval()  # ensure BN running stats are used
    _replace_layers(model)
    return model
