"""
Hardware-aware cost model for LCPAQ (Module A).
"Adaptive Quantization with Mixed-Precision Based on Low-Cost Proxy"
arXiv:2402.17706, 2024.

Builds a per-layer cost lookup table covering model size, BOPs, and
analytical latency for each candidate bit-width.

Latency model:
  peak_throughput(2-bit) = 4x baseline (INT2 parallelism)
  peak_throughput(4-bit) = 2x baseline
  peak_throughput(8-bit) = 1x baseline
"""

import torch
import torch.nn as nn


# Relative throughput multiplier for each bit-width (vs 8-bit baseline)
_THROUGHPUT_MULTIPLIER = {
    2: 4.0,
    4: 2.0,
    8: 1.0,
}


def _compute_layer_macs(model, input_shape=(1, 3, 32, 32)):
    """Compute per-layer MAC counts via forward hooks.

    Args:
        model: Model on which to measure.
        input_shape: Shape of dummy input (batch, C, H, W).

    Returns:
        Dict {layer_name: num_macs}.
    """
    macs_dict = {}
    hooks = []

    def make_hook(name):
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
            hooks.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    dummy = torch.zeros(*input_shape, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return macs_dict


def _count_params(module):
    """Count number of weight parameters in a module."""
    return module.weight.numel()


def build_cost_table(model, input_shape=(1, 3, 32, 32), bit_choices=None):
    """Build hardware-aware cost table for every quantizable layer.

    For each layer l and each candidate bit-width b, computes:
      model_size_l(b) = num_params(l) * b / 8  (bytes)
      bops_l(b)       = MACs(l) * b * b         (bit-operations)
      latency_l(b)    = bops_l(b) / throughput(b) (normalized)

    Args:
        model: Pretrained model (not modified).
        input_shape: Dummy input shape for MAC profiling.
        bit_choices: Candidate bit-widths (default: [2, 4, 8]).

    Returns:
        Dict[str, Dict[int, Dict[str, float]]]:
            cost_table[layer_name][bits] = {
                "size": float,      # bytes
                "bops": float,      # bit-operations
                "latency": float,   # normalized latency
                "macs": int,        # raw MACs
                "params": int,      # number of weight params
            }
    """
    if bit_choices is None:
        bit_choices = [2, 4, 8]

    macs_dict = _compute_layer_macs(model, input_shape)
    cost_table = {}

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if name not in macs_dict:
            continue

        macs = macs_dict[name]
        params = _count_params(module)
        layer_costs = {}

        for b in bit_choices:
            bops = macs * b * b
            size_bytes = params * b / 8.0
            throughput = _THROUGHPUT_MULTIPLIER.get(b, 1.0)
            latency = bops / throughput

            layer_costs[b] = {
                "size": size_bytes,
                "bops": float(bops),
                "latency": latency,
                "macs": macs,
                "params": params,
            }

        cost_table[name] = layer_costs

    return cost_table


def summarize_cost_table(cost_table, bit_assignment=None):
    """Print a formatted summary of the cost table.

    If bit_assignment is given, highlights the selected bit-width per layer.
    """
    layer_names = list(cost_table.keys())
    bit_choices = sorted(next(iter(cost_table.values())).keys())

    print(f"  {'Layer':<30s}", end="")
    for b in bit_choices:
        print(f" | {b}b BOPs{'':<6s} {b}b Size", end="")
    if bit_assignment:
        print(f" | Selected", end="")
    print()

    print(f"  {'-'*30}", end="")
    for _ in bit_choices:
        print(f" | {'-'*22}", end="")
    if bit_assignment:
        print(f" | {'-'*8}", end="")
    print()

    for name in layer_names:
        short = name.rsplit(".", 1)[-1] if "." in name else name
        print(f"  {short:<30s}", end="")
        for b in bit_choices:
            c = cost_table[name][b]
            print(f" | {c['bops']:>10,.0f} {c['size']:>9,.0f}B", end="")
        if bit_assignment:
            sel = bit_assignment.get(name, "?")
            print(f" | {sel:>5}b", end="")
        print()
