"""
Analytical dtype assignment for AQUA v2.

For each layer, tries every candidate dtype and measures the
sensitivity-weighted quantization MSE (excluding outlier weights).
Then uses a greedy Pareto solver to find the cheapest assignment
that stays within the target BOPs budget.

The key insight: because outlier weights are protected in fp16,
the remaining (non-outlier) weights can be more aggressively quantized
without catastrophic accuracy loss.
"""

import torch
import torch.nn as nn

from .quantizers import DTYPE_CATALOG, make_quantizer_bank


def compute_dtype_errors(model, allowed_dtypes, hessian_traces,
                         outlier_masks=None, device="cpu"):
    """Compute per-layer, per-dtype quantization error (sensitivity-weighted MSE).

    For each layer l and dtype d:
        error(l, d) = H_trace(l) * MSE(w_normal, Q_d(w_normal))

    where w_normal = weights with outliers zeroed out.

    Args:
        model: Pretrained FP32 model.
        allowed_dtypes: List of dtype name strings.
        hessian_traces: Dict[layer_name -> float] from Hessian estimator.
        outlier_masks: Optional Dict[layer_name -> bool tensor].
        device: Torch device for quantizer forward pass.

    Returns:
        Dict[layer_name -> Dict[dtype_name -> float]] of weighted errors.
    """
    errors = {}

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if name not in hessian_traces:
            continue

        w = module.weight.detach().to(device)
        h_trace = hessian_traces[name]

        mask = outlier_masks.get(name) if outlier_masks else None
        if mask is not None:
            mask = mask.to(device)
            w_normal = w.clone()
            w_normal[mask] = 0.0
            n_normal = (~mask).sum().item()
        else:
            w_normal = w
            n_normal = w.numel()

        if n_normal == 0:
            n_normal = 1

        layer_errors = {}
        for dt_name in allowed_dtypes:
            entry = DTYPE_CATALOG[dt_name]
            q = entry["quantizer_cls"](**entry["args"]).to(device)
            q.init_scale(w_normal)

            with torch.no_grad():
                w_q = q(w_normal)
                diff = w_normal - w_q
                if mask is not None:
                    diff[mask] = 0.0
                mse = diff.pow(2).sum().item() / n_normal

            layer_errors[dt_name] = h_trace * mse

        errors[name] = layer_errors

    return errors


def build_bops_table(model, allowed_dtypes, input_shape=(1, 3, 32, 32)):
    """Compute BOPs for each layer x dtype combination.

    BOPs(l, d) = MACs(l) * bits(d)^2

    Reference BOPs is always computed against fp32 (32-bit) regardless
    of which dtypes are allowed, so that target_bops_ratio has a stable
    meaning: 1.0 = full-precision fp32 cost.

    Args:
        model: Pretrained model.
        allowed_dtypes: List of dtype name strings.
        input_shape: Dummy input shape for MAC profiling.

    Returns:
        (bops_table, ref_bops) where
            bops_table: Dict[layer_name -> Dict[dtype_name -> float]]
            ref_bops: float (total BOPs at fp32 = 32 bits)
    """
    hooks = []
    macs_dict = {}

    def make_hook(lname):
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
                macs_dict[lname] = int(macs)
            elif isinstance(mod, nn.Linear):
                macs_dict[lname] = mod.in_features * mod.out_features
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    dev = next(model.parameters()).device
    dummy = torch.zeros(*input_shape, device=dev)
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()

    REF_BITS = 32  # always fp32 baseline

    bops_table = {}
    ref_bops = 0.0
    for name in macs_dict:
        layer_bops = {}
        for dt in allowed_dtypes:
            bits = DTYPE_CATALOG[dt]["bits"]
            layer_bops[dt] = macs_dict[name] * bits * bits
        bops_table[name] = layer_bops
        ref_bops += macs_dict[name] * REF_BITS * REF_BITS

    return bops_table, max(ref_bops, 1.0)


def avg_bits_to_bops_ratio(target_avg_bits):
    """Convert a target average bit-width to a BOPs ratio vs fp32.

    BOPs ratio = (target_bits / 32)^2

    Examples: 4 bits -> 0.0156, 6 bits -> 0.0352, 8 bits -> 0.0625
    """
    return (target_avg_bits / 32.0) ** 2


def solve_dtype_assignment(layer_names, dtype_errors, bops_table,
                           ref_bops, target_bops_ratio=0.0156,
                           allowed_dtypes=None):
    """Greedy Pareto assignment: minimize total weighted error under BOPs budget.

    Strategy:
    1. Start every layer at the cheapest (lowest-bit) dtype.
    2. Greedily upgrade layers where the error-reduction-per-BOPs-increase
       ratio is highest, until the budget is exhausted.

    Note: target_bops_ratio is relative to fp32 (32-bit) baseline.
    Common values:
        0.0156 = ~4 avg bits, 0.0352 = ~6 avg bits, 0.0625 = ~8 avg bits

    Args:
        layer_names: List of layer name strings (ordered).
        dtype_errors: Dict[layer -> Dict[dtype -> float]].
        bops_table: Dict[layer -> Dict[dtype -> float]].
        ref_bops: Reference fp32 total BOPs.
        target_bops_ratio: Fraction of fp32 ref_bops allowed.
        allowed_dtypes: Ordered list of dtype names.

    Returns:
        Dict[layer_name -> dtype_name] assignment.
    """
    if allowed_dtypes is None:
        allowed_dtypes = list(dtype_errors[layer_names[0]].keys())

    dtypes_by_cost = sorted(allowed_dtypes,
                            key=lambda d: DTYPE_CATALOG[d]["bits"])

    budget = ref_bops * target_bops_ratio

    assignment = {name: dtypes_by_cost[0] for name in layer_names}
    current_bops = sum(bops_table[n][assignment[n]] for n in layer_names)

    upgrade_options = {name: 0 for name in layer_names}

    improved = True
    while improved:
        improved = False
        best_ratio = -1.0
        best_layer = None
        best_dtype = None
        best_delta_bops = 0
        best_delta_error = 0
        best_next_idx = 0

        for name in layer_names:
            cur_idx = upgrade_options[name]
            cur_dt = dtypes_by_cost[cur_idx]
            cur_err = dtype_errors[name][cur_dt]
            cur_bop = bops_table[name][cur_dt]

            for next_idx in range(cur_idx + 1, len(dtypes_by_cost)):
                next_dt = dtypes_by_cost[next_idx]
                next_err = dtype_errors[name][next_dt]
                next_bop = bops_table[name][next_dt]

                delta_bops = next_bop - cur_bop
                delta_error = cur_err - next_err

                if delta_error <= 0:
                    continue
                if current_bops + delta_bops > budget:
                    continue

                # Same-cost dtypes (e.g. int8 vs fp8): prefer lower error
                if delta_bops == 0:
                    ratio = delta_error * 1e15
                else:
                    ratio = delta_error / delta_bops

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_layer = name
                    best_dtype = next_dt
                    best_delta_bops = delta_bops
                    best_delta_error = delta_error
                    best_next_idx = next_idx

        if best_layer is not None:
            assignment[best_layer] = best_dtype
            upgrade_options[best_layer] = best_next_idx
            current_bops += best_delta_bops
            improved = True

    return assignment


def compute_assignment_summary(assignment, dtype_errors, bops_table,
                               ref_bops, hessian_traces):
    """Summarize a dtype assignment for logging.

    Returns:
        Dict with avg_bits, total_bops, bops_ratio, total_error,
        dtype_counts, per_layer details.
    """
    dtype_counts = {}
    total_bops = 0.0
    total_error = 0.0
    per_layer = {}
    total_params = 0
    weighted_bits = 0.0

    for name, dt in assignment.items():
        bits = DTYPE_CATALOG[dt]["bits"]
        bops = bops_table[name][dt]
        err = dtype_errors[name][dt]
        sens = hessian_traces.get(name, 0.0)

        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        total_bops += bops
        total_error += err

        numel = bops_table[name][dt] / (bits * bits) if bits > 0 else 0
        weighted_bits += bits * numel
        total_params += numel

        per_layer[name] = {
            "dtype": dt,
            "bits": bits,
            "bops": bops,
            "error": err,
            "sensitivity": sens,
        }

    avg_bits = weighted_bits / max(total_params, 1)

    return {
        "avg_bits": avg_bits,
        "total_bops": total_bops,
        "bops_ratio": total_bops / max(ref_bops, 1.0),
        "total_error": total_error,
        "dtype_counts": dtype_counts,
        "per_layer": per_layer,
    }
