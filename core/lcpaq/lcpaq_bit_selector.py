"""
Hessian-aware greedy Pareto bit selector for LCPAQ (Module B).
"Adaptive Quantization with Mixed-Precision Based on Low-Cost Proxy"
arXiv:2402.17706, 2024.

Pipeline:
  1. Compute per-layer Hessian trace (sensitivity) — reuses Hutchinson estimator
  2. Compute per-layer per-bit quantization error (relative MSE)
  3. Compute sensitivity-weighted error for each (layer, bits) pair
  4. Greedy Pareto selection: starting from all-8-bit, iteratively reduce
     the layer that gives the best accuracy-cost tradeoff, until BOPs budget met
"""

import torch
import torch.nn as nn

from ..hessian_sensitivity import compute_hessian_trace  # noqa: F401 — re-export


def compute_quant_errors(model, bit_choices=None):
    """Compute relative quantization MSE per layer per bit-width.

    For each layer l and bits b:
      W_q = uniform_quantize(W_l, bits=b)
      error_l(b) = ||W_l - W_q||_F^2 / ||W_l||_F^2

    Args:
        model: Pretrained model (not modified).
        bit_choices: Candidate bit-widths (default: [2, 4, 8]).

    Returns:
        Dict[str, Dict[int, float]]:
            quant_errors[layer_name][bits] = relative_mse
    """
    if bit_choices is None:
        bit_choices = [2, 4, 8]

    quant_errors = {}

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        w = module.weight.detach()
        w_norm_sq = w.pow(2).sum().item()
        if w_norm_sq < 1e-12:
            quant_errors[name] = {b: 0.0 for b in bit_choices}
            continue

        layer_errors = {}
        for b in bit_choices:
            w_q = _uniform_quantize(w, b)
            mse = (w - w_q).pow(2).sum().item()
            layer_errors[b] = mse / w_norm_sq

        quant_errors[name] = layer_errors

    return quant_errors


def _uniform_quantize(w, bits):
    """Symmetric uniform quantization of a weight tensor.

    Args:
        w: Weight tensor (float).
        bits: Bit-width.

    Returns:
        Fake-quantized tensor.
    """
    q_min = -(2 ** (bits - 1))
    q_max = 2 ** (bits - 1) - 1

    r_min = w.min().item()
    r_max = w.max().item()
    r_min = min(r_min, 0.0)
    r_max = max(r_max, 0.0)
    if r_max == r_min:
        return w.clone()

    scale = (r_max - r_min) / (q_max - q_min)
    scale = max(scale, 1e-8)
    w_q = torch.clamp(torch.round(w / scale), q_min, q_max)
    return w_q * scale


def greedy_pareto_selection(model, hessian_traces, cost_table,
                            target_bops_ratio=0.5, bit_choices=None):
    """Greedy Pareto-front bit-width selection.

    Starting from all layers at max bits, iteratively selects the layer
    whose bit reduction yields the best ratio of accuracy-cost savings
    to hardware-cost savings, until the BOPs budget is met.

    accuracy_cost  = sum_l( S_l * quant_error_l(b_l) )
    hardware_cost  = sum_l( bops_l(b_l) )

    At each step, for each reducible layer, compute:
      delta_accuracy = S_l * (error(b-step) - error(b))     [increase in error]
      delta_bops     = bops(b) - bops(b-step)               [decrease in BOPs]
      ratio          = delta_accuracy / delta_bops           [lower is better]

    Pick the layer with smallest ratio (least accuracy hurt per BOPs saved).

    Args:
        model: Pretrained model (for computing quantization errors).
        hessian_traces: Dict {layer_name: float} from Hutchinson estimator.
        cost_table: Output of build_cost_table().
        target_bops_ratio: Target total BOPs as fraction of all-max-bit BOPs.
        bit_choices: Candidate bit-widths (default: [2, 4, 8]).

    Returns:
        Dict {layer_name: assigned_bits}.
    """
    if bit_choices is None:
        bit_choices = [2, 4, 8]

    bit_choices_sorted = sorted(bit_choices, reverse=True)  # e.g. [8, 4, 2]
    max_bits = bit_choices_sorted[0]
    layer_names = list(cost_table.keys())

    # Compute quantization errors
    quant_errors = compute_quant_errors(model, bit_choices)

    # Start with all layers at max bits
    assignment = {name: max_bits for name in layer_names}

    # Compute BOPs budget
    total_bops_max = sum(cost_table[n][max_bits]["bops"] for n in layer_names)
    target_bops = target_bops_ratio * total_bops_max

    def current_bops():
        return sum(cost_table[n][assignment[n]]["bops"] for n in layer_names)

    # Track search steps for analysis
    search_steps = []

    # Greedy reduction loop
    while current_bops() > target_bops:
        best_name = None
        best_ratio = float("inf")
        best_new_bits = None

        for name in layer_names:
            cur_bits = assignment[name]
            cur_idx = bit_choices_sorted.index(cur_bits)
            if cur_idx >= len(bit_choices_sorted) - 1:
                continue  # already at minimum bits

            new_bits = bit_choices_sorted[cur_idx + 1]
            sensitivity = hessian_traces.get(name, 1.0)

            # Accuracy cost increase
            err_cur = quant_errors[name].get(cur_bits, 0.0)
            err_new = quant_errors[name].get(new_bits, 1.0)
            delta_accuracy = sensitivity * (err_new - err_cur)

            # BOPs savings
            delta_bops = cost_table[name][cur_bits]["bops"] - \
                         cost_table[name][new_bits]["bops"]
            if delta_bops <= 0:
                continue

            ratio = delta_accuracy / delta_bops
            if ratio < best_ratio:
                best_ratio = ratio
                best_name = name
                best_new_bits = new_bits

        if best_name is None:
            break  # cannot reduce further

        old_bits = assignment[best_name]
        assignment[best_name] = best_new_bits
        search_steps.append({
            "layer": best_name,
            "from": old_bits,
            "to": best_new_bits,
            "ratio": best_ratio,
            "bops_after": current_bops(),
        })

    return assignment, search_steps
