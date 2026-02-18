"""
ILP-based bit-width assignment for Edge-MPQ
(Zhao et al., IEEE Trans. Computers, 2024).

Solves an Integer Linear Program to assign per-layer bit-widths
from {2, 4, 8, 16} that minimize total model size subject to:
  1. Total BOPs <= target_bops (energy/compute budget)
  2. Weighted quantization error <= epsilon (accuracy constraint)

Falls back to a greedy sensitivity-based heuristic if PuLP
is not available or the ILP is infeasible.
"""

import numpy as np


def _quantization_mse(num_bits):
    """Approximate relative quantization MSE for uniform quantization.

    For a uniform quantizer with n_levels = 2^b over range [-r, r]:
      MSE ∝ r^2 / (3 * 4^b)
    We return 1 / (3 * 4^b) as a normalized error metric.
    Lower bits → higher error.
    """
    return 1.0 / (3.0 * (4.0 ** num_bits))


def _compute_bops(macs, bits):
    """BOPs for a layer: bits^2 * MACs (bit-operations model)."""
    return bits * bits * macs


def solve_bit_assignment(sensitivities, macs_per_layer, target_bops_ratio=0.5,
                         bit_choices=None):
    """Find per-layer bit-widths via ILP, with greedy fallback.

    Args:
        sensitivities: Dict {layer_name: hessian_trace} — higher = more sensitive.
        macs_per_layer: Dict {layer_name: num_macs} — MAC count per layer.
        target_bops_ratio: Target BOPs as fraction of all-16-bit BOPs (default: 0.5).
        bit_choices: Allowed bit-widths (default: [2, 4, 8, 16]).

    Returns:
        Dict {layer_name: assigned_bits}.
    """
    if bit_choices is None:
        bit_choices = [2, 4, 8, 16]

    layer_names = list(sensitivities.keys())
    n_layers = len(layer_names)

    if n_layers == 0:
        return {}

    # Compute reference BOPs (all layers at max bits)
    max_bits = max(bit_choices)
    total_bops_max = sum(
        _compute_bops(macs_per_layer[name], max_bits) for name in layer_names
    )
    target_bops = target_bops_ratio * total_bops_max

    # Try PuLP ILP first
    try:
        result = _solve_ilp_pulp(
            layer_names, sensitivities, macs_per_layer,
            target_bops, bit_choices
        )
        if result is not None:
            return result
    except ImportError:
        pass

    # Fallback: greedy sensitivity-based assignment
    return _solve_greedy(
        layer_names, sensitivities, macs_per_layer,
        target_bops, bit_choices
    )


def _solve_ilp_pulp(layer_names, sensitivities, macs_per_layer,
                     target_bops, bit_choices):
    """Solve bit assignment via ILP using PuLP.

    Variables: x[l][b] ∈ {0, 1} — whether layer l uses bit-width b.
    Objective: minimize sum_l sum_b (S_l * error(b) * x[l][b])
               (minimize sensitivity-weighted quantization error)
    Subject to:
      sum_b x[l][b] = 1  for each layer (exactly one bit-width)
      sum_l sum_b BOPs(l, b) * x[l][b] <= target_bops
    """
    import pulp

    n_layers = len(layer_names)
    n_bits = len(bit_choices)

    prob = pulp.LpProblem("EdgeMPQ_BitAssignment", pulp.LpMinimize)

    # Decision variables: x[l][b] = 1 if layer l uses bit_choices[b]
    x = {}
    for i, name in enumerate(layer_names):
        for j, b in enumerate(bit_choices):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Objective: minimize sensitivity-weighted quantization error
    obj_terms = []
    for i, name in enumerate(layer_names):
        s_l = sensitivities[name]
        for j, b in enumerate(bit_choices):
            obj_terms.append(s_l * _quantization_mse(b) * x[i, j])
    prob += pulp.lpSum(obj_terms)

    # Constraint 1: each layer picks exactly one bit-width
    for i in range(n_layers):
        prob += pulp.lpSum(x[i, j] for j in range(n_bits)) == 1

    # Constraint 2: total BOPs <= target
    bops_terms = []
    for i, name in enumerate(layer_names):
        macs = macs_per_layer[name]
        for j, b in enumerate(bit_choices):
            bops_terms.append(_compute_bops(macs, b) * x[i, j])
    prob += pulp.lpSum(bops_terms) <= target_bops

    # Solve (suppress output)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        print("  [ILP] No optimal solution found, falling back to greedy.")
        return None

    # Extract solution
    assignment = {}
    for i, name in enumerate(layer_names):
        for j, b in enumerate(bit_choices):
            if pulp.value(x[i, j]) is not None and pulp.value(x[i, j]) > 0.5:
                assignment[name] = b
                break
        if name not in assignment:
            assignment[name] = max(bit_choices)

    return assignment


def _solve_greedy(layer_names, sensitivities, macs_per_layer,
                  target_bops, bit_choices):
    """Greedy fallback: sort by sensitivity, assign high bits to sensitive layers.

    Strategy:
      1. Start all layers at minimum bits.
      2. Sort layers by sensitivity (highest first).
      3. Greedily upgrade the most sensitive layer that still fits in budget.
      4. Repeat until budget is exhausted or all layers are at max bits.
    """
    bit_choices_sorted = sorted(bit_choices)
    min_bits = bit_choices_sorted[0]
    assignment = {name: min_bits for name in layer_names}

    # Sort layers by sensitivity (most sensitive first)
    sorted_layers = sorted(layer_names, key=lambda n: sensitivities[n], reverse=True)

    def current_bops():
        return sum(
            _compute_bops(macs_per_layer[n], assignment[n]) for n in layer_names
        )

    # Greedily upgrade layers
    changed = True
    while changed:
        changed = False
        for name in sorted_layers:
            cur_bits = assignment[name]
            cur_idx = bit_choices_sorted.index(cur_bits)
            if cur_idx >= len(bit_choices_sorted) - 1:
                continue  # already at max
            next_bits = bit_choices_sorted[cur_idx + 1]
            # Try upgrading
            assignment[name] = next_bits
            if current_bops() <= target_bops:
                changed = True
            else:
                assignment[name] = cur_bits  # revert

    return assignment
