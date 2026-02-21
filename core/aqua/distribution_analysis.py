"""
Weight distribution analysis for AQUA v2.

Characterizes per-layer weight distributions using statistical moments
(kurtosis, skewness) and derives format recommendations:

- Gaussian-like distributions (kurtosis ~ 3) -> INT (uniform) is near-optimal
- Heavy-tailed distributions (kurtosis > 4)  -> FP (non-uniform) is better
- This follows from rate-distortion theory: uniform quantization minimizes
  MSE for Gaussian sources, while non-uniform (log-spaced) is better for
  super-Gaussian / Laplacian sources.
"""

import torch
import torch.nn as nn

KURTOSIS_THRESHOLD = 4.0


def analyze_layer_distributions(model):
    """Compute distribution statistics for each quantizable layer.

    Args:
        model: Pretrained FP32 model (not modified).

    Returns:
        Dict[layer_name -> stats_dict] where stats_dict contains:
            kurtosis, skewness, sparsity, outlier_ratio, std, mean_abs
    """
    stats = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        w = module.weight.detach().flatten().float()
        n = w.numel()
        if n < 4:
            continue

        mu = w.mean()
        centered = w - mu
        var = centered.pow(2).mean()
        std = var.sqrt().clamp(min=1e-10)

        skew = centered.pow(3).mean() / std.pow(3)
        kurt = centered.pow(4).mean() / var.pow(2)

        mean_abs = w.abs().mean().item()
        sparsity = (w.abs() < 1e-6).float().mean().item()
        outlier_ratio = (w.abs() > 3 * std).float().mean().item()

        stats[name] = {
            "kurtosis": kurt.item(),
            "skewness": abs(skew.item()),
            "std": std.item(),
            "mean_abs": mean_abs,
            "sparsity": sparsity,
            "outlier_ratio": outlier_ratio,
            "numel": n,
        }
    return stats


def recommend_format(kurtosis):
    """Recommend INT or FP based on excess kurtosis.

    Gaussian kurtosis = 3.0, Laplacian = 6.0.
    Below threshold: INT (uniform spacing suits symmetric thin-tail).
    Above threshold: FP (log-spaced levels suit heavy tails).
    """
    return "fp" if kurtosis > KURTOSIS_THRESHOLD else "int"


def recommend_dtype_candidates(layer_stats, allowed_dtypes):
    """For each layer, rank dtype candidates using distribution heuristics.

    Returns Dict[layer_name -> list of dtype names] ordered by expected quality.
    INT formats first for Gaussian-like, FP formats first for heavy-tailed.
    """
    from .quantizers import DTYPE_CATALOG

    recs = {}
    for name, st in layer_stats.items():
        preferred = recommend_format(st["kurtosis"])
        preferred_list = []
        other_list = []
        for dt in allowed_dtypes:
            etype = DTYPE_CATALOG[dt]["energy_type"]
            if etype == preferred:
                preferred_list.append(dt)
            else:
                other_list.append(dt)
        recs[name] = preferred_list + other_list
    return recs
