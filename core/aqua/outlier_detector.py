"""
Outlier detection for AQUA v2.

Identifies the most sensitive weights per layer using a combined score
of weight magnitude and Hessian diagonal sensitivity.  Outlier weights
are protected in fp16 while the rest are aggressively quantized.

    score_i = |w_i| * sqrt(H_ii)

The top ``outlier_pct`` fraction of weights (by score) are masked as
outliers and stored separately in half precision.
"""

import torch
import torch.nn as nn


def compute_fisher_diagonal(model, loss_fn, data_loader, device,
                            n_batches=5):
    """Approximate Hessian diagonal via empirical Fisher information.

    The Fisher diagonal is: F_ii = E[ (dL/dw_i)^2 ]
    This is cheaper than full Hutchinson diagonal and provides a good
    proxy for weight sensitivity.

    Args:
        model: Pretrained model (not modified, but gradients are computed).
        loss_fn: Loss function (e.g. nn.CrossEntropyLoss).
        data_loader: DataLoader for calibration data.
        device: Torch device.
        n_batches: Number of batches to average over.

    Returns:
        Dict[layer_name -> tensor of shape weight.shape] with diagonal entries.
    """
    model.eval()
    model.to(device)

    target_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers[name] = module

    accum = {name: torch.zeros_like(m.weight.data)
             for name, m in target_layers.items()}
    count = 0

    for i, (inputs, targets) in enumerate(data_loader):
        if i >= n_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, module in target_layers.items():
            if module.weight.grad is not None:
                accum[name] += module.weight.grad.data.pow(2)
        count += 1

    for name in accum:
        accum[name] /= max(count, 1)

    return accum


def compute_outlier_masks(model, fisher_diag, outlier_pct=0.01):
    """Compute per-layer boolean outlier masks.

    For each layer, the sensitivity score is ``|w| * sqrt(F_ii)``
    where F_ii is the Fisher diagonal.  The top ``outlier_pct``
    fraction of weights are marked as outliers.

    Args:
        model: Pretrained FP32 model.
        fisher_diag: Dict from :func:`compute_fisher_diagonal`.
        outlier_pct: Fraction of weights to protect (default 1%).

    Returns:
        Dict[layer_name -> bool tensor of shape weight.shape].
        True = outlier (protect in fp16).
    """
    masks = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if name not in fisher_diag:
            continue

        w = module.weight.detach()
        f_diag = fisher_diag[name]
        score = w.abs() * f_diag.sqrt().clamp(min=1e-12)

        n = score.numel()
        k = max(int(n * outlier_pct), 1)

        threshold = score.flatten().topk(k).values[-1]
        masks[name] = score >= threshold

    return masks


def compute_outlier_stats(model, outlier_masks):
    """Summary statistics about outliers per layer.

    Returns:
        Dict[layer_name -> dict with count, pct, mean_magnitude].
    """
    stats = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if name not in outlier_masks:
            continue

        mask = outlier_masks[name]
        w = module.weight.detach()
        n = w.numel()
        n_outlier = mask.sum().item()
        mean_mag = w[mask].abs().mean().item() if n_outlier > 0 else 0.0
        normal_mag = w[~mask].abs().mean().item() if n_outlier < n else 0.0

        stats[name] = {
            "n_outlier": int(n_outlier),
            "n_total": n,
            "pct": 100.0 * n_outlier / n,
            "outlier_mean_abs": mean_mag,
            "normal_mean_abs": normal_mag,
            "magnitude_ratio": mean_mag / max(normal_mag, 1e-10),
        }
    return stats
