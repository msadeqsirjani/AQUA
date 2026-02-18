"""
End-to-end ASQ training script (Zhou et al. 2025).

Usage: python -m scripts.asq.train

Steps:
1. Load pretrained FP32 ResNet-18 on CIFAR-10
2. Create two models: ASQ+POST (paper) and LSQ (baseline)
3. Initialize scales via calibration
4. Fine-tune both for 30 epochs (MLP frozen for first 5)
5. Compare 4-bit ASQ vs 4-bit LSQ vs FP32
"""

import argparse
import os
import copy
import time
import json

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models import get_model
from utils import (
    get_dataloaders, setup_device,
    load_fp32_model, save_results,
)
from utils.console import console, banner, section, print_config, metric, info, success, file_saved
from utils.args import add_common_args, get_result_dir
from utils.config import load_config
from core.asq.asq_quantizer import ASQActivationQuantizer, POSTWeightQuantizer
from core.asq.asq_model import (
    replace_with_asq, replace_with_lsq,
    init_quantizer_scales,
    ASQConv2d, ASQLinear, LSQConv2d, LSQLinear,
)


# ---------------------------------------------------------------------------
# Training with separate LR groups
# ---------------------------------------------------------------------------

def build_param_groups(model, lr_weights, lr_scale, lr_mlp):
    """Build parameter groups with different learning rates.

    - Model weights (conv, linear, BN): lr_weights
    - Quantizer s_base / s / scale params: lr_scale
    - ASQ MLP parameters: lr_mlp
    """
    weight_params = []
    scale_params = []
    mlp_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".mlp." in name:
            mlp_params.append(param)
        elif name.endswith(".s_base") or name.endswith(".s") or name.endswith(".scale"):
            scale_params.append(param)
        else:
            weight_params.append(param)

    groups = [
        {"params": weight_params, "lr": lr_weights},
        {"params": scale_params, "lr": lr_scale},
    ]
    if mlp_params:
        groups.append({"params": mlp_params, "lr": lr_mlp})

    return groups


def train_quantized(model, train_loader, val_loader, device, epochs=30,
                    lr_weights=1e-4, lr_scale=1e-3, lr_mlp=5e-4,
                    mlp_warmup_epochs=5, label="Model"):
    """Fine-tune a quantized model with separate LR groups.

    For ASQ models: freeze MLP for first mlp_warmup_epochs to stabilize s_base.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Freeze MLP during warmup (only affects ASQ models)
    has_mlp = False
    for m in model.modules():
        if isinstance(m, ASQActivationQuantizer):
            m.freeze_mlp()
            has_mlp = True

    param_groups = build_param_groups(model, lr_weights, lr_scale, lr_mlp)
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "delta_s_stats": []}

    for epoch in range(1, epochs + 1):
        # Unfreeze MLP after warmup
        if epoch == mlp_warmup_epochs + 1 and has_mlp:
            for m in model.modules():
                if isinstance(m, ASQActivationQuantizer):
                    m.unfreeze_mlp()
            # Rebuild optimizer to include MLP params
            param_groups = build_param_groups(model, lr_weights, lr_scale, lr_mlp)
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
            # Reset scheduler for remaining epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1
            )
            info(f"[{label}] MLP unfrozen at epoch {epoch}")

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # skip corrupted batches
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += inputs.size(0)
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Collect delta_s stats for ASQ models
        if has_mlp and epoch % 5 == 0:
            delta_vals = _collect_delta_s(model, train_loader, device)
            history["delta_s_stats"].append({
                "epoch": epoch,
                "mean": float(np.mean(delta_vals)),
                "std": float(np.std(delta_vals)),
                "min": float(np.min(delta_vals)),
                "max": float(np.max(delta_vals)),
            })

        if epoch % 5 == 0 or epoch == 1:
            info(f"[{label}] Epoch {epoch:2d}/{epochs} | "
                 f"Loss: {train_loss:.4f} | "
                 f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return best_acc, history


@torch.no_grad()
def _collect_delta_s(model, data_loader, device, n_batches=5):
    """Collect delta_s values from ASQ quantizers to check MLP is learning."""
    model.eval()
    delta_values = []

    hooks = []

    def make_hook(quantizer):
        def hook_fn(module, inp, out):
            x = inp[0]
            if x.dim() < 2:
                return
            B = x.size(0)
            C = x.size(1)
            if x.dim() == 4:
                x_mean = x.mean(dim=[2, 3])
                x_std = x.std(dim=[2, 3])
                x_max = x.amax(dim=[2, 3])
            else:
                x_mean = x
                x_std = torch.zeros_like(x)
                x_max = x.abs()
            features = torch.cat([x_mean, x_std, x_max], dim=1)
            ds = quantizer.mlp(features)
            ds = torch.tanh(ds) * quantizer.delta_scale_bound
            delta_values.extend(ds.cpu().numpy().flatten().tolist())
        return hook_fn

    for m in model.modules():
        if isinstance(m, (ASQConv2d, ASQLinear)):
            hooks.append(m.register_forward_hook(make_hook(m.act_quantizer)))

    for i, (inputs, _) in enumerate(data_loader):
        if i >= n_batches:
            break
        model(inputs.to(device))

    for h in hooks:
        h.remove()

    return delta_values if delta_values else [0.0]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def benchmark_forward(model, device, input_shape=(64, 3, 32, 32), n_iter=100):
    """Benchmark forward pass latency."""
    model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / n_iter * 1000  # ms per forward


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(asq_hist, lsq_hist, save_path):
    """Plot training curves for ASQ vs LSQ."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(asq_hist["val_acc"]) + 1)

    ax1.plot(epochs, asq_hist["train_loss"], label="ASQ train loss", linewidth=2)
    ax1.plot(epochs, lsq_hist["train_loss"], label="LSQ train loss", linewidth=2,
             linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, asq_hist["val_acc"], label="ASQ val acc", linewidth=2)
    ax2.plot(epochs, lsq_hist["val_acc"], label="LSQ val acc", linewidth=2,
             linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("4-bit ASQ+POST vs LSQ on CIFAR-10 ResNet-18")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_delta_s(asq_hist, save_path):
    """Plot delta_s statistics over training to verify MLP is learning."""
    stats = asq_hist.get("delta_s_stats", [])
    if not stats:
        info("No delta_s stats to plot (MLP may not have been active).")
        return

    epochs = [s["epoch"] for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    mins = [s["min"] for s in stats]
    maxs = [s["max"] for s in stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(epochs, mins, maxs, alpha=0.2, color="#2196F3", label="min-max range")
    ax.fill_between(epochs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.4, color="#2196F3", label="mean ± std")
    ax.plot(epochs, means, "o-", color="#1565C0", linewidth=2, label="mean delta_s")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("delta_s value")
    ax.set_title("ASQ delta_s Distribution Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASQ training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("AQUA — ASQ Training", f"Config: {args.config}")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "asq")
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        data_root=cfg.data_root,
    )

    # ===== Step 1: Load pretrained FP32 =====
    section(f"Load Pretrained FP32 {cfg.model} ({cfg.dataset})", step=1)

    model_fp32, fp32_acc = load_fp32_model(
        cfg.model, device, val_loader,
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dataset_name=cfg.dataset,
    )

    # ===== Step 2: Create ASQ and LSQ models =====
    section("Create 4-bit ASQ+POST and LSQ Models", step=2)

    bits = cfg.bits

    asq_model = copy.deepcopy(model_fp32)
    asq_model = replace_with_asq(asq_model, bits=bits)
    n_asq = sum(1 for m in asq_model.modules() if isinstance(m, (ASQConv2d, ASQLinear)))
    metric("ASQ+POST model", f"{n_asq} quantized layers")

    lsq_model = copy.deepcopy(model_fp32)
    lsq_model = replace_with_lsq(lsq_model, bits=bits)
    n_lsq = sum(1 for m in lsq_model.modules() if isinstance(m, (LSQConv2d, LSQLinear)))
    metric("LSQ baseline model", f"{n_lsq} quantized layers")

    # Count MLP overhead
    mlp_params = sum(p.numel() for n, p in asq_model.named_parameters() if ".mlp." in n)
    total_params = sum(p.numel() for p in asq_model.parameters())
    metric("ASQ MLP parameter overhead",
           f"{mlp_params:,} / {total_params:,} ({100*mlp_params/total_params:.2f}%)")

    # ===== Step 3: Initialize scales =====
    section("Initialize Quantizer Scales (200 batches)", step=3)

    info("ASQ model:")
    init_quantizer_scales(asq_model, train_loader, device, n_batches=200)
    info("LSQ model:")
    init_quantizer_scales(lsq_model, train_loader, device, n_batches=200)

    # ===== Step 4: Fine-tune =====
    section("Fine-Tune (30 epochs, MLP frozen for first 5)", step=4)

    info("--- ASQ+POST ---")
    asq_acc, asq_hist = train_quantized(
        asq_model, train_loader, val_loader, device,
        epochs=30, lr_weights=1e-4, lr_scale=1e-3, lr_mlp=5e-4,
        mlp_warmup_epochs=5, label="ASQ",
    )

    info("--- LSQ Baseline ---")
    lsq_acc, lsq_hist = train_quantized(
        lsq_model, train_loader, val_loader, device,
        epochs=30, lr_weights=1e-4, lr_scale=1e-3, lr_mlp=5e-4,
        mlp_warmup_epochs=0, label="LSQ",
    )

    # ===== Step 5: Overhead benchmark =====
    section("Forward Pass Overhead", step=5)

    asq_time = benchmark_forward(asq_model, device)
    lsq_time = benchmark_forward(lsq_model, device)
    overhead = (asq_time / lsq_time - 1.0) * 100
    metric("ASQ forward", f"{asq_time:.2f} ms")
    metric("LSQ forward", f"{lsq_time:.2f} ms")
    metric("ASQ overhead", f"{overhead:+.1f}%")
    if overhead < 5.0:
        success("Overhead check: PASS (< 5%)")
    else:
        info(f"Overhead check: {overhead:.1f}% (target < 5%)")

    # ===== Results =====
    section("Results Summary")
    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("4-bit LSQ", f"{lsq_acc:.2f}%")
    metric("4-bit ASQ+POST", f"{asq_acc:.2f}%")
    metric("ASQ improvement", f"{asq_acc - lsq_acc:+.2f}%")

    if asq_acc > lsq_acc:
        success("Status: PASS (ASQ beats LSQ)")
    else:
        info("Status: ASQ did not beat LSQ (expected 0.3-0.8% improvement)")

    # ===== Plots =====
    section("Generating Plots")

    plot_comparison(asq_hist, lsq_hist, os.path.join(result_dir, "asq_vs_lsq_comparison.png"))
    plot_delta_s(asq_hist, os.path.join(result_dir, "asq_delta_s_distribution.png"))

    # Save results
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "lsq_4bit_acc": lsq_acc,
        "asq_4bit_acc": asq_acc,
        "asq_improvement": asq_acc - lsq_acc,
        "asq_overhead_pct": overhead,
        "mlp_param_overhead_pct": 100 * mlp_params / total_params,
        "bits": bits,
        "asq_delta_s_stats": asq_hist.get("delta_s_stats", []),
        "summary": {
            "method": "ASQ",
            "fp32_acc": fp32_acc,
            "method_acc": asq_acc,
            "int8_acc": None,
            "int4_acc": lsq_acc,
            "avg_bits": float(bits),
            "per_layer_bits": {},
            "training_history": [
                {"epoch": e + 1, "val_acc": va}
                for e, va in enumerate(asq_hist["val_acc"])
            ],
            "category": "learnable",
        },
    }
    save_results(results, os.path.join(result_dir, "asq_results.json"))

    model_path = os.path.join(result_dir, f"{cfg.model}_asq_{bits}bit.pt")
    torch.save(asq_model.state_dict(), model_path)
    file_saved(model_path)


if __name__ == "__main__":
    main()
