"""
AQUA v2: Outlier-Aware Distribution-Guided Mixed-Precision Quantization.

Analytical pipeline:
  1. Load FP32 model
  2. Compute Hessian / Fisher sensitivity per layer
  3. Analyze weight distributions (kurtosis, skewness, outlier ratio)
  4. Detect outliers (top-k by |w| * sqrt(Fisher_diag))
  5. Compute quantization error per layer x dtype (excluding outliers)
  6. Solve dtype assignment via greedy Pareto under BOPs budget
  7. Build AQUA model with fixed dtypes + outlier masks
  8. Fine-tune with simple QAT + knowledge distillation
  9. Save results + plots

Usage:
    python -m scripts.aqua.train --config configs/cifar10_resnet18_aqua.yaml
"""

import argparse
import copy
import os
import json
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..', '..')))

from rich.table import Table
from rich import box

from models import get_model
from utils import (get_dataloaders, setup_device, load_fp32_model,
                   save_results, evaluate)
from utils.console import (console, banner, section, print_config, metric,
                            info, success, warning, file_saved)
from utils.args import add_common_args, get_result_dir
from utils.config import load_config

from core.aqua.quantizers import resolve_dtypes, DTYPE_CATALOG
from core.aqua.distribution_analysis import (
    analyze_layer_distributions, recommend_format,
)
from core.aqua.outlier_detector import (
    compute_fisher_diagonal, compute_outlier_masks, compute_outlier_stats,
)
from core.aqua.dtype_selector import (
    compute_dtype_errors, build_bops_table, solve_dtype_assignment,
    compute_assignment_summary,
)
from core.aqua.aqua_model import (
    replace_with_aqua, AQUAConv2d, AQUALinear,
    get_aqua_layer_stats, compute_per_layer_quant_error,
    get_avg_bits, get_dtype_counts, count_aqua_layers,
)
from core.aqua.aqua_trainer import AQUATrainer
from core.hessian_sensitivity import compute_hessian_trace


# ===================================================================
# Console helpers
# ===================================================================

def _short_name(name):
    return name.rsplit(".", 1)[-1] if "." in name else name


def _dtype_counts_str(dtype_counts):
    parts = [f"{dt}:{n}" for dt, n in sorted(dtype_counts.items(),
                                              key=lambda x: -x[1])]
    return " ".join(parts)


def log_epoch(epoch, total, stats, val_acc):
    console.print(
        f"  Epoch {epoch:3d}/{total}"
        f"  Loss [yellow]{stats['train_loss']:.4f}[/yellow]"
        f"  Train [cyan]{stats['train_acc']:.1f}%[/cyan]"
        f"  Val [green]{val_acc:.1f}%[/green]"
        f"  CE={stats['l_ce']:.4f}  KL={stats['l_kl']:.4f}"
        f"  LR={stats['lr_weights']:.6f}"
    )


def print_assignment_table(assignment_summary, layer_stats, title=None):
    """Rich table showing the analytical dtype assignment."""
    table = Table(
        title=title or "Per-Layer Dtype Assignment",
        box=box.ROUNDED, border_style="blue",
        header_style="bold cyan", show_lines=False,
    )
    table.add_column("Layer", style="white", min_width=22)
    table.add_column("Dtype", justify="center", style="magenta")
    table.add_column("Bits", justify="right", style="yellow")
    table.add_column("Kurtosis", justify="right", style="dim")
    table.add_column("Format Rec", justify="center", style="dim")
    table.add_column("Sensitivity", justify="right", style="red")
    table.add_column("Outliers", justify="right", style="cyan")

    per_layer = assignment_summary["per_layer"]
    for name, info_dict in per_layer.items():
        short = _short_name(name)
        st = layer_stats.get(name, {})
        kurt = st.get("kurtosis", 0)
        rec = recommend_format(kurt)
        sens = info_dict.get("sensitivity", 0)
        outlier_str = ""
        table.add_row(
            short,
            info_dict["dtype"],
            str(info_dict["bits"]),
            f"{kurt:.1f}",
            rec.upper(),
            f"{sens:.2e}",
            outlier_str,
        )

    dt_counts = assignment_summary["dtype_counts"]
    dt_summary = " / ".join(f"{n} {dt}" for dt, n in
                            sorted(dt_counts.items(), key=lambda x: -x[1]))
    table.add_section()
    table.add_row(
        "[bold]Summary[/bold]", f"[bold]{dt_summary}[/bold]",
        f"[bold]{assignment_summary['avg_bits']:.2f}[/bold]",
        "", "",
        f"BOPs ratio: {assignment_summary['bops_ratio']:.3f}",
        "",
    )
    console.print(table)


def print_outlier_table(outlier_stats):
    """Rich table showing outlier detection results."""
    table = Table(
        title="Outlier Detection Results",
        box=box.ROUNDED, border_style="cyan",
        header_style="bold cyan", show_lines=False,
    )
    table.add_column("Layer", style="white", min_width=22)
    table.add_column("Outliers", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="dim")
    table.add_column("Pct", justify="right", style="magenta")
    table.add_column("Outlier |w|", justify="right", style="red")
    table.add_column("Normal |w|", justify="right", style="green")
    table.add_column("Ratio", justify="right", style="cyan")

    for name, st in outlier_stats.items():
        short = _short_name(name)
        table.add_row(
            short,
            str(st["n_outlier"]),
            str(st["n_total"]),
            f"{st['pct']:.2f}%",
            f"{st['outlier_mean_abs']:.4f}",
            f"{st['normal_mean_abs']:.4f}",
            f"{st['magnitude_ratio']:.1f}x",
        )
    console.print(table)


def print_distribution_table(dist_stats):
    """Rich table showing weight distribution analysis."""
    table = Table(
        title="Weight Distribution Analysis",
        box=box.ROUNDED, border_style="green",
        header_style="bold green", show_lines=False,
    )
    table.add_column("Layer", style="white", min_width=22)
    table.add_column("Kurtosis", justify="right", style="yellow")
    table.add_column("Skewness", justify="right", style="dim")
    table.add_column("Std", justify="right", style="dim")
    table.add_column("Sparsity", justify="right", style="cyan")
    table.add_column("Outlier %", justify="right", style="red")
    table.add_column("Recommendation", justify="center", style="magenta")

    for name, st in dist_stats.items():
        short = _short_name(name)
        rec = recommend_format(st["kurtosis"])
        table.add_row(
            short,
            f"{st['kurtosis']:.2f}",
            f"{st['skewness']:.2f}",
            f"{st['std']:.4f}",
            f"{st['sparsity']*100:.1f}%",
            f"{st['outlier_ratio']*100:.2f}%",
            rec.upper(),
        )
    console.print(table)


# ===================================================================
# Plotting helpers
# ===================================================================

def plot_training_curves(history, save_path):
    """Loss and accuracy curves for QAT fine-tuning."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history],
            lw=2, label="Total", color="#333")
    ax.plot(epochs, [h["l_ce"] for h in history],
            lw=1.5, ls="--", label="CE", color="#2196F3")
    ax.plot(epochs, [h["l_kl"] for h in history],
            lw=1.5, ls="--", label="KL", color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("QAT Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [h["train_acc"] for h in history],
            lw=2, label="Train", color="#2196F3")
    ax.plot(epochs, [h["val_acc"] for h in history],
            lw=2, label="Val", color="#4CAF50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("QAT Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("AQUA v2 Fine-Tuning", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_dtype_assignment(assignment_summary, save_path):
    """Per-layer dtype assignment bar chart."""
    per_layer = assignment_summary["per_layer"]
    names = [_short_name(n) for n in per_layer]
    bits = [per_layer[n]["bits"] for n in per_layer]
    dtypes = [per_layer[n]["dtype"] for n in per_layer]

    dtype_colors = {
        "fp32": "#E91E63", "fp16": "#FF5722", "fp8": "#FF9800",
        "int8": "#2196F3", "fp4": "#FFC107", "int4": "#4CAF50",
        "int2": "#009688", "int1": "#607D8B",
    }
    colors = [dtype_colors.get(dt, "#9E9E9E") for dt in dtypes]

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                    figsize=(max(10, len(names) * 0.6), 8))

    ax1.bar(range(len(names)), bits, color=colors,
            edgecolor="white", lw=0.5)
    ax1.set_ylabel("Bit-Width")
    ax1.set_title("Assigned Bit-Width Per Layer")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.axhline(assignment_summary["avg_bits"], color="red", ls="--",
                alpha=0.5, label=f"avg={assignment_summary['avg_bits']:.2f}")
    ax1.legend()

    sensitivities = [per_layer[n]["sensitivity"] for n in per_layer]
    ax2.bar(range(len(names)), sensitivities, color="#F44336",
            edgecolor="white", lw=0.5)
    ax2.set_ylabel("Hessian Sensitivity")
    ax2.set_title("Layer Sensitivity (higher = needs more bits)")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.2, axis="y")

    from matplotlib.patches import Patch
    unique_dtypes = sorted(set(dtypes))
    legend_elements = [Patch(facecolor=dtype_colors.get(dt, "#9E9E9E"),
                             label=dt) for dt in unique_dtypes]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_distribution_analysis(dist_stats, save_path):
    """Kurtosis and skewness per layer."""
    names = [_short_name(n) for n in dist_stats]
    kurtosis = [dist_stats[n]["kurtosis"] for n in dist_stats]
    skewness = [dist_stats[n]["skewness"] for n in dist_stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(len(names)), kurtosis, color="#FF9800",
            edgecolor="white", lw=0.5)
    ax1.axhline(4.0, color="red", ls="--", alpha=0.7,
                label="Threshold (4.0)")
    ax1.set_ylabel("Kurtosis")
    ax1.set_title("Weight Kurtosis Per Layer")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis="y")

    ax2.bar(range(len(names)), skewness, color="#9C27B0",
            edgecolor="white", lw=0.5)
    ax2.set_ylabel("|Skewness|")
    ax2.set_title("Weight |Skewness| Per Layer")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_quant_error(quant_errors, save_path):
    """Per-layer quantization error bar chart."""
    names = [_short_name(e["name"]) for e in quant_errors]
    mses = [e["mse"] for e in quant_errors]
    bits_list = [e["bits"] for e in quant_errors]

    fig, ax1 = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
    x = np.arange(len(names))

    ax1.bar(x, mses, color="#F44336", alpha=0.7, edgecolor="white")
    ax1.set_ylabel("MSE", color="#F44336")
    ax1.set_xlabel("Layer")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax1.tick_params(axis="y", labelcolor="#F44336")

    ax2 = ax1.twinx()
    ax2.plot(x, bits_list, "o-", color="#4CAF50", lw=2, markersize=5,
             label="Bit-width")
    ax2.set_ylabel("Bit-width", color="#4CAF50")
    ax2.tick_params(axis="y", labelcolor="#4CAF50")

    fig.suptitle("Per-Layer Quantization Error vs Bit-Width",
                 fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.92))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_lr_schedule(history, save_path):
    """Learning rate schedule."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, [h["lr_weights"] for h in history],
            lw=2, label="Weights (SGD)", color="#2196F3")
    ax.plot(epochs, [h["lr_scales"] for h in history],
            lw=2, label="Scales (Adam)", color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AQUA v2: Outlier-Aware Distribution-Guided Mixed-Precision")
    add_common_args(parser)
    parser.add_argument("--allowed-dtypes", type=str, nargs="+", default=None,
                        help="Allowed dtypes (e.g. int8 fp8 int4 fp4 int2)")
    parser.add_argument("--blocked-dtypes", type=str, nargs="+", default=None,
                        help="Blocked dtypes (e.g. fp32 fp16 int1)")
    parser.add_argument("--target-avg-bits", type=float, default=None,
                        help="Target average bit-width (default 4.0). "
                             "Converted to BOPs ratio internally.")
    parser.add_argument("--target-bops-ratio", type=float, default=None,
                        help="Target BOPs ratio vs fp32 (overrides --target-avg-bits). "
                             "e.g. 0.0156=~4bit, 0.0625=~8bit")
    parser.add_argument("--outlier-pct", type=float, default=None,
                        help="Fraction of weights to protect as outliers (default 0.01)")
    parser.add_argument("--finetune-epochs", type=int, default=None,
                        help="QAT fine-tuning epochs (default 30)")
    parser.add_argument("--finetune-lr", type=float, default=None,
                        help="QAT learning rate for weights (default 0.0001)")
    parser.add_argument("--kd-temperature", type=float, default=None,
                        help="Knowledge distillation temperature (default 4.0)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="KL distillation weight (default 0.5)")
    parser.add_argument("--hessian-samples", type=int, default=None,
                        help="Hessian trace estimator samples (default 100)")
    parser.add_argument("--fisher-batches", type=int, default=None,
                        help="Fisher diagonal batches (default 5)")
    args = parser.parse_args()
    cfg = load_config(args.config, args)

    banner("AQUA v2 -- Outlier-Aware Distribution-Guided Mixed-Precision",
           f"Config: {args.config}")
    device = setup_device()

    # ---- hyperparameters (CLI > config > default) ----
    def _hp(cli_val, cfg_name, default):
        if cli_val is not None:
            return cli_val
        return getattr(cfg, cfg_name, default)

    # BOPs budget: --target-bops-ratio overrides --target-avg-bits
    if args.target_bops_ratio is not None:
        target_bops_ratio = args.target_bops_ratio
        target_avg_bits = 32.0 * (target_bops_ratio ** 0.5)
    else:
        target_avg_bits = _hp(args.target_avg_bits, "target_avg_bits", 4.0)
        from core.aqua.dtype_selector import avg_bits_to_bops_ratio
        target_bops_ratio = avg_bits_to_bops_ratio(target_avg_bits)

    outlier_pct = _hp(args.outlier_pct, "outlier_pct", 0.01)
    finetune_epochs = _hp(args.finetune_epochs, "finetune_epochs", 30)
    finetune_lr = _hp(args.finetune_lr, "finetune_lr", 0.0001)
    kd_temperature = _hp(args.kd_temperature, "kd_temperature", 4.0)
    alpha = _hp(args.alpha, "alpha", 0.5)
    hessian_samples = _hp(args.hessian_samples, "hessian_samples", 100)
    fisher_batches = _hp(args.fisher_batches, "fisher_batches", 5)
    run_tag = args.run_tag

    # ---- dtype resolution ----
    cfg_allowed = getattr(cfg, "allowed_dtypes", None)
    cfg_blocked = getattr(cfg, "blocked_dtypes", None)
    allowed_dtypes = resolve_dtypes(
        allowed=args.allowed_dtypes or cfg_allowed,
        blocked=args.blocked_dtypes or cfg_blocked,
    )

    print_config(cfg, extra={
        "Allowed dtypes": ", ".join(allowed_dtypes),
        "Target avg bits": f"{target_avg_bits:.1f} (BOPs ratio: {target_bops_ratio:.4f} vs fp32)",
        "Outlier %": f"{outlier_pct*100:.1f}%",
        "Finetune epochs": finetune_epochs,
        "Finetune LR": finetune_lr,
        "KD temperature": kd_temperature,
        "alpha (KL)": alpha,
        "Hessian samples": hessian_samples,
        "Fisher batches": fisher_batches,
        "run_tag": run_tag or "(default)",
    })

    result_dir = get_result_dir(cfg.dataset, cfg.model, "aqua", run_tag=run_tag)
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size, data_root=cfg.data_root,
    )

    # ===== Step 1: Load pretrained FP32 =====
    section(f"Load Pretrained FP32 {cfg.model} ({cfg.dataset})", step=1)
    model_fp32, fp32_acc = load_fp32_model(
        cfg.model, device, val_loader,
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dataset_name=cfg.dataset,
    )

    # ===== Step 2: Hessian sensitivity =====
    section("Compute Hessian Trace Sensitivity", step=2)
    t0 = time.time()
    hessian_traces = compute_hessian_trace(
        model_fp32, nn.CrossEntropyLoss(), train_loader,
        n_samples=hessian_samples, device=device,
    )
    t_hessian = time.time() - t0
    metric("Hessian computation time", f"{t_hessian:.1f}s")
    metric("Layers analyzed", len(hessian_traces))

    sens_sorted = sorted(hessian_traces.items(), key=lambda x: -x[1])
    info(f"Most sensitive: {_short_name(sens_sorted[0][0])} "
         f"({sens_sorted[0][1]:.2e})")
    info(f"Least sensitive: {_short_name(sens_sorted[-1][0])} "
         f"({sens_sorted[-1][1]:.2e})")

    # ===== Step 3: Weight distribution analysis =====
    section("Analyze Weight Distributions", step=3)
    dist_stats = analyze_layer_distributions(model_fp32)
    metric("Layers analyzed", len(dist_stats))
    print_distribution_table(dist_stats)

    n_fp_rec = sum(1 for st in dist_stats.values()
                   if recommend_format(st["kurtosis"]) == "fp")
    n_int_rec = len(dist_stats) - n_fp_rec
    info(f"Format recommendations: {n_int_rec} INT, {n_fp_rec} FP")

    # ===== Step 4: Outlier detection =====
    section("Detect Sensitive Outlier Weights", step=4)
    t0 = time.time()
    fisher_diag = compute_fisher_diagonal(
        model_fp32, nn.CrossEntropyLoss(), train_loader,
        device=device, n_batches=fisher_batches,
    )
    outlier_masks = compute_outlier_masks(model_fp32, fisher_diag,
                                          outlier_pct=outlier_pct)
    t_outlier = time.time() - t0
    metric("Outlier detection time", f"{t_outlier:.1f}s")

    outlier_st = compute_outlier_stats(model_fp32, outlier_masks)
    print_outlier_table(outlier_st)

    total_outliers = sum(s["n_outlier"] for s in outlier_st.values())
    total_weights = sum(s["n_total"] for s in outlier_st.values())
    metric("Total outlier weights",
           f"{total_outliers:,} / {total_weights:,} "
           f"({100*total_outliers/max(total_weights,1):.2f}%)")

    # ===== Step 5: Compute quantization errors =====
    section("Compute Per-Layer Per-Dtype Quantization Errors", step=5)
    t0 = time.time()
    dtype_errors = compute_dtype_errors(
        model_fp32, allowed_dtypes, hessian_traces,
        outlier_masks=outlier_masks, device=device,
    )
    t_errors = time.time() - t0
    metric("Error computation time", f"{t_errors:.1f}s")
    metric("Combinations tested",
           f"{len(dtype_errors)} layers x {len(allowed_dtypes)} dtypes "
           f"= {len(dtype_errors)*len(allowed_dtypes)}")

    # ===== Step 6: Solve dtype assignment =====
    section("Solve Dtype Assignment (Greedy Pareto)", step=6)

    img_size = cfg.img_size
    in_channels = 3
    input_shape = (1, in_channels, img_size, img_size)
    bops_table, ref_bops = build_bops_table(
        model_fp32, allowed_dtypes, input_shape=input_shape,
    )

    layer_names = [n for n in dtype_errors if n in bops_table]

    assignment = solve_dtype_assignment(
        layer_names, dtype_errors, bops_table,
        ref_bops, target_bops_ratio=target_bops_ratio,
        allowed_dtypes=allowed_dtypes,
    )

    assignment_summary = compute_assignment_summary(
        assignment, dtype_errors, bops_table, ref_bops, hessian_traces,
    )

    print_assignment_table(assignment_summary, dist_stats)
    metric("Average bit-width", f"{assignment_summary['avg_bits']:.2f}")
    metric("BOPs ratio", f"{assignment_summary['bops_ratio']:.4f} "
           f"(target: {target_bops_ratio})")
    metric("Total weighted error", f"{assignment_summary['total_error']:.4e}")

    # ===== Step 7: Build AQUA model =====
    section("Build AQUA Model with Fixed Dtypes + Outlier Masks", step=7)
    aqua_model = copy.deepcopy(model_fp32)
    aqua_model = replace_with_aqua(aqua_model, assignment,
                                    outlier_masks=outlier_masks)
    aqua_model.to(device)

    n_aqua = count_aqua_layers(aqua_model)
    avg_bits = get_avg_bits(aqua_model)
    dt_counts = get_dtype_counts(aqua_model)
    total_params = sum(p.numel() for p in aqua_model.parameters())

    metric("AQUA layers", n_aqua)
    metric("Avg bits (param-weighted)", f"{avg_bits:.2f}")
    metric("Dtype distribution", _dtype_counts_str(dt_counts))
    metric("Total parameters", f"{total_params:,}")

    pre_qat_acc = evaluate(aqua_model, val_loader, device)
    metric("Pre-QAT accuracy", f"{pre_qat_acc:.2f}% "
           f"(delta: {pre_qat_acc - fp32_acc:+.2f}%)")

    # ===== Step 8: QAT Fine-tuning =====
    section(f"QAT Fine-Tuning ({finetune_epochs} epochs)", step=8)
    info(f"CE + {alpha}*T^2*KL, T={kd_temperature}")
    info(f"SGD(weights, lr={finetune_lr}) + Adam(scales, lr=0.001)")
    info(f"CosineAnnealingLR, grad clip=1.0")

    trainer = AQUATrainer(
        aqua_model, model_fp32,
        alpha=alpha,
        kd_temperature=kd_temperature,
        lr_weights=finetune_lr,
        lr_scales=0.001,
        epochs=finetune_epochs,
    )

    best_acc = pre_qat_acc
    best_state = None
    history = []

    for epoch in range(1, finetune_epochs + 1):
        stats = trainer.train_epoch(train_loader, device, epoch)
        val_acc = trainer.validate(val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(aqua_model.state_dict())

        record = {"epoch": epoch, "val_acc": val_acc, **stats}
        history.append(record)

        log_epoch(epoch, finetune_epochs, stats, val_acc)

    if best_state is not None:
        aqua_model.load_state_dict(best_state)

    # ===== Step 9: Results summary =====
    section("Results Summary", step=9)
    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("Pre-QAT accuracy", f"{pre_qat_acc:.2f}%")
    metric("AQUA best (post-QAT)", f"{best_acc:.2f}%")
    metric("Delta vs FP32", f"{best_acc - fp32_acc:+.2f}%")

    final_avg_bits = get_avg_bits(aqua_model)
    final_dt_counts = get_dtype_counts(aqua_model)
    metric("Average bit-width", f"{final_avg_bits:.2f}")
    metric("Dtype split", _dtype_counts_str(final_dt_counts))

    quant_errors = compute_per_layer_quant_error(aqua_model)
    avg_mse = np.mean([e["mse"] for e in quant_errors])
    metric("Avg quantization error (MSE)", f"{avg_mse:.6f}")

    if best_acc >= fp32_acc - 1.0:
        success(f"Accuracy within 1% of FP32 at {final_avg_bits:.1f} avg bits!")
    else:
        warning(f"Accuracy gap: {fp32_acc - best_acc:.2f}% "
                f"-- consider increasing target_bops_ratio or reducing outlier_pct")

    layer_stats_final = get_aqua_layer_stats(aqua_model)

    err_table = Table(title="Per-Layer Quantization Error (Final)",
                      box=box.ROUNDED, border_style="red",
                      header_style="bold red")
    err_table.add_column("Layer", style="white", min_width=22)
    err_table.add_column("Dtype", justify="center", style="magenta")
    err_table.add_column("Bits", justify="right", style="yellow")
    err_table.add_column("Outliers", justify="right", style="cyan")
    err_table.add_column("MSE", justify="right", style="red")
    err_table.add_column("Weight Norm", justify="right", style="dim")

    for e, ls in zip(quant_errors, layer_stats_final):
        short = _short_name(e["name"])
        err_table.add_row(
            short, e["dtype"], str(e["bits"]),
            str(ls["n_outlier"]),
            f"{e['mse']:.6f}", f"{e['w_norm']:.3f}",
        )
    console.print(err_table)

    # ===== Step 10: Plots =====
    section("Generating Plots", step=10)

    plot_training_curves(history,
                         os.path.join(result_dir, "aqua_training_curves.png"))
    plot_dtype_assignment(assignment_summary,
                          os.path.join(result_dir, "aqua_dtype_assignment.png"))
    plot_distribution_analysis(dist_stats,
                               os.path.join(result_dir,
                                            "aqua_distribution_analysis.png"))
    plot_quant_error(quant_errors,
                     os.path.join(result_dir, "aqua_quant_error.png"))
    plot_lr_schedule(history,
                     os.path.join(result_dir, "aqua_lr_schedule.png"))

    # ===== Step 11: Save results =====
    section("Saving Results", step=11)

    per_layer_dtype = {s["name"]: s["dtype"] for s in layer_stats_final}
    per_layer_bits = {s["name"]: s["bits"] for s in layer_stats_final}

    results = {
        "run_tag": run_tag,
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "pre_qat_acc": pre_qat_acc,
        "aqua_acc": best_acc,
        "delta_vs_fp32": best_acc - fp32_acc,
        "avg_bits": float(final_avg_bits),
        "avg_quant_error_mse": float(avg_mse),
        "allowed_dtypes": allowed_dtypes,
        "target_avg_bits": target_avg_bits,
        "target_bops_ratio": target_bops_ratio,
        "actual_bops_ratio": assignment_summary["bops_ratio"],
        "outlier_pct": outlier_pct,
        "dtype_counts": final_dt_counts,
        "per_layer_dtype": per_layer_dtype,
        "per_layer_bits": per_layer_bits,
        "per_layer_quant_error": {
            e["name"]: e["mse"] for e in quant_errors
        },
        "distribution_analysis": {
            name: {
                "kurtosis": s["kurtosis"],
                "skewness": s["skewness"],
                "format_recommendation": recommend_format(s["kurtosis"]),
            }
            for name, s in dist_stats.items()
        },
        "outlier_stats": {
            name: {"n_outlier": s["n_outlier"], "pct": s["pct"]}
            for name, s in outlier_st.items()
        },
        "hessian_traces": {name: float(v) for name, v in hessian_traces.items()},
        "hyperparameters": {
            "target_avg_bits": target_avg_bits,
            "target_bops_ratio": target_bops_ratio,
            "outlier_pct": outlier_pct,
            "finetune_epochs": finetune_epochs,
            "finetune_lr": finetune_lr,
            "kd_temperature": kd_temperature,
            "alpha": alpha,
            "hessian_samples": hessian_samples,
            "fisher_batches": fisher_batches,
            "optimizer_weights": "SGD(momentum=0.9, wd=1e-4)",
            "optimizer_scales": "Adam(lr=0.001)",
        },
        "timing": {
            "hessian_seconds": round(t_hessian, 1),
            "outlier_detection_seconds": round(t_outlier, 1),
            "error_computation_seconds": round(t_errors, 1),
        },
        "summary": {
            "method": "AQUA_v2",
            "fp32_acc": fp32_acc,
            "method_acc": best_acc,
            "avg_bits": float(final_avg_bits),
            "per_layer_bits": per_layer_bits,
            "per_layer_dtype": per_layer_dtype,
            "training_history": [
                {"epoch": h["epoch"], "val_acc": h["val_acc"],
                 "train_loss": h["train_loss"]}
                for h in history
            ],
            "category": "analytical+qat",
        },
    }
    save_results(results, os.path.join(result_dir, "aqua_results.json"))

    ckpt_path = os.path.join(result_dir, f"{cfg.model}_aqua.pt")
    torch.save({"model": aqua_model.state_dict()}, ckpt_path)
    file_saved(ckpt_path)

    success("AQUA v2 training complete!")


if __name__ == "__main__":
    main()
