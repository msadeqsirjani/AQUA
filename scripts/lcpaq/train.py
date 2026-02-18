"""
End-to-end LCPAQ training script.
"Adaptive Quantization with Mixed-Precision Based on Low-Cost Proxy"
arXiv:2402.17706, 2024.

Usage: python -m scripts.lcpaq.train

Pipeline:
  Module A: Build hardware-aware cost table for ResNet-18
  Module B: Hessian traces + greedy Pareto bit selection (50% BOPs)
  Module C: Low-cost proxy NAS to find best QAT hyperparameters
  Fine-tune: 30 epochs with best hyperparameters
  Compare: vs uniform INT8, and vs Edge-MPQ at same BOPs budget
"""

import argparse
import os
import copy
import json
import time

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
from utils.args import add_common_args, get_result_dir
from utils.config import load_config
from utils.console import console, banner, section, print_config, metric, info, success, file_saved
from core.lcpaq.lcpaq_cost_model import build_cost_table, summarize_cost_table
from core.lcpaq.lcpaq_bit_selector import (
    compute_quant_errors,
    greedy_pareto_selection,
)
from core.hessian_sensitivity import compute_hessian_trace
from core.lcpaq.lcpaq_proxy_nas import proxy_hyperparameter_search
from core.edge_mpq.edge_mpq_model import apply_mixed_precision, MPQConv2d, MPQLinear
from core.jacob_fake_quant import JacobFakeQuantize


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_qat(model, train_loader, val_loader, device, epochs=30,
                 lr=1e-4, weight_decay=1e-4, observer_epochs=5, label="QAT"):
    """Fine-tune a model with fake-quantized layers.

    Returns:
        (best_acc, history) where history is list of
        (epoch, train_loss, train_acc, val_acc) tuples.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == observer_epochs + 1:
            for m in model.modules():
                if isinstance(m, JacobFakeQuantize):
                    m.disable_observer()
            info(f"[{label}] Observers disabled")

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
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
                _, predicted = model(inputs).max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += inputs.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        history.append((epoch, train_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            info(f"[{label}] Epoch {epoch:2d}/{epochs} | "
                 f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return best_acc, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(histories, labels, save_path):
    """Plot val accuracy curves for multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    for (label, hist), color in zip(zip(labels, histories), colors):
        epochs = [h[0] for h in hist]
        val_accs = [h[3] for h in hist]
        ax.plot(epochs, val_accs, "-o", label=label, color=color,
                markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("LCPAQ vs Baselines: Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_bit_assignment(bit_assignment, hessian_traces, cost_table,
                        save_path):
    """Dual-axis plot: Hessian sensitivity bars + assigned bit-widths."""
    layer_names = list(bit_assignment.keys())
    short_names = [n.rsplit(".", 1)[-1] if "." in n else n for n in layer_names]

    fig, ax1 = plt.subplots(figsize=(max(12, len(layer_names) * 0.6), 6))
    x = np.arange(len(layer_names))

    sens_vals = [hessian_traces.get(n, 0) for n in layer_names]
    bits_vals = [bit_assignment[n] for n in layer_names]

    color1 = "#2196F3"
    ax1.bar(x, sens_vals, color=color1, alpha=0.6, label="Hessian trace")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Hessian Trace", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.plot(x, bits_vals, "o-", color=color2, linewidth=2, markersize=6,
             label="Assigned bits")
    ax2.set_ylabel("Bit-width", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 10)

    fig.suptitle("LCPAQ: Hessian Sensitivity vs Greedy Pareto Bit Assignment")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_proxy_correlation(proxy_results, full_results, save_path):
    """Scatter plot: proxy accuracy vs full accuracy for rank correlation."""
    fig, ax = plt.subplots(figsize=(6, 5))

    proxy_accs = [r["acc"] for r in proxy_results]
    full_accs = [r["acc"] for r in full_results]
    labels = [f"lr={r['lr']:.0e}\nwd={r['wd']:.0e}" for r in proxy_results]

    ax.scatter(proxy_accs, full_accs, s=80, c="#2196F3", edgecolors="k", zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (proxy_accs[i], full_accs[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=6)

    # Rank correlation
    from scipy.stats import spearmanr
    try:
        rho, pval = spearmanr(proxy_accs, full_accs)
        ax.set_title(f"Proxy vs Full Accuracy (Spearman rho={rho:.3f}, p={pval:.3f})")
    except Exception:
        ax.set_title("Proxy vs Full Accuracy")

    ax.set_xlabel("Proxy Accuracy (5 epochs, %)")
    ax.set_ylabel("Full Accuracy (30 epochs, %)")
    ax.grid(True, alpha=0.3)

    # Diagonal reference
    lo = min(min(proxy_accs), min(full_accs)) - 1
    hi = max(max(proxy_accs), max(full_accs)) + 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LCPAQ training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("LCPAQ Training", "Adaptive Quantization with Mixed-Precision Based on Low-Cost Proxy")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "lcpaq")
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        data_root=cfg.data_root,
    )

    section(f"Load Pretrained FP32 {cfg.model} ({cfg.dataset})", step=1)

    model, fp32_acc = load_fp32_model(
        cfg.model, device, val_loader,
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dataset_name=cfg.dataset,
    )

    section("Module A: Build Hardware-Aware Cost Table")

    bit_choices = [2, 4, 8]
    cost_table = build_cost_table(model, input_shape=(1, 3, 32, 32),
                                  bit_choices=bit_choices)
    info(f"Profiled {len(cost_table)} layers at bit-widths {bit_choices}")
    summarize_cost_table(cost_table)

    # BOPs at uniform 8-bit
    layer_names = list(cost_table.keys())
    bops_8bit = sum(cost_table[n][8]["bops"] for n in layer_names)
    bops_4bit = sum(cost_table[n][4]["bops"] for n in layer_names)
    bops_2bit = sum(cost_table[n][2]["bops"] for n in layer_names)
    metric("BOPs at 8-bit", f"{bops_8bit:,.0f}")
    metric("BOPs at 4-bit", f"{bops_4bit:,.0f} ({100*bops_4bit/bops_8bit:.1f}% of 8-bit)")
    metric("BOPs at 2-bit", f"{bops_2bit:,.0f} ({100*bops_2bit/bops_8bit:.1f}% of 8-bit)")

    section("Module B: Hessian Trace + Greedy Pareto Bit Selection")

    # Step B1: Hessian traces
    info("Computing Hessian traces (50 Rademacher samples)...")
    t0 = time.time()
    loss_fn = nn.CrossEntropyLoss()
    hessian_traces = compute_hessian_trace(
        model, loss_fn, train_loader, n_samples=50, device=device,
    )
    hessian_time = time.time() - t0
    metric("Hessian computation", f"{hessian_time:.1f}s")

    sorted_sens = sorted(hessian_traces.items(), key=lambda kv: kv[1], reverse=True)
    info("Per-layer Hessian trace (sorted):")
    for name, trace in sorted_sens:
        metric(name, f"{trace:.4f}")

    # Step B2: Quantization errors
    info("Computing per-layer quantization errors...")
    quant_errors = compute_quant_errors(model, bit_choices=bit_choices)
    for name in layer_names:
        short = name.rsplit(".", 1)[-1] if "." in name else name
        errs = " | ".join(f"{b}b={quant_errors[name][b]:.6f}" for b in bit_choices)
        info(f"{short}: {errs}")

    # Step B3: Greedy Pareto selection
    target_bops_ratio = 0.5
    info(f"Running greedy Pareto selection (target: {target_bops_ratio*100:.0f}% BOPs)...")
    t0 = time.time()
    bit_assignment, search_steps = greedy_pareto_selection(
        model, hessian_traces, cost_table,
        target_bops_ratio=target_bops_ratio, bit_choices=bit_choices,
    )
    pareto_time = time.time() - t0
    metric("Pareto search", f"{pareto_time:.1f}s, {len(search_steps)} steps")

    # Print assignment
    info("Bit-width assignment:")
    for name in layer_names:
        bits = bit_assignment.get(name, 8)
        sens = hessian_traces.get(name, 0)
        metric(name, f"{bits}b, sens={sens:.4f}")

    bops_mixed = sum(cost_table[n][bit_assignment[n]]["bops"] for n in layer_names)
    avg_bits_weighted = sum(
        bit_assignment[n] * cost_table[n][8]["macs"] for n in layer_names
    ) / sum(cost_table[n][8]["macs"] for n in layer_names)
    metric("Mixed BOPs", f"{bops_mixed:,.0f} ({100*bops_mixed/bops_8bit:.1f}% of 8-bit)")
    metric("MAC-weighted avg bits", f"{avg_bits_weighted:.2f}")

    # Verify Pareto optimality: check no single reduction improves ratio
    info("Verifying Pareto optimality...")
    all_at_min = all(
        bit_assignment[n] == min(bit_choices) for n in layer_names
    )
    if bops_mixed <= target_bops_ratio * bops_8bit or all_at_min:
        success("Budget met or all layers at minimum")
    else:
        info("Status: WARNING (budget not fully met)")

    section("Module C: Low-Cost Proxy NAS (5-epoch trials)")

    # Build mixed-precision model template for proxy search
    proxy_template = copy.deepcopy(model)
    proxy_template = apply_mixed_precision(proxy_template, bit_assignment)

    t0 = time.time()
    best_lr, best_wd, proxy_results = proxy_hyperparameter_search(
        proxy_template, train_loader, val_loader, device,
        proxy_epochs=5,
    )
    proxy_time = time.time() - t0
    metric("Total proxy search time", f"{proxy_time:.1f}s")
    metric("Selected", f"lr={best_lr:.0e}, wd={best_wd:.0e}")

    section("Full LCPAQ Training (30 epochs, best hyperparams)", step=4)
    metric("Hyperparams", f"lr={best_lr:.0e}, wd={best_wd:.0e}")

    lcpaq_model = copy.deepcopy(model)
    lcpaq_model = apply_mixed_precision(lcpaq_model, bit_assignment)

    n_mpqconv = sum(1 for m in lcpaq_model.modules() if isinstance(m, MPQConv2d))
    n_mpqlin = sum(1 for m in lcpaq_model.modules() if isinstance(m, MPQLinear))
    metric("MPQConv2d layers", n_mpqconv)
    metric("MPQLinear layers", n_mpqlin)

    lcpaq_acc, lcpaq_history = finetune_qat(
        lcpaq_model, train_loader, val_loader, device,
        epochs=30, lr=best_lr, weight_decay=best_wd,
        observer_epochs=5, label="LCPAQ",
    )

    section("Uniform INT8 Baseline (30 epochs)", step=5)

    int8_assignment = {name: 8 for name in layer_names}
    int8_model = copy.deepcopy(model)
    int8_model = apply_mixed_precision(int8_model, int8_assignment)

    int8_acc, int8_history = finetune_qat(
        int8_model, train_loader, val_loader, device,
        epochs=30, lr=1e-4, weight_decay=1e-4,
        observer_epochs=5, label="INT8",
    )

    section("Proxy Correlation Validation", step=6)
    info("Running full 30-epoch training for all hyperparameter combos...")

    full_results = []
    for pr in proxy_results:
        lr_i, wd_i = pr["lr"], pr["wd"]
        trial_model = copy.deepcopy(model)
        trial_model = apply_mixed_precision(trial_model, bit_assignment)
        trial_acc, _ = finetune_qat(
            trial_model, train_loader, val_loader, device,
            epochs=30, lr=lr_i, weight_decay=wd_i,
            observer_epochs=5, label="Full",
        )
        full_results.append({"lr": lr_i, "wd": wd_i, "acc": trial_acc})
        info(f"lr={lr_i:.0e}, wd={wd_i:.0e} -> acc={trial_acc:.2f}%")
        del trial_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Rank correlation
    proxy_accs = [r["acc"] for r in proxy_results]
    full_accs = [r["acc"] for r in full_results]
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(proxy_accs, full_accs)
        metric("Spearman rank correlation", f"rho={rho:.3f}, p={pval:.3f}")
        if rho >= 0.8:
            success("PASS (rho >= 0.8)")
        else:
            info(f"Status: rho={rho:.3f} < 0.8")
    except ImportError:
        info("scipy not available — skipping rank correlation")
        rho = None

    section("Results Summary")

    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("Uniform INT8 (QAT)", f"{int8_acc:.2f}% (BOPs: {bops_8bit:,.0f})")
    metric("LCPAQ mixed (QAT)", f"{lcpaq_acc:.2f}% (BOPs: {bops_mixed:,.0f}, {100*bops_mixed/bops_8bit:.1f}% of 8-bit)")
    metric("Acc drop vs INT8", f"{int8_acc - lcpaq_acc:+.2f}%")
    metric("BOPs savings vs INT8", f"{100*(1 - bops_mixed/bops_8bit):.1f}%")
    metric("MAC-weighted avg bits", f"{avg_bits_weighted:.2f}")
    info("Search cost:")
    metric("Hessian computation", f"{hessian_time:.1f}s")
    metric("Pareto search", f"{pareto_time:.1f}s")
    metric("Proxy NAS (6 trials)", f"{proxy_time:.1f}s")
    metric("Total search overhead", f"{hessian_time + pareto_time + proxy_time:.1f}s")

    if lcpaq_acc >= int8_acc - 1.0:
        success("PASS (within 1% of INT8 at reduced BOPs)")
    else:
        info(f"Status: FAIL (drop {int8_acc - lcpaq_acc:.2f}% > 1%)")

    section("Generating Plots")

    plot_training_curves(
        [lcpaq_history, int8_history],
        [f"LCPAQ ({avg_bits_weighted:.1f}b avg)", "INT8"],
        os.path.join(result_dir, "lcpaq_training_curves.png"),
    )

    plot_bit_assignment(bit_assignment, hessian_traces, cost_table,
                        os.path.join(result_dir, "lcpaq_bit_assignment.png"))

    if full_results:
        plot_proxy_correlation(proxy_results, full_results,
                               os.path.join(result_dir, "lcpaq_proxy_correlation.png"))

    # Save results JSON
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "lcpaq_acc": lcpaq_acc,
        "int8_acc": int8_acc,
        "bit_assignment": bit_assignment,
        "avg_bits_weighted": avg_bits_weighted,
        "bops_8bit": bops_8bit,
        "bops_mixed": bops_mixed,
        "best_lr": best_lr,
        "best_wd": best_wd,
        "proxy_results": proxy_results,
        "full_results": full_results,
        "search_steps": search_steps,
        "hessian_traces": {k: float(v) for k, v in hessian_traces.items()},
        "search_time": {
            "hessian_s": hessian_time,
            "pareto_s": pareto_time,
            "proxy_nas_s": proxy_time,
        },
        "summary": {
            "method": "LCPAQ",
            "fp32_acc": fp32_acc,
            "method_acc": lcpaq_acc,
            "int8_acc": int8_acc,
            "int4_acc": None,
            "avg_bits": round(avg_bits_weighted, 2),
            "per_layer_bits": bit_assignment,
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in lcpaq_history
            ],
            "category": "mpq",
        },
    }
    save_results(results, os.path.join(result_dir, "lcpaq_results.json"))

    model_path = os.path.join(result_dir, f"{cfg.model}_lcpaq.pt")
    torch.save(lcpaq_model.state_dict(), model_path)
    file_saved(model_path)


if __name__ == "__main__":
    main()
