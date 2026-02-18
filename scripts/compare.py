"""
Unified comparison script for all AQUA quantization algorithms.

Usage:
    python -m scripts.compare                                     # defaults from config
    python -m scripts.compare --dataset cifar100 --model vgg16
    python -m scripts.compare --config configs/tiny_imagenet.yaml

Reads results/{dataset}/{model}/{algo}/{algo}_results.json from each algorithm,
extracts the standardized "summary" block, and generates:
  1. Accuracy bar chart (FP32, INT8, each method)
  2. Accuracy vs average bit-width scatter (Pareto front)
  3. Training curves overlay (val accuracy over epochs)
  4. Per-layer bit-width comparison (mixed-precision methods)

All figures saved to results/{dataset}/{model}/comparison/.
"""

import argparse
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import list_models
from utils.data import list_datasets
from utils.config import load_config
from utils.console import (
    console, banner, section, print_config, metric, info, success,
    file_saved, results_table,
)
from rich.table import Table
from rich import box

ALGO_NAMES = [
    "haq",
    "jacob_qat",
    "efficient_qat",
    "edge_mpq",
    "lcpaq",
    "adapt",
    "asq",
    "sdp",
]

# Colors for each method (consistent across all figures)
METHOD_COLORS = {
    "Jacob QAT":     "#1976D2",
    "EfficientQAT":  "#388E3C",
    "HAQ":           "#D32F2F",
    "Edge-MPQ":      "#7B1FA2",
    "LCPAQ":         "#F57C00",
    "AdaPT":         "#00796B",
    "ASQ":           "#C2185B",
    "SDP":           "#455A64",
}

CATEGORY_MARKERS = {
    "qat":       "s",
    "mpq":       "D",
    "adaptive":  "^",
    "learnable": "o",
}


def _result_path(dataset_name, model_name, algo):
    """Return the expected result JSON path for a dataset/model/algo combination.

    Checks dataset-aware path first, then model-only, then legacy flat.
    """
    candidates = [
        os.path.join("results", dataset_name, model_name, algo,
                      f"{algo}_results.json"),
        os.path.join("results", model_name, algo, f"{algo}_results.json"),
        os.path.join("results", algo, f"{algo}_results.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # preferred (may not exist yet)


def load_summaries(dataset_name, model_name):
    """Load all available result JSONs and extract summary blocks."""
    summaries = {}
    for algo in ALGO_NAMES:
        path = _result_path(dataset_name, model_name, algo)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if "summary" not in data:
            from utils.console import warning
            warning(f"{path} has no 'summary' block, skipping")
            continue
        summaries[algo] = data["summary"]
        info(f"Loaded {algo}: {data['summary']['method']} "
             f"({data['summary']['method_acc']:.2f}%)")
    return summaries


# ---------------------------------------------------------------------------
# Figure 1: Accuracy bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_bars(summaries, save_path, model_name=""):
    """Grouped bar chart: FP32, INT8, then each method's accuracy."""
    methods = []
    method_accs = []
    colors = []

    fp32_accs = [s["fp32_acc"] for s in summaries.values() if s["fp32_acc"]]
    fp32_avg = np.mean(fp32_accs) if fp32_accs else 0

    int8_accs = [s["int8_acc"] for s in summaries.values()
                 if s["int8_acc"] is not None]
    int8_avg = np.mean(int8_accs) if int8_accs else None

    for s in summaries.values():
        methods.append(s["method"])
        method_accs.append(s["method_acc"])
        colors.append(METHOD_COLORS.get(s["method"], "#888888"))

    labels = []
    values = []
    bar_colors = []

    labels.append("FP32")
    values.append(fp32_avg)
    bar_colors.append("#4CAF50")

    if int8_avg is not None:
        labels.append("INT8\n(uniform)")
        values.append(int8_avg)
        bar_colors.append("#9E9E9E")

    for name, acc, c in zip(methods, method_accs, colors):
        labels.append(name)
        values.append(acc)
        bar_colors.append(c)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    title = "Quantization Methods — Final Accuracy Comparison"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ymin = min(values) - 5
    ymax = max(values) + 3
    ax.set_ylim(max(0, ymin), min(100, ymax))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    pass  # caller handles file_saved


# ---------------------------------------------------------------------------
# Figure 2: Accuracy vs average bit-width (Pareto scatter)
# ---------------------------------------------------------------------------

def plot_pareto_scatter(summaries, save_path, model_name=""):
    """Scatter plot: X = avg bits, Y = accuracy."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for s in summaries.values():
        name = s["method"]
        color = METHOD_COLORS.get(name, "#888888")
        marker = CATEGORY_MARKERS.get(s.get("category", ""), "o")
        ax.scatter(s["avg_bits"], s["method_acc"],
                   color=color, marker=marker, s=120, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(name,
                    (s["avg_bits"], s["method_acc"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color, fontweight="bold")

    fp32_accs = [s["fp32_acc"] for s in summaries.values()]
    if fp32_accs:
        fp32 = np.mean(fp32_accs)
        ax.axhline(fp32, color="#4CAF50", linestyle="--", alpha=0.5,
                    label=f"FP32 ({fp32:.1f}%)")

    ax.set_xlabel("Average Bit-Width")
    ax.set_ylabel("Accuracy (%)")
    title = "Accuracy vs Bit-Width — Efficiency Frontier"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    pass  # caller handles file_saved


# ---------------------------------------------------------------------------
# Figure 3: Training curves overlay
# ---------------------------------------------------------------------------

def plot_training_curves(summaries, save_path, model_name=""):
    """Overlay val accuracy curves for all methods on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for s in summaries.values():
        name = s["method"]
        hist = s.get("training_history", [])
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        accs = [h["val_acc"] for h in hist]
        color = METHOD_COLORS.get(name, "#888888")
        ax.plot(epochs, accs, "o-", markersize=2, linewidth=1.5,
                label=f"{name} ({max(accs):.1f}%)", color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    title = "Training Curves — All Quantization Methods"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    pass  # caller handles file_saved


# ---------------------------------------------------------------------------
# Figure 4: Per-layer bit-width comparison (MPQ methods)
# ---------------------------------------------------------------------------

def plot_per_layer_bits(summaries, save_path):
    """Side-by-side per-layer bit-width bar chart for mixed-precision methods."""
    mpq_methods = []
    for s in summaries.values():
        if s.get("per_layer_bits") and len(s["per_layer_bits"]) > 0:
            mpq_methods.append(s)

    if not mpq_methods:
        info("No methods with per-layer bit data; skipping per-layer plot")
        return

    all_layers = []
    for s in mpq_methods:
        for layer in s["per_layer_bits"]:
            if layer not in all_layers:
                all_layers.append(layer)

    n_methods = len(mpq_methods)
    n_layers = len(all_layers)

    if n_layers == 0:
        info("No per-layer data to plot; skipping")
        return

    fig, ax = plt.subplots(figsize=(max(12, n_layers * 0.5), 6))
    x = np.arange(n_layers)
    width = 0.8 / n_methods

    for i, s in enumerate(mpq_methods):
        name = s["method"]
        color = METHOD_COLORS.get(name, "#888888")
        bits = [s["per_layer_bits"].get(layer, 0) for layer in all_layers]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, bits, width, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Bit-Width")
    ax.set_title("Per-Layer Bit-Width Assignment Comparison")
    ax.set_xticks(x)

    short_names = []
    for layer in all_layers:
        parts = layer.rsplit(".", 1)
        short_names.append(parts[-1] if len(parts) > 1 else layer)
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=6)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    pass  # caller handles file_saved


# ---------------------------------------------------------------------------
# Summary table (text)
# ---------------------------------------------------------------------------

def print_summary_table(summaries):
    """Print a formatted comparison table using rich."""
    table = Table(
        title="Algorithm Comparison",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
    )
    table.add_column("Method", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Avg Bits", justify="right")
    table.add_column("FP32 %", justify="right", style="green")
    table.add_column("INT8 %", justify="right")
    table.add_column("Method %", justify="right", style="bold yellow")
    table.add_column("Drop", justify="right")

    for s in summaries.values():
        int8_str = f"{s['int8_acc']:.2f}" if s["int8_acc"] is not None else "[dim]N/A[/dim]"
        drop = s["fp32_acc"] - s["method_acc"]
        drop_style = "red" if drop > 1.0 else "green"
        table.add_row(
            s["method"],
            s.get("category", ""),
            f"{s['avg_bits']:.2f}",
            f"{s['fp32_acc']:.2f}",
            int8_str,
            f"{s['method_acc']:.2f}",
            f"[{drop_style}]{drop:+.2f}[/{drop_style}]",
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare all AQUA quantization algorithms",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/cifar10_resnet18_4bit.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list_datasets(),
        help="Dataset name (default: from config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list_models(),
        help="Model architecture to compare results for (default: from config)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args)

    label = f"{cfg.model} / {cfg.dataset}"

    banner("AQUA — Algorithm Comparison", label)
    print_config(cfg)

    section("Loading Results")
    summaries = load_summaries(cfg.dataset, cfg.model)

    if not summaries:
        console.print(
            f"\n[bold red]No results found[/bold red] for {label}. "
            "Run training scripts first:"
        )
        for algo in ALGO_NAMES:
            info(f"python -m scripts.{algo}.train --config {args.config}")
        return

    success(f"Found {len(summaries)} algorithm result(s)")

    out_dir = os.path.join("results", cfg.dataset, cfg.model, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    print_summary_table(summaries)

    section("Generating Figures")
    plot_accuracy_bars(summaries,
                       os.path.join(out_dir, "accuracy_comparison.png"),
                       model_name=label)
    file_saved(os.path.join(out_dir, "accuracy_comparison.png"))
    plot_pareto_scatter(summaries,
                        os.path.join(out_dir, "pareto_bits_vs_accuracy.png"),
                        model_name=label)
    file_saved(os.path.join(out_dir, "pareto_bits_vs_accuracy.png"))
    plot_training_curves(summaries,
                         os.path.join(out_dir, "training_curves_overlay.png"),
                         model_name=label)
    file_saved(os.path.join(out_dir, "training_curves_overlay.png"))
    plot_per_layer_bits(summaries,
                        os.path.join(out_dir, "per_layer_bits_comparison.png"))

    # Save combined summary as JSON
    combined = {algo: s for algo, s in summaries.items()}
    json_path = os.path.join(out_dir, "comparison_summary.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    file_saved(json_path)

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
