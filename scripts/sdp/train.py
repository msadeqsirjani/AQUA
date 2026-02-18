"""
End-to-end SDP (Structured Dynamic Precision) training script.
ACM Transactions on Design Automation of Electronic Systems, 2022.
DOI: https://dl.acm.org/doi/10.1145/3549535

Usage: python -m scripts.sdp.train

Steps:
1. Load pretrained FP32 ResNet-18 on CIFAR-10
2. Fine-tune with SDP quantization (N=8, M=4, sparsity=0.5)
3. Fine-tune uniform INT8 and INT4 baselines for comparison
4. Log effective average bits per layer and mask statistics
5. Plot training curves and per-layer effective bits
"""

import argparse
import os
import copy
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
from utils.args import add_common_args, get_result_dir
from utils.config import load_config
from utils.console import banner, section, print_config, metric, info, success, file_saved, warning
from core.sdp.sdp_quantizer import SDPQuantizer
from core.sdp.sdp_model import replace_with_sdp, SDPConv2d, SDPLinear
from core.jacob_fake_quant import JacobFakeQuantize


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_sdp(model, train_loader, val_loader, device, epochs=30, lr=1e-4,
                 observer_epochs=5, label="SDP"):
    """Fine-tune a model with SDP/fake-quantized layers.

    Returns:
        (best_acc, history) where history is list of
        (epoch, train_loss, train_acc, val_acc) tuples.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # Disable observers after warmup
        if epoch == observer_epochs + 1:
            for m in model.modules():
                if isinstance(m, (SDPQuantizer, JacobFakeQuantize)):
                    m.disable_observer()
            info(f"[{label}] Observers disabled (scale frozen)")

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
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += inputs.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        history.append((epoch, train_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            info(f"[{label}] Epoch {epoch:2d}/{epochs} | "
                 f"Loss: {train_loss:.4f} | "
                 f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return best_acc, history


# ---------------------------------------------------------------------------
# Mask and effective-bits statistics
# ---------------------------------------------------------------------------

def collect_sdp_stats(model, data_loader, device, n_batches=10):
    """Run a few forward passes and collect per-layer SDP mask statistics.

    Returns:
        dict: {layer_name: {"mask_ratio": float, "eff_bits": float,
                             "total_bits": int, "high_bits": int}}
    """
    model.to(device).eval()
    stats = {}

    # Identify SDP weight quantizers
    sdp_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (SDPConv2d, SDPLinear)):
            sdp_layers[name] = module.weight_quantizer

    # Forward pass to populate mask stats
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    for name, quantizer in sdp_layers.items():
        mask_ratio = quantizer.last_mask_ratio
        N = quantizer.total_bits
        M = quantizer.high_bits
        L = quantizer.low_bits
        eff = M + mask_ratio * L
        stats[name] = {
            "mask_ratio": mask_ratio,
            "eff_bits": eff,
            "total_bits": N,
            "high_bits": M,
        }

    return stats


def verify_mask_counts(model, data_loader, device, group_size, sparsity,
                       n_batches=5):
    """Verify that ~sparsity fraction of elements per group keep full precision."""
    model.to(device).eval()

    ratios = []
    for name, module in model.named_modules():
        if isinstance(module, (SDPConv2d, SDPLinear)):
            ratios.append(module.weight_quantizer.last_mask_ratio)

    if not ratios:
        return

    avg_ratio = sum(ratios) / len(ratios)
    expected = sparsity
    metric("Mask ratio check", f"avg={avg_ratio:.4f}, expected~={expected:.4f}")
    if abs(avg_ratio - expected) < 0.15:
        success("OK (within tolerance)")
    else:
        warning("Mask ratio deviates from expected sparsity")


# ---------------------------------------------------------------------------
# Uniform INT-K baseline via Jacob fake-quant (for comparison)
# ---------------------------------------------------------------------------

def build_uniform_model(model_fp32, bits, device):
    """Create a uniform fake-quantized model at given bit-width.

    Uses JacobFakeQuantize wrappers (same as Jacob QAT).
    """
    from core.jacob_qat.bn_fold import prepare_model_for_qat
    from core.jacob_qat.jacob_quantized_layers import QConv2d as JQConv2d, QLinear as JQLinear

    qmodel = copy.deepcopy(model_fp32)
    qmodel = prepare_model_for_qat(qmodel)

    # Override bit-widths on all quantizers
    for m in qmodel.modules():
        if isinstance(m, JacobFakeQuantize):
            m.num_bits = bits
            if m.mode == "symmetric":
                m.q_min = -(2 ** (bits - 1))
                m.q_max = 2 ** (bits - 1) - 1
            else:
                m.q_min = 0
                m.q_max = 2 ** bits - 1

    return qmodel


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
    ax.set_title("SDP vs Uniform Quantization: Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_effective_bits(sdp_stats, save_path):
    """Bar chart of effective bits per layer."""
    names = list(sdp_stats.keys())
    eff_bits = [sdp_stats[n]["eff_bits"] for n in names]
    total_bits = [sdp_stats[n]["total_bits"] for n in names]

    short_names = [n.rsplit(".", 1)[-1] if "." in n else n for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.7), 5))
    x = np.arange(len(names))
    width = 0.4

    ax.bar(x - width / 2, total_bits, width, label="Total bits (N)",
           color="#2196F3", alpha=0.5)
    ax.bar(x + width / 2, eff_bits, width, label="Effective bits",
           color="#FF5722", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Bits")
    ax.set_title("SDP: Total vs Effective Bits per Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDP training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("AQUA — SDP Training", f"Config: {args.config}")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "sdp")
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

    section("SDP Quantization (N=8, M=4, G=8, sparsity=0.5)", step=2)

    total_bits = 8
    high_bits = 4
    group_size = 8
    sparsity = 0.5
    eff_bits_expected = high_bits + sparsity * (total_bits - high_bits)

    metric("Total bits", f"N={total_bits}, M={high_bits}, L={total_bits - high_bits}")
    metric("Group size", f"G={group_size}")
    metric("Sparsity", sparsity)
    metric("Expected effective bits", f"{eff_bits_expected:.1f}")

    sdp_model = copy.deepcopy(model)
    sdp_model = replace_with_sdp(sdp_model, total_bits=total_bits,
                                  high_bits=high_bits, group_size=group_size,
                                  sparsity=sparsity)

    n_sdpconv = sum(1 for m in sdp_model.modules() if isinstance(m, SDPConv2d))
    n_sdplin = sum(1 for m in sdp_model.modules() if isinstance(m, SDPLinear))
    metric("SDPConv2d layers", n_sdpconv)
    metric("SDPLinear layers", n_sdplin)

    section("Fine-Tune SDP (30 epochs)", step=3)

    sdp_acc, sdp_history = finetune_sdp(
        sdp_model, train_loader, val_loader, device,
        epochs=30, lr=1e-4, observer_epochs=5, label="SDP",
    )

    # Collect SDP mask stats after training
    sdp_stats = collect_sdp_stats(sdp_model, val_loader, device)

    info("Per-layer SDP statistics:")
    for name, st in sdp_stats.items():
        metric(name, f"{100*st['mask_ratio']:.1f}% mask, {st['eff_bits']:.2f} eff bits")

    avg_eff = sum(s["eff_bits"] for s in sdp_stats.values()) / len(sdp_stats)
    metric("Average effective bits", f"{avg_eff:.2f}")

    verify_mask_counts(sdp_model, val_loader, device, group_size, sparsity)

    section("Uniform INT8 Baseline (30 epochs)", step=4)

    int8_model = build_uniform_model(model, bits=8, device=device)
    int8_acc, int8_history = finetune_sdp(
        int8_model, train_loader, val_loader, device,
        epochs=30, lr=1e-4, observer_epochs=5, label="INT8",
    )

    section("Uniform INT4 Baseline (30 epochs)", step=5)

    int4_model = build_uniform_model(model, bits=4, device=device)
    int4_acc, int4_history = finetune_sdp(
        int4_model, train_loader, val_loader, device,
        epochs=30, lr=1e-4, observer_epochs=5, label="INT4",
    )

    section("Results Summary")
    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("Uniform INT8 (8-bit, QAT)", f"{int8_acc:.2f}%")
    metric(f"SDP (eff ~{avg_eff:.1f}-bit, QAT)", f"{sdp_acc:.2f}%")
    metric("Uniform INT4 (4-bit, QAT)", f"{int4_acc:.2f}%")
    metric("SDP vs INT8 drop", f"{int8_acc - sdp_acc:+.2f}%")
    metric("SDP vs INT4 gain", f"{sdp_acc - int4_acc:+.2f}%")
    metric("Bit savings vs INT8", f"{8 - avg_eff:.1f} bits avg")

    if sdp_acc >= int8_acc - 0.5:
        success("SDP Status: PASS (within 0.5% of INT8)")
    else:
        warning(f"SDP Status: FAIL (drop {int8_acc - sdp_acc:.2f}% > 0.5%)")

    section("Generating Plots")

    plot_training_curves(
        [sdp_history, int8_history, int4_history],
        [f"SDP (eff ~{avg_eff:.1f}b)", "INT8", "INT4"],
        os.path.join(result_dir, "sdp_training_curves.png"),
    )

    plot_effective_bits(sdp_stats, os.path.join(result_dir, "sdp_effective_bits.png"))

    # Build per-layer effective bits for summary
    sdp_per_layer_bits = {
        name: round(st.get("eff_bits", total_bits), 2)
        for name, st in sdp_stats.items()
    }

    # Save results JSON
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "sdp_acc": sdp_acc,
        "int8_acc": int8_acc,
        "int4_acc": int4_acc,
        "sdp_config": {
            "total_bits": total_bits,
            "high_bits": high_bits,
            "group_size": group_size,
            "sparsity": sparsity,
            "expected_eff_bits": eff_bits_expected,
            "measured_avg_eff_bits": avg_eff,
        },
        "per_layer_stats": {
            name: {k: float(v) for k, v in st.items()}
            for name, st in sdp_stats.items()
        },
        "summary": {
            "method": "SDP",
            "fp32_acc": fp32_acc,
            "method_acc": sdp_acc,
            "int8_acc": int8_acc,
            "int4_acc": int4_acc,
            "avg_bits": round(avg_eff, 2),
            "per_layer_bits": sdp_per_layer_bits,
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in sdp_history
            ],
            "category": "learnable",
        },
    }
    save_results(results, os.path.join(result_dir, "sdp_results.json"))

    model_path = os.path.join(result_dir, f"{cfg.model}_sdp.pt")
    torch.save(sdp_model.state_dict(), model_path)
    file_saved(model_path)


if __name__ == "__main__":
    main()
