"""
End-to-end Edge-MPQ training script
(Zhao et al., IEEE Trans. Computers, 2024).

Usage: python -m scripts.edge_mpq.train

Steps:
1. Load pretrained FP32 ResNet-18 on CIFAR-10
2. Compute Hessian trace sensitivity for all quantizable layers
3. Solve ILP for bit-width assignment at 50% BOPs budget
4. Apply mixed-precision and fine-tune with QAT for 30 epochs
5. Compare to uniform INT8 baseline
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
from utils.console import console, banner, section, print_config, metric, info, success, file_saved
from core.hessian_sensitivity import compute_hessian_trace
from core.edge_mpq.ilp_bit_assignment import solve_bit_assignment, _compute_bops
from core.edge_mpq.edge_mpq_model import apply_mixed_precision, get_layer_macs, MPQConv2d, MPQLinear


# ---------------------------------------------------------------------------
# Fine-tuning (reuse Jacob QAT trainer logic)
# ---------------------------------------------------------------------------

def finetune_qat(model, train_loader, val_loader, device, epochs=30, lr=1e-4):
    """Fine-tune a model with fake-quantized layers."""
    from core.jacob_fake_quant import JacobFakeQuantize

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Observer control: enable for first 5 epochs, then freeze
    observer_epochs = 5
    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == observer_epochs + 1:
            for m in model.modules():
                if isinstance(m, JacobFakeQuantize):
                    m.disable_observer()
            info("Observers disabled (scale/zero_point frozen)")

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

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        history.append((epoch, train_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            info(f"Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} | "
                 f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return best_acc, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity_and_bits(sensitivities, bit_assignment, layer_names,
                              save_path):
    """Dual-axis plot: Hessian sensitivity bars + bit-width line."""
    fig, ax1 = plt.subplots(figsize=(max(12, len(layer_names) * 0.6), 6))

    x = np.arange(len(layer_names))
    sens_vals = [sensitivities[n] for n in layer_names]
    bits_vals = [bit_assignment[n] for n in layer_names]

    # Sensitivity bars
    color1 = "#2196F3"
    ax1.bar(x, sens_vals, color=color1, alpha=0.6, label="Hessian trace")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Hessian Trace (sensitivity)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    short_names = [n.rsplit(".", 1)[-1] if "." in n else n for n in layer_names]
    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    # Bit-width line on secondary axis
    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.plot(x, bits_vals, "o-", color=color2, linewidth=2, markersize=6,
             label="Assigned bits")
    ax2.set_ylabel("Bit-width", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 18)

    fig.suptitle("Edge-MPQ: Hessian Sensitivity vs Assigned Bit-Width")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Edge-MPQ training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("Edge-MPQ", "Hessian-guided mixed-precision quantization")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "edge_mpq")
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        data_root=cfg.data_root,
    )

    # ===== Step 1: Load pretrained FP32 =====
    section(f"Load Pretrained FP32 {cfg.model} ({cfg.dataset})", step=1)

    model, fp32_acc = load_fp32_model(
        cfg.model, device, val_loader,
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dataset_name=cfg.dataset,
    )

    # ===== Step 2: Hessian trace sensitivity =====
    section("Compute Hessian Trace Sensitivity", step=2)
    info("Running Hutchinson estimator (100 Rademacher samples)...")

    loss_fn = nn.CrossEntropyLoss()
    sensitivities = compute_hessian_trace(
        model, loss_fn, train_loader, n_samples=100, device=device
    )

    # Print sensitivities sorted
    sorted_sens = sorted(sensitivities.items(), key=lambda kv: kv[1], reverse=True)
    info("Per-layer Hessian trace (sorted by sensitivity):")
    for name, trace in sorted_sens:
        metric(name, f"{trace:.4f}")

    # ===== Step 3: ILP bit assignment =====
    section("ILP Bit-Width Assignment (50% BOPs budget)", step=3)

    # Get per-layer MACs
    macs_per_layer = get_layer_macs(model, input_h=32, input_w=32)

    bit_assignment = solve_bit_assignment(
        sensitivities, macs_per_layer,
        target_bops_ratio=0.5,
        bit_choices=[2, 4, 8, 16],
    )

    # Print assignment
    layer_names = list(sensitivities.keys())
    console.print()
    info("Per-layer bit-width assignment:")
    console.print(f"  [dim]{'Layer':<30s} {'Bits':>5s} {'Sensitivity':>15s} {'MACs':>15s}[/dim]")
    console.print(f"  [dim]{'-'*30} {'-'*5} {'-'*15} {'-'*15}[/dim]")
    for name in layer_names:
        bits = bit_assignment.get(name, 8)
        sens = sensitivities[name]
        macs = macs_per_layer.get(name, 0)
        console.print(f"  {name:<30s} {bits:>5d} {sens:>15.4f} {macs:>15,d}")

    # BOPs analysis
    max_bits = 16
    bops_max = sum(_compute_bops(macs_per_layer[n], max_bits) for n in layer_names)
    bops_int8 = sum(_compute_bops(macs_per_layer[n], 8) for n in layer_names)
    bops_mixed = sum(
        _compute_bops(macs_per_layer[n], bit_assignment.get(n, 8))
        for n in layer_names
    )
    metric("BOPs (16-bit ref)", f"{bops_max:,.0f}")
    metric("BOPs (INT8)", f"{bops_int8:,.0f} ({100*bops_int8/bops_max:.1f}%)")
    metric("BOPs (mixed)", f"{bops_mixed:,.0f} ({100*bops_mixed/bops_max:.1f}%)")
    metric("Mixed/INT8 ratio", f"{100*bops_mixed/bops_int8:.1f}%")

    # Average bit-width
    total_params = sum(macs_per_layer.get(n, 1) for n in layer_names)
    avg_bits = sum(
        bit_assignment.get(n, 8) * macs_per_layer.get(n, 1) for n in layer_names
    ) / total_params
    metric("MAC-weighted avg bits", f"{avg_bits:.2f}")

    # ===== Step 4: Apply mixed precision & fine-tune =====
    section("Apply Mixed Precision & Fine-Tune (30 epochs)", step=4)

    mpq_model = copy.deepcopy(model)
    mpq_model = apply_mixed_precision(mpq_model, bit_assignment)

    n_mpqconv = sum(1 for m in mpq_model.modules() if isinstance(m, MPQConv2d))
    n_mpqlin = sum(1 for m in mpq_model.modules() if isinstance(m, MPQLinear))
    metric("MPQConv2d layers", str(n_mpqconv))
    metric("MPQLinear layers", str(n_mpqlin))

    info("Fine-tuning with QAT...")
    mpq_acc, mpq_history = finetune_qat(mpq_model, train_loader, val_loader, device,
                                        epochs=30, lr=1e-4)

    # ===== Step 5: Uniform INT8 comparison =====
    section("Uniform INT8 Comparison", step=5)

    int8_assignment = {name: 8 for name in layer_names}
    int8_model = copy.deepcopy(model)
    int8_model = apply_mixed_precision(int8_model, int8_assignment)

    info("Fine-tuning uniform INT8...")
    int8_acc, _ = finetune_qat(int8_model, train_loader, val_loader, device,
                               epochs=30, lr=1e-4)

    # ===== Results =====
    section("Results Summary")

    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("Uniform INT8 (QAT)", f"{int8_acc:.2f}% (BOPs: {100*bops_int8/bops_max:.1f}% of 16-bit)")
    metric("Edge-MPQ mixed (QAT)", f"{mpq_acc:.2f}% (BOPs: {100*bops_mixed/bops_max:.1f}% of 16-bit)")
    metric("Acc drop vs INT8", f"{int8_acc - mpq_acc:.2f}%")
    metric("BOPs savings vs INT8", f"{100*(1 - bops_mixed/bops_int8):.1f}%")

    if mpq_acc >= int8_acc - 1.0:
        success("PASS (within 1% of uniform INT8)")
    else:
        info(f"Status: FAIL (drop {int8_acc - mpq_acc:.2f}% > 1%)")

    # ===== Plots =====
    section("Generating Plots")

    plot_sensitivity_and_bits(sensitivities, bit_assignment, layer_names,
                              os.path.join(result_dir, "edge_mpq_sensitivity_bits.png"))

    # Save results
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "int8_acc": int8_acc,
        "mpq_acc": mpq_acc,
        "bit_assignment": bit_assignment,
        "sensitivities": {k: float(v) for k, v in sensitivities.items()},
        "bops_int8": bops_int8,
        "bops_mixed": bops_mixed,
        "bops_max": bops_max,
        "summary": {
            "method": "Edge-MPQ",
            "fp32_acc": fp32_acc,
            "method_acc": mpq_acc,
            "int8_acc": int8_acc,
            "int4_acc": None,
            "avg_bits": round(avg_bits, 2),
            "per_layer_bits": {n: bit_assignment[n] for n in layer_names},
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in mpq_history
            ],
            "category": "mpq",
        },
    }
    save_results(results, os.path.join(result_dir, "edge_mpq_results.json"))

    model_path = os.path.join(result_dir, f"{cfg.model}_edge_mpq.pt")
    torch.save(mpq_model.state_dict(), model_path)
    file_saved(model_path)


if __name__ == "__main__":
    main()
