"""
End-to-end EfficientQAT training script for ResNet-18 on CIFAR-10.
Adapted from Chen et al., "EfficientQAT" (ACL 2025 Main).
Original repo: https://github.com/OpenGVLab/EfficientQAT

Usage: python -m scripts.efficient_qat.train

Two-phase pipeline:
  Phase 1 (Block-AP): Block-wise training of all parameters (5 epochs/block)
  Phase 2 (E2E-QP):   End-to-end quantization parameter training (15 epochs)

Comparison:
  - Single-phase full-model QAT at same bit-width (30 epochs)
  - Reports per-phase accuracy improvement
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
from core.efficient_qat.efficient_qat_resnet import (
    BlockAPTrainer, EQATConv2d, EQATLinear,
    set_quant_state,
)
from core.efficient_qat.efficient_qat_e2e import E2EQPTrainer
from core.jacob_fake_quant import JacobFakeQuantize


# ---------------------------------------------------------------------------
# Single-phase QAT baseline (for comparison)
# ---------------------------------------------------------------------------

def finetune_single_phase_qat(model, train_loader, val_loader, device,
                               bits=4, epochs=30, lr=1e-4, label="QAT"):
    """Standard single-phase full-model QAT at given bit-width.

    Uses Jacob-style fake quantization (BN folding + fake quant).
    """
    from core.jacob_qat.bn_fold import prepare_model_for_qat

    qmodel = copy.deepcopy(model)
    qmodel = prepare_model_for_qat(qmodel)

    # Override bit-widths
    for m in qmodel.modules():
        if isinstance(m, JacobFakeQuantize):
            m.num_bits = bits
            if m.mode == "symmetric":
                m.q_min = -(2 ** (bits - 1))
                m.q_max = 2 ** (bits - 1) - 1
            else:
                m.q_min = 0
                m.q_max = 2 ** bits - 1

    qmodel.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        qmodel.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    observer_epochs = 5
    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == observer_epochs + 1:
            for m in qmodel.modules():
                if isinstance(m, JacobFakeQuantize):
                    m.disable_observer()
            info(f"[{label}] Observers disabled")

        qmodel.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = qmodel(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        qmodel.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, predicted = qmodel(inputs).max(1)
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


from utils.training import evaluate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_block_ap_losses(all_histories, save_path):
    """Plot MSE loss per epoch for each block in Phase 1."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    for (block_name, history), color in zip(all_histories.items(), colors):
        epochs = [h[0] for h in history]
        losses = [h[1] for h in history]
        ax.plot(epochs, losses, "o-", label=block_name, color=color,
                markersize=4, linewidth=1.5)

    ax.set_xlabel("Epoch (within block)")
    ax.set_ylabel("MSE Loss")
    ax.set_title("EfficientQAT Phase 1: Block-AP MSE Loss per Block")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_training_curves(eqat_p2_history, single_phase_history,
                         eqat_p1_end_acc, save_path):
    """Plot val accuracy: E2E-QP (Phase 2) vs single-phase QAT."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Phase 2 curve
    e2e_epochs = [h[0] for h in eqat_p2_history]
    e2e_accs = [h[3] for h in eqat_p2_history]
    ax.plot(e2e_epochs, e2e_accs, "-o", label="EfficientQAT Phase 2 (E2E-QP)",
            color="#2196F3", markersize=3, linewidth=1.5)

    # Single-phase curve
    sp_epochs = [h[0] for h in single_phase_history]
    sp_accs = [h[3] for h in single_phase_history]
    ax.plot(sp_epochs, sp_accs, "-o", label="Single-Phase QAT",
            color="#FF5722", markersize=3, linewidth=1.5)

    # Phase 1 end accuracy as horizontal line
    ax.axhline(y=eqat_p1_end_acc, color="#4CAF50", linestyle="--",
               label=f"After Phase 1 (Block-AP): {eqat_p1_end_acc:.2f}%")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("EfficientQAT Two-Phase vs Single-Phase QAT")
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
    parser = argparse.ArgumentParser(description="EfficientQAT training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("AQUA — EfficientQAT Training", f"Config: {args.config}")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "efficient_qat")
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        data_root=cfg.data_root,
    )
    bits = cfg.bits

    # ===== Step 1: Load pretrained FP32 =====
    section(f"Load Pretrained FP32 {cfg.model} ({cfg.dataset})", step=1)

    model, fp32_acc = load_fp32_model(
        cfg.model, device, val_loader,
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dataset_name=cfg.dataset,
    )

    # ===== Step 2: Phase 1 — Block-AP =====
    section(f"Phase 1 — Block-AP ({bits}-bit, 5 epochs/block)", step=2)

    block_ap = BlockAPTrainer(
        quant_lr=1e-4,
        weight_lr=1e-5,
        min_lr_factor=20,
    )

    eqat_model, block_histories = block_ap.train_all_blocks(
        model, train_loader, val_loader, device,
        bits=bits, epochs_per_block=5,
    )

    # Evaluate after Phase 1
    set_quant_state(eqat_model, True)
    p1_acc = evaluate(eqat_model, val_loader, device)
    metric("Phase 1 (Block-AP) accuracy", f"{p1_acc:.2f}%")

    # Print block-level MSE summary
    info("Block-AP MSE summary:")
    for name, hist in block_histories.items():
        init_mse = hist[0][1]
        final_mse = hist[-1][1]
        metric(name, f"Initial: {init_mse:.6f}, Final: {final_mse:.6f}")

    # ===== Step 3: Phase 2 — E2E-QP =====
    section("Phase 2 — E2E-QP (15 epochs, quant params only)", step=3)

    # Count trainable params before/after freeze
    total_params = sum(p.numel() for p in eqat_model.parameters())
    e2e_trainer = E2EQPTrainer(lr=2e-5, warmup_ratio=0.03, max_grad_norm=0.3)
    n_quant_params = e2e_trainer.freeze_weights(eqat_model)

    # Restore requires_grad for the count, then re-freeze in train()
    # (train() will call freeze_weights again)
    for param in eqat_model.parameters():
        param.requires_grad = True

    metric("Total model params", f"{total_params:,}")
    metric("Quant params to train", str(n_quant_params))
    metric("Param reduction", f"{100*(1 - n_quant_params/total_params):.1f}%")

    p2_acc, p2_history = e2e_trainer.train(
        eqat_model, train_loader, val_loader, device, epochs=15,
    )

    metric("Phase 2 (E2E-QP) accuracy", f"{p2_acc:.2f}%")
    metric("Phase 2 improvement", f"{p2_acc - p1_acc:+.2f}%")

    # ===== Step 4: Single-phase QAT baseline =====
    section(f"Single-Phase QAT Baseline ({bits}-bit, 30 epochs)", step=4)

    sp_acc, sp_history = finetune_single_phase_qat(
        model, train_loader, val_loader, device,
        bits=bits, epochs=30, lr=1e-4, label=f"QAT-{bits}b",
    )

    # ===== Results =====
    section("Results Summary")

    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("EfficientQAT Phase 1 (Block-AP)", f"{p1_acc:.2f}% (5 epochs x 6 blocks = 30 block-epochs)")
    metric("EfficientQAT Phase 2 (E2E-QP)", f"{p2_acc:.2f}% (+15 epochs, quant params only)")
    metric(f"Single-phase QAT {bits}-bit", f"{sp_acc:.2f}% (30 epochs, all params)")
    console.print()
    metric("EfficientQAT total", f"{p2_acc:.2f}%")
    metric("Single-phase QAT", f"{sp_acc:.2f}%")
    metric("Advantage", f"{p2_acc - sp_acc:+.2f}%")
    metric("Phase 2 contribution", f"{p2_acc - p1_acc:+.2f}%")

    if p2_acc >= sp_acc:
        success("PASS (EfficientQAT matches or beats single-phase)")
    else:
        info(f"EfficientQAT is {sp_acc - p2_acc:.2f}% below single-phase")

    # ===== Plots =====
    section("Generating Plots")

    plot_block_ap_losses(block_histories,
                         os.path.join(result_dir, "efficient_qat_block_ap_losses.png"))

    plot_training_curves(p2_history, sp_history, p1_acc,
                         os.path.join(result_dir, "efficient_qat_training_curves.png"))

    # Save results JSON
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "phase1_acc": p1_acc,
        "phase2_acc": p2_acc,
        "single_phase_acc": sp_acc,
        "bits": bits,
        "phase2_improvement": p2_acc - p1_acc,
        "vs_single_phase": p2_acc - sp_acc,
        "block_histories": {
            name: [{"epoch": e, "mse": l} for e, l in hist]
            for name, hist in block_histories.items()
        },
        "summary": {
            "method": "EfficientQAT",
            "fp32_acc": fp32_acc,
            "method_acc": p2_acc,
            "int8_acc": None,
            "int4_acc": sp_acc,
            "avg_bits": float(bits),
            "per_layer_bits": {},
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in p2_history
            ],
            "category": "qat",
        },
    }
    save_results(results, os.path.join(result_dir, "efficient_qat_results.json"))

    model_path = os.path.join(result_dir, f"{cfg.model}_efficient_qat.pt")
    torch.save(eqat_model.state_dict(), model_path)
    file_saved(model_path)


if __name__ == "__main__":
    main()
