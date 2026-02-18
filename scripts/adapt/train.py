"""
End-to-end Adaptive Precision Training (AdaPT) script.
Kummer et al., arXiv:2107.13490, 2021.

Usage: python -m scripts.adapt.train

Steps:
1. Load pretrained FP32 ResNet-18 on CIFAR-10
2. Replace layers with AdaPT fixed-point quantized versions (all start at 8 bits)
3. Fine-tune for 30 epochs with precision controller checking every 5 epochs
4. Compare to uniform INT8 and INT4 baselines
5. Plot per-layer bit convergence and training curves
"""

import argparse
import os
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils.console import (
    console, banner, section, print_config, metric, info, success, file_saved,
)
from core.adapt.adapt_fixed_point import FixedPointQuantizer
from core.adapt.adapt_precision_controller import AdaPTPrecisionController


# ---------------------------------------------------------------------------
# AdaPT layer wrappers
# ---------------------------------------------------------------------------

class AdaPTConv2d(nn.Conv2d):
    """Conv2d with AdaPT fixed-point weight and activation quantizers."""

    def __init__(self, *args, total_bits=8, min_bits=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = FixedPointQuantizer(
            total_bits=total_bits, min_bits=min_bits, is_weight=True,
        )
        self.act_quantizer = FixedPointQuantizer(
            total_bits=total_bits, min_bits=min_bits, is_weight=False,
        )

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class AdaPTLinear(nn.Linear):
    """Linear with AdaPT fixed-point weight and activation quantizers."""

    def __init__(self, *args, total_bits=8, min_bits=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = FixedPointQuantizer(
            total_bits=total_bits, min_bits=min_bits, is_weight=True,
        )
        self.act_quantizer = FixedPointQuantizer(
            total_bits=total_bits, min_bits=min_bits, is_weight=False,
        )

    def forward(self, x):
        x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)


# ---------------------------------------------------------------------------
# Model preparation (BN folding + replace with AdaPT layers)
# ---------------------------------------------------------------------------

def _fold_bn_into_conv(conv, bn):
    """Fold BN into Conv2d, return AdaPTConv2d with folded weights."""
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps

    inv_std = gamma / torch.sqrt(var + eps)

    new_conv = AdaPTConv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups,
        bias=True,
    )
    new_conv.weight.data = conv.weight.data * inv_std.view(-1, 1, 1, 1)
    if conv.bias is not None:
        new_conv.bias.data = (conv.bias.data - mean) * inv_std + beta
    else:
        new_conv.bias.data = beta - mean * inv_std

    return new_conv


def _replace_layers_adapt(module):
    """Recursively replace Conv2d+BN pairs and standalone layers."""
    children = list(module.named_children())
    i = 0
    while i < len(children):
        name, child = children[i]

        if isinstance(child, nn.Conv2d) and not isinstance(child, AdaPTConv2d):
            # Look for BN as next sibling
            bn = None
            bn_name = None
            if i + 1 < len(children):
                next_name, next_child = children[i + 1]
                if isinstance(next_child, nn.BatchNorm2d):
                    bn = next_child
                    bn_name = next_name

            if bn is not None:
                new_conv = _fold_bn_into_conv(child, bn)
                setattr(module, name, new_conv)
                setattr(module, bn_name, nn.Identity())
                i += 2
                continue
            else:
                new_conv = AdaPTConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups,
                    bias=child.bias is not None,
                )
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)
                setattr(module, name, new_conv)
                i += 1
                continue

        elif isinstance(child, nn.Linear):
            new_lin = AdaPTLinear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
            )
            new_lin.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_lin.bias.data.copy_(child.bias.data)
            setattr(module, name, new_lin)
            i += 1
            continue

        else:
            _replace_layers_adapt(child)
            i += 1


def prepare_model_for_adapt(model):
    """Prepare FP32 model for AdaPT training.

    Folds BN, replaces Conv2d/Linear with AdaPT versions (all start at 8 bits).
    """
    model.eval()
    _replace_layers_adapt(model)
    return model


# ---------------------------------------------------------------------------
# Uniform baseline (Jacob QAT at fixed bit-width)
# ---------------------------------------------------------------------------

def build_uniform_model(model_fp32, bits):
    """Create a uniform fake-quantized model using Jacob QAT."""
    from core.jacob_qat.bn_fold import prepare_model_for_qat
    from core.jacob_fake_quant import JacobFakeQuantize

    qmodel = copy.deepcopy(model_fp32)
    qmodel = prepare_model_for_qat(qmodel)

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


def finetune_uniform(model, train_loader, val_loader, device, epochs=30,
                     lr=1e-4, label="Uniform"):
    """Fine-tune a Jacob-QAT model (uniform bit-width)."""
    from core.jacob_fake_quant import JacobFakeQuantize

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    observer_epochs = 5
    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == observer_epochs + 1:
            for m in model.modules():
                if isinstance(m, JacobFakeQuantize):
                    m.disable_observer()
            info(f"[{label}] Observers disabled")

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
                 f"Loss: {train_loss:.4f} | "
                 f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return best_acc, history


# ---------------------------------------------------------------------------
# AdaPT training loop
# ---------------------------------------------------------------------------

def train_adapt(model, controller, train_loader, val_loader, device,
                epochs=30, lr=1e-4):
    """Fine-tune with AdaPT: precision controller checks every check_interval.

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

    # Record initial bits
    controller.record_bits(0)

    for epoch in range(1, epochs + 1):
        # Train one epoch
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Gradient hooks fire during backward — grad_norms updated
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

        # Log
        avg_bits = controller.get_avg_bits()
        if epoch % 5 == 0 or epoch == 1:
            info(f"[AdaPT] Epoch {epoch:2d}/{epochs} | "
                 f"Loss: {train_loss:.4f} | "
                 f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                 f"AvgBits: {avg_bits:.2f}")

        # Precision check at interval
        if epoch % controller.check_interval == 0:
            controller.check_and_update(epoch)

    return best_acc, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(histories, labels, save_path):
    """Plot val accuracy curves for multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for (label, hist), color in zip(zip(labels, histories), colors):
        epochs = [h[0] for h in hist]
        val_accs = [h[3] for h in hist]
        ax.plot(epochs, val_accs, "-o", label=label, color=color,
                markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("AdaPT vs Uniform Quantization: Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_bits_convergence(controller, save_path):
    """Plot per-layer bit-width vs epoch (convergence plot)."""
    if not controller.bits_history:
        return

    names = list(controller.layer_info.keys())
    short = [n.rsplit(".", 1)[-1] if "." in n else n for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_list = [e for e, _ in controller.bits_history]

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        bits_vals = [snap.get(name, 8) for _, snap in controller.bits_history]
        ax.plot(epochs_list, bits_vals, "o-", label=short[i], color=colors[i],
                markersize=4, linewidth=1.5)

    # Average line
    avg_vals = []
    for _, snap in controller.bits_history:
        vals = [snap.get(n, 8) for n in names]
        avg_vals.append(sum(vals) / len(vals))
    ax.plot(epochs_list, avg_vals, "k--", label="Average", linewidth=2.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bit-width")
    ax.set_title("AdaPT: Per-Layer Bit-Width Convergence")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AdaPT training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("AQUA — AdaPT Training", f"Config: {args.config}")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "adapt")
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

    # ===== Step 2: Prepare AdaPT model =====
    section("Prepare AdaPT Model (all layers start at 8 bits)", step=2)

    adapt_model = copy.deepcopy(model)
    adapt_model = prepare_model_for_adapt(adapt_model)

    n_aconv = sum(1 for m in adapt_model.modules() if isinstance(m, AdaPTConv2d))
    n_alin = sum(1 for m in adapt_model.modules() if isinstance(m, AdaPTLinear))
    metric("AdaPTConv2d layers", n_aconv)
    metric("AdaPTLinear layers", n_alin)

    # Create precision controller
    controller = AdaPTPrecisionController(
        adapt_model,
        snr_threshold=20.0,
        grad_threshold=1e-4,
        check_interval=5,
        min_bits=2,
    )
    info(f"Controller: SNR>={controller.snr_threshold}dB, "
         f"grad>={controller.grad_threshold}, "
         f"check every {controller.check_interval} epochs")
    metric("Tracked layers", len(controller.layer_info))

    # ===== Step 3: AdaPT training =====
    section("AdaPT Training (30 epochs)", step=3)

    adapt_acc, adapt_history = train_adapt(
        adapt_model, controller, train_loader, val_loader, device,
        epochs=30, lr=1e-4,
    )

    # Print final bits table
    info("Per-layer bit-width evolution:")
    controller.print_bits_table()

    final_avg = controller.get_avg_bits()
    final_bits = controller.get_current_bits()
    metric("Final average bits", f"{final_avg:.2f}")

    # Clean up hooks
    controller.remove_hooks()

    # ===== Step 4: Uniform INT8 baseline =====
    section("Uniform INT8 Baseline (30 epochs)", step=4)

    int8_model = build_uniform_model(model, bits=8)
    int8_acc, int8_history = finetune_uniform(
        int8_model, train_loader, val_loader, device,
        epochs=30, lr=1e-4, label="INT8",
    )

    # ===== Step 5: Uniform INT4 baseline =====
    section("Uniform INT4 Baseline (30 epochs)", step=5)

    int4_model = build_uniform_model(model, bits=4)
    int4_acc, int4_history = finetune_uniform(
        int4_model, train_loader, val_loader, device,
        epochs=30, lr=1e-4, label="INT4",
    )

    # ===== Results =====
    section("Results Summary")
    metric("FP32 baseline", f"{fp32_acc:.2f}%")
    metric("Uniform INT8 (8-bit, QAT)", f"{int8_acc:.2f}%")
    metric(f"AdaPT (avg {final_avg:.1f}-bit, adaptive)", f"{adapt_acc:.2f}%")
    metric("Uniform INT4 (4-bit, QAT)", f"{int4_acc:.2f}%")
    metric("AdaPT vs INT8 drop", f"{int8_acc - adapt_acc:+.2f}%")
    metric("AdaPT vs INT4 gain", f"{adapt_acc - int4_acc:+.2f}%")
    metric("Bit savings vs INT8", f"{8 - final_avg:.1f} bits avg")

    if adapt_acc > int4_acc:
        success("AdaPT vs INT4: PASS (beats uniform INT4)")
    else:
        info("AdaPT vs INT4: FAIL (below uniform INT4)")

    # ===== Plots =====
    section("Generating Plots")

    plot_training_curves(
        [adapt_history, int8_history, int4_history],
        [f"AdaPT (avg {final_avg:.1f}b)", "INT8", "INT4"],
        os.path.join(result_dir, "adapt_training_curves.png"),
    )

    plot_bits_convergence(controller, os.path.join(result_dir, "adapt_bits_convergence.png"))

    # Save results JSON
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "adapt_acc": adapt_acc,
        "int8_acc": int8_acc,
        "int4_acc": int4_acc,
        "adapt_config": {
            "snr_threshold": controller.snr_threshold,
            "grad_threshold": controller.grad_threshold,
            "check_interval": controller.check_interval,
            "min_bits": controller.min_bits,
        },
        "final_avg_bits": final_avg,
        "final_per_layer_bits": final_bits,
        "bits_history": [
            {"epoch": e, "bits": snap}
            for e, snap in controller.bits_history
        ],
        "summary": {
            "method": "AdaPT",
            "fp32_acc": fp32_acc,
            "method_acc": adapt_acc,
            "int8_acc": int8_acc,
            "int4_acc": int4_acc,
            "avg_bits": round(final_avg, 2),
            "per_layer_bits": final_bits,
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in adapt_history
            ],
            "category": "adaptive",
        },
    }
    save_results(results, os.path.join(result_dir, "adapt_results.json"))

    model_path = os.path.join(result_dir, f"{cfg.model}_adapt.pt")
    torch.save(adapt_model.state_dict(), model_path)
    file_saved(model_path)


if __name__ == "__main__":
    main()
