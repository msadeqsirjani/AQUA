"""
End-to-end training script for Jacob et al. 2018 QAT.

Usage: python -m scripts.jacob_qat.train

Steps:
1. Train FP32 ResNet-18 baseline on CIFAR-10 (or load checkpoint)
2. Fold BatchNorm and prepare model for QAT
3. Run QAT fine-tuning for 30 epochs
4. Report FP32 vs INT8 QAT accuracy
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

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models import get_model
from utils import (
    get_dataloaders, setup_device, evaluate,
    load_fp32_model, save_results, plot_training_curves,
)
from utils.args import add_common_args, get_result_dir, get_fp32_path
from utils.config import load_config
from utils.console import console, banner, section, print_config, metric, info, success, file_saved
from core.jacob_qat import prepare_model_for_qat, JacobQATTrainer


def train_fp32_baseline(model, train_loader, val_loader, device,
                        epochs=200, lr=0.1, checkpoint_path=None):
    """Train FP32 baseline to ~93% on CIFAR-10 (or load checkpoint)."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        info(f"Loading FP32 checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device)
        acc = evaluate(model, val_loader, device)
        metric("FP32 baseline accuracy", f"{acc:.2f}%")
        return acc

    info(f"Training FP32 baseline for {epochs} epochs...")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
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

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            info(f"[FP32] Epoch {epoch:3d}/{epochs} | "
                 f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | "
                 f"Val: {val_acc:.2f}% | LR: {lr_now:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    success(f"FP32 training complete. Best val accuracy: {best_acc:.2f}%")
    return best_acc



def main():
    parser = argparse.ArgumentParser(description="Jacob QAT training")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("Jacob QAT", f"{cfg.model} on {cfg.dataset}")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "jacob_qat")
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        data_root=cfg.data_root,
    )

    # Step 1: FP32 baseline
    section(f"FP32 Baseline Training ({cfg.model}, {cfg.dataset})", step=1)
    model = get_model(cfg.model, num_classes=cfg.num_classes,
                      img_size=cfg.img_size)
    fp32_ckpt = get_fp32_path(cfg.dataset, cfg.model)
    fp32_acc = train_fp32_baseline(model, train_loader, val_loader, device,
                                   checkpoint_path=fp32_ckpt)

    # Step 2: Prepare for QAT (BN folding + fake quantization)
    section("Preparing model for QAT (BN folding)", step=2)
    qat_model = copy.deepcopy(model)
    qat_model = prepare_model_for_qat(qat_model)
    info("Model prepared for QAT. Architecture:")
    from core.jacob_qat.jacob_quantized_layers import QConv2d, QLinear
    n_qconv = sum(1 for m in qat_model.modules() if isinstance(m, QConv2d))
    n_qlin = sum(1 for m in qat_model.modules() if isinstance(m, QLinear))
    metric("QConv2d layers", n_qconv)
    metric("QLinear layers", n_qlin)

    # Step 3: QAT fine-tuning
    section("QAT Fine-tuning (30 epochs)", step=3)
    trainer = JacobQATTrainer(
        qat_model, train_loader, val_loader, device,
        lr=1e-4, epochs=30, observer_epochs=5,
    )
    qat_acc, qat_history = trainer.train()

    # Step 4: Results
    section("Results")
    metric("FP32 baseline accuracy", f"{fp32_acc:.2f}%")
    metric("INT8 QAT accuracy", f"{qat_acc:.2f}%")
    metric("Accuracy drop", f"{fp32_acc - qat_acc:.2f}%")

    if qat_acc >= 92.5:
        success("Status: PASS (>= 92.5%)")
    else:
        info("Status: FAIL (< 92.5%)")

    # Save outputs
    plot_training_curves(
        [qat_history], ["INT8 QAT"],
        os.path.join(result_dir, "jacob_qat_training_curves.png"),
        title=f"Jacob QAT on {cfg.model} ({cfg.dataset})",
    )

    ckpt_path = os.path.join(result_dir, f"{cfg.model}_qat_int8.pt")
    torch.save(qat_model.state_dict(), ckpt_path)
    file_saved(ckpt_path)

    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "int8_qat_acc": qat_acc,
        "accuracy_drop": fp32_acc - qat_acc,
        "summary": {
            "method": "Jacob QAT",
            "fp32_acc": fp32_acc,
            "method_acc": qat_acc,
            "int8_acc": qat_acc,
            "int4_acc": None,
            "avg_bits": 8.0,
            "per_layer_bits": {},
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in qat_history
            ],
            "category": "qat",
        },
    }
    save_results(results, os.path.join(result_dir, "jacob_qat_results.json"))


if __name__ == "__main__":
    main()
