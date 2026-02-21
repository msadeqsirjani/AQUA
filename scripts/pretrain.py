"""
Train an FP32 baseline model on any supported dataset.

Usage:
    python -m scripts.pretrain --config configs/cifar10_resnet18_4bit.yaml
    python -m scripts.pretrain --config configs/cifar100_vgg16_4bit.yaml
    python -m scripts.pretrain --config configs/tiny_imagenet_resnet34_8bit.yaml

The --config flag is required. All settings (dataset, model, epochs, lr,
batch_size, etc.) come from the YAML config file.
See configs/ for all available presets.

Saves the checkpoint to results/{dataset}/{model}/{model}_fp32.pt.
All quantization algorithm scripts expect this checkpoint to exist.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from models import get_model
from utils.data import get_dataloaders
from utils.training import setup_device, evaluate
from utils.config import load_config
from utils.console import (
    console, banner, section, print_config, metric, success, file_saved,
    training_progress,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train FP32 baseline (all settings from config file)",
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file (e.g. configs/cifar10_resnet18_4bit.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    epochs = cfg.fp32_epochs
    lr = cfg.fp32_lr

    banner("AQUA — FP32 Pretraining", f"Config: {args.config}")
    device = setup_device()
    print_config(cfg)

    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, data_root=cfg.data_root,
    )

    save_dir = os.path.join("results", cfg.dataset, cfg.model)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{cfg.model}_fp32.pt")

    # Check for existing checkpoint
    if os.path.exists(ckpt_path):
        section("Existing Checkpoint Found")
        model = get_model(cfg.model, num_classes=cfg.num_classes,
                          img_size=cfg.img_size)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.to(device)
        acc = evaluate(model, val_loader, device)
        metric("Checkpoint", ckpt_path)
        metric("Accuracy", f"{acc:.2f}%")
        return

    # Check legacy paths (model-only and flat)
    for legacy in [
        os.path.join("results", cfg.model, f"{cfg.model}_fp32.pt"),
        os.path.join("results", f"{cfg.model}_fp32.pt"),
    ]:
        if os.path.exists(legacy):
            section("Legacy Checkpoint Migration")
            import shutil
            shutil.copy2(legacy, ckpt_path)
            model = get_model(cfg.model, num_classes=cfg.num_classes,
                              img_size=cfg.img_size)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            model.to(device)
            acc = evaluate(model, val_loader, device)
            metric("Migrated from", legacy)
            metric("New path", ckpt_path)
            metric("Accuracy", f"{acc:.2f}%")
            return

    # Train from scratch
    section("Training", step=1)
    console.print(
        f"  Training [bold]{cfg.model}[/bold] on [bold]{cfg.dataset}[/bold] "
        f"for [bold]{epochs}[/bold] epochs\n"
    )

    model = get_model(cfg.model, num_classes=cfg.num_classes,
                      img_size=cfg.img_size).to(device)

    label_smoothing = getattr(cfg, "fp32_label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optim_name = getattr(cfg, "fp32_optimizer", "sgd").lower()
    wd = getattr(cfg, "fp32_weight_decay", 5e-4)
    warmup_epochs = getattr(cfg, "fp32_warmup_epochs", 0)

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd,
        )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs,
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    best_acc = 0.0
    with training_progress(epochs) as progress:
        task = progress.add_task("Training", total=epochs)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
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

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)

            lr_now = optimizer.param_groups[0]["lr"]
            progress.update(
                task, advance=1,
                description=(
                    f"Epoch {epoch}/{epochs}  "
                    f"Loss [yellow]{train_loss:.4f}[/yellow]  "
                    f"Train [cyan]{train_acc:.1f}%[/cyan]  "
                    f"Val [green]{val_acc:.1f}%[/green]  "
                    f"LR [dim]{lr_now:.5f}[/dim]"
                ),
            )

    # Final summary
    section("Complete")
    metric("Best accuracy", f"{best_acc:.2f}%")
    file_saved(ckpt_path)


if __name__ == "__main__":
    main()
