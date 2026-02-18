"""
Shared training utilities used by all algorithm training scripts.

Provides device setup, model loading, evaluation, training loops,
result saving, and plotting -- eliminating ~120 lines of boilerplate
per script.
"""

import json
import os

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.console import (
    console, metric, success, file_saved, info, training_progress,
)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def setup_device():
    """Detect device (CUDA / CPU) and print it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = str(device)
    if device.type == "cuda":
        label += f" ({torch.cuda.get_device_name(0)})"
    console.print(f"  [dim]Device:[/dim] [bold magenta]{label}[/bold magenta]")
    return device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute top-1 accuracy (%) on a DataLoader.

    Args:
        model: nn.Module (switched to eval automatically).
        loader: DataLoader yielding (inputs, targets).
        device: torch device.

    Returns:
        Accuracy as a float in [0, 100].
    """
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# FP32 model loading
# ---------------------------------------------------------------------------

def load_fp32_model(model_name, device, val_loader, num_classes=10,
                    img_size=32, dataset_name=None):
    """Load a pretrained FP32 model and return (model, fp32_accuracy).

    Searches for the checkpoint in order:
      1. ``results/{dataset}/{model}/{model}_fp32.pt``   (dataset-aware)
      2. ``results/{model}/{model}_fp32.pt``              (model-only)
      3. ``results/{model}_fp32.pt``                      (legacy flat)

    Args:
        model_name: key in MODEL_REGISTRY (e.g. "resnet18").
        device: torch device.
        val_loader: validation DataLoader (used to compute FP32 accuracy).
        num_classes: number of output classes.
        img_size: input image size (passed to ViT).
        dataset_name: dataset name (optional, for path resolution).

    Returns:
        (model, fp32_acc) tuple.

    Raises:
        FileNotFoundError: if no checkpoint exists.
    """
    from models import get_model

    model = get_model(model_name, num_classes=num_classes, img_size=img_size)

    candidates = []
    if dataset_name:
        candidates.append(
            os.path.join("results", dataset_name, model_name,
                         f"{model_name}_fp32.pt")
        )
    candidates.append(
        os.path.join("results", model_name, f"{model_name}_fp32.pt")
    )
    candidates.append(
        os.path.join("results", f"{model_name}_fp32.pt")
    )

    ckpt = None
    for p in candidates:
        if os.path.exists(p):
            ckpt = p
            break

    if ckpt is None:
        raise FileNotFoundError(
            f"No FP32 checkpoint found for '{model_name}'.\n"
            f"  Expected: {candidates[0]}\n"
            f"  Run: python -m scripts.pretrain --config <your_config>.yaml"
        )

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    fp32_acc = evaluate(model, val_loader, device)
    metric("Loaded FP32", f"{model_name} from [cyan]{ckpt}[/cyan]")
    metric("FP32 accuracy", f"{fp32_acc:.2f}%")
    return model, fp32_acc


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch.

    Returns:
        (avg_loss, accuracy_pct) tuple.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
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
    return total_loss / total, 100.0 * correct / total


def standard_train_loop(model, train_loader, val_loader, device, *,
                        epochs=30, optimizer=None, scheduler=None,
                        lr=1e-4, label="QAT"):
    """Generic train-val loop with history tracking and rich progress bar.

    If *optimizer* / *scheduler* are ``None``, sensible defaults are created
    (SGD + CosineAnnealing).

    Returns:
        (best_acc, history)  where *history* is a list of
        ``(epoch, train_loss, train_acc, val_acc)`` tuples.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
        )
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
        )

    best_acc = 0.0
    history = []

    with training_progress(epochs) as progress:
        task = progress.add_task(f"{label}", total=epochs)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc

            history.append((epoch, train_loss, train_acc, val_acc))

            progress.update(
                task, advance=1,
                description=(
                    f"[{label}] Epoch {epoch}/{epochs}  "
                    f"Loss [yellow]{train_loss:.4f}[/yellow]  "
                    f"Train [cyan]{train_acc:.1f}%[/cyan]  "
                    f"Val [green]{val_acc:.1f}%[/green]"
                ),
            )

    return best_acc, history


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(results_dict, path):
    """Save a results dict as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    file_saved(path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(histories, labels, save_path, title="Training Curves"):
    """Plot validation accuracy curves for one or more training runs.

    Args:
        histories: list of history lists.  Each history is a list of
            ``(epoch, train_loss, train_acc, val_acc)`` tuples.
        labels: list of string labels (same length as *histories*).
        save_path: path for the saved PNG.
        title: figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for hist, lbl in zip(histories, labels):
        epochs = [h[0] for h in hist]
        val_accs = [h[3] for h in hist]
        ax.plot(epochs, val_accs, "o-", markersize=3, linewidth=2, label=lbl)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)
