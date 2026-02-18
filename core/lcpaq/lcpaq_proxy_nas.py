"""
Low-cost proxy NAS for LCPAQ (Module C).
"Adaptive Quantization with Mixed-Precision Based on Low-Cost Proxy"
arXiv:2402.17706, 2024.

Runs short proxy training (5 epochs) with each hyperparameter combination
to rank them, then selects the best for full training. This avoids the
cost of running full 30-epoch training for each candidate.

Search space:
  lr:           {1e-5, 5e-5, 1e-4}
  weight_decay: {1e-4, 1e-5}
  Total: 6 combinations (each runs 5 proxy epochs)
"""

import copy
import time

import torch
import torch.nn as nn

from ..jacob_fake_quant import JacobFakeQuantize


# Default search space
DEFAULT_LR_CHOICES = [1e-5, 5e-5, 1e-4]
DEFAULT_WD_CHOICES = [1e-4, 1e-5]


def _run_proxy(model, train_loader, val_loader, device, lr, weight_decay,
               proxy_epochs=5):
    """Run a short proxy training and return validation accuracy.

    Enables observers for the full proxy duration (since it's short).

    Args:
        model: Quantized model (will be trained in-place).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Torch device.
        lr: Learning rate.
        weight_decay: Weight decay.
        proxy_epochs: Number of proxy epochs (default: 5).

    Returns:
        float: Best validation accuracy during proxy training (%).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=proxy_epochs,
    )

    best_acc = 0.0
    for epoch in range(1, proxy_epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, predicted = model(inputs).max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.size(0)

        val_acc = 100.0 * correct / total
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc


def proxy_hyperparameter_search(model_template, train_loader, val_loader,
                                device, proxy_epochs=5,
                                lr_choices=None, wd_choices=None):
    """Search for best QAT hyperparameters using short proxy training.

    For each (lr, weight_decay) combination, deep-copies the model,
    runs proxy_epochs of training, and records validation accuracy.
    Returns the best combination.

    Args:
        model_template: Quantized model (not modified — deep-copied per trial).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Torch device.
        proxy_epochs: Epochs per proxy trial (default: 5).
        lr_choices: List of learning rates to try.
        wd_choices: List of weight decays to try.

    Returns:
        (best_lr, best_wd, results) where results is a list of
        {"lr": float, "wd": float, "acc": float, "time": float} dicts.
    """
    if lr_choices is None:
        lr_choices = DEFAULT_LR_CHOICES
    if wd_choices is None:
        wd_choices = DEFAULT_WD_CHOICES

    results = []
    best_acc = 0.0
    best_lr = lr_choices[0]
    best_wd = wd_choices[0]

    n_total = len(lr_choices) * len(wd_choices)
    trial = 0

    for lr in lr_choices:
        for wd in wd_choices:
            trial += 1
            print(f"    [Proxy {trial}/{n_total}] lr={lr:.0e}, wd={wd:.0e} ...",
                  end=" ", flush=True)

            # Deep copy to avoid contaminating the template
            trial_model = copy.deepcopy(model_template)
            t0 = time.time()
            acc = _run_proxy(trial_model, train_loader, val_loader, device,
                             lr=lr, weight_decay=wd, proxy_epochs=proxy_epochs)
            elapsed = time.time() - t0

            results.append({
                "lr": lr, "wd": wd, "acc": acc, "time": elapsed,
            })
            print(f"acc={acc:.2f}% ({elapsed:.1f}s)")

            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_wd = wd

            # Free GPU memory
            del trial_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"    Best proxy: lr={best_lr:.0e}, wd={best_wd:.0e}, "
          f"acc={best_acc:.2f}%")

    return best_lr, best_wd, results
