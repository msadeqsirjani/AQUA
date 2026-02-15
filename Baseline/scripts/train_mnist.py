"""
Baseline — MNIST Experiment (full-precision training).

This serves as the baseline for comparison with EQAT.
No quantization, no energy modeling — standard cross-entropy training.

Run from project root:
  python -m Baseline.scripts.train_mnist
"""

import ssl
import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context

from Baseline.models.models import SimpleCNN5
from Baseline.core.energy import BaselineEnergyModel

# ── Hyper-parameters ───────────────────────────────────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 64
LR             = 1.0          # Adadelta (same as EQAT for fair comparison)
SEED           = 42
SAVE_DIR       = 'results/Baseline/MNIST'
BITWIDTH       = 8.0          # 8-bit uniform quantization baseline (paper standard)


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, device, test_loader):
    """Evaluate on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == '__main__':
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SimpleCNN5(in_channels=1, num_classes=10).to(DEVICE)

    # Energy model
    layers = model.get_compute_layers()
    spatial_sizes = model.get_spatial_sizes(in_channels=1)
    energy_model = BaselineEnergyModel(layers, spatial_sizes, bitwidth=BITWIDTH)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────
    bitwidth_str = f'{int(BITWIDTH)}-bit' if BITWIDTH == 8 else 'FP32'
    print(f'Baseline MNIST  |  device={DEVICE}  |  seed={SEED}  |  {bitwidth_str}')
    hdr = f"{'Ep':>4} | {'TrainAcc%':>10} | {'TestAcc%':>9} | {'Loss':>8} | {'E(norm)':>8} | {'Bits':>5}"
    print(hdr)
    print('-' * len(hdr))

    best_acc = 0.
    results = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, DEVICE, train_ld, optimizer, criterion, epoch)
        test_acc = evaluate(model, DEVICE, test_ld)
        energy_norm = energy_model.normalized()

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'energy_normalized': energy_norm,
            'bitwidth': BITWIDTH,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{SAVE_DIR}/best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>4} | {train_acc:>10.2f} | {test_acc:>9.2f} | {train_loss:>8.4f} | "
                  f"{energy_norm:>8.2f} | {int(BITWIDTH):>5}")

    # Save final results (PyTorch checkpoint)
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'best_accuracy': best_acc,
        'energy_normalized': energy_norm,
        'bitwidth': BITWIDTH,
        'energy_stats': energy_model.get_stats(),
    }, f'{SAVE_DIR}/final.pt')

    # Save summary as JSON (human-readable)
    summary = {
        'method': 'Baseline',
        'dataset': 'MNIST',
        'best_test_accuracy': float(best_acc),
        'final_train_accuracy': float(results[-1]['train_acc']),
        'energy_normalized': float(energy_norm),
        'energy_absolute': float(energy_model.compute_energy()),
        'bitwidth_weights': float(BITWIDTH),
        'bitwidth_activations': float(BITWIDTH),
        'hyperparameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'optimizer': 'Adadelta',
            'seed': SEED,
        },
        'energy_stats': energy_model.get_stats(),
    }
    with open(f'{SAVE_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-epoch results as CSV
    with open(f'{SAVE_DIR}/results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print('-' * len(hdr))
    print(f'Best test accuracy : {best_acc:.2f}%')
    print(f'Energy (normalized): {energy_norm:.2f}')
    print(f'Energy (absolute)  : {energy_model.compute_energy():,.0f}')
    print(f'Bit-width          : {int(BITWIDTH)} bits')
    print(f'\nResults saved to {SAVE_DIR}/')
    print(f'  - final.pt       : PyTorch checkpoint')
    print(f'  - summary.json   : Human-readable summary')
    print(f'  - results.csv    : Per-epoch training log')
