"""
Baseline — CIFAR-10 Experiment (full-precision training).

This serves as the baseline for comparison with EQAT.
No quantization, no energy modeling — standard cross-entropy training.

Run from project root:
  python -m Baseline.scripts.train_cifar10
"""

import ssl
import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context

from Baseline.models.models import SimpleCNN5
from Baseline.core.energy import BaselineEnergyModel

# ── Hyper-parameters ───────────────────────────────────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 64
LR             = 0.001        # Adam (same as EQAT for fair comparison)
SEED           = 42
SAVE_DIR       = 'results/Baseline/CIFAR-10'
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
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_ld  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SimpleCNN5(in_channels=3, num_classes=10).to(DEVICE)

    # Energy model
    layers = model.get_compute_layers()
    spatial_sizes = model.get_spatial_sizes(in_channels=3)
    energy_model = BaselineEnergyModel(layers, spatial_sizes, bitwidth=BITWIDTH)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────
    bitwidth_str = f'{int(BITWIDTH)}-bit' if BITWIDTH == 8 else 'FP32'
    print(f'Baseline CIFAR-10  |  device={DEVICE}  |  seed={SEED}  |  {bitwidth_str}')
    hdr = f"{'Ep':>4} | {'TrainAcc%':>10} | {'TestAcc%':>9} | {'Loss':>8} | {'E(norm)':>8} | {'Bits':>5} | {'LR':>8}"
    print(hdr)
    print('-' * len(hdr))

    best_acc = 0.
    results = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, DEVICE, train_ld, optimizer, criterion, epoch)
        test_acc = evaluate(model, DEVICE, test_ld)
        current_lr = optimizer.param_groups[0]['lr']
        energy_norm = energy_model.normalized()

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'lr': current_lr,
            'energy_normalized': energy_norm,
            'bitwidth': BITWIDTH,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{SAVE_DIR}/best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>4} | {train_acc:>10.2f} | {test_acc:>9.2f} | "
                  f"{train_loss:>8.4f} | {energy_norm:>8.2f} | {int(BITWIDTH):>5} | {current_lr:>8.6f}")

        scheduler.step()

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
        'dataset': 'CIFAR-10',
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
            'optimizer': 'Adam',
            'lr_schedule': 'CosineAnnealing',
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
