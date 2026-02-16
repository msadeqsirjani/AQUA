"""
Unified training script.

Usage:
  python -m scripts.train --dataset mnist --model simplecnn5 --mode baseline
  python -m scripts.train --dataset cifar10 --model simplecnn5 --mode baseline
  python -m scripts.train --dataset cifar100 --model resnet18 --mode baseline
  python -m scripts.train --dataset cifar10 --model resnet18 --mode baseline --epochs 200 --lr 0.01
"""

import argparse
import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.data import get_dataloaders
from utils.factory import get_model, get_energy_model
from utils.training import train_epoch, evaluate

# ── Sensible defaults per dataset ──────────────────────────────────────────
_DEFAULTS = {
    'mnist':    {'batch_size': 64,  'lr': 1.0,   'optimizer': 'adadelta'},
    'cifar10':  {'batch_size': 64,  'lr': 0.001, 'optimizer': 'adam'},
    'cifar100': {'batch_size': 128, 'lr': 0.001, 'optimizer': 'adam'},
}


def parse_args():
    p = argparse.ArgumentParser(description='AQUA Training')
    p.add_argument('--dataset',   required=True, choices=['mnist', 'cifar10', 'cifar100'])
    p.add_argument('--model',     required=True, choices=['simplecnn5', 'resnet18'])
    p.add_argument('--mode',      default='baseline', choices=['baseline'])
    p.add_argument('--epochs',    type=int,   default=100)
    p.add_argument('--batch-size',type=int,   default=None, help='default: auto per dataset')
    p.add_argument('--lr',        type=float, default=None, help='default: auto per dataset')
    p.add_argument('--optimizer', default=None, choices=['adam', 'adadelta'], help='default: auto per dataset')
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--bitwidth',  type=float, default=8.0)
    p.add_argument('--save-dir',  default=None, help='default: results/<mode>/<dataset>')
    return p.parse_args()


def main():
    args = parse_args()

    # Fill defaults from dataset if not overridden
    defaults = _DEFAULTS[args.dataset]
    batch_size = args.batch_size or defaults['batch_size']
    lr         = args.lr         or defaults['lr']
    opt_name   = args.optimizer  or defaults['optimizer']
    save_dir   = args.save_dir   or f'results/{args.mode}/{args.dataset}'

    torch.manual_seed(args.seed)
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────
    train_ld, test_ld, in_channels, num_classes = get_dataloaders(
        args.dataset, batch_size
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(args.model, in_channels, num_classes).to(device)
    energy_model = get_energy_model(model, args.model, in_channels, args.bitwidth)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = None
    if opt_name == 'adam':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    bitwidth_str = f'{int(args.bitwidth)}-bit' if args.bitwidth == 8 else 'FP32'
    print(f'{args.mode} {args.dataset} ({args.model})  |  device={device}  |  '
          f'seed={args.seed}  |  {bitwidth_str}')

    has_lr_col = scheduler is not None
    hdr = f"{'Ep':>4} | {'TrainAcc%':>10} | {'TestAcc%':>9} | {'Loss':>8} | {'E(norm)':>8} | {'Bits':>5}"
    if has_lr_col:
        hdr += f" | {'LR':>8}"
    print(hdr)
    print('-' * len(hdr))

    best_acc = 0.
    results = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_ld, optimizer, criterion, epoch)
        test_acc = evaluate(model, device, test_ld)
        energy_norm = energy_model.normalized()

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'energy_normalized': energy_norm,
            'bitwidth': args.bitwidth,
        }
        if has_lr_col:
            row['lr'] = optimizer.param_groups[0]['lr']
        results.append(row)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{save_dir}/best.pt')

        if epoch % 5 == 0 or epoch == 1:
            line = (f"{epoch:>4} | {train_acc:>10.2f} | {test_acc:>9.2f} | "
                    f"{train_loss:>8.4f} | {energy_norm:>8.2f} | {int(args.bitwidth):>5}")
            if has_lr_col:
                line += f" | {optimizer.param_groups[0]['lr']:>8.6f}"
            print(line)

        if scheduler:
            scheduler.step()

    # ── Save results ──────────────────────────────────────────────────────
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'best_accuracy': best_acc,
        'energy_normalized': energy_norm,
        'bitwidth': args.bitwidth,
        'energy_stats': energy_model.get_stats(),
    }, f'{save_dir}/final.pt')

    summary = {
        'method': args.mode,
        'dataset': args.dataset,
        'model': args.model,
        'best_test_accuracy': float(best_acc),
        'final_train_accuracy': float(results[-1]['train_acc']),
        'energy_normalized': float(energy_norm),
        'energy_absolute': float(energy_model.compute_energy()),
        'bitwidth_weights': float(args.bitwidth),
        'bitwidth_activations': float(args.bitwidth),
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': opt_name,
            'lr_schedule': 'CosineAnnealing' if scheduler else 'none',
            'seed': args.seed,
        },
        'energy_stats': energy_model.get_stats(),
    }
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(f'{save_dir}/results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print('-' * len(hdr))
    print(f'Best test accuracy : {best_acc:.2f}%')
    print(f'Energy (normalized): {energy_norm:.2f}')
    print(f'Energy (absolute)  : {energy_model.compute_energy():,.0f}')
    print(f'Bit-width          : {int(args.bitwidth)} bits')
    print(f'\nResults saved to {save_dir}/')


if __name__ == '__main__':
    main()
