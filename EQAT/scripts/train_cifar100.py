"""
EQAT — CIFAR-100 Experiment (Table 1, row 3 of the paper).

Target results:
  Accuracy : ~69.98 %  (baseline 65.72 %, +4.26% improvement)
  Energy   : ~0.97  (normalised vs 8-bit baseline, 3% reduction)
  Avg bits : ~6.75

Run from project root:
  python -m EQAT.scripts.train_cifar100
"""

import ssl, sys, os
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import csv
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from EQAT.models.models import ResNet18EQAT
from EQAT.core.energy   import EnergyModel
from EQAT.core.loss     import EQATLoss
from EQAT.core.trainer  import EQATTrainer

# ── Hyper-parameters (Section 4.1 of the paper) ──────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 128
LR_WEIGHTS     = 0.001        # Adam lr
LR_BITWIDTH    = 0.001        # Adam lr for q̃
ALPHA          = 0.7          # KL weight for CIFAR-100
BETA           = 0.01         # max energy weight (ramps from 0.001→0.01 via warmup)
WARMUP         = 10
FREEZE_BW      = 20
Q_MIN, Q_MAX   = 2.0, 8.0
SEED           = 42
SAVE_DIR       = 'results/EQAT/CIFAR-100'


if __name__ == '__main__':
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = datasets.CIFAR100('./data', train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_ld  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ResNet18EQAT(num_classes=100, q_min=Q_MIN, q_max=Q_MAX)

    # Spatial sizes (H, W) entering each _EQATResConv, in forward order.
    # ResNet-18 on CIFAR-100 (32×32 input):
    #   stem  : 32×32 → conv7×7 stride=2 → 16×16 → maxpool → 8×8
    #   layer1: block0(conv1:8×8, conv2:8×8),  block1(conv1:8×8,  conv2:8×8)
    #   layer2: block0(conv1:8×8, conv2:4×4, ds:8×8), block1(conv1:4×4, conv2:4×4)
    #   layer3: block0(conv1:4×4, conv2:2×2, ds:4×4), block1(conv1:2×2, conv2:2×2)
    #   layer4: block0(conv1:2×2, conv2:1×1, ds:2×2), block1(conv1:1×1, conv2:1×1)
    #   fc    : Linear (no spatial)
    #
    # in_hws list must match the order returned by model.eqat_blocks():
    #   stem, then for each residual block: unit_a, unit_b, [downsample]
    in_hws = [
        # stem
        (32, 32),
        # layer1: block0 (no downsample)
        (8,  8),    # unit_a (after maxpool, 8×8)
        (8,  8),    # unit_b
        # layer1: block1
        (8,  8),    # unit_a
        (8,  8),    # unit_b
        # layer2: block0 (downsample, stride=2: 8×8→4×4)
        (8,  8),    # unit_a
        (4,  4),    # unit_b
        (8,  8),    # downsample
        # layer2: block1
        (4,  4),    # unit_a
        (4,  4),    # unit_b
        # layer3: block0 (downsample, stride=2: 4×4→2×2)
        (4,  4),    # unit_a
        (2,  2),    # unit_b
        (4,  4),    # downsample
        # layer3: block1
        (2,  2),    # unit_a
        (2,  2),    # unit_b
        # layer4: block0 (downsample, stride=2: 2×2→1×1)
        (2,  2),    # unit_a
        (1,  1),    # unit_b
        (2,  2),    # downsample
        # layer4: block1
        (1,  1),    # unit_a
        (1,  1),    # unit_b
        # fc (Linear, no spatial)
        (),
    ]

    energy_model = EnergyModel(model.eqat_blocks(), in_hws)
    loss_fn      = EQATLoss(alpha=ALPHA, beta=BETA, warmup_epochs=WARMUP)

    trainer = EQATTrainer(
        model               = model,
        energy_model        = energy_model,
        loss_fn             = loss_fn,
        lr_weights          = LR_WEIGHTS,
        lr_bitwidth         = LR_BITWIDTH,
        device              = DEVICE,
        optimizer_cls       = torch.optim.Adam,
        weight_optim_kwargs = {},
        freeze_bw_epochs    = FREEZE_BW,
        # CosineAnnealingLR for CIFAR as specified in paper Section 4.1
        scheduler_cls       = CosineAnnealingLR,
        scheduler_kwargs    = {'T_max': EPOCHS},
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'EQAT CIFAR-100  |  device={DEVICE}  |  seed={SEED}  |  bit-width frozen for first {FREEZE_BW} epochs')
    hdr = f"{'Ep':>4} | {'Acc%':>7} | {'E(norm)':>8} | {'AvgBits':>8} | {'L_CE':>7} | {'L_KL':>7}"
    print(hdr)
    print('-' * len(hdr))

    best_acc = 0.
    results = []

    for epoch in range(1, EPOCHS + 1):
        losses  = trainer.train_epoch(train_ld, epoch)
        acc     = trainer.evaluate(test_ld)
        bits    = trainer.get_bitwidths()
        avg_bw  = sum(bits) / len(bits)
        e_norm  = trainer.get_energy()

        # Store results
        results.append({
            'epoch': epoch,
            'test_acc': acc,
            'energy_normalized': e_norm,
            'avg_bitwidth': avg_bw,
            'bitwidths': bits,
            'loss_ce': losses['L_CE'],
            'loss_kl': losses['L_KL'],
        })

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{SAVE_DIR}/best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>4} | {acc:>7.2f} | {e_norm:>8.3f} | {avg_bw:>8.2f} | "
                  f"{losses['L_CE']:>7.4f} | {losses['L_KL']:>7.4f}")

    # Save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'best_accuracy': best_acc,
    }, f'{SAVE_DIR}/final.pt')

    # Save summary as JSON
    final_bits = trainer.get_bitwidths()
    final_energy = trainer.get_energy()
    summary = {
        'method': 'EQAT',
        'dataset': 'CIFAR-100',
        'best_test_accuracy': float(best_acc),
        'final_test_accuracy': float(results[-1]['test_acc']),
        'energy_normalized': float(final_energy),
        'avg_bitwidth': float(sum(final_bits) / len(final_bits)),
        'bitwidths_per_layer': [float(b) for b in final_bits],
        'hyperparameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr_weights': LR_WEIGHTS,
            'lr_bitwidth': LR_BITWIDTH,
            'alpha': ALPHA,
            'beta': BETA,
            'warmup': WARMUP,
            'freeze_bw': FREEZE_BW,
            'q_min': Q_MIN,
            'q_max': Q_MAX,
            'optimizer_weights': 'Adam',
            'optimizer_bitwidth': 'Adam',
            'lr_schedule': 'CosineAnnealing',
            'seed': SEED,
        },
        'paper_targets': {
            'accuracy': 69.98,
            'energy': 0.97,
            'avg_bits': 6.75,
        }
    }
    with open(f'{SAVE_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-epoch results as CSV
    csv_results = []
    for r in results:
        csv_row = {
            'epoch': r['epoch'],
            'test_acc': r['test_acc'],
            'energy_normalized': r['energy_normalized'],
            'avg_bitwidth': r['avg_bitwidth'],
            'loss_ce': r['loss_ce'],
            'loss_kl': r['loss_kl'],
        }
        csv_results.append(csv_row)

    with open(f'{SAVE_DIR}/results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)

    print('-' * len(hdr))
    print(f'Best accuracy      : {best_acc:.2f}%')
    print(f'Final energy (norm): {final_energy:.3f}')
    print(f'Final avg bits     : {sum(final_bits)/len(final_bits):.2f}')
    print(f'Final bits/layer   : {final_bits}')
    print()
    print('Paper targets: acc≈69.98% (+4.26% vs baseline),  E≈0.97,  avg bits≈6.75')
    print()
    print(f'Results saved to {SAVE_DIR}/')
    print(f'  - final.pt       : PyTorch checkpoint')
    print(f'  - summary.json   : Human-readable summary')
    print(f'  - results.csv    : Per-epoch training log')
