"""
EQAT — CIFAR-10 Experiment (Table 1, row 2 of the paper).

Target results:
  Accuracy : ~86.23 %  (baseline 87.02 %)
  Energy   : ~0.65  (normalised vs 8-bit baseline, 35% reduction)
  Avg bits : ~5.85

Run from project root:
  python -m EQAT.scripts.train_cifar10
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

from EQAT.models.models import SimpleCNN5
from EQAT.core.energy   import EnergyModel
from EQAT.core.loss     import EQATLoss
from EQAT.core.trainer  import EQATTrainer

# ── Hyper-parameters (Section 4.1 of the paper) ──────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 64
LR_WEIGHTS     = 0.001        # Adam lr
LR_BITWIDTH    = 0.001        # Adam lr for q̃
ALPHA          = 0.8          # KL weight for CIFAR-10
BETA           = 0.01         # max energy weight (ramps from 0.001→0.01 via warmup)
WARMUP         = 10
FREEZE_BW      = 20
Q_MIN, Q_MAX   = 2.0, 8.0
SEED           = 42
SAVE_DIR       = 'results/EQAT/CIFAR-10'


if __name__ == '__main__':
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_ld  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    # CIFAR-10: 3-channel 32×32, after 3 pools → 4×4, flatten=2048
    model = SimpleCNN5(in_channels=3, num_classes=10, q_min=Q_MIN, q_max=Q_MAX)

    # Spatial sizes (H, W) entering each block
    in_hws = [
        (32, 32),   # block1 conv sees 32×32
        (16, 16),   # block2 conv sees 16×16  (after pool)
        (8,  8),    # block3 conv sees 8×8    (after pool)
        (),         # fc1
        (),         # fc2
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
    print(f'EQAT CIFAR-10  |  device={DEVICE}  |  seed={SEED}  |  bit-width frozen for first {FREEZE_BW} epochs')
    hdr = f"{'Ep':>4} | {'Acc%':>7} | {'E(norm)':>8} | {'AvgBits':>8} | {'L_CE':>7} | {'L_KL':>7} | bits/layer"
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
                  f"{losses['L_CE']:>7.4f} | {losses['L_KL']:>7.4f} | {bits}")

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
        'dataset': 'CIFAR-10',
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
            'accuracy': 86.23,
            'energy': 0.65,
            'avg_bits': 5.85,
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
    print('Paper targets: acc≈86.23%,  E≈0.65,  avg bits≈5.85')
    print()
    print(f'Results saved to {SAVE_DIR}/')
    print(f'  - final.pt       : PyTorch checkpoint')
    print(f'  - summary.json   : Human-readable summary')
    print(f'  - results.csv    : Per-epoch training log')
