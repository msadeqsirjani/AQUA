"""
EQAT — MNIST Experiment (Table 1, row 1 of the paper).

Target results:
  Accuracy : ~99.45 %
  Energy   : ~0.60  (normalised vs 8-bit baseline)
  Avg bits : ~6.0

Run from project root:
  python -m EQAT.scripts.train_mnist
"""

import ssl, sys, os
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from EQAT.models.models import SimpleCNN5
from EQAT.core.energy   import EnergyModel
from EQAT.core.loss     import EQATLoss
from EQAT.core.trainer  import EQATTrainer

# ── Hyper-parameters (Section 4.1 of the paper) ──────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 64
LR_WEIGHTS     = 1.0          # Adadelta (paper: lr=1.0 for MNIST)
LR_BITWIDTH    = 0.0001       # Adam for q̃  (lower to prevent collapse)
ALPHA          = 0.95         # KL weight
BETA           = 0.005        # max energy weight (reduced to prevent collapse)
WARMUP         = 30           # epochs to ramp β from 0 → BETA (longer warmup)
FREEZE_BW      = 25           # freeze bit-widths until model converges (longer freeze)
Q_MIN, Q_MAX   = 2.0, 8.0
SEED           = 42
SAVE_DIR       = 'results/EQAT/MNIST'


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
    model = SimpleCNN5(in_channels=1, num_classes=10, q_min=Q_MIN, q_max=Q_MAX)

    # Spatial sizes (H, W) entering each block — after preceding pool layers
    # MNIST 28×28: conv1→28×28 (no pool before), then pool→14, pool→7
    in_hws = [
        (28, 28),   # block1 conv sees 28×28
        (14, 14),   # block2 conv sees 14×14  (after pool)
        (7,  7),    # block3 conv sees 7×7    (after pool)
        (),         # fc1  — Linear, no spatial
        (),         # fc2  — Linear, no spatial
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
        optimizer_cls       = torch.optim.Adadelta,
        weight_optim_kwargs = {},
        freeze_bw_epochs    = FREEZE_BW,
        # No LR scheduler for MNIST (paper uses plain Adadelta)
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'EQAT MNIST  |  device={DEVICE}  |  seed={SEED}  |  bit-width frozen for first {FREEZE_BW} epochs')
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
        'dataset': 'MNIST',
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
            'optimizer_weights': 'Adadelta',
            'optimizer_bitwidth': 'Adam',
            'seed': SEED,
        },
        'paper_targets': {
            'accuracy': 99.45,
            'energy': 0.60,
            'avg_bits': 6.0,
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
    print('Paper targets: acc≈99.45%,  E≈0.60,  avg bits≈6.0')
    print()
    print(f'Results saved to {SAVE_DIR}/')
    print(f'  - final.pt       : PyTorch checkpoint')
    print(f'  - summary.json   : Human-readable summary')
    print(f'  - results.csv    : Per-epoch training log')
