"""
EQAT Trainer — Algorithm 1 from the paper.

Key points implemented correctly:
1. Full-precision forward: model(x, quantize=False) — no hacks.
2. Quantized forward: model(x, quantize=True) — weights + activations.
3. One backward pass covers both weight and bit-width gradients.
4. Bit-width optimiser is FROZEN for `freeze_bw_epochs` to let the model
   converge at the initial precision (~6.7 bits) before reducing bits.
5. Adam for bit-widths with low lr (avoids the collapse seen with lr=0.01).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from EQAT.core.loss   import EQATLoss
from EQAT.core.energy import EnergyModel


class EQATTrainer:
    def __init__(
        self,
        model:              nn.Module,
        energy_model:       EnergyModel,
        loss_fn:            EQATLoss,
        lr_weights:         float  = 1.0,
        lr_bitwidth:        float  = 0.001,
        device:             str    = 'cpu',
        optimizer_cls              = None,     # for weights
        weight_optim_kwargs: dict  = None,
        freeze_bw_epochs:   int    = 20,       # freeze bit-widths this many epochs
        scheduler_cls              = None,     # e.g. CosineAnnealingLR (for CIFAR)
        scheduler_kwargs:   dict   = None,
    ):
        self.model        = model.to(device)
        self.energy_model = energy_model
        self.loss_fn      = loss_fn
        self.device       = device
        self.freeze_bw_epochs = freeze_bw_epochs

        # ── Separate weight params from bit-width / PACT params ──────────
        bw_ids   = {id(p) for p in model.bitwidth_params()}
        pact_ids = {id(p) for p in model.pact_params()}
        special  = bw_ids | pact_ids

        weight_params  = [p for p in model.parameters() if id(p) not in special]
        bw_params      = model.bitwidth_params()
        pact_params    = model.pact_params()

        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        kw = weight_optim_kwargs or {}

        self.opt_w    = optimizer_cls(weight_params, lr=lr_weights, **kw)
        self.opt_bw   = torch.optim.Adam(bw_params + pact_params, lr=lr_bitwidth)

        # Optional LR scheduler on opt_w (e.g. CosineAnnealingLR for CIFAR)
        skw = scheduler_kwargs or {}
        self.scheduler = scheduler_cls(self.opt_w, **skw) if scheduler_cls else None

    # ── Training ─────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        totals = {'L_CE': 0., 'L_KL': 0., 'E_norm': 0., 'total': 0.}
        n = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            # 1. Full-precision forward (no quantization, no grad needed)
            with torch.no_grad():
                y_full = self.model(x, quantize=False)

            # 2. Quantized forward — weights and activations quantized
            y_quant = self.model(x, quantize=True)

            # 3. Differentiable energy
            e_norm = self.energy_model()

            # 4. 3-term loss
            loss, comps = self.loss_fn(y_quant, y_full, y, e_norm, epoch)

            # 5. Backward
            self.opt_w.zero_grad()
            self.opt_bw.zero_grad()

            # Check for NaN in loss before backward
            if not torch.isfinite(loss):
                print(f"Warning: NaN/Inf in loss at epoch {epoch}, skipping batch")
                continue

            loss.backward()

            # Clip gradients more aggressively
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Check for NaN in gradients
            has_nan = False
            for param in self.model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan = True
                    break

            if has_nan:
                print(f"Warning: NaN in gradients at epoch {epoch}, skipping batch")
                continue

            # 6. Weight update (always)
            self.opt_w.step()

            # 7. Bit-width update (frozen for first freeze_bw_epochs)
            if epoch > self.freeze_bw_epochs:
                self.opt_bw.step()

            for k in ('L_CE', 'L_KL', 'E_norm'):
                totals[k] += comps.get(k, 0.)
            totals['total'] += loss.item()
            n += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v / n for k, v in totals.items()}

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds    = self.model(x, quantize=True).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return 100. * correct / total

    def get_bitwidths(self) -> list:
        return self.model.get_bitwidths()

    def get_energy(self) -> float:
        return self.energy_model.normalised()
