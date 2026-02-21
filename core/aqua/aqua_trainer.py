"""
AQUA v2 QAT Trainer -- simple quantization-aware fine-tuning with knowledge distillation.

No learned dtype selection, no energy penalty, no entropy penalty.
Dtype assignment is already fixed analytically before training starts.

Loss::

    L = L_CE + alpha * T^2 * KL(softmax(y_q/T) || softmax(y_fp32/T))

Optimisers:
    - SGD (momentum=0.9, weight_decay=1e-4) for model weights
    - Adam for quantizer scale parameters
    - CosineAnnealingLR for both
"""

import torch
import torch.nn.functional as F

from .aqua_model import AQUAConv2d, AQUALinear, collect_quantizer_params


class AQUATrainer:

    def __init__(
        self, model, fp32_model, *,
        alpha=0.5,
        kd_temperature=4.0,
        lr_weights=0.0001,
        lr_scales=0.001,
        epochs=30,
    ):
        self.model = model
        self.fp32_model = fp32_model
        self.fp32_model.eval()
        for p in self.fp32_model.parameters():
            p.requires_grad_(False)

        self.alpha = alpha
        self.temperature = kd_temperature
        self.epochs = epochs

        weight_params = []
        scale_param_ids = set()
        for m in model.modules():
            if isinstance(m, (AQUAConv2d, AQUALinear)):
                for qp in m.quantizer.parameters():
                    scale_param_ids.add(id(qp))

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) not in scale_param_ids:
                weight_params.append(p)

        scale_params = collect_quantizer_params(model)

        self.opt_weights = torch.optim.SGD(
            weight_params, lr=lr_weights,
            momentum=0.9, weight_decay=1e-4,
        )
        self.opt_scales = torch.optim.Adam(scale_params, lr=lr_scales)

        self.sched_weights = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_weights, T_max=epochs,
        )
        self.sched_scales = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_scales, T_max=epochs,
        )
        self._first_epoch_done = False

    def _compute_loss(self, y_quant, y_true, y_fp32):
        l_ce = F.cross_entropy(y_quant, y_true)

        T = self.temperature
        l_kl = F.kl_div(
            F.log_softmax(y_quant / T, dim=1),
            F.softmax(y_fp32 / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        loss = l_ce + self.alpha * l_kl
        return loss, {"l_ce": l_ce.item(), "l_kl": l_kl.item()}

    def train_epoch(self, train_loader, device, epoch):
        self.model.train()

        total_loss = 0.0
        correct, total = 0, 0
        comp_accum = {"l_ce": 0.0, "l_kl": 0.0}

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            y_quant = self.model(inputs)
            with torch.no_grad():
                y_fp32 = self.fp32_model(inputs)

            loss, comp = self._compute_loss(y_quant, targets, y_fp32)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.opt_weights.zero_grad()
            self.opt_scales.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0,
            )

            self.opt_weights.step()
            self.opt_scales.step()

            bs = inputs.size(0)
            total_loss += loss.item() * bs
            _, predicted = y_quant.max(1)
            correct += predicted.eq(targets).sum().item()
            total += bs
            for k in comp_accum:
                comp_accum[k] += comp[k] * bs

        if self._first_epoch_done:
            self.sched_weights.step()
            self.sched_scales.step()
        self._first_epoch_done = True

        n = max(total, 1)
        return {
            "train_loss": total_loss / n,
            "train_acc": 100.0 * correct / n,
            **{k: v / n for k, v in comp_accum.items()},
            "lr_weights": self.opt_weights.param_groups[0]["lr"],
            "lr_scales": self.opt_scales.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self, val_loader, device):
        self.model.eval()
        correct, total = 0, 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
        return 100.0 * correct / max(total, 1)
