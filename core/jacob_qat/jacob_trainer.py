"""
QAT training loop implementing the fine-tuning procedure
from Jacob et al. 2018.
"""

import torch
import torch.nn as nn

from ..jacob_fake_quant import JacobFakeQuantize


class JacobQATTrainer:
    """Quantization-Aware Training trainer.

    Fine-tunes a model prepared with fake quantization layers.
    Uses low learning rate (fine-tuning), SGD with cosine annealing.
    Observers run for first `observer_epochs`, then freeze.
    """

    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-4, epochs=30, observer_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.observer_epochs = observer_epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def _get_quantizers(self):
        """Collect all JacobFakeQuantize modules."""
        return [m for m in self.model.modules() if isinstance(m, JacobFakeQuantize)]

    def _disable_observers(self):
        """Freeze all observers — scale/zero_point stop updating."""
        for q in self._get_quantizers():
            q.disable_observer()
        print("  [QAT] Observers disabled — scale/zero_point frozen")

    def _print_scale_diagnostics(self, epoch):
        """Print scale/zero_point for first conv and last linear quantizers."""
        from core.jacob_qat.jacob_quantized_layers import QConv2d, QLinear

        first_conv = None
        last_linear = None
        for m in self.model.modules():
            if isinstance(m, QConv2d) and first_conv is None:
                first_conv = m
            if isinstance(m, QLinear):
                last_linear = m

        print(f"  [Diagnostics @ epoch {epoch}]")
        if first_conv is not None:
            wq = first_conv.weight_quantizer
            aq = first_conv.act_quantizer
            print(f"    First QConv2d weight: scale={wq.scale.item():.6f}, zp={wq.zero_point.item()}")
            print(f"    First QConv2d act:    scale={aq.scale.item():.6f}, zp={aq.zero_point.item()}")
        if last_linear is not None:
            wq = last_linear.weight_quantizer
            aq = last_linear.act_quantizer
            print(f"    Last QLinear weight:  scale={wq.scale.item():.6f}, zp={wq.zero_point.item()}")
            print(f"    Last QLinear act:     scale={aq.scale.item():.6f}, zp={aq.zero_point.item()}")

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        return 100.0 * correct / total

    def train(self):
        """Run full QAT training loop.

        Returns:
            (best_acc, history) where history is a list of
            (epoch, train_loss, train_acc, val_acc) tuples.
        """
        print(f"Starting QAT: {self.epochs} epochs, observer_epochs={self.observer_epochs}")
        best_acc = 0.0
        history = []

        for epoch in range(1, self.epochs + 1):
            if epoch == self.observer_epochs + 1:
                self._disable_observers()

            train_loss, train_acc = self._train_one_epoch()
            val_acc = self._validate()
            self.scheduler.step()
            history.append((epoch, train_loss, train_acc, val_acc))

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | "
                  f"Val: {val_acc:.2f}% | LR: {lr:.6f}")

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch in (5, self.epochs):
                self._print_scale_diagnostics(epoch)

        print(f"\nQAT complete. Best val accuracy: {best_acc:.2f}%")
        return best_acc, history
