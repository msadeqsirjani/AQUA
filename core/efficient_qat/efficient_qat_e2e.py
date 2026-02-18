"""
EfficientQAT Phase 2: End-to-End Quantization Parameter Training (E2E-QP).
Adapted from Chen et al., "EfficientQAT" (ACL 2025 Main).
Original repo: https://github.com/OpenGVLab/EfficientQAT

After Block-AP (Phase 1), all weights and quantization parameters have
been initialized. Phase 2 freezes ALL weight parameters and only trains
the quantization parameters (scales and zero_points) end-to-end using
the standard cross-entropy loss.

This is fast because:
  - No weight gradient computation (fewer parameters, smaller backward)
  - Only scale/zp parameters have gradients
  - Can run for fewer epochs (10-15)
"""

import torch
import torch.nn as nn

from .efficient_qat_resnet import EQATConv2d, EQATLinear, set_quant_state


class E2EQPTrainer:
    """Phase 2: End-to-end quantization parameter training.

    Freezes all model weights. Only scale and zero_point parameters
    are trainable. Uses standard cross-entropy loss end-to-end.

    Following EfficientQAT's E2E-QP:
      - Freeze all parameters, then unfreeze only scales/zero_points
      - AdamW optimizer with cosine LR schedule
      - Warmup for 3% of total steps
      - Gradient clipping at max_norm=0.3
    """

    def __init__(self, lr=2e-5, warmup_ratio=0.03, max_grad_norm=0.3):
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm

    def freeze_weights(self, model):
        """Freeze everything except quantizer scale/zero_point params.

        Following EfficientQAT:
          1. Freeze ALL parameters
          2. Unfreeze only 'scale' and 'zero_point' in EQAT layers
        """
        # Step 1: Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze only quantizer parameters
        n_unfrozen = 0
        for name, module in model.named_modules():
            if isinstance(module, (EQATConv2d, EQATLinear)):
                for pname, param in module.weight_quantizer.named_parameters():
                    param.requires_grad = True
                    n_unfrozen += 1

        return n_unfrozen

    def _get_quant_params(self, model):
        """Collect trainable quantizer parameters."""
        params = []
        for module in model.modules():
            if isinstance(module, (EQATConv2d, EQATLinear)):
                for param in module.weight_quantizer.parameters():
                    if param.requires_grad:
                        params.append(param)
        return params

    def train(self, model, train_loader, val_loader, device, epochs=15):
        """Run E2E-QP training.

        Args:
            model: Quantized model from Block-AP (Phase 1).
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: Torch device.
            epochs: Number of E2E-QP epochs (default: 15).

        Returns:
            (best_acc, history) where history is list of
            (epoch, train_loss, train_acc, val_acc) tuples.
        """
        model.to(device)

        # Enable quantization for forward pass
        set_quant_state(model, True)

        # Freeze weights, unfreeze only quant params
        n_unfrozen = self.freeze_weights(model)
        print(f"    [E2E-QP] Frozen all weights. "
              f"Unfrozen {n_unfrozen} quantizer parameters.")

        # Optimizer on quant params only
        quant_params = self._get_quant_params(model)
        if not quant_params:
            print("    [E2E-QP] WARNING: No trainable quantizer params found!")
            return 0.0, []

        optimizer = torch.optim.AdamW(
            quant_params, lr=self.lr, weight_decay=0,
        )

        # Cosine schedule with linear warmup
        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            import math
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        history = []

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping (following EfficientQAT)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        quant_params, self.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.size(0)

            train_loss = total_loss / total
            train_acc = 100.0 * correct / total

            # Validate
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(targets).sum().item()
                    val_total += inputs.size(0)

            val_acc = 100.0 * val_correct / val_total
            history.append((epoch, train_loss, train_acc, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % 5 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"    [E2E-QP] Epoch {epoch:2d}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                      f"LR: {lr_now:.2e}")

        return best_acc, history
