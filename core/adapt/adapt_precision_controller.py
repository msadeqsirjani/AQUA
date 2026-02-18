"""
Precision controller for Adaptive Precision Training (AdaPT).
Kummer et al., arXiv:2107.13490, 2021.

Monitors per-layer SNR (signal vs quantization noise) and gradient
norms. When BOTH conditions are met for a layer, its bit-width is
reduced by 1. Bits only decrease (never increase) during training.

Conditions for reduction:
  1. SNR >= snr_threshold (quantization error is small relative to signal)
  2. grad_norm >= grad_threshold (layer is still actively learning)
"""

import math

import torch
import torch.nn as nn

from .adapt_fixed_point import FixedPointQuantizer


class AdaPTPrecisionController:
    """Per-layer adaptive precision controller.

    Periodically checks information-theoretic conditions and reduces
    bit-widths for layers where it is safe.

    Args:
        model: Model containing FixedPointQuantizer modules (via
               AdaPTConv2d / AdaPTLinear layers).
        snr_threshold: Minimum SNR in dB for reduction (default: 20.0).
        grad_threshold: Minimum gradient L2 norm for reduction (default: 1e-4).
        check_interval: Check conditions every N epochs (default: 5).
        min_bits: Floor for bit-width reduction (default: 2).
    """

    def __init__(self, model, snr_threshold=20.0, grad_threshold=1e-4,
                 check_interval=5, min_bits=2):
        self.model = model
        self.snr_threshold = snr_threshold
        self.grad_threshold = grad_threshold
        self.check_interval = check_interval
        self.min_bits = min_bits

        # Discover all quantized weight layers
        self.layer_info = {}  # name -> (parent_module, weight_quantizer)
        for name, module in model.named_modules():
            wq = getattr(module, "weight_quantizer", None)
            if isinstance(wq, FixedPointQuantizer) and wq.is_weight:
                self.layer_info[name] = module

        # Gradient norms (updated each backward via hooks)
        self.grad_norms = {}

        # History: bits per layer per epoch (for plotting)
        self.bits_history = []  # list of (epoch, {name: bits})

        # Install gradient hooks
        self._hooks = []
        self._install_grad_hooks()

    # ------------------------------------------------------------------
    # Gradient hooks
    # ------------------------------------------------------------------

    def _install_grad_hooks(self):
        """Register backward hooks to capture per-layer gradient norms."""
        for name, module in self.layer_info.items():
            hook = module.weight.register_hook(self._make_grad_hook(name))
            self._hooks.append(hook)

    def _make_grad_hook(self, name):
        def hook_fn(grad):
            self.grad_norms[name] = grad.detach().norm(2).item()
        return hook_fn

    def remove_hooks(self):
        """Remove gradient hooks (call when training is done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # SNR computation
    # ------------------------------------------------------------------

    def compute_snr(self, layer_name):
        """Compute SNR(l) = 10 * log10(signal_power / noise_power).

        signal_power = mean(W^2)
        noise_power  = mean((W - Q(W))^2)

        Returns:
            float: SNR in dB (returns inf if noise is zero).
        """
        module = self.layer_info[layer_name]
        wq = module.weight_quantizer
        w = module.weight.detach()

        signal_power = w.pow(2).mean().item()
        if signal_power < 1e-12:
            return 0.0

        # Quantize at current bits
        f = wq.find_best_fractional(w)
        w_q = wq.quantize_fixed_point(w, wq.current_bits, f)
        noise_power = (w - w_q).pow(2).mean().item()

        if noise_power < 1e-12:
            return float("inf")

        return 10.0 * math.log10(signal_power / noise_power)

    def compute_grad_norm(self, layer_name):
        """Return the last-recorded gradient L2 norm for this layer.

        Must be called after loss.backward() and before optimizer.step(),
        because gradient hooks fire during backward.
        """
        return self.grad_norms.get(layer_name, 0.0)

    # ------------------------------------------------------------------
    # Precision check & update
    # ------------------------------------------------------------------

    def record_bits(self, epoch):
        """Record current per-layer bits for history tracking."""
        snapshot = {}
        for name, module in self.layer_info.items():
            snapshot[name] = module.weight_quantizer.current_bits
        self.bits_history.append((epoch, snapshot))

    def check_and_update(self, epoch):
        """Check conditions and reduce bits where safe.

        Called every check_interval epochs. For each layer:
          - Compute SNR and gradient norm
          - If both conditions met: reduce by 1 bit
          - Print summary of changes

        Returns:
            dict: {layer_name: (old_bits, new_bits)} for layers that changed.
        """
        changes = {}
        print(f"\n  [AdaPT] Precision check at epoch {epoch}:")
        print(f"  {'Layer':<30s} {'Bits':>5s} {'SNR(dB)':>9s} "
              f"{'GradNorm':>10s} {'Action':>12s}")
        print(f"  {'-'*30} {'-'*5} {'-'*9} {'-'*10} {'-'*12}")

        for name, module in self.layer_info.items():
            wq = module.weight_quantizer
            old_bits = wq.current_bits

            snr = self.compute_snr(name)
            grad_norm = self.compute_grad_norm(name)

            cond_snr = snr >= self.snr_threshold
            cond_grad = grad_norm >= self.grad_threshold

            action = "keep"
            if old_bits <= self.min_bits:
                action = "at minimum"
            elif cond_snr and cond_grad:
                wq.reduce_bits()
                action = f"{old_bits} -> {wq.current_bits}"
                changes[name] = (old_bits, wq.current_bits)

            print(f"  {name:<30s} {old_bits:>5d} {snr:>9.2f} "
                  f"{grad_norm:>10.6f} {action:>12s}")

        if changes:
            print(f"  => Reduced bits for {len(changes)} layer(s)")
        else:
            print(f"  => No changes")

        self.record_bits(epoch)
        return changes

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_current_bits(self):
        """Return {layer_name: current_bits} dict."""
        return {name: module.weight_quantizer.current_bits
                for name, module in self.layer_info.items()}

    def get_avg_bits(self):
        """Return simple average of current bits across all layers."""
        bits = self.get_current_bits()
        if not bits:
            return 0.0
        return sum(bits.values()) / len(bits)

    def print_bits_table(self):
        """Print a table of per-layer bits across all recorded epochs."""
        if not self.bits_history:
            return

        names = list(self.layer_info.keys())
        short = [n.rsplit(".", 1)[-1] if "." in n else n for n in names]

        # Header
        header = f"  {'Epoch':>6s}"
        for s in short:
            header += f" {s:>8s}"
        header += f" {'Avg':>6s}"
        print(header)
        print(f"  {'-'*6}" + f" {'-'*8}" * len(short) + f" {'-'*6}")

        for epoch, snapshot in self.bits_history:
            row = f"  {epoch:>6d}"
            vals = []
            for name in names:
                b = snapshot.get(name, 8)
                row += f" {b:>8d}"
                vals.append(b)
            avg = sum(vals) / len(vals) if vals else 0
            row += f" {avg:>6.2f}"
            print(row)
