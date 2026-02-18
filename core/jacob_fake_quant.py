"""
Fake quantization module implementing Jacob et al. 2018
"Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference" (CVPR 2018).

Implements Equations 1-5 from the paper with STE gradients.
"""

import torch
import torch.nn as nn


class JacobFakeQuantize(nn.Module):
    """Fake quantization with learnable scale/zero-point via EMA observers.

    Weight quantization: INT8 symmetric [-128, 127], zero_point=0
    Activation quantization: UINT8 asymmetric [0, 255]
    """

    def __init__(self, num_bits=8, mode="symmetric", is_weight=False, ema_momentum=0.99):
        super().__init__()
        self.num_bits = num_bits
        self.mode = mode
        self.is_weight = is_weight
        self.ema_momentum = ema_momentum

        if mode == "symmetric":
            self.q_min = -(2 ** (num_bits - 1))  # -128
            self.q_max = 2 ** (num_bits - 1) - 1  # 127
        else:
            self.q_min = 0
            self.q_max = 2 ** num_bits - 1  # 255

        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("running_min", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("running_max", torch.tensor(float("-inf"), dtype=torch.float32))
        self.register_buffer("observer_enabled", torch.tensor(1, dtype=torch.int32))

    def compute_scale_zp(self, r_min, r_max):
        """Compute scale S and zero-point Z from real min/max range.

        S = (r_max - r_min) / (q_max - q_min)          [Eq. 1]
        Z = round(q_min - r_min / S), clamped to [q_min, q_max]  [Eq. 2]
        """
        r_min = min(r_min, 0.0)
        r_max = max(r_max, 0.0)

        if r_max == r_min:
            return torch.tensor(1.0), torch.tensor(0, dtype=torch.int32)

        scale = (r_max - r_min) / (self.q_max - self.q_min)
        scale = max(scale, 1e-8)

        if self.mode == "symmetric":
            zero_point = torch.tensor(0, dtype=torch.int32)
        else:
            zero_point = self.q_min - round(r_min / scale)
            zero_point = int(max(self.q_min, min(self.q_max, zero_point)))
            zero_point = torch.tensor(zero_point, dtype=torch.int32)

        return torch.tensor(scale, dtype=torch.float32), zero_point

    def forward(self, x):
        if self.observer_enabled.item():
            if self.is_weight:
                # Weights: use actual min/max of current tensor (no EMA)
                r_min = x.detach().min().item()
                r_max = x.detach().max().item()
            else:
                # Activations: update EMA running min/max
                batch_min = x.detach().min().item()
                batch_max = x.detach().max().item()
                if self.running_min.item() == float("inf"):
                    self.running_min.fill_(batch_min)
                    self.running_max.fill_(batch_max)
                else:
                    self.running_min.mul_(self.ema_momentum).add_(
                        batch_min * (1 - self.ema_momentum)
                    )
                    self.running_max.mul_(self.ema_momentum).add_(
                        batch_max * (1 - self.ema_momentum)
                    )
                r_min = self.running_min.item()
                r_max = self.running_max.item()

            scale, zero_point = self.compute_scale_zp(r_min, r_max)
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)

        # Fake quantize: x_q = clamp(round(x/S) + Z, q_min, q_max)
        #                x_fq = S * (x_q - Z)
        inv_scale = 1.0 / self.scale
        x_q = torch.clamp(
            torch.round(x * inv_scale) + self.zero_point.float(),
            self.q_min,
            self.q_max,
        )
        x_fq = self.scale * (x_q - self.zero_point.float())

        # STE: gradient passes through as identity in valid range
        return x + (x_fq - x).detach()

    def disable_observer(self):
        """Freeze scale and zero_point — stop updating observers."""
        self.observer_enabled.fill_(0)

    def extra_repr(self):
        return (
            f"bits={self.num_bits}, mode={self.mode}, is_weight={self.is_weight}, "
            f"scale={self.scale.item():.6f}, zp={self.zero_point.item()}"
        )
