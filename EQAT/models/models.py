"""
EQAT model architectures (Section 4.1 of the paper).

SimpleCNN5    : 5-layer CNN (3 conv + 2 fc) for MNIST / CIFAR-10.
ResNet18EQAT  : ResNet-18 for CIFAR-100 with per-layer EQAT quantization.

Every quantizable Conv and Linear layer has:
  - learnable bit-width  q̃  (shared for weights + activations, Eq. 13)
  - learnable PACT clip  α   (for activation quantization)

Full-precision forward: call model.forward(x, quantize=False).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from EQAT.core.quantizer import EQATConvBlock, EQATLinear, fake_quant_weight, _FakeQuantAct


# ── 5-Layer CNN ───────────────────────────────────────────────────────────────

class SimpleCNN5(nn.Module):
    """
    5-layer CNN: Conv→Conv→Conv→FC→FC.
    MaxPool after each conv layer.

    MNIST  (1-ch, 28×28): spatial after 3 pools → 3×3, flatten=1152
    CIFAR-10 (3-ch, 32×32): spatial after 3 pools → 4×4, flatten=2048
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 q_min: float = 2.0, q_max: float = 8.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # 3 convolutional blocks (each with BN + ReLU + PACT inside)
        self.block1 = EQATConvBlock(in_channels, 32, q_min=q_min, q_max=q_max)
        self.block2 = EQATConvBlock(32, 64,       q_min=q_min, q_max=q_max)
        self.block3 = EQATConvBlock(64, 128,      q_min=q_min, q_max=q_max)

        # Flatten size: 128 × (3×3 for MNIST, 4×4 for CIFAR-10)
        flat = 128 * (3 * 3 if in_channels == 1 else 4 * 4)

        # 2 fully-connected blocks
        self.fc1 = EQATLinear(flat, 256,        activate=True,  q_min=q_min, q_max=q_max)
        self.fc2 = EQATLinear(256, num_classes, activate=False, q_min=q_min, q_max=q_max)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        x = self.pool(self.block1(x, quantize))
        x = self.pool(self.block2(x, quantize))
        x = self.pool(self.block3(x, quantize))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x, quantize)
        x = self.dropout(x)
        x = self.fc2(x, quantize)
        return x

    # ── helpers for energy model and optimisers ───────────────────────────

    def eqat_blocks(self) -> list:
        """All EQAT blocks (conv + fc), in forward order."""
        return [self.block1, self.block2, self.block3, self.fc1, self.fc2]

    def bitwidth_params(self) -> list:
        """Learnable q̃ parameters (one per block)."""
        return [b.q_tilde for b in self.eqat_blocks()]

    def pact_params(self) -> list:
        """Learnable PACT α parameters."""
        return [b.alpha for b in self.eqat_blocks()]

    def get_bitwidths(self) -> list:
        return [round(b.get_bitwidth(), 2) for b in self.eqat_blocks()]


# ── ResNet-18 with EQAT ───────────────────────────────────────────────────────
#
# Each residual block contains two "EQAT units":
#   Unit A: Conv3×3 → BN → ReLU → PACT    (q̃_a, α_a)
#   Unit B: Conv3×3 → BN → (skip add) → ReLU → PACT    (q̃_b, α_b)
#
# Downsample (1×1 conv + BN) gets its own q̃ for weight quantization.
# The ResNet stem (conv7×7 → BN → ReLU) and final FC are also EQAT-quantized.

class _EQATResConv(nn.Module):
    """
    Single EQAT unit for ResNet: weight fake-quantization + optional PACT output.
    Stores its own q̃ and α for weight and activation quantization.
    Reads self._quantize flag (set externally by ResNet18EQAT.forward).
    """

    def __init__(self, conv: nn.Conv2d, bn: nn.Module,
                 q_min: float = 2.0, q_max: float = 8.0,
                 has_relu: bool = True):
        super().__init__()
        self.conv      = conv
        self.bn        = bn
        self.has_relu  = has_relu
        self.q_min     = q_min
        self.q_max     = q_max
        self.q_tilde   = nn.Parameter(torch.tensor(1.5))   # σ(1.5)≈0.82 → q≈6.9 bits
        self.alpha     = nn.Parameter(torch.tensor(6.0))   # PACT clip
        self._quantize = True                               # toggled by ResNet18EQAT

    def get_q(self) -> torch.Tensor:
        return self.q_min + (self.q_max - self.q_min) * torch.sigmoid(self.q_tilde)

    def get_bitwidth(self) -> float:
        return self.get_q().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q()
        if self._quantize:
            scale = self.conv.weight.abs().max() + 1e-8
            w_q   = fake_quant_weight(self.conv.weight / scale, q) * scale
        else:
            w_q = self.conv.weight

        out = F.conv2d(x, w_q, self.conv.bias,
                       self.conv.stride, self.conv.padding,
                       self.conv.dilation, self.conv.groups)
        out = self.bn(out)

        if self.has_relu:
            out = F.relu(out)
            if self._quantize:
                out = _FakeQuantAct.apply(out, q, self.alpha.abs())

        return out


class _EQATResBlock(nn.Module):
    """
    ResNet-18 BasicBlock with EQAT.
    Two EQAT conv units (A and B) plus optional EQAT downsample.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int,
                 stride: int = 1,
                 q_min: float = 2.0, q_max: float = 8.0):
        super().__init__()

        # Unit A: conv1 + bn1 + ReLU + PACT
        self.unit_a = _EQATResConv(
            nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            q_min=q_min, q_max=q_max, has_relu=True,
        )

        # Unit B: conv2 + bn2 — ReLU + PACT applied AFTER the skip add
        self.unit_b = _EQATResConv(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            q_min=q_min, q_max=q_max, has_relu=False,  # skip add comes first
        )

        # Separate PACT + q̃ for the post-skip activation (Unit B output)
        self.q_b_out   = nn.Parameter(torch.tensor(1.5))
        self.alpha_b_out = nn.Parameter(torch.tensor(6.0))
        self.q_min     = q_min
        self.q_max     = q_max

        # Downsample (when stride > 1 or channel mismatch)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = _EQATResConv(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                q_min=q_min, q_max=q_max, has_relu=False,
            )

    def _get_q_b_out(self) -> torch.Tensor:
        return self.q_min + (self.q_max - self.q_min) * torch.sigmoid(self.q_b_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.unit_a(x)
        out = self.unit_b(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)

        # PACT on block output (activations entering the next block)
        if self.unit_a._quantize:
            q_out = self._get_q_b_out()
            out   = _FakeQuantAct.apply(out, q_out, self.alpha_b_out.abs())

        return out

    def eqat_units(self) -> list:
        """All _EQATResConv units in this block (including downsample)."""
        units = [self.unit_a, self.unit_b]
        if self.downsample is not None:
            units.append(self.downsample)
        return units


class _EQATResLayer(nn.Module):
    """Sequence of _EQATResBlock with quantize-flag threading."""

    def __init__(self, blocks: list):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class ResNet18EQAT(nn.Module):
    """
    ResNet-18 with per-layer EQAT weight + activation quantization.
    Used for CIFAR-100 experiments (Section 4.1).

    Architecture matches torchvision ResNet-18 channel dimensions:
      stem  → layer1 (64)  → layer2 (128, stride=2)
            → layer3 (256, stride=2) → layer4 (512, stride=2) → fc

    Quantize flag is broadcast to all EQAT units before the forward pass.
    """

    def __init__(self, num_classes: int = 100,
                 q_min: float = 2.0, q_max: float = 8.0):
        super().__init__()
        self.q_min = q_min
        self.q_max = q_max

        # Stem: Conv7×7(3→64, stride=2) + BN + ReLU + PACT
        self.stem = _EQATResConv(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            q_min=q_min, q_max=q_max, has_relu=True,
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers
        self.layer1 = _EQATResLayer(self._make_blocks(64,  64,  2, stride=1, q_min=q_min, q_max=q_max))
        self.layer2 = _EQATResLayer(self._make_blocks(64,  128, 2, stride=2, q_min=q_min, q_max=q_max))
        self.layer3 = _EQATResLayer(self._make_blocks(128, 256, 2, stride=2, q_min=q_min, q_max=q_max))
        self.layer4 = _EQATResLayer(self._make_blocks(256, 512, 2, stride=2, q_min=q_min, q_max=q_max))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier (weight-only quant; no activation — outputs logits)
        self.fc = EQATLinear(512, num_classes, activate=False, q_min=q_min, q_max=q_max)

        self._init_weights()

    @staticmethod
    def _make_blocks(in_planes, planes, num_blocks, stride, q_min, q_max):
        blocks = [_EQATResBlock(in_planes, planes, stride=stride, q_min=q_min, q_max=q_max)]
        for _ in range(1, num_blocks):
            blocks.append(_EQATResBlock(planes, planes, stride=1, q_min=q_min, q_max=q_max))
        return blocks

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _set_quantize(self, flag: bool):
        """Broadcast the quantize flag to every _EQATResConv in the model."""
        for m in self.modules():
            if isinstance(m, _EQATResConv):
                m._quantize = flag

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        self._set_quantize(quantize)

        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, quantize)
        return x

    # ── helpers ──────────────────────────────────────────────────────────

    def eqat_blocks(self) -> list:
        """All EQAT units (conv + fc), in forward order, for energy model."""
        units: list = [self.stem]
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in layer.blocks:
                units.extend(blk.eqat_units())
        units.append(self.fc)
        return units

    def bitwidth_params(self) -> list:
        params = [b.q_tilde for b in self.eqat_blocks()]
        # Also include the post-skip q̃ parameters in each residual block
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in layer.blocks:
                params.append(blk.q_b_out)
        return params

    def pact_params(self) -> list:
        params = [b.alpha for b in self.eqat_blocks()]
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in layer.blocks:
                params.append(blk.alpha_b_out)
        return params

    def get_bitwidths(self) -> list:
        return [round(b.get_bitwidth(), 2) for b in self.eqat_blocks()]
