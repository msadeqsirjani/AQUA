"""
EfficientQAT Phase 1: Block-wise Training of All Parameters (Block-AP).
Adapted from Chen et al., "EfficientQAT" (ACL 2025 Main).
Original repo: https://github.com/OpenGVLab/EfficientQAT

Adapted from LLM transformer blocks to ResNet-18 residual blocks.

Core idea: train each block independently using a block-local MSE loss
between quantized and full-precision block outputs. This is efficient
because each block is small and only its parameters receive gradients.

Block definition for ResNet-18 on CIFAR-10:
  Block 0: conv1 + bn1  (stem)
  Block 1: layer1        (2 residual BasicBlocks)
  Block 2: layer2        (2 residual BasicBlocks)
  Block 3: layer3        (2 residual BasicBlocks)
  Block 4: layer4        (2 residual BasicBlocks)
  Block 5: fc            (classifier head)
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..jacob_fake_quant import JacobFakeQuantize


# ---------------------------------------------------------------------------
# Learnable-scale quantizer (following EfficientQAT's UniformAffineQuantizer)
# ---------------------------------------------------------------------------

class LearnableQuantizer(nn.Module):
    """Uniform affine quantizer with learnable scale and zero_point.

    Adapted from EfficientQAT's quantizer.py:
      - Asymmetric: qmin=0, qmax=2^b - 1
      - Scale and zero_point are nn.Parameters (receive gradients)
      - STE: round_ste(x) = x + (round(x) - x).detach()
      - Init from min/max of the weight tensor

    Per-tensor quantization (EfficientQAT uses per-group for LLMs,
    but per-tensor is standard for Conv2d in CNNs).
    """

    def __init__(self, n_bits=4, weight=None):
        super().__init__()
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1

        # Init scale and zero_point from weight statistics
        if weight is not None:
            w = weight.detach().float()
            xmin = w.min().clamp(max=0.0)
            xmax = w.max().clamp(min=0.0)
            scale = ((xmax - xmin) / (self.qmax - self.qmin)).clamp(min=1e-4)
            zero_point = (-xmin / scale).round().clamp(-1e4, 1e4)
        else:
            scale = torch.tensor(1.0)
            zero_point = torch.tensor(0.0)

        self.scale = nn.Parameter(scale)
        self.zero_point = nn.Parameter(zero_point)

    def forward(self, x):
        # STE round and clamp
        x_int = self._round_ste(x / self.scale) + self._round_ste(self.zero_point)
        x_int = self._clamp_ste(x_int, self.qmin, self.qmax)
        x_dq = (x_int - self._round_ste(self.zero_point)) * self.scale
        return x_dq

    @staticmethod
    def _round_ste(x):
        return x + (x.round() - x).detach()

    @staticmethod
    def _clamp_ste(x, qmin, qmax):
        return x + (x.clamp(qmin, qmax) - x).detach()

    def extra_repr(self):
        return (f"bits={self.n_bits}, scale={self.scale.item():.6f}, "
                f"zp={self.zero_point.item():.2f}")


# ---------------------------------------------------------------------------
# Quantized layer wrappers (with use_weight_quant toggle like EfficientQAT)
# ---------------------------------------------------------------------------

class EQATConv2d(nn.Module):
    """Conv2d with learnable quantizer and FP/quantized toggle.

    Following EfficientQAT's QuantLinear pattern:
      - weight is an nn.Parameter (trainable in Block-AP)
      - bias is a buffer (frozen)
      - use_weight_quant toggles between FP and quantized forward
    """

    def __init__(self, org_module, n_bits=4):
        super().__init__()
        self.weight = nn.Parameter(org_module.weight.data.clone())
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias.data.clone())
        else:
            self.bias = None

        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

        self.weight_quantizer = LearnableQuantizer(n_bits, weight=org_module.weight)
        self.use_weight_quant = False

    def forward(self, x):
        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class EQATLinear(nn.Module):
    """Linear with learnable quantizer and FP/quantized toggle."""

    def __init__(self, org_module, n_bits=4):
        super().__init__()
        self.weight = nn.Parameter(org_module.weight.data.clone())
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias.data.clone())
        else:
            self.bias = None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.weight_quantizer = LearnableQuantizer(n_bits, weight=org_module.weight)
        self.use_weight_quant = False

    def forward(self, x):
        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


# ---------------------------------------------------------------------------
# Model preparation: replace layers with EQAT versions
# ---------------------------------------------------------------------------

def _replace_with_eqat(module, n_bits=4):
    """Recursively replace Conv2d and Linear with EQAT versions."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, EQATConv2d(child, n_bits))
        elif isinstance(child, nn.Linear):
            setattr(module, name, EQATLinear(child, n_bits))
        else:
            _replace_with_eqat(child, n_bits)


def set_quant_state(model, enabled):
    """Toggle use_weight_quant on all EQAT layers."""
    for m in model.modules():
        if isinstance(m, (EQATConv2d, EQATLinear)):
            m.use_weight_quant = enabled


# ---------------------------------------------------------------------------
# Block splitting for ResNet-18
# ---------------------------------------------------------------------------

def split_resnet_into_blocks(model):
    """Split a ResNet-18 into sequential blocks for Block-AP training.

    Returns:
        List of (block_name, block_forward_fn, block_modules) tuples.
        Each block_forward_fn takes a tensor and returns the block output.
    """
    blocks = []

    # Block 0: stem (conv1 + bn1 + relu)
    class StemBlock(nn.Module):
        def __init__(self, conv1, bn1):
            super().__init__()
            self.conv1 = conv1
            self.bn1 = bn1

        def forward(self, x):
            return F.relu(self.bn1(self.conv1(x)))

    blocks.append(("stem", StemBlock(model.conv1, model.bn1)))

    # Block 1-4: residual layers
    for i, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"]):
        blocks.append((layer_name, getattr(model, layer_name)))

    # Block 5: classifier (avgpool + flatten + fc)
    class HeadBlock(nn.Module):
        def __init__(self, avgpool, fc):
            super().__init__()
            self.avgpool = avgpool
            self.fc = fc

        def forward(self, x):
            out = self.avgpool(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    blocks.append(("head", HeadBlock(model.avgpool, model.fc)))

    return blocks


# ---------------------------------------------------------------------------
# Block-AP Trainer
# ---------------------------------------------------------------------------

class BlockAPTrainer:
    """Phase 1: Block-wise training of all parameters.

    For each block sequentially:
      1. Compute FP reference output (quantization disabled)
      2. Enable quantization, train block with MSE loss
      3. Freeze block and move to next

    Following EfficientQAT:
      - Separate LR for quant params (scale/zp) and weight params
      - AdamW optimizer with cosine LR schedule per step
      - MSE loss between FP and quantized block output
    """

    def __init__(self, quant_lr=1e-4, weight_lr=1e-5, min_lr_factor=20):
        self.quant_lr = quant_lr
        self.weight_lr = weight_lr
        self.min_lr_factor = min_lr_factor

    def train_block(self, block_name, block, train_loader, device,
                    epochs=5, head_block=None, labels_available=False):
        """Train a single block with block-local MSE loss.

        Args:
            block_name: Name for logging.
            block: The block module (already has EQAT layers).
            train_loader: DataLoader yielding (block_input, fp_target) pairs,
                          or (block_input, fp_target, labels) if labels_available.
            device: Torch device.
            epochs: Number of epochs for this block.
            head_block: If provided, adds CE loss through head (for non-final blocks).
            labels_available: Whether labels are in the dataloader.

        Returns:
            List of (epoch, mse_loss) tuples.
        """
        block.to(device)
        set_quant_state(block, True)  # enable quantization

        # Build optimizer with separate LR groups (following EfficientQAT)
        quant_params = []
        weight_params = []
        for name, param in block.named_parameters():
            if "scale" in name or "zero_point" in name:
                quant_params.append(param)
            else:
                weight_params.append(param)

        param_groups = []
        if quant_params:
            param_groups.append({
                "params": quant_params,
                "lr": self.quant_lr,
            })
        if weight_params:
            param_groups.append({
                "params": weight_params,
                "lr": self.weight_lr,
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1),
            eta_min=self.quant_lr / self.min_lr_factor,
        )

        mse_criterion = nn.MSELoss()
        ce_criterion = nn.CrossEntropyLoss()
        history = []

        for epoch in range(1, epochs + 1):
            block.train()
            total_mse = 0.0
            n_batches = 0

            for batch in train_loader:
                if labels_available:
                    block_input, fp_target, labels = batch
                    labels = labels.to(device)
                else:
                    block_input, fp_target = batch
                    labels = None

                block_input = block_input.to(device)
                fp_target = fp_target.to(device)

                optimizer.zero_grad()

                # Quantized block output
                q_output = block(block_input)

                # MSE loss (block-local reconstruction)
                loss = mse_criterion(q_output, fp_target)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_mse += loss.item()
                n_batches += 1

            avg_mse = total_mse / max(n_batches, 1)
            history.append((epoch, avg_mse))

            if epoch == 1 or epoch == epochs:
                print(f"      [{block_name}] Epoch {epoch}/{epochs} | "
                      f"MSE: {avg_mse:.6f}")

        return history

    def train_all_blocks(self, model, train_loader, val_loader, device,
                         bits=4, epochs_per_block=5):
        """Run Block-AP on all blocks sequentially.

        Args:
            model: FP32 pretrained ResNet-18.
            train_loader: CIFAR-10 training DataLoader.
            val_loader: CIFAR-10 validation DataLoader.
            device: Torch device.
            bits: Quantization bit-width.
            epochs_per_block: Epochs to train each block.

        Returns:
            (quantized_model, all_histories) where all_histories is a dict
            mapping block_name -> list of (epoch, loss).
        """
        # Deep copy and replace layers with EQAT versions
        qmodel = copy.deepcopy(model)
        _replace_with_eqat(qmodel, n_bits=bits)
        qmodel.to(device)

        # Split into blocks
        blocks = split_resnet_into_blocks(qmodel)

        # Collect calibration data: full-precision forward on training set
        print("    Collecting calibration inputs...")
        all_inputs = []
        all_labels = []
        qmodel.eval()
        set_quant_state(qmodel, False)  # FP mode
        with torch.no_grad():
            for inputs, targets in train_loader:
                all_inputs.append(inputs)
                all_labels.append(targets)
        all_inputs = torch.cat(all_inputs, dim=0)  # (N, 3, 32, 32)
        all_labels = torch.cat(all_labels, dim=0)

        all_histories = {}

        # Process each block sequentially
        current_input = all_inputs
        for block_idx, (block_name, block) in enumerate(blocks):
            print(f"\n    --- Block {block_idx}: {block_name} ---")

            # Step 1: Compute FP reference output (quant disabled)
            set_quant_state(block, False)
            block.to(device).eval()
            fp_outputs = []
            batch_size = 256
            with torch.no_grad():
                for i in range(0, current_input.size(0), batch_size):
                    batch = current_input[i:i + batch_size].to(device)
                    fp_out = block(batch)
                    fp_outputs.append(fp_out.cpu())
            fp_output = torch.cat(fp_outputs, dim=0)

            # Step 2: Build block-local dataloader
            block_dataset = torch.utils.data.TensorDataset(
                current_input, fp_output,
            )
            block_loader = torch.utils.data.DataLoader(
                block_dataset, batch_size=128, shuffle=True,
                num_workers=0, pin_memory=True,
            )

            # Step 3: Train this block with MSE loss
            history = self.train_block(
                block_name, block, block_loader, device,
                epochs=epochs_per_block,
            )
            all_histories[block_name] = history

            # Step 4: After training, update inputs for next block
            # Run quantized block on all inputs to get next block's inputs
            set_quant_state(block, True)
            block.eval()
            next_inputs = []
            with torch.no_grad():
                for i in range(0, current_input.size(0), batch_size):
                    batch = current_input[i:i + batch_size].to(device)
                    out = block(batch)
                    next_inputs.append(out.cpu())
            current_input = torch.cat(next_inputs, dim=0)

        return qmodel, all_histories
