"""
HAQ RL Environment (Wang et al., CVPR 2019).
Modernized from lib/env/quantize_env.py.

Implements:
  - 9-dimensional state vector (Section 3.2)
  - Analytical energy model replacing Pixel 1 hardware LUT
  - Reward = accuracy if energy <= budget, else large penalty (Eq. 1)
  - Proxy reward using validation on 1000 random samples (no finetune)
"""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from .haq_quantize_utils import QConv2d, QLinear, QModule


# ---------------------------------------------------------------------------
# Analytical energy model (replaces real hardware LUT)
# ---------------------------------------------------------------------------

def compute_layer_macs(layer, input_h, input_w):
    """Compute number of MAC operations for a layer.

    Conv2d: c_in * c_out * k^2 * h_out * w_out / groups
    Linear: in_features * out_features
    """
    if isinstance(layer, (nn.Conv2d, QConv2d)):
        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding
        d_h, d_w = layer.dilation
        h_out = (input_h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (input_w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
        macs = (
            layer.in_channels
            * layer.out_channels
            * k_h
            * k_w
            * h_out
            * w_out
            / layer.groups
        )
        return int(macs), h_out, w_out
    elif isinstance(layer, (nn.Linear, QLinear)):
        macs = layer.in_features * layer.out_features
        return int(macs), 1, 1
    return 0, input_h, input_w


def compute_layer_energy(macs, w_bits, a_bits):
    """Energy = w_bits * a_bits * num_MACs (analytical model)."""
    return w_bits * a_bits * macs


# ---------------------------------------------------------------------------
# Model profiling — extract quantizable layers + geometry
# ---------------------------------------------------------------------------

class LayerInfo:
    """Stores metadata about a single quantizable layer."""

    def __init__(self, module_idx, layer, is_conv, in_channels, out_channels,
                 kernel_size, stride, input_h, input_w, n_params, n_macs):
        self.module_idx = module_idx
        self.layer = layer
        self.is_conv = is_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_h = input_h
        self.input_w = input_w
        self.n_params = n_params
        self.n_macs = n_macs


def profile_model(model, input_h=32, input_w=32):
    """Profile model to extract quantizable layer info.

    Runs a dummy forward pass with hooks to capture input spatial dims.
    Returns list of LayerInfo for each quantizable layer.
    """
    layer_infos = []
    hooks = []
    module_list = list(model.modules())

    # Attach hooks to capture input shapes
    input_shapes = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                x = inp[0]
                if x.dim() == 4:
                    input_shapes[idx] = (x.shape[2], x.shape[3])
                elif x.dim() == 2:
                    input_shapes[idx] = (x.shape[1], 1)
        return hook_fn

    for i, m in enumerate(module_list):
        if isinstance(m, (nn.Conv2d, QConv2d, nn.Linear, QLinear)):
            hooks.append(m.register_forward_hook(make_hook(i)))

    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, input_h, input_w, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    # Build LayerInfo list
    for i, m in enumerate(module_list):
        if isinstance(m, (nn.Conv2d, QConv2d)):
            h_in, w_in = input_shapes.get(i, (input_h, input_w))
            macs, _, _ = compute_layer_macs(m, h_in, w_in)
            layer_infos.append(LayerInfo(
                module_idx=i,
                layer=m,
                is_conv=True,
                in_channels=m.in_channels,
                out_channels=m.out_channels,
                kernel_size=m.kernel_size[0],
                stride=m.stride[0],
                input_h=h_in,
                input_w=w_in,
                n_params=m.weight.numel(),
                n_macs=macs,
            ))
        elif isinstance(m, (nn.Linear, QLinear)):
            h_in, w_in = input_shapes.get(i, (m.in_features, 1))
            macs = m.in_features * m.out_features
            layer_infos.append(LayerInfo(
                module_idx=i,
                layer=m,
                is_conv=False,
                in_channels=m.in_features,
                out_channels=m.out_features,
                kernel_size=1,
                stride=1,
                input_h=h_in,
                input_w=w_in,
                n_params=m.weight.numel(),
                n_macs=macs,
            ))

    return layer_infos


# ---------------------------------------------------------------------------
# HAQ RL Environment
# ---------------------------------------------------------------------------

class HAQEnvironment:
    """RL environment for hardware-aware mixed-precision quantization.

    State: 9-dimensional vector (Section 3.2 of paper)
    Action: continuous in [0, 1], mapped to bits in [b_min, b_max]
    Reward: accuracy if energy <= budget, else -inf (Eq. 1)

    Args:
        model: Pretrained FP32 model (will be deepcopied)
        val_loader: Validation DataLoader (uses first 1000 samples as proxy)
        device: torch device
        b_min: Minimum bit-width (default: 2)
        b_max: Maximum bit-width (default: 8)
        target_ratio: Target energy as fraction of full INT8 energy (default: 0.5)
        input_h, input_w: Input spatial dimensions (default: 32x32 for CIFAR)
    """

    def __init__(self, model, val_loader, device, b_min=2, b_max=8,
                 target_ratio=0.5, input_h=32, input_w=32):
        self.device = device
        self.b_min = b_min
        self.b_max = b_max
        self.target_ratio = target_ratio

        # Store pretrained weights
        self.pretrained_state = deepcopy(model.state_dict())
        self.model = model.to(device)
        self.val_loader = val_loader

        # Proxy validation: take first 1000 samples
        self._build_proxy_set()

        # Profile the model
        self.layer_infos = profile_model(self.model, input_h, input_w)
        self.n_layers = len(self.layer_infos)

        # Compute reference INT8 energy (all layers at 8-bit)
        self.int8_energy = sum(
            compute_layer_energy(info.n_macs, 8, 8) for info in self.layer_infos
        )
        self.target_energy = self.target_ratio * self.int8_energy

        # Compute normalization constants for state vector
        self._compute_norm_constants()

        # Episode state
        self.cur_ind = 0
        self.strategy = []  # list of (w_bit, a_bit) per layer
        self.last_action = self.b_max
        self.best_reward = -math.inf
        self.best_policy = []

        # Get baseline accuracy
        self.model.load_state_dict(self.pretrained_state)
        self.org_acc = self._proxy_validate()
        print(f"  [HAQ Env] Baseline proxy accuracy: {self.org_acc:.2f}%")
        print(f"  [HAQ Env] INT8 energy: {self.int8_energy:.0f}")
        print(f"  [HAQ Env] Target energy (50%): {self.target_energy:.0f}")
        print(f"  [HAQ Env] Quantizable layers: {self.n_layers}")

    def _build_proxy_set(self):
        """Extract first 1000 validation samples for fast proxy evaluation."""
        images_list, labels_list = [], []
        count = 0
        for imgs, labs in self.val_loader:
            images_list.append(imgs)
            labels_list.append(labs)
            count += imgs.size(0)
            if count >= 1000:
                break
        self.proxy_images = torch.cat(images_list, dim=0)[:1000].to(self.device)
        self.proxy_labels = torch.cat(labels_list, dim=0)[:1000].to(self.device)

    def _compute_norm_constants(self):
        """Compute max values for state normalization."""
        self.k_max = max(info.kernel_size for info in self.layer_infos)
        self.c_in_max = max(info.in_channels for info in self.layer_infos)
        self.c_out_max = max(info.out_channels for info in self.layer_infos)
        self.h_max = max(info.input_h for info in self.layer_infos)
        self.params_max = max(info.n_params for info in self.layer_infos)

    def _get_state(self):
        """Build 9-dimensional normalized state vector for current layer.

        s_t = [t/T, is_conv, k/k_max, c_in/c_in_max, c_out/c_out_max,
               h/h_max, params/params_max, budget_remaining/budget_total,
               a_{t-1}]
        """
        info = self.layer_infos[self.cur_ind]

        # Compute remaining energy budget
        used_energy = 0.0
        for i, (wb, ab) in enumerate(self.strategy):
            used_energy += compute_layer_energy(
                self.layer_infos[i].n_macs, wb, ab
            )
        budget_remaining = max(0.0, self.target_energy - used_energy)

        # Normalize previous action to [0, 1]
        prev_action = (self.last_action - self.b_min) / (self.b_max - self.b_min)

        state = np.array([
            self.cur_ind / max(self.n_layers - 1, 1),     # layer index
            1.0 if info.is_conv else 0.0,                  # layer type
            info.kernel_size / max(self.k_max, 1),         # kernel size
            info.in_channels / max(self.c_in_max, 1),      # input channels
            info.out_channels / max(self.c_out_max, 1),    # output channels
            info.input_h / max(self.h_max, 1),             # feature map height
            info.n_params / max(self.params_max, 1),       # parameter count
            budget_remaining / max(self.target_energy, 1), # remaining budget
            prev_action,                                    # previous action
        ], dtype=np.float32)

        return state

    def _action_to_bits(self, action):
        """Map continuous action [0, 1] to discrete bit-width [b_min, b_max]."""
        action = float(np.clip(action, 0.0, 1.0))
        bits = round(action * (self.b_max - self.b_min) + self.b_min)
        bits = max(self.b_min, min(self.b_max, bits))
        return bits

    def _apply_strategy(self):
        """Apply the current mixed-precision strategy to the model."""
        module_list = list(self.model.modules())
        for i, (w_bit, a_bit) in enumerate(self.strategy):
            layer = module_list[self.layer_infos[i].module_idx]
            if isinstance(layer, QModule):
                layer.w_bit = w_bit
                layer.a_bit = a_bit
                layer.weight_range.data[0] = -1.0  # force recalibration

    def _compute_strategy_energy(self):
        """Compute total energy of current strategy."""
        total = 0.0
        for i, (w_bit, a_bit) in enumerate(self.strategy):
            total += compute_layer_energy(
                self.layer_infos[i].n_macs, w_bit, a_bit
            )
        return total

    @torch.no_grad()
    def _proxy_validate(self):
        """Fast proxy validation on 1000 samples."""
        self.model.eval()
        outputs = self.model(self.proxy_images)
        _, predicted = outputs.max(1)
        correct = predicted.eq(self.proxy_labels).sum().item()
        return 100.0 * correct / self.proxy_labels.size(0)

    def step(self, action):
        """Take one step: assign bits to current layer.

        Args:
            action: float in [0, 1] from DDPG actor

        Returns:
            (next_state, reward, done, info)
        """
        bits = self._action_to_bits(action)
        self.strategy.append((bits, bits))  # same bits for weight and activation
        self.last_action = bits

        is_final = self.cur_ind == self.n_layers - 1

        if is_final:
            # Apply strategy to model and evaluate
            self.model.load_state_dict(self.pretrained_state)
            self._apply_strategy()

            # Calibrate the model
            self.model.eval()
            with torch.no_grad():
                # Run a small batch for calibration
                for module in self.model.modules():
                    if isinstance(module, QModule):
                        module.set_calibrate(True)
                self.model(self.proxy_images[:128])
                for module in self.model.modules():
                    if isinstance(module, QModule):
                        module.set_calibrate(False)

            acc = self._proxy_validate()
            energy = self._compute_strategy_energy()

            # Reward (Eq. 1): accuracy if under budget, else penalty
            if energy <= self.target_energy:
                reward = (acc - self.org_acc) * 0.1
            else:
                reward = -1.0  # penalty for exceeding budget

            info = {
                "accuracy": acc,
                "energy": energy,
                "energy_ratio": energy / self.int8_energy,
                "strategy": list(self.strategy),
            }

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_policy = list(self.strategy)

            obs = self._get_state()  # terminal state
            return obs, reward, True, info
        else:
            # Intermediate step — no reward yet
            self.cur_ind += 1
            obs = self._get_state()
            return obs, 0.0, False, {"energy": 0.0}

    def reset(self):
        """Reset environment for new episode."""
        self.model.load_state_dict(self.pretrained_state)
        self.cur_ind = 0
        self.strategy = []
        self.last_action = self.b_max
        return self._get_state()

    def get_best_policy(self):
        """Return the best policy found so far."""
        return self.best_policy
