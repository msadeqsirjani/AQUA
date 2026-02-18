"""
End-to-end HAQ training script (Wang et al., CVPR 2019).

Usage: python -m scripts.haq.train

Steps:
1. Load pretrained FP32 ResNet-18 on CIFAR-10
2. Convert to quantizable model (QConv2d/QLinear)
3. Run DDPG for 300 episodes to find mixed-precision policy
4. Fine-tune model with learned policy for 30 epochs
5. Compare to uniform INT8 baseline
6. Generate plots: reward curve + per-layer bit-width bar chart
"""

import argparse
import os
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models import get_model
from utils import (
    get_dataloaders, setup_device, evaluate,
    load_fp32_model, save_results,
)
from utils.args import add_common_args, get_result_dir, get_fp32_path
from utils.config import load_config
from utils.console import banner, section, print_config, metric, info, success, file_saved
from core.haq.haq_quantize_utils import QConv2d, QLinear, QModule, calibrate
from core.haq.haq_env import HAQEnvironment, profile_model, compute_layer_energy
from core.haq.haq_ddpg import DDPGAgent


# ---------------------------------------------------------------------------
# Convert standard model to quantizable model
# ---------------------------------------------------------------------------

def convert_to_quantizable(model):
    """Replace all Conv2d -> QConv2d and Linear -> QLinear in a model.

    Preserves pretrained weights. Default bit-widths set to 8.
    """
    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and not isinstance(child, QConv2d):
                qconv = QConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups,
                    bias=child.bias is not None,
                    w_bit=8, a_bit=8,
                )
                qconv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qconv.bias.data.copy_(child.bias.data)
                setattr(module, name, qconv)
            elif isinstance(child, nn.Linear) and not isinstance(child, QLinear):
                qlin = QLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    w_bit=8, a_bit=8,
                )
                qlin.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qlin.bias.data.copy_(child.bias.data)
                setattr(module, name, qlin)
            else:
                _replace(child)

    _replace(model)
    return model


# ---------------------------------------------------------------------------
# Fine-tuning with QAT after RL policy found
# ---------------------------------------------------------------------------

def finetune_with_policy(model, train_loader, val_loader, device,
                         epochs=30, lr=1e-3):
    """Fine-tune quantized model with learned mixed-precision policy."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

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

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        history.append((epoch, train_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            info(f"[Finetune] Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} | "
                 f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return best_acc, history


# ---------------------------------------------------------------------------
# Evaluate uniform INT8 baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_uniform_int8(model, val_loader, device):
    """Evaluate model with uniform 8-bit quantization."""
    model.eval()
    for m in model.modules():
        if isinstance(m, QModule):
            m.w_bit = 8
            m.a_bit = 8
            m.weight_range.data[0] = -1.0

    # Quick calibration
    inputs, _ = next(iter(val_loader))
    for m in model.modules():
        if isinstance(m, QModule):
            m.set_calibrate(True)
    model(inputs.to(device))
    for m in model.modules():
        if isinstance(m, QModule):
            m.set_calibrate(False)

    correct, total = 0, 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reward_curve(rewards, save_path):
    """Plot RL reward over episodes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, label="Episode reward")
    # Smoothed
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(rewards)), smoothed, linewidth=2, label=f"Smoothed (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("HAQ DDPG Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


def plot_bitwidth_policy(policy, layer_names, save_path):
    """Plot per-layer bit-width as bar chart."""
    w_bits = [p[0] for p in policy]
    a_bits = [p[1] for p in policy]

    fig, ax = plt.subplots(figsize=(max(12, len(policy) * 0.6), 5))
    x = np.arange(len(policy))
    width = 0.35
    bars1 = ax.bar(x - width/2, w_bits, width, label="Weight bits", color="#2196F3")
    bars2 = ax.bar(x + width/2, a_bits, width, label="Activation bits", color="#FF9800")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Bit-width")
    ax.set_title("HAQ Learned Mixed-Precision Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    file_saved(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HAQ training")
    add_common_args(parser, default_epochs=30, default_lr=1e-3)
    args = parser.parse_args()
    cfg = load_config(args.config, args)
    banner("HAQ Training", f"{cfg.model} on {cfg.dataset}")

    device = setup_device()
    print_config(cfg)
    result_dir = get_result_dir(cfg.dataset, cfg.model, "haq")
    train_loader, val_loader = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size,
        data_root=cfg.data_root,
    )
    fp32_path = get_fp32_path(cfg.dataset, cfg.model)

    # ===== Step 1: Load pretrained FP32 model =====
    section(f"Load Pretrained FP32 {cfg.model} ({cfg.dataset})", step=1)

    model, fp32_acc = load_fp32_model(
        cfg.model, device, val_loader,
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dataset_name=cfg.dataset,
    )

    # ===== Step 2: Convert to quantizable model =====
    section("Convert to Quantizable Model", step=2)

    q_model = copy.deepcopy(model)
    q_model = convert_to_quantizable(q_model)
    q_model.to(device)

    n_qconv = sum(1 for m in q_model.modules() if isinstance(m, QConv2d))
    n_qlin = sum(1 for m in q_model.modules() if isinstance(m, QLinear))
    metric("QConv2d layers", n_qconv)
    metric("QLinear layers", n_qlin)

    # Evaluate uniform INT8 baseline
    int8_model = copy.deepcopy(q_model)
    int8_acc = evaluate_uniform_int8(int8_model, val_loader, device)  # noqa: F811
    metric("Uniform INT8 accuracy (no finetune)", f"{int8_acc:.2f}%")

    # Compute INT8 energy baseline
    layer_infos = profile_model(q_model, 32, 32)
    int8_energy = sum(compute_layer_energy(li.n_macs, 8, 8) for li in layer_infos)
    metric("INT8 total energy", f"{int8_energy:.0f}")

    # ===== Step 3: DDPG RL search =====
    section("DDPG Mixed-Precision Search (300 episodes)", step=3)

    env = HAQEnvironment(
        model=q_model,
        val_loader=val_loader,
        device=device,
        b_min=2,
        b_max=8,
        target_ratio=0.5,
        input_h=32,
        input_w=32,
    )

    state_dim = 9
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=1,
        device=device,
        lr_actor=1e-4,
        lr_critic=1e-3,
        tau=0.01,
        gamma=1.0,
        buffer_size=2000,
        batch_size=64,
        warmup_episodes=20,
        init_delta=0.5,
        delta_decay=0.99,
    )

    n_episodes = 300
    episode_rewards = []
    best_reward = -float("inf")
    best_policy = []

    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        trajectory = []  # store (s, a, r, s', done) for end-of-episode update

        done = False
        while not done:
            action = agent.select_action(state, episode=ep)
            next_state, reward, done, step_info = env.step(action[0] if isinstance(action, np.ndarray) else action)

            trajectory.append((state.copy(), action, reward, next_state.copy(), done))
            episode_reward += reward
            state = next_state

        # At end of episode, store all transitions with final reward
        # (matching original HAQ: use final reward for all steps in trajectory)
        final_reward = trajectory[-1][2]
        for s, a, r, s2, d in trajectory:
            agent.store_transition(s, a, final_reward, s2, d)

        # Update policy after warmup
        if ep >= agent.warmup_episodes:
            agent.update_policy()

        episode_rewards.append(final_reward)

        if final_reward > best_reward:
            best_reward = final_reward
            best_policy = step_info.get("strategy", [])

        if ep % 30 == 0 or ep == n_episodes - 1:
            energy_ratio = step_info.get("energy_ratio", 0)
            acc = step_info.get("accuracy", 0)
            info(f"Episode {ep:3d}/{n_episodes} | Reward: {final_reward:.4f} | Acc: {acc:.2f}% | "
                 f"Energy ratio: {energy_ratio:.3f} | Best reward: {best_reward:.4f}")

    # Get best policy
    if not best_policy:
        best_policy = env.get_best_policy()

    metric("Best reward", f"{best_reward:.4f}")
    info("Best policy (per-layer w_bit, a_bit):")
    for i, (wb, ab) in enumerate(best_policy):
        li = layer_infos[i]
        layer_type = "Conv2d" if li.is_conv else "Linear"
        info(f"Layer {i:2d} ({layer_type:6s} {li.in_channels:4d}->{li.out_channels:4d}): w={wb}, a={ab}")

    # Compute energy of best policy
    best_energy = sum(
        compute_layer_energy(layer_infos[i].n_macs, wb, ab)
        for i, (wb, ab) in enumerate(best_policy)
    )
    metric("Best policy energy", f"{best_energy:.0f} ({100*best_energy/int8_energy:.1f}% of INT8)")

    # ===== Step 4: Fine-tune with learned policy =====
    section("Fine-tune with Learned Policy (30 epochs)", step=4)

    # Reload pretrained weights and apply best policy
    ft_model = copy.deepcopy(q_model)
    ft_model.load_state_dict(torch.load(fp32_path, map_location=device, weights_only=True), strict=False)
    ft_model = convert_to_quantizable(ft_model)
    ft_model.to(device)

    # Apply bit-width policy
    module_list = list(ft_model.modules())
    ft_layer_infos = profile_model(ft_model, 32, 32)
    for i, (w_bit, a_bit) in enumerate(best_policy):
        layer = module_list[ft_layer_infos[i].module_idx]
        if isinstance(layer, QModule):
            layer.w_bit = w_bit
            layer.a_bit = a_bit
            layer.weight_range.data[0] = -1.0

    # Calibrate
    ft_model = calibrate(ft_model, train_loader, device=device)

    # Fine-tune
    ft_acc, ft_history = finetune_with_policy(ft_model, train_loader, val_loader, device,
                                              epochs=30, lr=1e-3)

    # Also fine-tune uniform INT8 for fair comparison
    info("Fine-tuning uniform INT8 for comparison...")
    int8_ft_model = copy.deepcopy(q_model)
    int8_ft_model.load_state_dict(torch.load(fp32_path, map_location=device, weights_only=True), strict=False)
    int8_ft_model = convert_to_quantizable(int8_ft_model)
    int8_ft_model.to(device)
    for m in int8_ft_model.modules():
        if isinstance(m, QModule):
            m.w_bit = 8
            m.a_bit = 8
            m.weight_range.data[0] = -1.0
    int8_ft_model = calibrate(int8_ft_model, train_loader, device=device)
    int8_ft_acc, _ = finetune_with_policy(int8_ft_model, train_loader, val_loader, device,
                                          epochs=30, lr=1e-3)

    # ===== Step 5: Results =====
    section("Results Summary", step=5)
    metric("FP32 baseline accuracy", f"{fp32_acc:.2f}%")
    metric("Uniform INT8 (finetuned)", f"{int8_ft_acc:.2f}% (energy: 100.0%)")
    metric("HAQ mixed-precision (finetuned)", f"{ft_acc:.2f}% (energy: {100*best_energy/int8_energy:.1f}%)")
    metric("Accuracy drop vs INT8", f"{int8_ft_acc - ft_acc:.2f}%")
    metric("Energy savings vs INT8", f"{100*(1 - best_energy/int8_energy):.1f}%")

    if ft_acc >= int8_ft_acc - 1.0 and best_energy / int8_energy < 0.6:
        success("Status: PASS (within 1% of INT8, <60% energy)")
    else:
        if ft_acc < int8_ft_acc - 1.0:
            info(f"Status: accuracy drop {int8_ft_acc - ft_acc:.2f}% > 1% threshold")
        if best_energy / int8_energy >= 0.6:
            info(f"Status: energy ratio {100*best_energy/int8_energy:.1f}% >= 60% threshold")

    # ===== Step 6: Plots =====
    section("Generating Plots", step=6)

    plot_reward_curve(episode_rewards, os.path.join(result_dir, "haq_reward_curve.png"))

    layer_names = []
    for i, li in enumerate(layer_infos):
        lt = "Conv" if li.is_conv else "FC"
        layer_names.append(f"L{i}:{lt}")
    plot_bitwidth_policy(best_policy, layer_names,
                         os.path.join(result_dir, "haq_bitwidth_policy.png"))

    # Compute average bit-width weighted by MACs
    total_macs = sum(li.n_macs for li in layer_infos)
    avg_bits = sum(
        ((w + a) / 2.0) * li.n_macs
        for (w, a), li in zip(best_policy, layer_infos)
    ) / max(total_macs, 1)

    # Build per-layer bits dict
    per_layer_bits = {}
    for i, (w_bit, a_bit) in enumerate(best_policy):
        lt = "Conv" if layer_infos[i].is_conv else "FC"
        per_layer_bits[f"L{i}:{lt}"] = (w_bit + a_bit) / 2.0

    # Save results
    results = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "fp32_acc": fp32_acc,
        "int8_ft_acc": int8_ft_acc,
        "haq_ft_acc": ft_acc,
        "best_policy": best_policy,
        "best_energy_ratio": best_energy / int8_energy,
        "episode_rewards": [float(r) for r in episode_rewards],
        "summary": {
            "method": "HAQ",
            "fp32_acc": fp32_acc,
            "method_acc": ft_acc,
            "int8_acc": int8_ft_acc,
            "int4_acc": None,
            "avg_bits": round(avg_bits, 2),
            "per_layer_bits": per_layer_bits,
            "training_history": [
                {"epoch": e, "val_acc": va}
                for e, _, _, va in ft_history
            ],
            "category": "mpq",
        },
    }
    save_results(results, os.path.join(result_dir, "haq_results.json"))

    # Save model
    model_path = os.path.join(result_dir, f"{cfg.model}_haq_mixed.pt")
    torch.save(ft_model.state_dict(), model_path)
    file_saved(model_path)

    agent_path = os.path.join(result_dir, "haq_ddpg_agent.pt")
    agent.save(agent_path)
    file_saved(agent_path)


if __name__ == "__main__":
    main()
