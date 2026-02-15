# EQAT: Energy-Efficient Quantization-Aware Training

Implementation of **"Energy-Efficient Quantization-Aware Training with Dynamic Bit-Width Optimization"**,
Ali Karkehabadi and Avesta Sasan, UC Davis — GLSVLSI '25.

## Overview

EQAT jointly optimizes model weights and per-layer bit-widths during training to balance accuracy and energy efficiency. Three key innovations:

1. **3-term loss** combining cross-entropy, KL divergence (quantized vs. full-precision), and an explicit energy penalty
2. **Differentiable bit-width** — each layer's bit-width `q_i` is a learnable continuous parameter with a sigmoid reparameterization, enabling gradient-based optimization
3. **Analytical energy gradient** derived from a hardware-aware energy model (computation + data movement)

## Repository Structure

```
GLSVLSI2026/
└── AUQA/
    ├── EQAT/
    │   ├── loss.py          # 3-term loss (Eq. 9) + progressive β schedule
    │   ├── quantizer.py     # Fake-quant STE for weights (Eq. 16-18) + PACT activations
    │   ├── energy.py        # Differentiable energy model (Eq. 10-12, 19-20)
    │   ├── models.py        # SimpleCNN5 (MNIST/CIFAR-10), ResNet18EQAT (CIFAR-100)
    │   ├── trainer.py       # Training loop implementing Algorithm 1
    │   ├── train_mnist.py   # MNIST experiment
    │   ├── train_cifar10.py # CIFAR-10 experiment
    │   └── train_cifar100.py# CIFAR-100 experiment
    └── docs/
        └── EQAT.pdf         # Paper
```

## Method

**Loss function (Eq. 9):**

```
L_total = L_CE(y_quant, y_true)
        + α · KL(y_quant ‖ y_full)
        + β_epoch · E_total({q})
```

where `β_epoch = β · min(1, epoch / warmup_epochs)` ramps the energy penalty progressively.

**Bit-width reparameterization (Eq. 13):**

```
q_i = q_min + (q_max - q_min) · σ(q̃_i)
```

`q̃_i` is a learnable scalar; the sigmoid keeps `q_i ∈ [q_min, q_max]` during optimization.

**Energy model (Eq. 11-12):**

```
E_comp = MACs × E_MAC × q²
E_data = DataSize × E_access × q
```

Energy scales quadratically with bit-width for compute and linearly for memory — matching real hardware behavior.

## Results

| Dataset   | Model       | Accuracy       | Energy (norm.) | Avg bits |
|-----------|-------------|----------------|----------------|----------|
| MNIST     | SimpleCNN5  | ~99.45%        | ~0.60          | ~6.0     |
| CIFAR-10  | SimpleCNN5  | ~86.23%        | ~0.65          | ~5.85    |
| CIFAR-100 | ResNet-18   | ~69.98% (+4.26% vs baseline) | ~0.97 | ~6.75 |

Energy is normalized against an 8-bit uniform quantization baseline.

## Usage

Run all experiments from the `AUQA/` directory:

```bash
# MNIST
python -m EQAT.train_mnist

# CIFAR-10
python -m EQAT.train_cifar10

# CIFAR-100
python -m EQAT.train_cifar100
```

Best model checkpoints are saved as `eqat_<dataset>_best.pt` in the working directory.

## Hyperparameters

| Parameter        | MNIST  | CIFAR-10 | CIFAR-100 |
|------------------|--------|----------|-----------|
| Optimizer        | Adadelta (lr=1.0) | Adam (lr=1e-3) | Adam (lr=1e-3) |
| α (KL weight)    | 0.95   | 0.80     | 0.70      |
| β (energy weight)| 0.001→0.01 | 0.001→0.01 | 0.001→0.01 |
| Warmup epochs    | 10     | 10       | 10        |
| Bit-width freeze | 20     | 20       | 20        |
| q_min / q_max    | 2 / 8  | 2 / 8    | 2 / 8     |
| Epochs           | 100    | 100      | 100       |
| LR schedule      | —      | CosineAnnealing | CosineAnnealing |

## Requirements

```
torch
torchvision
```
