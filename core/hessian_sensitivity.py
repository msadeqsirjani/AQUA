"""
Hessian trace sensitivity estimation for Edge-MPQ
(Zhao et al., IEEE Trans. Computers, 2024).

Uses the Hutchinson stochastic trace estimator:
  Tr(H) ≈ (1/n) * sum_i  v_i^T H v_i
where v_i are Rademacher random vectors (+1/-1).

The Hessian-vector product Hv is computed via two back-props
(Pearlmutter's trick) without materializing the full Hessian.
"""

import torch
import torch.nn as nn


def _hessian_vector_product(loss, params, vector):
    """Compute Hessian-vector product Hv via two backprops.

    Given loss L and parameters W:
      grad_L = dL/dW
      Hv = d(grad_L . v) / dW

    Args:
        loss: Scalar loss tensor (must have grad_fn).
        params: List of parameter tensors.
        vector: List of tensors same shape as params (the vector v).

    Returns:
        List of tensors: Hv for each parameter.
    """
    # First backprop: get gradient of loss w.r.t. params
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Dot product: grad_L . v
    grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vector))

    # Second backprop: d(grad_L . v) / dW = Hv
    hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=True)

    return hvp


def compute_hessian_trace(model, loss_fn, data_loader, n_samples=100,
                          device="cuda"):
    """Compute per-layer Hessian trace using Hutchinson estimator.

    For each quantizable layer (Conv2d, Linear), estimates:
      S_l = Tr(H_l) ≈ (1/n) * sum_{i=1}^{n} v_i^T H_l v_i

    Uses 1 batch of data for the loss computation.
    Uses n_samples Rademacher vectors for the trace estimate.

    Args:
        model: Pretrained model (will not be modified).
        loss_fn: Loss function, e.g. nn.CrossEntropyLoss().
        data_loader: DataLoader to get one batch from.
        n_samples: Number of Rademacher vectors (default: 100).
        device: Torch device.

    Returns:
        Dict mapping layer_name -> hessian_trace_estimate (float).
        Higher value = more sensitive = needs more bits.
    """
    model.eval()
    model.to(device)

    # Identify quantizable layers and their parameters
    target_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers[name] = module

    if not target_layers:
        return {}

    # Get one batch
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    sensitivities = {}

    for layer_name, layer in target_layers.items():
        params = [layer.weight]
        trace_sum = 0.0

        for _ in range(n_samples):
            # Zero all gradients
            model.zero_grad()

            # Forward pass (must recompute each time for fresh graph)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Sample Rademacher vector: v_i in {-1, +1}^d
            v = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1
                 for p in params]

            # Hessian-vector product: Hv
            hvp = _hessian_vector_product(loss, params, v)

            # Trace estimate: v^T H v
            vhv = sum((vi * hvi).sum().item() for vi, hvi in zip(v, hvp))
            trace_sum += vhv

        sensitivities[layer_name] = abs(trace_sum / n_samples)

    return sensitivities
