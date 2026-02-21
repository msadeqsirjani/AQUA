"""
Format-aware energy model for AQUA — multi-dtype categorical version.

Energy per layer is the probability-weighted sum of per-dtype energy costs::

    E_layer_i = sum_k  p_k * [MACs_i * E_mac(dtype_k) * (b_k/8)^alpha
                              + DataSize_i * E_access * b_k]

Hardware energy constants from Yang et al. 2017 (45 nm CMOS, pJ).

The entropy penalty is now categorical over K dtypes, encouraging each
layer to commit to a single dtype.
"""

import torch
import torch.nn as nn

from .quantizers import DTYPE_CATALOG

# Published hardware energy constants (pJ, 45 nm CMOS, Yang et al. 2017)
E_INT_MAC = 3.7    # 8-bit integer multiply-accumulate
E_FP_MAC = 5.2     # 8-bit floating-point multiply-accumulate
E_ACCESS = 0.9     # SRAM access per bit


def _dtype_mac_energy(bits, energy_type):
    """Energy per MAC for a single dtype."""
    q_norm = bits / 8.0
    if energy_type == "int":
        return E_INT_MAC * q_norm ** 2
    else:
        return E_FP_MAC * q_norm ** 1.5


def compute_layer_info(model):
    """Extract MAC and data-size proxies for each AQUA layer.

    Uses ``weight.numel()`` as a proportional proxy for both MACs and
    data size — exact spatial dimensions are not needed since the energy
    model only requires *relative* layer costs.

    Returns:
        List of dicts with keys ``name``, ``macs``, ``data_size``.
    """
    from .aqua_model import AQUAConv2d, AQUALinear

    infos = []
    for name, m in model.named_modules():
        if isinstance(m, (AQUAConv2d, AQUALinear)):
            numel = m.weight.numel()
            infos.append({
                "name": name,
                "macs": numel,
                "data_size": numel,
            })
    return infos


def compute_energy(layer_infos, probs_list, bits_list, energy_types_list):
    """Differentiable total energy for all AQUA layers.

    Args:
        layer_infos: list from :func:`compute_layer_info`.
        probs_list: list of K-dim probability tensors per layer.
        bits_list: list of K-dim bit-width tensors per layer.
        energy_types_list: list of K-length string lists per layer.

    Returns:
        Scalar tensor (total energy, not yet normalised).
    """
    device = probs_list[0].device
    total = torch.tensor(0.0, device=device)

    for info, probs, bits, etypes in zip(layer_infos, probs_list,
                                          bits_list, energy_types_list):
        layer_energy = torch.tensor(0.0, device=device)
        for k in range(len(probs)):
            b = bits[k].item()
            e_mac = _dtype_mac_energy(b, etypes[k])
            e_comp = info["macs"] * e_mac
            e_data = info["data_size"] * E_ACCESS * b
            layer_energy = layer_energy + probs[k] * (e_comp + e_data)
        total = total + layer_energy

    return total


def compute_ref_energy(layer_infos, allowed_dtypes=None, q_max=None):
    """Reference energy for normalisation.

    Uses the most expensive dtype in the allowed set (highest bits, FP type).
    Falls back to fp32 if available, otherwise the highest-bits dtype.
    """
    if allowed_dtypes is None:
        allowed_dtypes = list(DTYPE_CATALOG.keys())

    max_bits = 0
    max_etype = "int"
    for name in allowed_dtypes:
        entry = DTYPE_CATALOG[name]
        if entry["bits"] > max_bits:
            max_bits = entry["bits"]
            max_etype = entry["energy_type"]

    ref = 0.0
    for info in layer_infos:
        e_mac = _dtype_mac_energy(max_bits, max_etype)
        ref += info["macs"] * e_mac
        ref += info["data_size"] * E_ACCESS * max_bits
    return max(ref, 1e-8)


def dtype_entropy(probs_list):
    """Categorical entropy penalty over K dtypes.

    Encourages each layer to commit to a single dtype (low entropy).

    ``H = -sum_k p_k log(p_k)``  averaged over layers.

    Args:
        probs_list: list of K-dim probability tensors per layer.

    Returns:
        Scalar tensor (mean per-layer entropy).
    """
    eps = 1e-6
    device = probs_list[0].device
    total = torch.tensor(0.0, device=device)
    for probs in probs_list:
        p = probs.clamp(eps, 1.0)
        total = total - (p * torch.log(p)).sum()
    return total / max(len(probs_list), 1)


# Keep old name for backward compat
format_entropy = dtype_entropy
