"""
Model and energy-model factories.
"""

import torch.nn as nn
from models.simplecnn5 import SimpleCNN5
from models.resnet18 import ResNet18
from core.energy import BaselineEnergyModel

_MODELS = {
    'simplecnn5': SimpleCNN5,
    'resnet18':   ResNet18,
}


def get_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    """Instantiate a model by name."""
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_MODELS.keys())}")

    if name == 'simplecnn5':
        return SimpleCNN5(in_channels=in_channels, num_classes=num_classes)
    elif name == 'resnet18':
        return ResNet18(num_classes=num_classes)


def get_energy_model(model: nn.Module, model_name: str, in_channels: int, bitwidth: float) -> BaselineEnergyModel:
    """Build a BaselineEnergyModel from a model instance."""
    layers = model.get_compute_layers()

    if model_name == 'simplecnn5':
        spatial_sizes = model.get_spatial_sizes(in_channels=in_channels)
    elif model_name == 'resnet18':
        spatial_sizes = model.get_spatial_sizes()
    else:
        raise ValueError(f"No energy model support for '{model_name}'")

    return BaselineEnergyModel(layers, spatial_sizes, bitwidth=bitwidth)
