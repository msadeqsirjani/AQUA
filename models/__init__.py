from .resnet import ResNet18, ResNet34
from .resnet18 import ResNet18 as _ResNet18_compat  # noqa: F811 keep old module loadable
from .vgg import VGG11, VGG16
from .mobilenetv2 import MobileNetV2
from .vit import ViTTiny

MODEL_REGISTRY = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "vgg11": VGG11,
    "vgg16": VGG16,
    "mobilenetv2": MobileNetV2,
    "vit_tiny": ViTTiny,
}


def get_model(name, num_classes=10, img_size=32):
    """Create a model by name from the registry.

    Args:
        name: key in MODEL_REGISTRY.
        num_classes: number of output classes.
        img_size: input image resolution (only used by ViT variants).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list_models()}"
        )
    if name == "vit_tiny":
        return ViTTiny(num_classes=num_classes, img_size=img_size)
    return MODEL_REGISTRY[name](num_classes=num_classes)


def list_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())
