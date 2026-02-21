"""
YAML-based configuration system for AQUA.

Loads a YAML config file and allows CLI arguments to override any value.
Config keys map directly to CLI flag names (with dashes replaced by
underscores).

Usage::

    from utils.config import load_config

    cfg = load_config(args.config, args)
    # cfg.dataset   -> "cifar10"
    # cfg.model     -> "resnet18"
    # cfg.bits      -> 4
    # cfg.num_classes -> 10  (auto-set from dataset)
    # cfg.img_size  -> 32   (auto-set from dataset)
"""

import os
import yaml


# ---------------------------------------------------------------------------
# Defaults (used if no config file is provided)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "dataset": "cifar10",
    "data_root": "./data",
    "batch_size": 128,
    "num_workers": 2,
    "model": "resnet18",
    "fp32_epochs": 200,
    "epochs": 30,
    "lr": 1e-4,
    "fp32_lr": 0.1,
    "fp32_optimizer": "sgd",
    "fp32_weight_decay": 5e-4,
    "fp32_warmup_epochs": 0,
    "fp32_label_smoothing": 0.0,
    "bits": 4,
}


class Config:
    """Simple attribute-access wrapper around a dict."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({items})"


def _flatten_yaml(data, prefix=""):
    """Flatten nested YAML dict into a flat dict.

    Example::

        {"dataset": {"name": "cifar10", "batch_size": 128}}
        -> {"dataset": "cifar10", "batch_size": 128}

    Special handling: the ``name`` key under a section is promoted to
    the section name itself (e.g. ``dataset.name`` -> ``dataset``).
    """
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                if subkey == "name":
                    flat[key] = subval
                else:
                    flat[subkey] = subval
        else:
            flat[key] = value
    return flat


def load_config(config_path=None, cli_args=None):
    """Load a YAML config file, merge with CLI overrides, return a Config.

    Priority (highest wins):  CLI args  >  YAML file  >  built-in defaults.

    Args:
        config_path: path to YAML file (``None`` uses built-in defaults).
        cli_args: argparse Namespace (``None`` means no CLI overrides).

    Returns:
        Config object with attribute access.
    """
    from utils.data import get_dataset_info

    cfg = dict(_DEFAULTS)

    # Layer 1: YAML file
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        flat = _flatten_yaml(raw)
        cfg.update(flat)

    # Layer 2: CLI overrides (only non-None values)
    if cli_args is not None:
        cli_dict = vars(cli_args)
        for key, value in cli_dict.items():
            if value is not None and key != "config":
                cfg[key] = value

    # Auto-derive dataset metadata
    info = get_dataset_info(cfg["dataset"])
    cfg["num_classes"] = info["num_classes"]
    cfg["img_size"] = info["img_size"]

    return Config(cfg)
