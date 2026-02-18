"""
Common argparse helpers shared by all training scripts.

Usage in a training script::

    import argparse
    from utils.args import add_common_args, get_result_dir, get_fp32_path
    from utils.config import load_config

    parser = argparse.ArgumentParser()
    add_common_args(parser)          # adds --config, --dataset, --model, etc.
    parser.add_argument(...)         # algorithm-specific flags
    args = parser.parse_args()
    cfg = load_config(args.config, args)

    result_dir = get_result_dir(cfg.dataset, cfg.model, "haq")
"""

import os


def add_common_args(parser, default_epochs=None, default_lr=None,
                    default_batch_size=None):
    """Add the standard flags shared by every algorithm script.

    Flags added:
        --config       Path to YAML config file.
        --dataset      Dataset name.
        --model        Model architecture name.
        --batch-size   Training batch size.
        --epochs       Number of fine-tuning epochs.
        --lr           Base learning rate.
        --bits         Default quantization bit-width.

    All flags default to ``None`` so that the YAML config layer takes
    precedence.  If explicit defaults are needed (e.g. for pretrain
    scripts), pass them via the ``default_*`` parameters.
    """
    from models import list_models
    from utils.data import list_datasets

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/cifar10_resnet18_4bit.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list_datasets(),
        help="Dataset name (default: from config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list_models(),
        help="Model architecture (default: from config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=default_batch_size,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--epochs", type=int, default=default_epochs,
        help="Fine-tuning epochs (default: from config)",
    )
    parser.add_argument(
        "--lr", type=float, default=default_lr,
        help="Base learning rate (default: from config)",
    )
    parser.add_argument(
        "--bits", type=int, default=None,
        help="Quantization bit-width (default: from config)",
    )
    return parser


def get_result_dir(dataset_name, model_name, algorithm_name):
    """Return ``results/{dataset}/{model}/{algo}/`` and create it.

    Example::

        get_result_dir("cifar100", "resnet18", "haq")
        # -> "results/cifar100/resnet18/haq"
    """
    path = os.path.join("results", dataset_name, model_name, algorithm_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_fp32_path(dataset_name, model_name):
    """Return the expected path for a pretrained FP32 checkpoint.

    Checks the new dataset-aware path first, then the model-only path,
    then the legacy flat path.

    Returns:
        Path string (e.g. ``results/cifar10/resnet18/resnet18_fp32.pt``).
    """
    new_path = os.path.join(
        "results", dataset_name, model_name, f"{model_name}_fp32.pt",
    )
    model_path = os.path.join("results", model_name, f"{model_name}_fp32.pt")
    legacy_path = os.path.join("results", f"{model_name}_fp32.pt")
    for p in (new_path, model_path, legacy_path):
        if os.path.exists(p):
            return p
    return new_path  # preferred location (may not exist yet)
