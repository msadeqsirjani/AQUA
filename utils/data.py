"""
Dataset loaders for AQUA.

Supports CIFAR-10, CIFAR-100, and Tiny-ImageNet with standard augmentation.
All loaders are dispatched through ``get_dataloaders()`` which picks the
correct dataset and augmentation based on the dataset name.

The legacy ``get_cifar10_loaders()`` is kept for backward compatibility.
"""

import os
import shutil
import zipfile
import glob

import torch
import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "cifar10": {
        "num_classes": 10,
        "img_size": 32,
        "channels": 3,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "num_classes": 100,
        "img_size": 32,
        "channels": 3,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "img_size": 64,
        "channels": 3,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
    },
}


def get_dataset_info(dataset_name):
    """Return metadata dict for a dataset (num_classes, img_size, channels).

    Raises:
        ValueError: if dataset_name is not in the registry.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list_datasets()}"
        )
    return DATASET_REGISTRY[dataset_name]


def list_datasets():
    """Return list of available dataset names."""
    return list(DATASET_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def get_dataloaders(dataset_name="cifar10", batch_size=128, num_workers=2,
                    data_root="./data"):
    """Get train and test DataLoaders for any supported dataset.

    Args:
        dataset_name: one of ``cifar10``, ``cifar100``, ``tiny_imagenet``.
        batch_size: batch size for both loaders.
        num_workers: number of data-loading workers.
        data_root: root directory for downloading / caching datasets.

    Returns:
        (train_loader, test_loader) tuple.
    """
    if dataset_name == "cifar10":
        return _get_cifar_loaders(
            torchvision.datasets.CIFAR10,
            batch_size, num_workers, data_root,
            DATASET_REGISTRY["cifar10"],
        )
    elif dataset_name == "cifar100":
        return _get_cifar_loaders(
            torchvision.datasets.CIFAR100,
            batch_size, num_workers, data_root,
            DATASET_REGISTRY["cifar100"],
        )
    elif dataset_name == "tiny_imagenet":
        return _get_tiny_imagenet_loaders(
            batch_size, num_workers, data_root,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list_datasets()}"
        )


# ---------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# ---------------------------------------------------------------------------

def _get_cifar_loaders(dataset_cls, batch_size, num_workers, data_root, info):
    """Shared loader for CIFAR-10 and CIFAR-100."""
    img_size = info["img_size"]
    mean, std = info["mean"], info["std"]

    train_transform = T.Compose([
        T.RandomCrop(img_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = dataset_cls(
        root=data_root, train=True, download=True, transform=train_transform,
    )
    test_set = dataset_cls(
        root=data_root, train=False, download=True, transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Tiny-ImageNet-200
# ---------------------------------------------------------------------------

_TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_tiny_imagenet(data_root):
    """Download and extract Tiny-ImageNet-200 if not already present.

    After extraction the validation set is reorganised into per-class
    subdirectories so that ``torchvision.datasets.ImageFolder`` works.
    """
    dest = os.path.join(data_root, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(dest, "train")):
        return dest

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"  Downloading Tiny-ImageNet from {_TINY_URL} ...")
        torch.hub.download_url_to_file(_TINY_URL, zip_path)

    if not os.path.isdir(dest):
        print("  Extracting Tiny-ImageNet ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)

    # Reorganise val/ into class sub-directories
    val_dir = os.path.join(dest, "val")
    val_annotations = os.path.join(val_dir, "val_annotations.txt")
    if os.path.exists(val_annotations):
        print("  Reorganising val/ into per-class directories ...")
        with open(val_annotations) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, cls = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, cls, "images")
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(val_dir, "images", fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
        # Clean up flat images dir and annotations file
        flat_dir = os.path.join(val_dir, "images")
        if os.path.isdir(flat_dir) and not os.listdir(flat_dir):
            os.rmdir(flat_dir)
        if os.path.exists(val_annotations):
            os.remove(val_annotations)

    return dest


def _get_tiny_imagenet_loaders(batch_size, num_workers, data_root):
    """Get Tiny-ImageNet-200 train/val loaders (64x64)."""
    info = DATASET_REGISTRY["tiny_imagenet"]
    mean, std = info["mean"], info["std"]
    img_size = info["img_size"]

    root = _download_tiny_imagenet(data_root)

    train_transform = T.Compose([
        T.RandomCrop(img_size, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.ImageFolder(
        os.path.join(root, "train"), transform=train_transform,
    )
    val_set = torchvision.datasets.ImageFolder(
        os.path.join(root, "val"), transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Legacy API (backward compatible)
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size=128, num_workers=2, data_root="./data"):
    """Get CIFAR-10 train and test dataloaders with standard augmentation.

    .. deprecated:: Use ``get_dataloaders("cifar10", ...)`` instead.
    """
    return get_dataloaders("cifar10", batch_size, num_workers, data_root)
