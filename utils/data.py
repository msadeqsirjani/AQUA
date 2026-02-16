"""
Dataset factory — returns dataloaders, channel count, and class count.
"""

import ssl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context

# Per-dataset normalization stats
_STATS = {
    'mnist':    ((0.1307,), (0.3081,)),
    'cifar10':  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

_META = {
    'mnist':    {'cls': datasets.MNIST,    'in_channels': 1, 'num_classes': 10,  'size': 28, 'workers': 0},
    'cifar10':  {'cls': datasets.CIFAR10,  'in_channels': 3, 'num_classes': 10,  'size': 32, 'workers': 2},
    'cifar100': {'cls': datasets.CIFAR100, 'in_channels': 3, 'num_classes': 100, 'size': 32, 'workers': 2},
}


def get_dataloaders(dataset: str, batch_size: int, data_dir: str = './data'):
    """
    Returns (train_loader, test_loader, in_channels, num_classes).

    Applies standard augmentation (RandomCrop + HorizontalFlip) for CIFAR datasets.
    """
    if dataset not in _META:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(_META.keys())}")

    meta = _META[dataset]
    mean, std = _STATS[dataset]
    size = meta['size']

    # Build transforms
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = meta['cls'](data_dir, train=True,  download=True, transform=transform_train)
    test_ds  = meta['cls'](data_dir, train=False, download=True, transform=transform_test)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=meta['workers'])
    test_ld  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=meta['workers'])

    return train_ld, test_ld, meta['in_channels'], meta['num_classes']
