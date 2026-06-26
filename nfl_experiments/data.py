"""
Data loading with IID and non-IID (Dirichlet) partitioning.
Returns per-client DataLoaders and a centralised test DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import config


def _mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _cifar10_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
    ])


def _load_raw(dataset: str):
    """Download and return raw (train, test) torch Datasets."""
    root = config.DATA_ROOT
    if dataset == "MNIST":
        train = datasets.MNIST(root, train=True,  download=True,
                               transform=_mnist_transforms())
        test  = datasets.MNIST(root, train=False, download=True,
                               transform=_mnist_transforms())
    elif dataset == "CIFAR10":
        train = datasets.CIFAR10(root, train=True,  download=True,
                                 transform=_cifar10_transforms(True))
        test  = datasets.CIFAR10(root, train=False, download=True,
                                 transform=_cifar10_transforms(False))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train, test


def _dirichlet_split(targets, num_clients: int, alpha: float, rng):
    """
    Partition sample indices among clients using a Dirichlet distribution.
    alpha → ∞  :  IID
    alpha → 0  :  each client gets one class
    Returns list of index arrays, one per client.
    """
    targets   = np.array(targets)
    classes   = np.unique(targets)
    client_idx = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        # Scale proportions to actual counts
        proportions = (proportions * len(cls_idx)).astype(int)
        # Fix rounding so sum == len(cls_idx)
        diff = len(cls_idx) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        splits = np.split(cls_idx, np.cumsum(proportions)[:-1])
        for cid, s in enumerate(splits):
            client_idx[cid].extend(s.tolist())

    return [np.array(idx) for idx in client_idx]


def get_data_loaders(dataset: str, alpha: float, seed: int = config.SEED):
    """
    Returns:
        client_loaders : list[DataLoader]  — one per client
        test_loader    : DataLoader        — global test set
    """
    rng = np.random.default_rng(seed)
    train_data, test_data = _load_raw(dataset)

    targets = (train_data.targets if hasattr(train_data, "targets")
               else train_data.labels)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    client_indices = _dirichlet_split(
        targets, config.NUM_CLIENTS, alpha, rng)

    client_loaders = []
    for idx in client_indices:
        subset = Subset(train_data, idx)
        loader = DataLoader(subset,
                            batch_size=config.LOCAL_BATCH_SIZE,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False)
        client_loaders.append(loader)

    test_loader = DataLoader(test_data,
                             batch_size=256,
                             shuffle=False,
                             num_workers=0)
    return client_loaders, test_loader


def get_client_weights(client_loaders):
    """Proportional weights p_i = n_i / sum(n_j)."""
    sizes  = np.array([len(cl.dataset) for cl in client_loaders],
                      dtype=float)
    return sizes / sizes.sum()
