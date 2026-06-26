"""
Neural network models for MNIST and CIFAR-10.
Each model exposes .layer_groups() so the NFL trainer knows which
parameters belong to (personalization, task, foundation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── MNIST — LeNet-style CNN ──────────────────────────────────────────────────
class MNISTNet(nn.Module):
    """
    3-block CNN for MNIST.
      Foundation      : conv1, conv2   (generic feature extractors)
      Task-specific   : fc1            (mid-level representation)
      Personalization : fc2            (output head, never transmitted)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Foundation layers — slow synchronisation
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        # Task layer — medium synchronisation
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        # Personalisation layer — local only
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def layer_groups(self):
        """Return dict mapping group name → list of parameter tensors."""
        return {
            "foundation":      list(self.conv1.parameters()) +
                               list(self.conv2.parameters()),
            "task":            list(self.fc1.parameters()),
            "personalization": list(self.fc2.parameters()),
        }

    def layer_group_names(self):
        """Return dict mapping group name → list of named parameters."""
        return {
            "foundation":      ["conv1.weight","conv1.bias",
                                 "conv2.weight","conv2.bias"],
            "task":            ["fc1.weight","fc1.bias"],
            "personalization": ["fc2.weight","fc2.bias"],
        }


# ─── CIFAR-10 — Wider CNN ─────────────────────────────────────────────────────
class CIFAR10Net(nn.Module):
    """
    4-block CNN for CIFAR-10.
      Foundation      : conv1, conv2, conv3  (generic visual features)
      Task-specific   : fc1                  (domain representation)
      Personalization : fc2                  (classifier head, local only)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Foundation
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.bn1   = nn.BatchNorm2d(32)
        self.bn2   = nn.BatchNorm2d(64)
        self.bn3   = nn.BatchNorm2d(128)
        # Task
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.drop  = nn.Dropout(0.5)
        # Personalisation
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 16→8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 8→4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def layer_groups(self):
        return {
            "foundation":      list(self.conv1.parameters()) +
                               list(self.conv2.parameters()) +
                               list(self.conv3.parameters()) +
                               list(self.bn1.parameters())  +
                               list(self.bn2.parameters())  +
                               list(self.bn3.parameters()),
            "task":            list(self.fc1.parameters()),
            "personalization": list(self.fc2.parameters()),
        }

    def layer_group_names(self):
        return {
            "foundation":      ["conv1.weight","conv1.bias",
                                 "conv2.weight","conv2.bias",
                                 "conv3.weight","conv3.bias",
                                 "bn1.weight","bn1.bias",
                                 "bn2.weight","bn2.bias",
                                 "bn3.weight","bn3.bias"],
            "task":            ["fc1.weight","fc1.bias"],
            "personalization": ["fc2.weight","fc2.bias"],
        }


def get_model(dataset: str):
    if dataset == "MNIST":
        return MNISTNet()
    elif dataset == "CIFAR10":
        return CIFAR10Net()
    raise ValueError(f"Unknown dataset: {dataset}")
