"""Shared agent utilities.

Provides:
    - MLP factory
    - Discrete replay buffer for off-policy methods
    - Parameter group partitioner for Nested-MARL (maps layers to update-rate groups)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


def make_mlp(in_dim: int, hidden: List[int], out_dim: int, activation: str = "relu") -> nn.Module:
    """Build a plain feed-forward MLP."""
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def partition_params_by_layer(
    module: nn.Sequential, group_bounds: List[int]
) -> List[List[nn.Parameter]]:
    """Partition a sequential MLP's Linear-layer weights into N groups by depth.

    Parameters
    ----------
    module : nn.Sequential
        Built by make_mlp.
    group_bounds : list of int
        Group boundaries as Linear-layer indices. E.g. [1, 2] means:
            group 0 -> Linear #0
            group 1 -> Linear #1
            group 2 -> Linear #2+
        Must be strictly increasing and < n_linear_layers.

    Returns
    -------
    List of lists of nn.Parameter. One list per group.
    """
    linears = [m for m in module if isinstance(m, nn.Linear)]
    n = len(linears)
    groups: List[List[nn.Parameter]] = [[] for _ in range(len(group_bounds) + 1)]
    for i, lin in enumerate(linears):
        # find which bucket i falls into
        g = 0
        for b in group_bounds:
            if i >= b:
                g += 1
        for p in lin.parameters():
            groups[g].append(p)
    return groups


class ReplayBuffer:
    """Simple circular buffer for (obs, act, r, next_obs, done) tuples per agent.

    Stores each agent's experience separately to support parameter-sharing or
    independent agents.
    """

    def __init__(self, capacity: int, obs_dim: int, n_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rews = np.zeros((capacity, n_agents), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, acts, rews, next_obs, done):
        self.obs[self.idx] = np.stack(obs)
        self.next_obs[self.idx] = np.stack(next_obs)
        self.acts[self.idx] = np.asarray(acts)
        self.rews[self.idx] = np.asarray(rews)
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple:
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.acts[idx]),
            torch.from_numpy(self.rews[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
        )


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())
