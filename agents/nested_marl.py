"""
Nested-MARL: Multi-Agent Reinforcement Learning with Multi-Timescale Updates.

This is the paper's main contribution. Each agent is an independent actor-critic
whose parameters are partitioned into three groups (L0: input representation,
L1: mid-level features, L2: decision head) updated at different rates
    alpha_0 << alpha_1 << alpha_2
matched to the environment's fast/medium/slow timescales.

An exponential moving average (EMA) across all parameters acts as a continuum
memory system in the sense of Behrouz et al. (2025) -- slow, stable baseline
that anchors the faster-adapting components.

The core update rule after each PPO-style policy gradient step g:
    theta_l_new = theta_l - alpha_l * g_l
    theta_l_new <- beta * theta_l_new + (1 - beta) * theta_l_EMA   (stabilisation)
    theta_l_EMA <- lambda * theta_l_EMA + (1 - lambda) * theta_l_new

This is the multi-agent extension of Nested Learning's principle that different
components of a learning system should update at frequencies matched to their
semantic role. Applied to MARL, it enables agents to preserve global coordination
structure (slow L0) while adapting quickly to local conditions (fast L2).
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import make_mlp, partition_params_by_layer


# ------------------------------------------------------------------
class NestedActorCritic(nn.Module):
    """Actor-critic with layer-wise parameter partitioning for nested updates."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int] = [64, 64]):
        super().__init__()
        self.actor = make_mlp(obs_dim, hidden, act_dim)
        self.critic = make_mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


# ------------------------------------------------------------------
class NestedAgent:
    """Single agent with nested multi-timescale updates."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: List[int] = [64, 64],
        alpha_l0: float = 1e-4,    # slow: input layer (anchors representation)
        alpha_l1: float = 5e-4,    # medium: hidden layer (matches IPPO baseline)
        alpha_l2: float = 2e-3,    # fast: output head (decision boundary)
        ema_lambda: float = 0.995,
        stabilise_beta: float = 0.95,
        gamma: float = 0.95,
        gae_lambda: float = 0.9,
        clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.net = NestedActorCritic(obs_dim, act_dim, hidden)
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)

        # Partition actor and critic each into 3 groups based on MLP depth.
        # hidden = [64, 64] => Linear layers at indices 0, 1, 2.
        # group_bounds = [1, 2]  => group 0 = lin0, group 1 = lin1, group 2 = lin2
        actor_groups = partition_params_by_layer(self.net.actor, group_bounds=[1, 2])
        critic_groups = partition_params_by_layer(self.net.critic, group_bounds=[1, 2])

        # Combine actor and critic groups at the same level (share timescale).
        self.groups: List[List[nn.Parameter]] = [
            actor_groups[i] + critic_groups[i] for i in range(3)
        ]
        self.alphas = [alpha_l0, alpha_l1, alpha_l2]
        self.ema_lambda = ema_lambda
        self.stabilise_beta = stabilise_beta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Per-group optimisers. We use Adam within each group: this is consistent
        # with Behrouz et al. (2025) who show momentum itself is a nested
        # optimisation (thus Adam is "nested" in spirit). Timescale separation
        # comes from the per-group learning rates, not from optimiser choice.
        self.optimisers = [
            torch.optim.Adam(g, lr=a) for g, a in zip(self.groups, self.alphas)
        ]

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            a, logp, v = self.net.act(obs_t)
        return int(a.item()), float(logp.item()), float(v.item())

    # ------------------------------------------------------------------
    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterm - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterm * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    def update(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        old_logp_batch: np.ndarray,
        adv_batch: np.ndarray,
        ret_batch: np.ndarray,
        n_epochs: int = 4,
    ) -> dict:
        obs_t = torch.from_numpy(obs_batch).float()
        act_t = torch.from_numpy(act_batch).long()
        old_logp_t = torch.from_numpy(old_logp_batch).float()
        adv_t = torch.from_numpy(adv_batch).float()
        ret_t = torch.from_numpy(ret_batch).float()

        # Normalise advantages for stability
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        last_loss = 0.0
        last_entropy = 0.0
        for _ in range(n_epochs):
            logits, values = self.net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_t)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Zero grads across all groups, compute gradient once, step each group
            for opt in self.optimisers:
                opt.zero_grad()
            loss.backward()
            # Clip combined gradient (conservative) before per-group step
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.groups for p in g], max_norm=0.5
            )
            for opt in self.optimisers:
                opt.step()
            last_loss = float(loss.item())
            last_entropy = float(entropy.item())

        # EMA stabilisation: blend current params toward EMA (inner-loop memory anchor)
        with torch.no_grad():
            # First: pull EMA toward current (update continuum memory)
            for p, pe in zip(self.net.parameters(), self.ema_net.parameters()):
                pe.data.mul_(self.ema_lambda).add_(p.data, alpha=1.0 - self.ema_lambda)
            # Second: stabilise current params *toward* EMA (this is what prevents forgetting)
            # Applied only to slowest group (L0) because that's the one we want anchored.
            if self.stabilise_beta < 1.0:
                l0_params = self.groups[0]
                # Build a map from id(param) to its EMA counterpart
                ema_params_list = list(self.ema_net.parameters())
                all_params_list = list(self.net.parameters())
                id_to_ema = {id(p): pe for p, pe in zip(all_params_list, ema_params_list)}
                for p in l0_params:
                    pe = id_to_ema[id(p)]
                    p.data.mul_(self.stabilise_beta).add_(
                        pe.data, alpha=1.0 - self.stabilise_beta
                    )

        return {"loss": last_loss, "entropy": last_entropy}


# ------------------------------------------------------------------
class NestedMARL:
    """Independent-learner MARL wrapper: one NestedAgent per base station.

    Training is done on-policy: collect trajectories, compute advantages,
    update each agent independently with its nested optimiser.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        hidden: List[int] = [64, 64],
        **kwargs,
    ):
        self.n_agents = n_agents
        # Allow seeds to be staggered across agents for diversity
        seed = kwargs.pop("seed", 0)
        self.agents = [
            NestedAgent(obs_dim, act_dim, hidden=hidden, seed=seed + i, **kwargs)
            for i in range(n_agents)
        ]

    def act_all(self, obs_list: List[np.ndarray]):
        acts, logps, vals = [], [], []
        for i, obs in enumerate(obs_list):
            a, lp, v = self.agents[i].act(obs)
            acts.append(a); logps.append(lp); vals.append(v)
        return acts, logps, vals

    def update_all(self, rollouts: List[dict], n_epochs: int = 4):
        """Each agent updates on its own slice of the rollout."""
        logs = []
        for i, agent in enumerate(self.agents):
            adv, ret = agent.compute_gae(
                rollouts[i]["rewards"],
                rollouts[i]["values"],
                rollouts[i]["dones"],
                rollouts[i]["last_value"],
            )
            log = agent.update(
                np.stack(rollouts[i]["obs"]),
                np.array(rollouts[i]["actions"]),
                np.array(rollouts[i]["logps"]),
                adv,
                ret,
                n_epochs=n_epochs,
            )
            logs.append(log)
        return logs
