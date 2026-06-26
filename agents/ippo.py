"""
Independent PPO (IPPO) baseline.

Each agent is an independent PPO learner with a *single* learning rate applied
uniformly to all parameters. This is the standard MARL-for-networks baseline
used in literature (e.g., for network slicing RL benchmarks).

IPPO with a single-timescale update is exactly what we claim Nested-MARL
improves upon under multi-timescale non-stationarity. Having the two share
almost all code (same actor-critic, same PPO update) means any performance
difference is attributable to the nested multi-rate updates and EMA anchor.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import make_mlp


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int] = [64, 64]):
        super().__init__()
        self.actor = make_mlp(obs_dim, hidden, act_dim)
        self.critic = make_mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value


class IPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: List[int] = [64, 64],
        lr: float = 5e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.9,
        clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        optimizer: str = "adam",
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, act_dim, hidden)
        if optimizer == "adam":
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(optimizer)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def act(self, obs):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            a, lp, v = self.net.act(obs_t)
        return int(a.item()), float(lp.item()), float(v.item())

    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        nv = last_value
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nv * nonterm - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterm * gae
            adv[t] = gae
            nv = values[t]
        returns = adv + np.array(values, dtype=np.float32)
        return adv, returns

    def update(self, obs, acts, old_logps, advs, rets, n_epochs=4):
        obs = torch.from_numpy(obs).float()
        acts = torch.from_numpy(acts).long()
        old_logps = torch.from_numpy(old_logps).float()
        advs = torch.from_numpy(advs).float()
        rets = torch.from_numpy(rets).float()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        last = 0.0
        for _ in range(n_epochs):
            logits, values = self.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts)
            ent = dist.entropy().mean()
            ratio = torch.exp(logp - old_logps)
            s1 = ratio * advs
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advs
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(values, rets)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
            last = float(loss.item())
        return {"loss": last}


class IPPO:
    def __init__(self, n_agents, obs_dim, act_dim, hidden=[64, 64], **kwargs):
        self.n_agents = n_agents
        seed = kwargs.pop("seed", 0)
        self.agents = [
            IPPOAgent(obs_dim, act_dim, hidden=hidden, seed=seed + i, **kwargs)
            for i in range(n_agents)
        ]

    def act_all(self, obs_list):
        acts, logps, vals = [], [], []
        for i, obs in enumerate(obs_list):
            a, lp, v = self.agents[i].act(obs)
            acts.append(a); logps.append(lp); vals.append(v)
        return acts, logps, vals

    def update_all(self, rollouts, n_epochs=4):
        logs = []
        for i, ag in enumerate(self.agents):
            adv, ret = ag.compute_gae(
                rollouts[i]["rewards"], rollouts[i]["values"],
                rollouts[i]["dones"], rollouts[i]["last_value"]
            )
            logs.append(ag.update(
                np.stack(rollouts[i]["obs"]),
                np.array(rollouts[i]["actions"]),
                np.array(rollouts[i]["logps"]),
                adv, ret, n_epochs=n_epochs
            ))
        return logs
