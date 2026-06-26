"""
EWC-IPPO: Independent PPO with online Elastic Weight Consolidation.

This is the continual-learning baseline. The paper frames Nested-MARL in
continual-learning terms (catastrophic forgetting, continuum memory), so a
reviewer reasonably expects a comparison against a standard continual-learning
mitigation. EWC (Kirkpatrick et al., 2017) is the canonical choice and is
already cited in the manuscript.

Online EWC: because the MTNS-MG has no explicit task boundaries, we consolidate
periodically (every `ewc_consolidate_every` episodes), estimating a diagonal
Fisher information from the most recent batch and anchoring parameters with a
quadratic penalty lambda * sum F (theta - theta*)^2 added to the PPO loss.

Shares the actor-critic, PPO update, and hyperparameters with IPPO so the only
difference is the EWC penalty. Drop-in compatible with train.py.
"""
from __future__ import annotations

from typing import List
import numpy as np
import torch
import torch.nn.functional as F

from .ippo import IPPOAgent


class EWCIPPOAgent(IPPOAgent):
    def __init__(self, *args, ewc_lambda: float = 50.0,
                 ewc_consolidate_every: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.ewc_consolidate_every = ewc_consolidate_every
        self.anchor = None      # list of param tensors
        self.fisher = None      # list of diagonal Fisher tensors
        self._update_count = 0

    def _ewc_penalty(self):
        if self.anchor is None:
            return torch.tensor(0.0)
        pen = 0.0
        for p, a, f in zip(self.net.parameters(), self.anchor, self.fisher):
            pen = pen + (f * (p - a) ** 2).sum()
        return self.ewc_lambda * pen

    def update(self, obs, acts, old_logps, advs, rets, n_epochs=4):
        obs_t = torch.from_numpy(obs).float()
        acts_t = torch.from_numpy(acts).long()
        old_t = torch.from_numpy(old_logps).float()
        adv_t = torch.from_numpy(advs).float()
        ret_t = torch.from_numpy(rets).float()
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        last = 0.0
        for _ in range(n_epochs):
            logits, values = self.net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts_t)
            ent = dist.entropy().mean()
            ratio = torch.exp(logp - old_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(values, ret_t)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent
            loss = loss + self._ewc_penalty()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
            last = float(loss.item())

        # periodic consolidation
        self._update_count += 1
        if self._update_count % self.ewc_consolidate_every == 0:
            self._consolidate(obs_t, acts_t)
        return {"loss": last}

    def _consolidate(self, obs_t, acts_t):
        logits, _ = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        ll = dist.log_prob(acts_t).sum()
        self.net.zero_grad()
        ll.backward()
        self.fisher = [p.grad.detach().clone() ** 2 if p.grad is not None
                       else torch.zeros_like(p) for p in self.net.parameters()]
        self.anchor = [p.detach().clone() for p in self.net.parameters()]
        self.net.zero_grad()


class EWCIPPO:
    def __init__(self, n_agents, obs_dim, act_dim, hidden=[64, 64], **kwargs):
        self.n_agents = n_agents
        seed = kwargs.pop("seed", 0)
        self.comm_msgs_per_step = 0  # independent learners: no inter-agent comm
        self.agents = [
            EWCIPPOAgent(obs_dim, act_dim, hidden=hidden, seed=seed + i, **kwargs)
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
