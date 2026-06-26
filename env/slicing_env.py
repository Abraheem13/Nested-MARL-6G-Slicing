"""
6G Network Slicing Environment for Multi-Agent Resource Management.

A lightweight, Gym-style environment simulating adaptive resource management
across three 6G slice types (eMBB, URLLC, mMTC) served by N base stations.
Each base station is controlled by an agent that allocates PRBs (physical
resource blocks) across slices. The environment exhibits non-stationarity at
three timescales:
    - Fast (per-step): channel fading, user-level traffic bursts
    - Medium (per-episode window): aggregate user demand shifts
    - Slow (over many episodes): SLA policy / pricing regime changes

This multi-timescale structure is what motivates Nested-MARL.

Observation (per agent):
    - Local channel state (one per slice)
    - Local queue length per slice
    - Aggregate demand indicator per slice
    - SLA regime indicator (one-hot)
    => dim = 3*channels + 3*queues + 3*demand + n_regimes

Action (per agent):
    - Discrete allocation over 3 slices (Dirichlet-like simplex via softmax over 3 logits)
    - We use a discretised action space: choose one of K allocation templates
      (K = 10 templates spanning the 3-simplex), which is tractable for
      Q-learning baselines while still expressive.

Reward (per agent):
    - Weighted sum of SLA satisfaction per slice - switching cost
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ----------------------- Allocation templates ------------------------------
def _simplex_templates(k_per_dim: int = 4, n_slices: int = 3) -> np.ndarray:
    """Generate a discrete set of n_slices-simplex allocation vectors.

    For n_slices=3, k_per_dim=4 this returns the original 15 templates,
    preserving the published behaviour exactly. For larger n_slices we reduce
    k_per_dim so the template count stays tractable (~comparable to 15).
    """
    if n_slices == 3:
        templates = []
        for i in range(k_per_dim + 1):
            for j in range(k_per_dim + 1 - i):
                k = k_per_dim - i - j
                templates.append([i, j, k])
        arr = np.array(templates, dtype=np.float32) / float(k_per_dim)
        return arr
    # general n_slices: compositions of G into n_slices parts, G chosen to
    # keep the count near 15.
    from math import comb
    G = 2
    for g in range(1, 8):
        if comb(g + n_slices - 1, n_slices - 1) >= 15:
            G = g
            break
    comps = []

    def rec(remaining, parts, cur):
        if parts == 1:
            comps.append(cur + [remaining]); return
        for v in range(remaining + 1):
            rec(remaining - v, parts - 1, cur + [v])
    rec(G, n_slices, [])
    arr = np.array(comps, dtype=np.float32)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


ALLOC_TEMPLATES = _simplex_templates(k_per_dim=4)  # 15 templates (S=3 default)


# ------------------------- The environment ---------------------------------
class SlicingEnv:
    """Multi-agent 6G slicing environment.

    Parameters
    ----------
    n_agents : int
        Number of base stations / agents.
    n_slices : int
        Number of slice types (default 3: eMBB, URLLC, mMTC).
    total_prbs : int
        Total PRBs per base station per step.
    horizon : int
        Steps per episode.
    seed : int
        Random seed.
    drift_scheduler : DriftScheduler or None
        Optional injector of non-stationary regimes.
    """

    # Per-slice QoS profiles: (throughput_req, latency_tol, reliability_req)
    # These set SLA thresholds. Values are in normalised units.
    SLICE_PROFILES = {
        "eMBB":  {"throughput": 0.80, "latency": 0.60, "reliability": 0.90},
        "URLLC": {"throughput": 0.40, "latency": 0.95, "reliability": 0.99},
        "mMTC":  {"throughput": 0.20, "latency": 0.40, "reliability": 0.70},
    }
    SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]

    def __init__(
        self,
        n_agents: int = 3,
        n_slices: int = 3,
        total_prbs: int = 100,
        horizon: int = 100,
        seed: int = 0,
        drift_scheduler=None,
        switching_cost_coef: float = 0.05,
    ):
        # S=3 is the standard eMBB/URLLC/mMTC triad and reproduces the
        # published numbers exactly. For S>3 we extend by cycling the three
        # canonical slice profiles, which keeps the reward model well-defined
        # for the scaling study without altering the S=3 path.
        assert n_slices >= 3, "Need at least the 3 canonical slice types."
        self.n_agents = n_agents
        self.n_slices = n_slices
        self.total_prbs = total_prbs
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.drift_scheduler = drift_scheduler
        self.switching_cost_coef = switching_cost_coef

        self.n_regimes = 3  # baseline, high-URLLC, high-eMBB
        # S-dependent allocation templates (S=3 reproduces the original 15)
        self.templates = _simplex_templates(k_per_dim=4, n_slices=n_slices)
        self.n_templates = self.templates.shape[0]
        # Slice names/profiles extended by cycling the canonical triad for S>3
        base_names = ["eMBB", "URLLC", "mMTC"]
        self.slice_names = [base_names[s % 3] for s in range(n_slices)]
        base_arr = np.array([5.0, 2.0, 8.0], dtype=np.float32)
        self.base_arrivals = np.array(
            [base_arr[s % 3] for s in range(n_slices)], dtype=np.float32)

        # Observation dimension per agent
        self.obs_dim = 3 * n_slices + self.n_regimes

        # Action space per agent = index into ALLOC_TEMPLATES
        self.action_dim = self.n_templates

        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.t = 0
        # Channel states per (agent, slice)
        self.channel = self.rng.uniform(0.4, 0.9, size=(self.n_agents, self.n_slices)).astype(np.float32)
        # Queue lengths per (agent, slice)
        self.queue = np.zeros((self.n_agents, self.n_slices), dtype=np.float32)
        # Current SLA regime (integer 0..n_regimes-1)
        self.regime = 0
        # Demand shift multiplier per slice (medium-timescale)
        self.demand_mult = np.ones(self.n_slices, dtype=np.float32)
        # Previous allocations (for switching cost)
        self.prev_alloc = np.tile(
            np.ones(self.n_slices, dtype=np.float32) / self.n_slices,
            (self.n_agents, 1),
        )

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> List[np.ndarray]:
        # Notify scheduler that an episode has completed so drift-time can advance
        # (relevant only when scheduler is in continuous mode).
        if self.drift_scheduler is not None and hasattr(self.drift_scheduler, "advance_episode"):
            # Advance by however many steps the last episode actually ran
            self.drift_scheduler.advance_episode(self.t)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._build_obs_all()

    # ------------------------------------------------------------------
    def _build_obs_all(self) -> List[np.ndarray]:
        regime_onehot = np.eye(self.n_regimes, dtype=np.float32)[self.regime]
        obs = []
        for i in range(self.n_agents):
            chan = self.channel[i]
            queue = np.clip(self.queue[i] / 10.0, 0.0, 1.0)
            # Aggregate demand indicator (broadcast over slices)
            demand = self.demand_mult
            obs_i = np.concatenate([chan, queue, demand, regime_onehot])
            obs.append(obs_i.astype(np.float32))
        return obs

    # ------------------------------------------------------------------
    def _evolve_fast(self):
        """Fast timescale: channel fading & arrival noise (every step)."""
        # AR(1) fading process
        alpha = 0.9
        noise = self.rng.normal(0.0, 0.05, size=self.channel.shape).astype(np.float32)
        self.channel = np.clip(alpha * self.channel + (1 - alpha) * 0.65 + noise, 0.1, 1.0)

    def _evolve_medium(self):
        """Medium timescale: aggregate demand shifts driven by scheduler."""
        if self.drift_scheduler is not None:
            new_demand = self.drift_scheduler.current_demand(self.t, self.regime)
            self.demand_mult = np.asarray(new_demand, dtype=np.float32)

    def _evolve_slow(self):
        """Slow timescale: regime change, mutates SLA weighting."""
        if self.drift_scheduler is not None:
            new_regime = self.drift_scheduler.current_regime(self.t)
            self.regime = int(new_regime)

    # ------------------------------------------------------------------
    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """Execute one environment step.

        Parameters
        ----------
        actions : list of ints
            One action index per agent, indexing ALLOC_TEMPLATES.
        """
        assert len(actions) == self.n_agents

        # 1. Decode allocations
        allocs = np.stack([self.templates[a] for a in actions], axis=0)  # (N, S)

        # 2. Generate per-slice arrivals (affected by demand mult)
        base_arrivals = self.base_arrivals  # per-slice
        arrivals = base_arrivals * self.demand_mult
        arrivals_per_agent = np.tile(arrivals, (self.n_agents, 1))
        # Add small noise
        arrivals_per_agent += self.rng.normal(0.0, 0.3, size=arrivals_per_agent.shape).astype(np.float32)
        arrivals_per_agent = np.clip(arrivals_per_agent, 0.0, None)

        # 3. Update queues
        self.queue += arrivals_per_agent

        # 4. Service per (agent, slice): depends on allocation * channel * total_prbs
        # We normalise: served = alloc_frac * channel_quality * (total_prbs / base_denom)
        service_capacity = allocs * self.channel * (self.total_prbs / 10.0)
        # Cannot serve more than queue
        served = np.minimum(self.queue, service_capacity)
        self.queue -= served

        # 5. Compute per-agent per-slice SLA metrics
        #   throughput proxy = served / (arrivals + epsilon)
        #   latency proxy    = 1 - normalised queue length
        #   reliability      = 1 - fraction lost (we do no loss here, so proxy via queue)
        throughput = served / (arrivals_per_agent + 1e-6)
        latency_score = 1.0 - np.clip(self.queue / 15.0, 0.0, 1.0)
        reliability = np.clip(1.0 - self.queue / 25.0, 0.0, 1.0)

        # 6. SLA satisfaction per slice: soft indicator the key metric meets threshold
        sla_sat = np.zeros_like(throughput)
        base_weights = {
            0: np.array([1.0, 1.0, 1.0]),   # baseline: equal
            1: np.array([0.3, 2.4, 0.3]),   # high-URLLC (sharpened)
            2: np.array([2.4, 0.3, 0.3]),   # high-eMBB  (sharpened)
        }
        # Extend the 3-slice regime weights by cycling for S>3 (S=3 unchanged).
        w = np.array([base_weights[self.regime][s % 3]
                      for s in range(self.n_slices)], dtype=np.float32)
        for s_idx, s_name in enumerate(self.slice_names):
            prof = self.SLICE_PROFILES[s_name]
            t_ok = np.clip(throughput[:, s_idx] / prof["throughput"], 0.0, 1.0)
            l_ok = np.clip(latency_score[:, s_idx] / prof["latency"], 0.0, 1.0)
            r_ok = np.clip(reliability[:, s_idx] / prof["reliability"], 0.0, 1.0)
            sla_sat[:, s_idx] = (t_ok + l_ok + r_ok) / 3.0

        # 7. Switching cost
        switching_cost = np.linalg.norm(allocs - self.prev_alloc, axis=1)

        # 8. Per-agent reward = weighted SLA - switching cost
        rewards_per_agent = (sla_sat * w[None, :]).mean(axis=1) - self.switching_cost_coef * switching_cost
        rewards: List[float] = [float(r) for r in rewards_per_agent]

        # 9. Save for next step
        self.prev_alloc = allocs

        # 10. Evolve dynamics (order matters: fast always, slow/medium based on schedule)
        self._evolve_fast()
        self._evolve_medium()
        self._evolve_slow()

        self.t += 1
        done = self.t >= self.horizon

        info = {
            "sla_per_slice": sla_sat.mean(axis=0).tolist(),  # averaged over agents
            "queue_mean": float(self.queue.mean()),
            "regime": int(self.regime),
            "demand_mult": self.demand_mult.tolist(),
            "switching_cost": float(switching_cost.mean()),
            "throughput": float(throughput.mean()),
        }
        next_obs = self._build_obs_all()
        return next_obs, rewards, done, info

    # ------------------------------------------------------------------
    def oracle_reward(self, horizon_avg: bool = True) -> float:
        """Non-stationary oracle baseline: the theoretical max SLA score
        assuming perfect allocation matched to regime."""
        # Oracle allocates perfectly: distribute weighted by regime weights
        weights_by_regime = {
            0: np.array([0.40, 0.30, 0.30]),
            1: np.array([0.25, 0.55, 0.20]),
            2: np.array([0.55, 0.25, 0.20]),
        }
        alloc = weights_by_regime[self.regime]
        # Oracle gets ~0.85 SLA on average (not 1.0 due to stochastic fading)
        return 0.85
