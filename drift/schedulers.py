"""
Multi-timescale drift schedulers.

Injects non-stationarity at three distinct timescales into the SlicingEnv:
    - Fast drift: handled by the env itself (per-step channel noise)
    - Medium drift: demand multipliers that evolve per window of steps
    - Slow drift: SLA regime changes at long intervals

The separation of timescales is the core signal that motivates Nested-MARL:
standard MARL with a single learning rate cannot track all three cleanly.
"""
from __future__ import annotations
import numpy as np


class MultiTimescaleDrift:
    """Drift scheduler with explicit fast / medium / slow timescales.

    Parameters
    ----------
    medium_period : int
    slow_period : int
    n_regimes : int
    severity : float
        Scales how sharply regimes differ in demand. 1.0 = default. Higher
        values make regime shifts more punishing for non-adaptive learners.
    continuous : bool
        If True, the drift clock does not reset between episodes.
    seed : int
    """

    def __init__(
        self,
        medium_period: int = 15,
        slow_period: int = 50,
        n_regimes: int = 3,
        n_slices: int = 3,
        severity: float = 1.0,
        continuous: bool = False,
        seed: int = 0,
    ):
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.n_regimes = n_regimes
        self.n_slices = n_slices
        self.severity = severity
        self.continuous = continuous
        self.rng = np.random.default_rng(seed)

        # Global clock (only used when continuous=True)
        self._global_t = 0

        # Precompute demand trajectories for reproducibility
        self._demand_cache: dict[int, np.ndarray] = {}
        self._regime_cache: dict[int, int] = {}

    def _effective_t(self, t_in_episode: int) -> int:
        """Map episode-local time to drift-schedule time."""
        if self.continuous:
            return self._global_t + t_in_episode
        return t_in_episode

    def advance_episode(self, episode_length: int):
        """Called by the env after each episode when continuous=True."""
        if self.continuous:
            self._global_t += episode_length

    # ------------------------------------------------------------------
    def current_demand(self, t: int, regime: int) -> np.ndarray:
        """Return the current medium-timescale demand multiplier vector."""
        t_eff = self._effective_t(t)
        window = t_eff // self.medium_period
        if window not in self._demand_cache:
            # Regime-specific demand bias; severity controls how extreme it is.
            # Baseline is all ones; off-baseline regimes pull away from 1 by `severity`.
            base_contrast = np.array([
                [ 0.0,  0.0,  0.0],   # baseline (no offset)
                [-0.5,  1.2, -0.5],   # high-URLLC (+URLLC, -others)
                [ 1.2, -0.5, -0.5],   # high-eMBB  (+eMBB, -others)
            ])
            row = base_contrast[regime % self.n_regimes]
            # extend to n_slices by cycling the canonical triad (S=3 unchanged)
            contrast = np.array([row[s % 3] for s in range(self.n_slices)]) * self.severity
            bias = 1.0 + contrast
            noise = self.rng.normal(0.0, 0.15, size=self.n_slices)
            demand = bias * (1.0 + noise)
            demand = np.clip(demand, 0.2, 3.5)
            self._demand_cache[window] = demand.astype(np.float32)
        return self._demand_cache[window]

    # ------------------------------------------------------------------
    def current_regime(self, t: int) -> int:
        """Return the current slow-timescale SLA regime."""
        t_eff = self._effective_t(t)
        window = t_eff // self.slow_period
        if window not in self._regime_cache:
            # Regime cycles deterministically with an occasional skip for realism
            r = window % self.n_regimes
            if self.rng.random() < 0.1:
                r = int(self.rng.integers(0, self.n_regimes))
            self._regime_cache[window] = int(r)
        return self._regime_cache[window]


class StationaryDrift:
    """No-op drift: constant demand and regime. For ablation."""

    def __init__(self, n_slices: int = 3):
        self._demand = np.ones(n_slices, dtype=np.float32)

    def current_demand(self, t: int, regime: int) -> np.ndarray:
        return self._demand

    def current_regime(self, t: int) -> int:
        return 0


class AbruptDrift:
    """Single abrupt regime shift at midpoint. For ablation."""

    def __init__(self, shift_step: int = 500, n_slices: int = 3, seed: int = 0):
        self.shift_step = shift_step
        self.rng = np.random.default_rng(seed)
        self._demand_before = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._demand_after = np.array([0.7, 1.8, 0.6], dtype=np.float32)

    def current_demand(self, t: int, regime: int) -> np.ndarray:
        return self._demand_after if t >= self.shift_step else self._demand_before

    def current_regime(self, t: int) -> int:
        return 1 if t >= self.shift_step else 0
