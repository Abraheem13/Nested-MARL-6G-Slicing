"""
Ablation variants of Nested-MARL.

These isolate which ingredient of the full method drives the performance gain:

    1. NestedMARL_NoEMA : multi-timescale updates but no continuum memory anchor.
       Tests whether the EMA/stabilisation is necessary.

    2. NestedMARL_NoTimescale : identical architecture and EMA, but all three
       parameter groups use the *same* learning rate. Tests whether the gain
       comes from timescale separation or merely from the EMA.

    3. NestedMARL_TwoLevel : only two timescales (merge L0 and L1). Tests
       whether three levels beat two.

All three are implemented by parameterising NestedAgent appropriately.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .nested_marl import NestedMARL


class NestedMARL_NoEMA(NestedMARL):
    """Multi-timescale updates but stabilise_beta = 1.0 (no anchoring)."""

    def __init__(self, n_agents, obs_dim, act_dim, hidden=[64, 64], **kwargs):
        kwargs["stabilise_beta"] = 1.0  # no stabilisation
        super().__init__(n_agents, obs_dim, act_dim, hidden=hidden, **kwargs)


class NestedMARL_NoTimescale(NestedMARL):
    """All parameter groups share the same learning rate, keep EMA.

    We set this rate to match IPPO's (5e-4) so the ablation is: 'what if we
    strip timescale separation from Nested-MARL but keep the EMA anchor?'
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden=[64, 64], lr: float = 5e-4, **kwargs):
        kwargs["alpha_l0"] = lr
        kwargs["alpha_l1"] = lr
        kwargs["alpha_l2"] = lr
        super().__init__(n_agents, obs_dim, act_dim, hidden=hidden, **kwargs)


class NestedMARL_TwoLevel(NestedMARL):
    """Effectively two timescales: merge L0 and L1 at slow rate."""

    def __init__(self, n_agents, obs_dim, act_dim, hidden=[64, 64], **kwargs):
        # Set L0 == L1 to slow rate, keep L2 fast.
        if "alpha_l0" in kwargs or "alpha_l1" in kwargs:
            # Respect overrides, else set sensible default
            pass
        kwargs.setdefault("alpha_l0", 1e-4)
        kwargs.setdefault("alpha_l1", 1e-4)  # merged with L0
        kwargs.setdefault("alpha_l2", 2e-3)
        super().__init__(n_agents, obs_dim, act_dim, hidden=hidden, **kwargs)
