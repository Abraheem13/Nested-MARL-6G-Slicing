"""
Unified training script.

Runs any of the agent classes on the 6G slicing environment. Uses on-policy
rollouts with per-agent GAE.

Usage:
    python train.py --method nested --seed 0 --episodes 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from env.slicing_env import SlicingEnv
from drift.schedulers import MultiTimescaleDrift, StationaryDrift, AbruptDrift
from agents.nested_marl import NestedMARL
from agents.ippo import IPPO
from agents.mappo import MAPPO
from agents.ewc_ippo import EWCIPPO
from agents.ablations import (
    NestedMARL_NoEMA,
    NestedMARL_NoTimescale,
    NestedMARL_TwoLevel,
)


METHOD_REGISTRY = {
    "nested": NestedMARL,
    "ippo": IPPO,
    "mappo": MAPPO,
    "ewc_ippo": EWCIPPO,
    "nested_no_ema": NestedMARL_NoEMA,
    "nested_no_timescale": NestedMARL_NoTimescale,
    "nested_two_level": NestedMARL_TwoLevel,
}


def make_env(drift_kind: str, horizon: int, n_agents: int, seed: int, severity: float = 1.0, n_slices: int = 3):
    if drift_kind == "multi":
        drift = MultiTimescaleDrift(
            medium_period=15, slow_period=50, severity=severity,
            continuous=True, seed=seed + 10000, n_slices=n_slices,
        )
    elif drift_kind == "stationary":
        drift = StationaryDrift(n_slices=n_slices)
    elif drift_kind == "abrupt":
        drift = AbruptDrift(shift_step=horizon // 2, seed=seed + 10000)
    else:
        raise ValueError(drift_kind)
    env = SlicingEnv(
        n_agents=n_agents,
        n_slices=n_slices,
        seed=seed,
        drift_scheduler=drift,
        horizon=horizon,
    )
    return env


def empty_rollout_dict(n_agents: int) -> List[dict]:
    return [
        {
            "obs": [], "actions": [], "logps": [], "values": [],
            "rewards": [], "dones": [], "last_value": 0.0,
        }
        for _ in range(n_agents)
    ]


def train(args):
    env = make_env(args.drift, args.horizon, args.n_agents, args.seed,
                   severity=getattr(args, "severity", 1.0),
                   n_slices=getattr(args, "n_slices", 3))
    obs = env.reset(seed=args.seed)

    AgentCls = METHOD_REGISTRY[args.method]
    # Allow per-method hyperparam via kwargs
    extra_kwargs: Dict = {}
    if args.method in ("ippo", "mappo", "ewc_ippo"):
        extra_kwargs["lr"] = args.lr_ippo
    # Nested variants pull their alphas from defaults; we keep them fixed per paper table.

    agents = AgentCls(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        act_dim=env.action_dim,
        hidden=[args.hidden, args.hidden],
        seed=args.seed,
        **extra_kwargs,
    )

    metrics = {
        "episode_reward": [],
        "sla_eMBB": [],
        "sla_URLLC": [],
        "sla_mMTC": [],
        "sla_all_slices": [],
        "queue_mean": [],
        "switching_cost": [],
        "regime_trace": [],
    }

    t0 = time.time()
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep * 997)
        rollouts = empty_rollout_dict(env.n_agents)
        ep_reward = np.zeros(env.n_agents)
        ep_sla = np.zeros(env.n_slices)
        ep_queue = 0.0
        ep_switch = 0.0
        regimes_this_ep = []

        for t in range(args.horizon):
            actions, logps, values = agents.act_all(obs)
            next_obs, rewards, done, info = env.step(actions)
            for i in range(env.n_agents):
                rollouts[i]["obs"].append(obs[i])
                rollouts[i]["actions"].append(actions[i])
                rollouts[i]["logps"].append(logps[i])
                rollouts[i]["values"].append(values[i])
                rollouts[i]["rewards"].append(rewards[i])
                rollouts[i]["dones"].append(float(done))
            ep_reward += np.array(rewards)
            ep_sla += np.array(info["sla_per_slice"])
            ep_queue += info["queue_mean"]
            ep_switch += info["switching_cost"]
            regimes_this_ep.append(info["regime"])
            obs = next_obs
            if done:
                break

        # Bootstrap last values for GAE
        for i in range(env.n_agents):
            _, _, v = agents.agents[i].act(obs[i])
            rollouts[i]["last_value"] = v

        agents.update_all(rollouts, n_epochs=args.ppo_epochs)

        metrics["episode_reward"].append(float(ep_reward.mean()))
        metrics["sla_eMBB"].append(float(ep_sla[0] / args.horizon))
        metrics["sla_URLLC"].append(float(ep_sla[1] / args.horizon))
        metrics["sla_mMTC"].append(float(ep_sla[2] / args.horizon))
        metrics["sla_all_slices"].append(float(ep_sla.mean() / args.horizon))
        metrics["queue_mean"].append(float(ep_queue / args.horizon))
        metrics["switching_cost"].append(float(ep_switch / args.horizon))
        metrics["regime_trace"].append(int(round(np.mean(regimes_this_ep))))

        if (ep + 1) % max(1, args.log_every) == 0 or ep == args.episodes - 1:
            print(
                f"[{args.method} s{args.seed}] ep {ep+1}/{args.episodes}"
                f" reward={metrics['episode_reward'][-1]:.3f}"
                f" sla=({metrics['sla_eMBB'][-1]:.2f},{metrics['sla_URLLC'][-1]:.2f},{metrics['sla_mMTC'][-1]:.2f})"
                f" q={metrics['queue_mean'][-1]:.2f}"
                f" sw={metrics['switching_cost'][-1]:.3f}"
                f" t={time.time()-t0:.1f}s"
            )

    # Overhead accounting for the scaling/overhead study
    try:
        if args.method == "mappo":
            params_per_agent = int(sum(p.numel() for p in agents.actors[0].parameters()))
            critic_params = int(sum(p.numel() for p in agents.critic.parameters()))
            optims_per_agent = 1
        else:
            a0 = agents.agents[0]
            params_per_agent = int(sum(p.numel() for p in a0.net.parameters()))
            critic_params = 0
            optims_per_agent = len(getattr(a0, "optimisers", [getattr(a0, "opt", None)]))
    except Exception:
        params_per_agent = None
        critic_params = None
        optims_per_agent = None
    comm = getattr(agents, "comm_msgs_per_step", 0)

    # Save
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sev_tag = f"_sev{getattr(args, 'severity', 1.0):.1f}" if getattr(args, 'severity', 1.0) != 1.0 else ""
    ns_tag = f"_N{args.n_agents}_S{args.n_slices}" if (args.n_agents != 3 or args.n_slices != 3) else ""
    outpath = outdir / f"{args.method}_drift-{args.drift}{sev_tag}{ns_tag}_seed{args.seed}.json"
    with open(outpath, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "metrics": metrics,
                "wall_time": time.time() - t0,
                "params_per_agent": params_per_agent,
                "critic_params": critic_params,
                "optims_per_agent": optims_per_agent,
                "comm_msgs_per_step": int(comm),
            },
            f,
            indent=2,
        )
    print(f"Saved -> {outpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="nested",
                   choices=list(METHOD_REGISTRY.keys()))
    p.add_argument("--drift", type=str, default="multi",
                   choices=["multi", "stationary", "abrupt"])
    p.add_argument("--n_agents", type=int, default=3)
    p.add_argument("--n_slices", type=int, default=3)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--lr_ippo", type=float, default=5e-4)
    p.add_argument("--severity", type=float, default=1.0,
                   help="Drift severity scaling factor (0.0 = no drift contrast, 1.0 = default, 2.0 = severe)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
