"""
Controlled compute-overhead timing.

The per-seed wall_time stored during the big sweep is unreliable for reporting
compute cost: runs were executed concurrently / with warm-up, so s/ep there is
noisy (e.g. an IPPO cell can spuriously read higher than MAPPO, which is
impossible structurally). This script times every method back-to-back in a
single quiet process, median of repeats, so the paper's overhead numbers are
defensible.

It reports, per (N, S, method):
  * params per agent (and centralised-critic params for MAPPO)
  * optimisers per agent
  * inter-agent communication messages per step (0 for independent learners; N for MAPPO)
  * median seconds per episode over `--repeats` short runs

Usage:
  python3 timing_overhead.py                 # default grid
  python3 timing_overhead.py --episodes 10 --repeats 3
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from env.slicing_env import SlicingEnv
from drift.schedulers import MultiTimescaleDrift
from agents.nested_marl import NestedMARL
from agents.ippo import IPPO
from agents.mappo import MAPPO
from agents.ewc_ippo import EWCIPPO

METHODS = {"ippo": IPPO, "mappo": MAPPO, "ewc_ippo": EWCIPPO, "nested": NestedMARL}


def time_one(method, N, S, episodes, horizon, seed=0):
    drift = MultiTimescaleDrift(15, 50, severity=1.5, continuous=True,
                                seed=seed + 10000, n_slices=S)
    env = SlicingEnv(n_agents=N, n_slices=S, seed=seed,
                     drift_scheduler=drift, horizon=horizon)
    obs = env.reset(seed=seed)
    cls = METHODS[method]
    kwargs = {}
    if method in ("ippo", "mappo", "ewc_ippo"):
        kwargs["lr"] = 5e-4
    agents = cls(n_agents=N, obs_dim=env.obs_dim, act_dim=env.action_dim,
                 hidden=[64, 64], seed=seed, **kwargs)

    # overhead facts
    if method == "mappo":
        params = sum(p.numel() for p in agents.actors[0].parameters())
        critic = sum(p.numel() for p in agents.critic.parameters())
        optims = 1
    else:
        a0 = agents.agents[0]
        params = sum(p.numel() for p in a0.net.parameters())
        critic = 0
        optims = len(getattr(a0, "optimisers", [getattr(a0, "opt", None)]))
    comm = getattr(agents, "comm_msgs_per_step", 0)

    t0 = time.perf_counter()
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep * 997)
        roll = [dict(obs=[], actions=[], logps=[], values=[], rewards=[],
                     dones=[], last_value=0.0) for _ in range(N)]
        for t in range(horizon):
            acts, logps, vals = agents.act_all(obs)
            nobs, rews, done, info = env.step(acts)
            for i in range(N):
                roll[i]["obs"].append(obs[i]); roll[i]["actions"].append(acts[i])
                roll[i]["logps"].append(logps[i]); roll[i]["values"].append(vals[i])
                roll[i]["rewards"].append(rews[i]); roll[i]["dones"].append(float(done))
            obs = nobs
        for i in range(N):
            _, _, v = agents.agents[i].act(obs[i])
            roll[i]["last_value"] = v
        agents.update_all(roll, n_epochs=4)
    sec_per_ep = (time.perf_counter() - t0) / episodes
    return dict(params=params, critic=critic, optims=optims,
                comm=comm, sec_per_ep=sec_per_ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid", default="3x3,6x3,10x3,3x5,6x5,10x5")
    args = ap.parse_args()

    grid = []
    for cell in args.grid.split(","):
        n, s = cell.split("x")
        grid.append((int(n), int(s)))

    print(f"{'N':>3}{'S':>3}{'method':>10}{'params':>9}{'critic':>8}"
          f"{'optims':>7}{'comm/step':>10}{'s/ep(med)':>11}")
    for (N, S) in grid:
        for m in ["ippo", "mappo", "ewc_ippo", "nested"]:
            reps = [time_one(m, N, S, args.episodes, args.horizon, seed=r)["sec_per_ep"]
                    for r in range(args.repeats)]
            info = time_one(m, N, S, 2, args.horizon, seed=0)  # for static facts
            med = float(np.median(reps))
            print(f"{N:>3}{S:>3}{m:>10}{info['params']:>9}{info['critic']:>8}"
                  f"{info['optims']:>7}{info['comm']:>10}{med:>11.4f}")


if __name__ == "__main__":
    main()
