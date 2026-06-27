"""
Batch experimental runner.

Executes the full experimental matrix in series:
    methods:  ippo, nested, nested_no_ema, nested_no_timescale, nested_two_level
    drifts:   multi (main), stationary (ablation)
    seeds:    0, 1, 2, 3, 4

Provides a single Python entry point with consistent logging and progress
reporting for the full experimental matrix.

Approx runtime on a laptop (Apple M-series or similar):
    - 150 episodes of horizon 200 per run = ~1-2 min per run
    - 5 methods x 5 seeds x 2 drifts = 50 runs = ~60-90 min total

For a faster smoke pass, use --episodes 50 --horizon 100  (~15 min total).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

# Import directly so we avoid spawning subprocesses (faster, no PYTHONPATH fuss)
from train import train


class Args:
    """Simple bag-of-attributes equivalent to argparse.Namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def run_one(method, drift, seed, episodes, horizon, outdir):
    a = Args(
        method=method,
        drift=drift,
        n_agents=3,
        horizon=horizon,
        episodes=episodes,
        seed=seed,
        hidden=64,
        ppo_epochs=4,
        lr_ippo=5e-4,
        log_every=max(1, episodes // 5),
        outdir=outdir,
    )
    train(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["ippo", "nested", "nested_no_ema",
                            "nested_no_timescale", "nested_two_level"])
    p.add_argument("--drifts", type=str, nargs="+",
                   default=["multi"])
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    total = len(args.methods) * len(args.seeds) * len(args.drifts)
    done = 0
    t_start = time.time()

    print("=" * 72)
    print(f"Total runs: {total}")
    print(f"Methods:    {args.methods}")
    print(f"Drifts:     {args.drifts}")
    print(f"Seeds:      {args.seeds}")
    print(f"Episodes:   {args.episodes}, horizon: {args.horizon}")
    print("=" * 72)

    for drift in args.drifts:
        for method in args.methods:
            for seed in args.seeds:
                done += 1
                elapsed = time.time() - t_start
                eta = (elapsed / max(done - 1, 1)) * (total - done + 1) if done > 1 else 0
                print(f"\n[{done}/{total}] method={method}  drift={drift}  seed={seed}"
                      f"   elapsed={elapsed/60:.1f}m   eta={eta/60:.1f}m")
                run_one(method, drift, seed, args.episodes, args.horizon, args.outdir)

    total_time = time.time() - t_start
    print(f"\nAll runs complete in {total_time/60:.1f} minutes.")


if __name__ == "__main__":
    main()
