"""
Full-paper experiment runner.

Runs everything the paper needs, in order of importance:

    Phase 1 -- Finish main comparison (nested seed 9)           (~2 min)
    Phase 2 -- Ablations at severity 1.5, seeds 0-4             (~30 min)
    Phase 3 -- Severity sweep at seeds 0-2, sev in {0.5, 1.0, 2.0}  (~30 min)

Total: ~60-65 minutes on a laptop CPU.

You can interrupt at any phase with Ctrl-C; already-completed runs are saved
and skipped on re-run (idempotent).
"""
from __future__ import annotations

import time
from pathlib import Path

# Import the training function directly
from train import train


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def base_args(**overrides):
    defaults = dict(
        method="nested", drift="multi", n_agents=3, horizon=150,
        episodes=80, seed=0, hidden=64, ppo_epochs=4, lr_ippo=5e-4,
        severity=1.5, log_every=80, outdir="results",
    )
    defaults.update(overrides)
    return Args(**defaults)


def result_exists(method, severity, seed, outdir="results") -> bool:
    sev_tag = f"_sev{severity:.1f}" if severity != 1.0 else ""
    p = Path(outdir) / f"{method}_drift-multi{sev_tag}_seed{seed}.json"
    return p.exists()


def run_if_missing(method, severity, seed):
    if result_exists(method, severity, seed):
        print(f"  [skip] {method} sev={severity} seed={seed} already exists.")
        return
    print(f"  [run]  {method} sev={severity} seed={seed}")
    args = base_args(method=method, severity=severity, seed=seed)
    train(args)


def main():
    t0 = time.time()
    print("=" * 72)
    print("PHASE 1 -- Finish main Nested-MARL seeds at severity 1.5")
    print("=" * 72)
    for seed in range(10):
        run_if_missing("nested", 1.5, seed)

    print("\n" + "=" * 72)
    print("PHASE 2 -- Ablations at severity 1.5, seeds 0-4")
    print("=" * 72)
    for seed in range(5):
        for method in ("nested_no_ema", "nested_no_timescale", "nested_two_level"):
            run_if_missing(method, 1.5, seed)

    print("\n" + "=" * 72)
    print("PHASE 3 -- Severity sweep, seeds 0-2, sev in {0.5, 1.0, 2.0}")
    print("=" * 72)
    for sev in (0.5, 1.0, 2.0):
        for seed in range(3):
            for method in ("ippo", "nested"):
                run_if_missing(method, sev, seed)

    print("\n" + "=" * 72)
    print(f"All phases complete in {(time.time()-t0)/60:.1f} minutes.")
    print("Now run:  python3 plots.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
