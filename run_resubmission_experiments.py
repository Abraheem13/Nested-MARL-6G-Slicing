"""
Resubmission experiment runner.

Adds exactly the runs the reviewers and the Associate Editor asked for, on top
of the existing published results in results/. Idempotent: already-completed
runs are skipped, so you can interrupt and resume.

New experiments
---------------
  Phase A -- New baselines at kappa=1.5 (R2.3, R1.5, AE):
             MAPPO (CTDE) and EWC-IPPO (continual learning), seeds 0-9.
  Phase B -- Scaling + overhead (R1.4/R2.4 + R1.3/R1.7 + AE):
             N in {6, 10} x S in {3, 5}, methods {ippo, mappo, nested},
             seeds 0-4. (N=3,S=3 already exists.)
  Phase C -- Severity sweep lifted to n=10 (fixes the n=3 soft spot):
             kappa in {0.5, 1.0, 2.0}, methods {ippo, nested}, seeds 3-9
             (seeds 0-2 already exist; kappa=1.5 already has n=10).

Estimated total on a laptop CPU: ~3-5 hours. Run overnight or in chunks:
  python3 run_resubmission_experiments.py --phase A
  python3 run_resubmission_experiments.py --phase B
  python3 run_resubmission_experiments.py --phase C
  python3 run_resubmission_experiments.py --phase all
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from train import train


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def base_args(**overrides):
    defaults = dict(
        method="nested", drift="multi", n_agents=3, n_slices=3, horizon=150,
        episodes=80, seed=0, hidden=64, ppo_epochs=4, lr_ippo=5e-4,
        severity=1.5, log_every=80, outdir="results",
    )
    defaults.update(overrides)
    return Args(**defaults)


def result_path(method, severity, seed, N=3, S=3, outdir="results") -> Path:
    sev_tag = f"_sev{severity:.1f}" if severity != 1.0 else ""
    ns_tag = f"_N{N}_S{S}" if (N != 3 or S != 3) else ""
    return Path(outdir) / f"{method}_drift-multi{sev_tag}{ns_tag}_seed{seed}.json"


def run_if_missing(method, severity, seed, N=3, S=3):
    p = result_path(method, severity, seed, N, S)
    if p.exists():
        print(f"  [skip] {p.name}")
        return
    print(f"  [run]  {p.name}")
    args = base_args(method=method, severity=severity, seed=seed, n_agents=N, n_slices=S)
    train(args)


def phase_A():
    print("\n" + "=" * 72)
    print("PHASE A -- New baselines (MAPPO, EWC-IPPO) at kappa=1.5, seeds 0-9")
    print("=" * 72)
    for seed in range(10):
        for method in ("mappo", "ewc_ippo"):
            run_if_missing(method, 1.5, seed)


def phase_B():
    print("\n" + "=" * 72)
    print("PHASE B -- Scaling + overhead: N in {6,10} x S in {3,5}, seeds 0-4")
    print("=" * 72)
    for N in (6, 10):
        for S in (3, 5):
            for method in ("ippo", "mappo", "nested"):
                for seed in range(5):
                    run_if_missing(method, 1.5, seed, N=N, S=S)
    # also S=5 at N=3 for a clean slice-only scaling axis
    for method in ("ippo", "mappo", "nested"):
        for seed in range(5):
            run_if_missing(method, 1.5, seed, N=3, S=5)


def phase_C():
    print("\n" + "=" * 72)
    print("PHASE C -- Severity sweep to n=10: kappa in {0.5,1.0,2.0}, seeds 3-9")
    print("=" * 72)
    for sev in (0.5, 1.0, 2.0):
        for seed in range(3, 10):
            for method in ("ippo", "nested"):
                run_if_missing(method, sev, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["A", "B", "C", "all"])
    args = ap.parse_args()
    t0 = time.time()
    if args.phase in ("A", "all"):
        phase_A()
    if args.phase in ("B", "all"):
        phase_B()
    if args.phase in ("C", "all"):
        phase_C()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Now run: python3 analyze_resubmission.py")


if __name__ == "__main__":
    main()
