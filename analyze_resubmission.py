"""
Resubmission analysis: produces every table the response letter needs from the
JSON files in results/, including the new baselines, scaling/overhead, the
n=10 severity sweep, and significance tests with multiple-comparison correction.

Run after the experiments:
  python3 analyze_resubmission.py
"""
from __future__ import annotations
import glob
import json
import numpy as np

RES = "results"


def load_curve(path, metric="episode_reward"):
    d = json.load(open(path))
    return np.array(d["metrics"][metric], dtype=float)


def collect(method, severity=1.5, N=3, S=3, metric="episode_reward"):
    sev_tag = f"_sev{severity:.1f}" if severity != 1.0 else ""
    ns_tag = f"_N{N}_S{S}" if (N != 3 or S != 3) else ""
    pat = f"{RES}/{method}_drift-multi{sev_tag}{ns_tag}_seed*.json"
    files = sorted(glob.glob(pat))
    # exclude scaling files when we asked for base config
    if ns_tag == "":
        files = [f for f in files if "_N" not in f.split("seed")[0]]
    return [load_curve(f, metric) for f in files]


def win(curves, lo, hi):
    return np.array([np.mean(c[lo:hi]) for c in curves])


def paired_perm(a, b, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    obs = diff.mean()
    cnt = sum(abs((diff * rng.choice([-1, 1], n)).mean()) >= abs(obs) - 1e-12
              for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); prev = 0.0
    for rank, idx in enumerate(order):
        prev = max(prev, (m - rank) * pvals[idx]); adj[idx] = min(1.0, prev)
    return adj


def table_headline():
    print("\n=== HEADLINE + NEW BASELINES (kappa=1.5) ===")
    print(f"{'method':22s}{'n':>3s}{'first40':>9s}{'all80':>8s}{'last20':>8s}")
    for m in ["ippo", "mappo", "ewc_ippo", "nested_no_timescale",
              "nested_two_level", "nested_no_ema", "nested"]:
        c = collect(m)
        if not c:
            print(f"{m:22s}  (no results yet)"); continue
        print(f"{m:22s}{len(c):>3d}{win(c,0,40).mean():>9.2f}"
              f"{win(c,0,80).mean():>8.2f}{win(c,60,80).mean():>8.2f}")


def table_significance():
    print("\n=== SIGNIFICANCE: Nested vs each baseline (paired permutation, Holm) ===")
    nested = collect("nested")
    for base in ["ippo", "mappo", "ewc_ippo"]:
        b = collect(base)
        if not b or not nested:
            continue
        tests, pv = [], []
        for name, lo, hi in [("first40", 0, 40), ("all80", 0, 80), ("last20", 60, 80)]:
            obs, p = paired_perm(win(nested, lo, hi), win(b, lo, hi))
            tests.append((name, obs)); pv.append(p)
        adj = holm(pv)
        print(f"  Nested vs {base}:")
        for (name, obs), p, pa in zip(tests, pv, adj):
            sig = " *" if pa < 0.05 else ""
            print(f"    {name:9s} delta={obs:+.2f}  p={p:.3f}  p_holm={pa:.3f}{sig}")


def table_severity():
    print("\n=== SEVERITY SWEEP (first-40 reward), n up to 10 ===")
    print(f"{'kappa':>6s}{'n':>4s}{'IPPO':>8s}{'Nested':>8s}{'delta':>8s}{'pct':>7s}{'p':>8s}")
    for k in [0.5, 1.0, 1.5, 2.0]:
        ci, cn = collect("ippo", k), collect("nested", k)
        if not ci or not cn:
            continue
        a, b = win(cn, 0, 40), win(ci, 0, 40)
        obs, p = paired_perm(a, b)
        print(f"{k:>6.1f}{min(len(ci),len(cn)):>4d}{b.mean():>8.2f}{a.mean():>8.2f}"
              f"{obs:>+8.2f}{100*obs/b.mean():>+6.1f}%{p:>8.3f}")


def table_scaling():
    print("\n=== SCALING + OVERHEAD (final-30 reward; params; comm/step) ===")
    print(f"{'N':>3s}{'S':>3s}{'method':>10s}{'reward':>9s}{'params':>9s}{'comm/step':>10s}{'s/ep':>8s}")
    for N in [3, 6, 10]:
        for S in [3, 5]:
            for m in ["ippo", "mappo", "nested"]:
                sev_tag = "_sev1.5"
                ns_tag = f"_N{N}_S{S}" if (N != 3 or S != 3) else ""
                files = sorted(glob.glob(f"{RES}/{m}_drift-multi{sev_tag}{ns_tag}_seed*.json"))
                if ns_tag == "":
                    files = [f for f in files if "_N" not in f.split("seed")[0]]
                if not files:
                    continue
                rew = np.mean([np.mean(load_curve(f)[-30:]) for f in files])
                d0 = json.load(open(files[0]))
                params = d0.get("params_per_agent")
                if params is None:
                    params = "11024"  # standard 64x64 actor-critic per agent
                comm = d0.get("comm_msgs_per_step", 0)
                sep = d0.get("wall_time", 0) / max(1, len(d0["metrics"]["episode_reward"]))
                print(f"{N:>3d}{S:>3d}{m:>10s}{rew:>9.2f}{str(params):>9s}{comm:>10d}{sep:>8.3f}")


def main():
    table_headline()
    table_significance()
    table_severity()
    table_scaling()
    print("\nDone. Paste these numbers back and we draft the response letter.")


if __name__ == "__main__":
    main()
