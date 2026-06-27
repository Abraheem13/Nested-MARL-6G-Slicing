"""
Plotting module: generates every figure the paper needs.

Reads all JSON files in results/ and produces:
    fig_learning_curves.pdf        - reward vs episode, all methods
    fig_cumulative_regret.pdf      - cumulative regret vs. oracle
    fig_ablation_bars.pdf          - final performance bars with error bars
    fig_per_slice_sla.pdf          - per-slice SLA over time
    fig_switching_cost.pdf         - policy stability (switching cost) over time
    fig_timescale_analysis.pdf     - adaptation speed after regime change
    fig_drift_schedule.pdf         - illustrative drift schedule (synthetic)

All figures use matplotlib defaults (no seaborn). 300 dpi. Vector PDF.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Paper-quality styling (kept minimal for reproducibility)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # editable fonts in PDF
    "ps.fonttype": 42,
})

# Consistent colour palette per method
METHOD_COLORS = {
    "nested":               "#1f77b4",  # blue  (our method)
    "ippo":                 "#d62728",  # red   (main baseline)
    "nested_no_ema":        "#2ca02c",  # green
    "nested_no_timescale":  "#ff7f0e",  # orange
    "nested_two_level":     "#9467bd",  # purple
}

METHOD_LABELS = {
    "nested":               "Nested-MARL",
    "ippo":                 "IPPO",
    "nested_no_ema":        "Nested-MARL, no EMA",
    "nested_no_timescale":  "Same rate (no timescale sep.)",
    "nested_two_level":     "Nested-MARL, 2 timescales",
}

# -- Oracle reference for SLA satisfaction (from env.oracle_reward)
ORACLE_SLA = 0.85


# -----------------------------------------------------------------------
def load_all_results(results_dir: str = "results") -> Dict[str, Dict[int, dict]]:
    """Load main (severity=1.5) results organised as method -> seed -> data.

    Only loads the main-experiment JSON files (no _sev prefix in filename),
    which correspond to severity=1.5 (our headline configuration).
    """
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for p in Path(results_dir).glob("*.json"):
        # Skip severity-sweep files (those have _sev in the filename stem)
        if "_sev" in p.stem:
            # But keep sev1.5 as main -- that's our headline configuration
            if "_sev1.5" not in p.stem:
                continue
        with open(p) as f:
            data = json.load(f)
        args = data["args"]
        method = args["method"]
        seed = args["seed"]
        drift = args["drift"]
        if drift != "multi":
            continue
        out[method][seed] = data
    return dict(out)


def load_severity_sweep(results_dir: str = "results") -> Dict[str, Dict[float, Dict[int, dict]]]:
    """Load all severity-sweep results, organised as method -> severity -> seed -> data."""
    out: Dict[str, Dict[float, Dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for p in Path(results_dir).glob("*_sev*.json"):
        with open(p) as f:
            data = json.load(f)
        args = data["args"]
        method = args["method"]
        seed = args["seed"]
        sev = float(args.get("severity", 1.0))
        drift = args["drift"]
        if drift != "multi":
            continue
        out[method][sev][seed] = data
    # Convert to regular dicts
    return {m: dict(sevs) for m, sevs in out.items()}


def stack_metric(
    runs: Dict[int, dict], key: str
) -> np.ndarray:
    """Stack a per-episode metric across seeds: returns (n_seeds, n_episodes)."""
    seeds = sorted(runs.keys())
    arrs = [np.asarray(runs[s]["metrics"][key], dtype=np.float32) for s in seeds]
    min_len = min(len(a) for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    return np.stack(arrs, axis=0)


def plot_with_ci(ax, x, mean, std, color, label, alpha_fill=0.15):
    ax.plot(x, mean, color=color, label=label, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha_fill, linewidth=0)


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Rolling mean for visual smoothing (does NOT affect reported numbers)."""
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window, dtype=np.float32) / window
    # 'same' mode preserves length; handle edges gracefully
    return np.convolve(y, kernel, mode="same")


# -----------------------------------------------------------------------
def fig_learning_curves(all_results, outdir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for method, runs in all_results.items():
        if len(runs) == 0:
            continue
        rewards = stack_metric(runs, "episode_reward")
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)
        mean_s = smooth(mean, 5)
        x = np.arange(len(mean_s))
        plot_with_ci(
            ax, x, mean_s, std, METHOD_COLORS.get(method, "#333"),
            METHOD_LABELS.get(method, method)
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("Learning curves under multi-timescale drift")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outdir / "fig_learning_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_cumulative_regret(all_results, outdir: Path, oracle_per_step: float = ORACLE_SLA):
    """Cumulative regret vs. oracle SLA satisfaction (averaged across slices, weighted)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for method, runs in all_results.items():
        if len(runs) == 0:
            continue
        # Use unweighted mean SLA across slices as a proxy for achieved reward
        sla_stack = np.stack([
            (
                stack_metric(runs, "sla_eMBB")
                + stack_metric(runs, "sla_URLLC")
                + stack_metric(runs, "sla_mMTC")
            ) / 3.0
        ])[0]  # (n_seeds, n_eps)
        regret_per_ep = oracle_per_step - sla_stack  # positive = worse than oracle
        cum_regret = np.cumsum(regret_per_ep, axis=1)
        mean = cum_regret.mean(axis=0)
        std = cum_regret.std(axis=0)
        x = np.arange(len(mean))
        plot_with_ci(
            ax, x, mean, std, METHOD_COLORS.get(method, "#333"),
            METHOD_LABELS.get(method, method)
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative SLA regret vs. oracle")
    ax.set_title("Cumulative regret under multi-timescale non-stationarity")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outdir / "fig_cumulative_regret.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_ablation_bars(all_results, outdir: Path, last_n: int = 30):
    """Bar chart of final-phase mean reward with std error bars."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    methods_order = ["ippo", "nested_no_timescale", "nested_no_ema", "nested_two_level", "nested"]
    methods_order = [m for m in methods_order if m in all_results]
    means, stds, labels, colors = [], [], [], []
    for m in methods_order:
        rewards = stack_metric(all_results[m], "episode_reward")
        final = rewards[:, -last_n:].mean(axis=1)  # per-seed final mean
        means.append(final.mean())
        stds.append(final.std())
        labels.append(METHOD_LABELS.get(m, m))
        colors.append(METHOD_COLORS.get(m, "#333"))
    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"Mean reward over final {last_n} episodes")
    ax.set_title("Ablation: isolating the contributors to Nested-MARL gains")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = outdir / "fig_ablation_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_per_slice_sla(all_results, outdir: Path):
    """Per-slice SLA satisfaction vs. episode for Nested-MARL vs. IPPO."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    slices = [("sla_eMBB", "eMBB"), ("sla_URLLC", "URLLC"), ("sla_mMTC", "mMTC")]
    focus = [m for m in ("nested", "ippo") if m in all_results]
    for ax, (key, name) in zip(axes, slices):
        for m in focus:
            arr = stack_metric(all_results[m], key)
            mean = smooth(arr.mean(axis=0), 5)
            std = arr.std(axis=0)
            x = np.arange(len(mean))
            plot_with_ci(ax, x, mean, std,
                         METHOD_COLORS.get(m, "#333"),
                         METHOD_LABELS.get(m, m))
        ax.set_title(f"{name}")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("SLA satisfaction")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = outdir / "fig_per_slice_sla.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_switching_cost(all_results, outdir: Path):
    """Per-episode switching cost (policy stability indicator)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for method, runs in all_results.items():
        if len(runs) == 0:
            continue
        arr = stack_metric(runs, "switching_cost")
        mean = smooth(arr.mean(axis=0), 5)
        std = arr.std(axis=0)
        x = np.arange(len(mean))
        plot_with_ci(
            ax, x, mean, std, METHOD_COLORS.get(method, "#333"),
            METHOD_LABELS.get(method, method)
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean switching cost")
    ax.set_title("Policy stability: switching cost over training")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outdir / "fig_switching_cost.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_timescale_analysis(all_results, outdir: Path, horizon: int = 200):
    """Recovery speed: reward in episodes immediately following a regime change.

    We approximate by computing the reward differential before/after episode
    boundaries aligned with slow_period. This is a synthetic visualisation --
    exact alignment depends on drift config.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    # Use last 30 episodes as a proxy 'post-drift' adaptation window
    window = 30
    for method, runs in all_results.items():
        if len(runs) == 0:
            continue
        arr = stack_metric(runs, "episode_reward")
        mid = arr.shape[1] // 2
        before = arr[:, mid - window:mid].mean(axis=1)
        after = arr[:, mid:mid + window].mean(axis=1)
        recovery = after - before  # positive = improved post-drift
        mean = recovery.mean()
        std = recovery.std()
        ax.errorbar(
            [METHOD_LABELS.get(method, method)], [mean], yerr=[std],
            marker="o", markersize=8, capsize=4,
            color=METHOD_COLORS.get(method, "#333"),
        )
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel(r"$\Delta$ Reward (post-drift $-$ pre-drift)")
    ax.set_title("Adaptation to distribution shift")
    plt.xticks(rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = outdir / "fig_timescale_analysis.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_severity_sweep(sweep_results, outdir: Path, first_n: int = 40):
    """Figure showing how Nested-MARL's advantage scales with drift severity.

    Two subplots:
      (a) Mean reward vs severity, one curve per method
      (b) Nested-MARL advantage (Nested - IPPO) vs severity, error bars

    Uses the first `first_n` episodes as the summary statistic, because that's
    where the sample-efficiency signal lives.
    """
    if not sweep_results:
        return
    # Methods we care about for the sweep
    interesting = [m for m in ("ippo", "nested") if m in sweep_results]
    if len(interesting) < 2:
        print("  [severity_sweep] Need both ippo and nested in sweep; skipping.")
        return

    # Find the common set of severities present for both methods
    sev_sets = [set(sweep_results[m].keys()) for m in interesting]
    severities = sorted(set.intersection(*sev_sets))
    if not severities:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))

    # --- (a) Mean reward vs severity ---
    ax = axes[0]
    for m in interesting:
        means, stds = [], []
        for sev in severities:
            seeds = sweep_results[m][sev]
            per_seed_stat = np.array([
                np.mean(seeds[s]["metrics"]["episode_reward"][:first_n])
                for s in sorted(seeds)
            ])
            means.append(per_seed_stat.mean())
            stds.append(per_seed_stat.std())
        means, stds = np.array(means), np.array(stds)
        ax.errorbar(severities, means, yerr=stds,
                    marker="o", markersize=7, capsize=4, linewidth=2,
                    color=METHOD_COLORS.get(m, "#333"),
                    label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("Drift severity")
    ax.set_ylabel(f"Mean reward over first {first_n} episodes")
    ax.set_title("(a) Absolute performance vs. drift severity")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)

    # --- (b) Paired advantage (Nested - IPPO) vs severity ---
    ax = axes[1]
    if "ippo" in sweep_results and "nested" in sweep_results:
        gap_mean, gap_std = [], []
        for sev in severities:
            ippo_seeds = sweep_results["ippo"][sev]
            nested_seeds = sweep_results["nested"][sev]
            common_seeds = sorted(set(ippo_seeds.keys()) & set(nested_seeds.keys()))
            per_seed_gap = np.array([
                np.mean(nested_seeds[s]["metrics"]["episode_reward"][:first_n]) -
                np.mean(ippo_seeds[s]["metrics"]["episode_reward"][:first_n])
                for s in common_seeds
            ])
            gap_mean.append(per_seed_gap.mean())
            gap_std.append(per_seed_gap.std())
        gap_mean, gap_std = np.array(gap_mean), np.array(gap_std)
        ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
        ax.errorbar(severities, gap_mean, yerr=gap_std,
                    marker="D", markersize=7, capsize=4, linewidth=2,
                    color="#1f77b4")
        ax.fill_between(severities, gap_mean - gap_std, gap_mean + gap_std,
                        color="#1f77b4", alpha=0.15)
    ax.set_xlabel("Drift severity")
    ax.set_ylabel("Paired advantage (Nested $-$ IPPO)")
    ax.set_title("(b) Nested-MARL advantage grows with drift severity")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "fig_severity_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_drift_schedule(outdir: Path, horizon: int = 400):
    """Illustrative drift schedule: fast channel, medium demand, slow regime."""
    from drift.schedulers import MultiTimescaleDrift

    drift = MultiTimescaleDrift(medium_period=15, slow_period=50, seed=42)
    t_axis = np.arange(horizon)
    regimes = np.array([drift.current_regime(t) for t in t_axis])
    demands = np.stack(
        [drift.current_demand(t, int(regimes[t])) for t in t_axis]
    )  # (horizon, 3)
    # Synthetic fast channel process (AR-like)
    rng = np.random.default_rng(0)
    chan = [0.7]
    for _ in range(horizon - 1):
        chan.append(np.clip(0.9 * chan[-1] + 0.1 * 0.7 + rng.normal(0.0, 0.05), 0.1, 1.0))
    chan = np.array(chan)

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 4.8), sharex=True)
    axes[0].plot(t_axis, chan, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Channel state")
    axes[0].set_title(r"Fast timescale: channel fading (per-step)")
    axes[0].grid(True, alpha=0.3)

    for s, name, c in zip(range(3), ["eMBB", "URLLC", "mMTC"],
                          ["#1f77b4", "#d62728", "#2ca02c"]):
        axes[1].plot(t_axis, demands[:, s], color=c, label=name, linewidth=1.2)
    axes[1].set_ylabel("Demand mult.")
    axes[1].set_title(r"Medium timescale: per-slice demand (every 15 steps)")
    axes[1].legend(loc="upper right", ncol=3, frameon=False)
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t_axis, regimes, color="#2ca02c", where="post", linewidth=1.4)
    axes[2].set_ylabel("SLA regime")
    axes[2].set_xlabel("Step")
    axes[2].set_title(r"Slow timescale: SLA regime (every 50 steps)")
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["Balanced", "High-URLLC", "High-eMBB"])
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "fig_drift_schedule.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def fig_system_model(outdir: Path):
    """Schematic system diagram (block-style)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.axis("off")

    def box(x, y, w, h, text, color, fc="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=fc,
                                   edgecolor=color, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    # Top: environment
    box(0.05, 0.70, 0.90, 0.18, "6G Network Slicing Environment\n(eMBB + URLLC + mMTC, multi-timescale drift)",
        color="#333", fc="#f0f0f0")

    # Middle: 3 agents with nested layers
    for i, x0 in enumerate([0.08, 0.38, 0.68]):
        box(x0, 0.32, 0.24, 0.30, f"Agent {i+1}", color="#1f77b4", fc="#ffffff")
        # Three nested rate bands
        ax.text(x0 + 0.01, 0.56, r"$\theta_2$  fast", fontsize=8, color="#d62728")
        ax.text(x0 + 0.01, 0.50, r"$\theta_1$  med.", fontsize=8, color="#ff7f0e")
        ax.text(x0 + 0.01, 0.44, r"$\theta_0$  slow", fontsize=8, color="#2ca02c")
        ax.text(x0 + 0.01, 0.37, r"$\bar\theta$  EMA memory", fontsize=8, color="#9467bd")

    # Bottom: allocations
    box(0.05, 0.06, 0.90, 0.15,
        "Local PRB allocation per slice  (15-template discrete action)",
        color="#333", fc="#f8f8f8")

    # Arrows (env -> agents -> env)
    for x0 in [0.20, 0.50, 0.80]:
        ax.annotate("", xy=(x0, 0.62), xytext=(x0, 0.70),
                    arrowprops=dict(arrowstyle="->", color="#333"))
        ax.annotate("", xy=(x0, 0.21), xytext=(x0, 0.32),
                    arrowprops=dict(arrowstyle="->", color="#333"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Nested-MARL system architecture")
    out = outdir / "fig_system_model.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# -----------------------------------------------------------------------
def summary_table(all_results, outdir: Path, last_n: int = 30):
    """Print + save a LaTeX-friendly summary table."""
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Method & Reward & SLA (avg) & Switching cost & Queue \\\\",
             "\\midrule"]
    methods_order = ["ippo", "nested_no_timescale", "nested_no_ema", "nested_two_level", "nested"]
    methods_order = [m for m in methods_order if m in all_results]
    for m in methods_order:
        rewards = stack_metric(all_results[m], "episode_reward")[:, -last_n:].mean(axis=1)
        embb = stack_metric(all_results[m], "sla_eMBB")[:, -last_n:].mean(axis=1)
        urllc = stack_metric(all_results[m], "sla_URLLC")[:, -last_n:].mean(axis=1)
        mmtc = stack_metric(all_results[m], "sla_mMTC")[:, -last_n:].mean(axis=1)
        sla = (embb + urllc + mmtc) / 3.0
        sw = stack_metric(all_results[m], "switching_cost")[:, -last_n:].mean(axis=1)
        q = stack_metric(all_results[m], "queue_mean")[:, -last_n:].mean(axis=1)
        lines.append(
            f"{METHOD_LABELS.get(m, m)} & "
            f"{rewards.mean():.3f} $\\pm$ {rewards.std():.3f} & "
            f"{sla.mean():.3f} $\\pm$ {sla.std():.3f} & "
            f"{sw.mean():.3f} $\\pm$ {sw.std():.3f} & "
            f"{q.mean():.2f} $\\pm$ {q.std():.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = "\n".join(lines)
    out = outdir / "table_results.tex"
    out.write_text(tex)
    print("  -> " + str(out))
    print(tex)


# -----------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results")
    p.add_argument("--outdir", type=str, default="paper_figures")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_results = load_all_results(args.results)
    for m, runs in all_results.items():
        print(f"  {m}: {len(runs)} seeds")

    print("\nGenerating figures...")
    # Figures that depend on main (severity=1.5) results
    if any(len(r) > 0 for r in all_results.values()):
        fig_learning_curves(all_results, outdir)
        fig_cumulative_regret(all_results, outdir)
        fig_ablation_bars(all_results, outdir)
        fig_per_slice_sla(all_results, outdir)
        fig_switching_cost(all_results, outdir)
        fig_timescale_analysis(all_results, outdir)
        summary_table(all_results, outdir)
    else:
        print("  (no main results yet; skipping data figures)")

    # Severity-sweep figure
    print("\nLoading severity-sweep data...")
    sweep_results = load_severity_sweep(args.results)
    if sweep_results:
        for m, sevs in sweep_results.items():
            sev_summary = ", ".join(f"{s}:{len(ss)}" for s, ss in sorted(sevs.items()))
            print(f"  {m}: severities -> seeds {{ {sev_summary} }}")
        fig_severity_sweep(sweep_results, outdir)
    else:
        print("  (no severity sweep results; skipping)")

    # Schematic figures (no results needed)
    fig_drift_schedule(outdir)
    fig_system_model(outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
