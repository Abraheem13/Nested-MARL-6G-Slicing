# Nested-MARL for 6G Network Slicing

[![Paper](https://img.shields.io/badge/IEEE-OJCOMS-blue)](https://github.com/Abraheem13/Nested-MARL-6G-Slicing)

Official implementation for:

> **Nested Multi-Agent Reinforcement Learning for Adaptive Resource Management in 6G Network Slicing: A Multi-Timescale Framework with Convergence Guarantees**  
> R. A. R. Ejaz, F. Iradat, and W. Iqbal  
> *IEEE Open Journal of the Communications Society (IEEE OJCOMS)*

This repository provides the simulation environment, training code, pre-computed experimental results, and figure-generation scripts used in the paper.

## Overview

We study adaptive multi-agent resource management in a 6G network-slicing setting with multi-timescale non-stationarity. **Nested-MARL** assigns each agent's parameters to three groups updated at distinct learning rates matched to fast, medium, and slow environmental drift, together with an exponential moving-average memory term for stability.

The codebase includes:

- A 3-agent 6G slicing simulator (eMBB / URLLC / mMTC)
- Multi-timescale drift schedulers
- Nested-MARL and baselines (IPPO, MAPPO, EWC-IPPO)
- Ablation variants (no EMA, no timescale separation, two-level nesting)
- Pre-computed results and scripts to regenerate all paper figures

## Repository Structure

```
├── env/slicing_env.py          # 6G network-slicing environment
├── drift/schedulers.py         # Multi-timescale and ablation drift models
├── agents/
│   ├── nested_marl.py          # Nested-MARL (proposed method)
│   ├── ippo.py                 # Independent PPO baseline
│   ├── mappo.py                # MAPPO (CTDE) baseline
│   ├── ewc_ippo.py             # EWC-IPPO continual-learning baseline
│   ├── ablations.py            # Ablation variants
│   └── common.py               # Shared networks, buffers, utilities
├── train.py                    # Single-run training entry point
├── run_experiments.py          # Batch runner for core experiment matrix
├── run_full_paper_experiments.py   # Idempotent runner for main paper experiments
├── run_extended_experiments.py     # Extended baselines, scaling, and severity sweeps
├── analyze_results.py          # Summary tables and significance analysis
├── plots.py                    # Generate all paper figures and LaTeX tables
├── timing_overhead.py          # Controlled compute-overhead measurements
├── results/                    # Pre-computed experiment logs (JSON)
└── paper_figures/              # Generated figures and tables
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy, Matplotlib, Pandas, PyYAML

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Verify that training runs correctly (~30 seconds):

```bash
python train.py --method ippo   --episodes 20 --horizon 100 --seed 0 --severity 1.5 --log_every 5
python train.py --method nested --episodes 20 --horizon 100 --seed 0 --severity 1.5 --log_every 5
```

Train a single configuration:

```bash
python train.py --method nested --episodes 80 --horizon 150 --seed 0 --severity 1.5
```

Available methods: `nested`, `ippo`, `mappo`, `ewc_ippo`, `nested_no_ema`, `nested_no_timescale`, `nested_two_level`.

## Reproducing Paper Results

Pre-computed logs for the main experiments are included in `results/`. To regenerate figures and tables from these logs:

```bash
python plots.py
```

Output is written to `paper_figures/`, including:

| File | Description |
|------|-------------|
| `fig_learning_curves.pdf` | Episode reward vs. training |
| `fig_cumulative_regret.pdf` | Cumulative regret vs. oracle |
| `fig_ablation_bars.pdf` | Ablation comparison |
| `fig_per_slice_sla.pdf` | Per-slice SLA satisfaction |
| `fig_switching_cost.pdf` | Policy stability over training |
| `fig_timescale_analysis.pdf` | Adaptation after regime change |
| `fig_severity_sweep.pdf` | Performance vs. drift severity |
| `fig_drift_schedule.pdf` | Illustrative drift schedule |
| `fig_system_model.pdf` | System architecture schematic |
| `table_results.tex` | Main results table (LaTeX) |

To re-run experiments from scratch (idempotent; completed runs are skipped):

```bash
# Core paper experiments (~1 hour on a laptop CPU)
python run_full_paper_experiments.py

# Extended experiments: MAPPO/EWC-IPPO baselines, scaling, severity sweeps (~3–5 hours)
python run_extended_experiments.py --phase all

# Generate summary tables and significance tests
python analyze_results.py
```

For controlled compute-overhead reporting:

```bash
python timing_overhead.py
```

## Default Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Drift severity | 1.5 | `--severity` |
| Episode horizon | 150 steps | `--horizon` |
| Training episodes | 80 | `--episodes` |
| Nested-MARL rates | α₀=1e-4, α₁=5e-4, α₂=2e-3 | `agents/nested_marl.py` |
| Drift periods | medium=15, slow=50 | `drift/schedulers.py` |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ejaz2026nestedmarl,
  author  = {Ejaz, Raja Abraheem Rashid and Iradat, Faisal and Iqbal, Waqar},
  title   = {Nested Multi-Agent Reinforcement Learning for Adaptive Resource Management in 6G Network Slicing: A Multi-Timescale Framework with Convergence Guarantees},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2026},
  publisher = {IEEE}
}
```

See also [`CITATION.bib`](CITATION.bib).

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

For questions regarding this implementation, please open an issue on GitHub or contact the authors.
