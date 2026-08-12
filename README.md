# Nested-MARL for 6G Network Slicing

[![Paper](https://img.shields.io/badge/IEEE-OJCOMS%202026-blue)](https://github.com/Abraheem13/Nested-MARL-6G-Slicing)
[![Access](https://img.shields.io/badge/code%20access-on%20request-orange)](mailto:2546393@brunel.ac.uk)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)](LICENSE)

Official repository for:

> **Nested Multi-Agent Reinforcement Learning for Adaptive Resource Management in 6G Network Slicing: A Multi-Timescale Framework with Convergence Guarantees**
> Abraheem Rashid, Faisal Iradat, Waseem Iqbal, Ikram Syed, Khalil Khan, and Khalid Yahya.
> *IEEE Open Journal of the Communications Society*, 2026.

---

## 🔒 Code and Data Access

**The source code, experimental data, and figure-generation scripts in this project are not publicly distributed.**

Access is granted on request, at the authors' discretion, for **non-commercial academic research and reproducibility verification**.

### How to request access

Email **[2546393@brunel.ac.uk](mailto:2546393@brunel.ac.uk)** with the subject line:

```
Nested-MARL-6G-Slicing — Access Request
```

Please include:

1. **Your full name and GitHub username**
2. **Your institutional affiliation** and academic email address
3. **Your intended use** — a short description of what you plan to do with the code
4. **Confirmation** that use will be non-commercial and that the paper will be cited in any resulting work

Requests are typically reviewed within a few working days. Approved requests receive read-only access to the private repository containing the full implementation, pre-computed results, and reproduction scripts.

> **Note:** Access is granted to individuals, not institutions, and may not be forwarded, mirrored, or redistributed. See [LICENSE](LICENSE) for the full terms.

---

## Overview

We study adaptive multi-agent resource management in a 6G network-slicing setting with multi-timescale non-stationarity. **Nested-MARL** assigns each agent's parameters to three groups updated at distinct learning rates matched to fast, medium, and slow environmental drift, together with an exponential moving-average memory term for stability.

The full (private) codebase includes:

- A 3-agent 6G slicing simulator (eMBB / URLLC / mMTC)
- Multi-timescale drift schedulers
- Nested-MARL and baselines (IPPO, MAPPO, EWC-IPPO)
- Ablation variants (no EMA, no timescale separation, two-level nesting)
- Pre-computed experimental logs and scripts to regenerate all paper figures

## Method Summary

Each agent's parameter vector is partitioned into three groups updated at separate learning rates:

| Group | Learning rate | Tracks |
|-------|---------------|--------|
| Slow | α₀ = 1e-4 | Long-horizon structural drift |
| Medium | α₁ = 5e-4 | Intermediate regime shifts |
| Fast | α₂ = 2e-3 | Rapid traffic fluctuation |

An exponential moving-average memory term stabilises the slow groups against fast-timescale noise, yielding the convergence guarantees established in the paper.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Drift severity | 1.5 |
| Episode horizon | 150 steps |
| Training episodes | 80 |
| Drift periods | medium = 15, slow = 50 |
| Slices | eMBB, URLLC, mMTC |
| Baselines | IPPO, MAPPO, EWC-IPPO |

## Results Reported in the Paper

The paper reports the following figures and tables, reproducible from the private repository:

| Artifact | Description |
|----------|-------------|
| `fig_learning_curves` | Episode reward vs. training |
| `fig_cumulative_regret` | Cumulative regret vs. oracle |
| `fig_ablation_bars` | Ablation comparison |
| `fig_per_slice_sla` | Per-slice SLA satisfaction |
| `fig_switching_cost` | Policy stability over training |
| `fig_timescale_analysis` | Adaptation after regime change |
| `fig_severity_sweep` | Performance vs. drift severity |
| `fig_drift_schedule` | Illustrative drift schedule |
| `fig_system_model` | System architecture schematic |
| `table_results` | Main results table |

## Requirements

Granted users will need:

- Python 3.9+
- PyTorch
- NumPy, Matplotlib, Pandas, PyYAML

Core paper experiments run in roughly one hour on a laptop CPU; the extended experiment suite takes approximately three to five hours.

## Citation

If you reference this work, please cite:

```bibtex
@article{rashid2026nestedmarl,
  author    = {Rashid, Abraheem and Iradat, Faisal and Iqbal, Waseem and Syed, Ikram and Khan, Khalil and Yahya, Khalid},
  title     = {Nested Multi-Agent Reinforcement Learning for Adaptive Resource Management in 6G Network Slicing: A Multi-Timescale Framework with Convergence Guarantees},
  journal   = {IEEE Open Journal of the Communications Society},
  year      = {2026},
  volume    = {},
  pages     = {},
  doi       = {},
  publisher = {IEEE}
}
```

See also [`CITATION.bib`](CITATION.bib).

## License

**All Rights Reserved.** This work is not open source. No permission is granted to use, copy, modify, or distribute the code or data without prior written consent from the authors. See [LICENSE](LICENSE) for full terms.

## Contact

**Abraheem Rashid** — [2546393@brunel.ac.uk](mailto:2546393@brunel.ac.uk)
Brunel University London

For access requests, please follow the procedure in [Code and Data Access](#-code-and-data-access) above.
