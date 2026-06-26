# Nested Federated Learning — Experiments

Complete reproducible experiment suite for the paper:
**"Nested Federated Learning: Layer-Wise Multi-Frequency Synchronization
for Privacy-Preserving Distributed Intelligence"**

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run All Experiments (recommended)

```bash
python run_experiments.py
```

This runs all 4 algorithms (FedAvg, FedProx, SCAFFOLD, NFL) on both
MNIST and CIFAR-10, under IID and Non-IID (α=0.5) conditions,
plus the unlearning experiment.

**Expected runtime:**
- GPU (RTX 3080 / A100): ~30–60 minutes
- CPU only: ~3–5 hours

---

## Quick Sanity Check (20 rounds)

```bash
python run_experiments.py --quick
```

---

## Single Dataset

```bash
python run_experiments.py --dataset MNIST
python run_experiments.py --dataset CIFAR10
```

---

## Generate LaTeX Tables

After experiments finish:

```bash
python generate_tables.py
```

Tables written to `./results/tables.tex` — include in your paper with:
```latex
\input{results/tables.tex}
```

---

## Output

| Path | Contents |
|------|----------|
| `./figures/` | All publication-quality figures (PDF + PNG) |
| `./results/MNIST_results.pkl` | Raw results for MNIST |
| `./results/CIFAR10_results.pkl` | Raw results for CIFAR-10 |
| `./results/tables.tex` | Auto-generated LaTeX tables |

---

## Figures Produced

| File | Description |
|------|-------------|
| `accuracy_rounds_<ds>_alpha<a>` | Test accuracy vs communication rounds |
| `accuracy_comm_<ds>_alpha<a>` | Test accuracy vs cumulative comm cost |
| `loss_rounds_<ds>_alpha<a>` | Test loss curves |
| `comm_overhead_<ds>` | Total communication cost bar chart |
| `heterogeneity_<ds>` | IID vs Non-IID accuracy grouped bar |
| `unlearning_results` | 3-panel unlearning evaluation figure |

---

## Key Configuration (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLIENTS` | 20 | Total federated clients |
| `CLIENTS_PER_ROUND` | 10 | Sampled per round |
| `NUM_ROUNDS` | 100 | Global training rounds |
| `DIRICHLET_ALPHA_NONIID` | 0.5 | Non-IID heterogeneity level |
| `NFL_SYNC_PERIODS` | {task:10, foundation:50} | Multi-frequency schedule |

---

## Reproducibility

All runs use `SEED=42`. Results are deterministic given same hardware.
