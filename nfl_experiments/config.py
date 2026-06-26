"""
Nested Federated Learning — Experiment Configuration
All hyperparameters for reproducibility.
"""

# ─── Federated Setup ──────────────────────────────────────────────────────────
NUM_CLIENTS        = 20       # total participating clients
CLIENTS_PER_ROUND  = 10       # sampled each round (50%)
NUM_ROUNDS         = 100      # global communication rounds
LOCAL_EPOCHS       = 5        # local SGD epochs per round
LOCAL_BATCH_SIZE   = 32

# ─── Optimisation ─────────────────────────────────────────────────────────────
LOCAL_LR           = 0.01     # client-side SGD learning rate
GLOBAL_LR          = 1.0      # server-side aggregation step
MOMENTUM           = 0.9      # SGD momentum
WEIGHT_DECAY       = 1e-4

# ─── Non-IID Control ──────────────────────────────────────────────────────────
# Dirichlet concentration: lower = more heterogeneous
DIRICHLET_ALPHA_IID    = 1000.0   # effectively IID
DIRICHLET_ALPHA_NONIID = 0.5      # heterogeneous (used in paper)

# ─── NFL — Federated Continuum Memory System ──────────────────────────────────
# Synchronisation periods U_k per layer group
# Layer 0 (personalization)  → U=inf (never sync)
# Layer 1 (task)             → every 10 rounds
# Layer 2 (foundation)       → every 50 rounds
NFL_SYNC_PERIODS = {
    "personalization": float("inf"),   # local only
    "task":            10,
    "foundation":      50,
}

# Stability coefficients λ_k  (0 = free, large = strict)
NFL_LAMBDA = {
    "personalization": 0.0,
    "task":            0.01,
    "foundation":      0.1,
}

# DFA momentum coefficient β
DFA_BETA = 0.9

# ─── FedProx ──────────────────────────────────────────────────────────────────
FEDPROX_MU = 0.01             # proximal regularisation coefficient

# ─── SCAFFOLD ─────────────────────────────────────────────────────────────────
SCAFFOLD_LR = 0.01

# ─── Datasets ─────────────────────────────────────────────────────────────────
DATASETS   = ["MNIST", "CIFAR10"]
DATA_ROOT  = "./data"

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
