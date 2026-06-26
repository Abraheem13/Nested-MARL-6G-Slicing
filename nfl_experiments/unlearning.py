"""
Nested Federated Unlearning experiment.
Demonstrates near-zero-cost client removal using the FCMS architecture:

  Layer 0 (personalization) — local only  → cost = 0 (just delete local file)
  Layer 1 (task)            — tracked     → rollback Δ contribution
  Layer K (foundation)      — averaged    → statistically negligible, no action

We measure:
  1. Unlearning time (wall-clock seconds)
  2. Model accuracy before and after unlearning
  3. Membership Inference Attack (MIA) success rate on unlearned client
     (lower = better unlearning; chance level ≈ 0.5)
"""

import copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import config
from train_utils import evaluate


# ─── Rollback Task-Layer Contribution ────────────────────────────────────────

def nfl_unlearn_client(global_model, contribution_log, client_id,
                       group_names, weights, device):
    """
    Remove client `client_id` from the global model by subtracting
    its accumulated task-layer deltas (weighted by its data proportion).

    Returns: (new_model, unlearn_time_seconds)
    """
    t0 = time.time()
    new_model = copy.deepcopy(global_model)
    new_sd    = new_model.state_dict()

    task_keys = [k for k in group_names["task"]
                 if k in new_sd]

    client_weight = weights[client_id]

    for (rnd, delta_dict) in contribution_log[client_id]:
        for k in task_keys:
            if k in delta_dict:
                # Subtract the client's weighted contribution
                new_sd[k] = (new_sd[k].float()
                             - client_weight * delta_dict[k].float().to(device))

    new_model.load_state_dict(new_sd)
    unlearn_time = time.time() - t0
    return new_model, unlearn_time


# ─── FedAvg unlearning baseline (full retrain) ─────────────────────────────-

def fedavg_unlearn_retrain_cost(num_rounds, num_clients, total_params):
    """
    FedAvg unlearning requires full retraining from scratch.
    Returns theoretical communication cost (in param units).
    """
    # 2 × params × clients_per_round × rounds
    return 2 * total_params * config.CLIENTS_PER_ROUND * num_rounds


# ─── Simple Membership Inference Attack ───────────────────────────────────────

def membership_inference_attack(model, member_loader, non_member_loader,
                                device, n_samples=200):
    """
    Threshold-based MIA: if loss on a sample < threshold → predict 'member'.
    Returns attack accuracy (0.5 = random / no information leakage).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    def get_losses(loader, n):
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                l = criterion(model(x), y).cpu().numpy()
                losses.extend(l.tolist())
                if len(losses) >= n:
                    break
        return np.array(losses[:n])

    member_losses     = get_losses(member_loader, n_samples)
    non_member_losses = get_losses(non_member_loader, n_samples)

    # Optimal threshold = midpoint of means
    threshold = (member_losses.mean() + non_member_losses.mean()) / 2

    # Predict member if loss < threshold
    tp = (member_losses     < threshold).mean()   # true positive rate
    tn = (non_member_losses >= threshold).mean()  # true negative rate
    attack_acc = 0.5 * (tp + tn)
    return float(attack_acc)


# ─── Full Unlearning Experiment ───────────────────────────────────────────────

def run_unlearning_experiment(global_model, nfl_history,
                              client_loaders, test_loader,
                              device, target_client=0):
    """
    Run the full unlearning evaluation and return a results dict.
    """
    from data import get_client_weights
    weights = get_client_weights(client_loaders)

    group_names = global_model.layer_group_names()
    contribution_log = nfl_history["contribution_log"]
    total_params = sum(p.numel() for p in global_model.parameters())

    results = {}

    # ── 1. Accuracy before unlearning ──────────────────────────────────────
    loss_before, acc_before = evaluate(global_model, test_loader, device)
    results["acc_before"] = acc_before
    results["loss_before"] = loss_before

    # ── 2. NFL unlearning (rollback task delta) ────────────────────────────
    unlearned_model, nfl_time = nfl_unlearn_client(
        global_model, contribution_log, target_client,
        group_names, weights, device)
    results["nfl_unlearn_time_s"]  = nfl_time

    loss_after, acc_after = evaluate(unlearned_model, test_loader, device)
    results["acc_after_nfl"]  = acc_after
    results["loss_after_nfl"] = loss_after

    # ── 3. FedAvg retraining cost (theoretical) ────────────────────────────
    retrain_comm = fedavg_unlearn_retrain_cost(
        config.NUM_ROUNDS, config.NUM_CLIENTS, total_params)
    results["fedavg_retrain_comm"] = retrain_comm

    if contribution_log[target_client]:
        nfl_unlearn_comm = sum(
            v.numel()
            for v in contribution_log[target_client][0][1].values()
        )
    else:
        nfl_unlearn_comm = 0
    results["nfl_unlearn_comm"] = nfl_unlearn_comm

    # ── 4. MIA on target client (before vs after unlearning) ───────────────
    # Use the client's own loader as 'member' and another client as 'non-member'
    non_target = (target_client + 1) % config.NUM_CLIENTS

    mia_before = membership_inference_attack(
        global_model,
        client_loaders[target_client],
        client_loaders[non_target],
        device)
    results["mia_before"] = mia_before

    mia_after = membership_inference_attack(
        unlearned_model,
        client_loaders[target_client],
        client_loaders[non_target],
        device)
    results["mia_after"] = mia_after

    # ── 5. Summary ──────────────────────────────────────────────────────────
    results["acc_drop_pct"] = (acc_before - acc_after) * 100
    results["mia_reduction"] = (mia_before - mia_after)

    print("\n" + "="*60)
    print("  UNLEARNING EXPERIMENT RESULTS")
    print("="*60)
    print(f"  Global acc before unlearning : {acc_before:.4f}")
    print(f"  Global acc after  unlearning : {acc_after:.4f}  "
          f"(drop: {results['acc_drop_pct']:.2f}%)")
    print(f"  NFL unlearning wall-time     : {nfl_time*1000:.2f} ms")
    print(f"  MIA accuracy before          : {mia_before:.4f}")
    print(f"  MIA accuracy after           : {mia_after:.4f}  "
          f"(closer to 0.5 = better)")
    print(f"  FedAvg retrain comm cost     : {retrain_comm:.2e} params")
    print(f"  NFL unlearn comm cost        : {nfl_unlearn_comm:.2e} params  (local only)")
    print("="*60)

    return results, unlearned_model
