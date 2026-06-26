"""
Nested Federated Learning (NFL) — Proposed Method.
Implements the Federated Continuum Memory System with:
  - Multi-Frequency Layer-wise Synchronisation Protocol
  - Delta Federated Aggregation (DFA) with momentum
  - Parameter isolation for efficient unlearning

Layer groups and sync periods follow config.NFL_SYNC_PERIODS.
"""

import copy, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from train_utils import evaluate, compute_delta


# ─── Layer-group utilities ────────────────────────────────────────────────────

def _params_in_group(model, group_names):
    """Return set of parameter tensor ids belonging to named layers."""
    ids = set()
    for name in group_names:
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        ids.add(id(getattr(obj, parts[-1])))
    return ids


def _get_group_sd(state_dict, group_keys):
    """Extract subset of state_dict for a layer group."""
    return {k: state_dict[k] for k in group_keys if k in state_dict}


def _set_group_sd(state_dict, group_sd):
    """Write group parameters back into a full state dict (in-place)."""
    for k, v in group_sd.items():
        state_dict[k] = v


# ─── Local NFL training ───────────────────────────────────────────────────────

def _nfl_local_train(model, global_sd, layer_group_names,
                     loader, device, round_idx,
                     lambda_k=config.NFL_LAMBDA,
                     epochs=config.LOCAL_EPOCHS):
    """
    Local training for NFL.
    - Personalization layers: free gradient, never regularised.
    - Task / Foundation layers: proximal regularisation with λ_k.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(),
                          lr=config.LOCAL_LR,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Build param-name → λ mapping
    lambda_map = {}
    for group, names in layer_group_names.items():
        lam = lambda_k.get(group, 0.0)
        for n in names:
            lambda_map[n] = lam

    global_tensors = {k: v.clone().float().to(device)
                      for k, v in global_sd.items()}

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            # Layer-specific proximal regularisation
            for name, param in model.named_parameters():
                lam = lambda_map.get(name, 0.0)
                if lam > 0 and name in global_tensors:
                    loss += (lam / 2) * torch.norm(
                        param - global_tensors[name]) ** 2
            loss.backward()
            optimizer.step()

    return model.state_dict()


# ─── Delta Federated Aggregation (DFA) ───────────────────────────────────────

def _dfa_aggregate(global_sd, deltas, weights, momentum_buf, group_keys,
                   beta=config.DFA_BETA, eta_g=config.GLOBAL_LR):
    """
    Momentum-corrected aggregation for a single layer group.
    m_r = β * m_{r-1} + weighted_avg(Δ_i)
    w_global += η_g * m_r
    """
    # Weighted average of client deltas for this group
    avg_delta = {}
    for k in group_keys:
        avg_delta[k] = sum(w * d[k].float()
                           for w, d in zip(weights, deltas))

    # Momentum update
    for k in group_keys:
        if k not in momentum_buf:
            momentum_buf[k] = torch.zeros_like(avg_delta[k])
        momentum_buf[k] = beta * momentum_buf[k] + avg_delta[k]

    # Apply to global model
    new_sd = {}
    for k in group_keys:
        new_sd[k] = global_sd[k].float() + eta_g * momentum_buf[k]
    return new_sd


# ─── Main NFL loop ────────────────────────────────────────────────────────────

def run_nfl(global_model, client_loaders, test_loader, device, seed=config.SEED):
    rng = random.Random(seed)
    weights = np.array([len(cl.dataset) for cl in client_loaders], dtype=float)
    weights /= weights.sum()

    # Layer group structure from the model
    group_names = global_model.layer_group_names()   # {group: [param_names]}
    all_keys    = set(global_model.state_dict().keys())

    # Sync periods
    sync_periods = config.NFL_SYNC_PERIODS  # {group: U_k}

    # Param-size per group (for comm cost accounting)
    group_param_count = {}
    for group, names in group_names.items():
        group_param_count[group] = sum(
            global_model.state_dict()[k].numel()
            for k in names if k in global_model.state_dict())

    total_params = sum(p.numel() for p in global_model.parameters())

    # Momentum buffers (DFA)
    momentum_buf = {}

    # Contribution tracking for unlearning (task layer only)
    # contribution_log[cid] = list of (round, delta_dict) for task group
    contribution_log = {cid: [] for cid in range(config.NUM_CLIENTS)}

    cum_comm = 0.0
    t0 = time.time()

    history = {
        "round":               [],
        "test_acc":            [],
        "test_loss":           [],
        "comm_cost":           [],
        "wall_time":           [],
        "active_groups":       [],   # which groups synced this round
        "contribution_log":    contribution_log,  # for unlearning experiment
    }

    for r in range(1, config.NUM_ROUNDS + 1):
        # Determine active groups K_r = {k : r mod U_k == 0}
        active_groups = []
        for group, U_k in sync_periods.items():
            if U_k == float("inf"):
                continue  # personalization: never sync
            if r % int(U_k) == 0:
                active_groups.append(group)

        sampled   = rng.sample(range(config.NUM_CLIENTS), config.CLIENTS_PER_ROUND)
        global_sd = global_model.state_dict()

        # ── Local training ─────────────────────────────────────────────────
        local_sds, local_w = [], []
        for cid in sampled:
            local_model = copy.deepcopy(global_model).to(device)
            sd = _nfl_local_train(local_model, global_sd, group_names,
                                  client_loaders[cid], device, r)
            local_sds.append(sd)
            local_w.append(weights[cid])

        local_w = np.array(local_w)
        local_w /= local_w.sum()

        # ── Compute deltas per client ───────────────────────────────────────
        deltas = []
        for sd in local_sds:
            d = compute_delta(global_sd, sd)
            deltas.append(d)

        # ── Log task-layer contributions for unlearning ────────────────────
        for idx, cid in enumerate(sampled):
            task_keys = group_names["task"]
            task_delta = {k: deltas[idx][k].clone() for k in task_keys
                          if k in deltas[idx]}
            contribution_log[cid].append((r, task_delta))

        # ── Selective aggregation (only active groups) ─────────────────────
        new_sd = copy.deepcopy(global_sd)
        round_comm = 0

        for group in active_groups:
            gkeys = [k for k in group_names[group]
                     if k in global_sd]
            group_deltas = [{k: d[k] for k in gkeys if k in d}
                            for d in deltas]
            updated_group = _dfa_aggregate(
                global_sd, group_deltas, local_w,
                momentum_buf, gkeys)
            _set_group_sd(new_sd, updated_group)

            # Communication: upload + download for this group, sampled clients
            round_comm += (2 * group_param_count[group]
                           * config.CLIENTS_PER_ROUND)

        global_model.load_state_dict(new_sd)
        cum_comm += round_comm

        # ── Evaluate ───────────────────────────────────────────────────────
        loss, acc = evaluate(global_model, test_loader, device)
        history["round"].append(r)
        history["test_acc"].append(acc)
        history["test_loss"].append(loss)
        history["comm_cost"].append(cum_comm)
        history["wall_time"].append(time.time() - t0)
        history["active_groups"].append(active_groups)

        if r % 10 == 0:
            sync_str = ", ".join(active_groups) if active_groups else "local only"
            print(f"  [NFL] Round {r:3d} | Acc {acc:.4f} | Loss {loss:.4f} "
                  f"| Sync: [{sync_str}] | CommCost {cum_comm:.2e}")

    return history
