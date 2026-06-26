"""
FedAvg baseline — McMahan et al. (2017).
All layers synchronise every round.
"""

import copy, random, time
import numpy as np
import torch
import config
from train_utils import evaluate, local_train_fedavg, weighted_average


def run_fedavg(global_model, client_loaders, test_loader, device, seed=config.SEED):
    rng = random.Random(seed)
    weights = np.array([len(cl.dataset) for cl in client_loaders], dtype=float)
    weights /= weights.sum()

    history = {
        "round":        [],
        "test_acc":     [],
        "test_loss":    [],
        "comm_cost":    [],   # cumulative bytes (relative units)
        "wall_time":    [],
    }

    # Model size in parameter count (used as proxy for comm cost)
    total_params = sum(p.numel() for p in global_model.parameters())
    cum_comm = 0.0
    t0 = time.time()

    for r in range(1, config.NUM_ROUNDS + 1):
        # Sample clients
        sampled = rng.sample(range(config.NUM_CLIENTS), config.CLIENTS_PER_ROUND)
        local_sds, local_w = [], []

        for cid in sampled:
            local_model = copy.deepcopy(global_model).to(device)
            sd = local_train_fedavg(local_model, client_loaders[cid], device)
            local_sds.append(sd)
            local_w.append(weights[cid])

        # Normalise weights of sampled clients
        local_w = np.array(local_w)
        local_w /= local_w.sum()

        # Aggregate
        new_sd = weighted_average(local_sds, local_w)
        global_model.load_state_dict(new_sd)

        # Communication: upload + download of full model for each sampled client
        cum_comm += 2 * total_params * config.CLIENTS_PER_ROUND

        loss, acc = evaluate(global_model, test_loader, device)
        history["round"].append(r)
        history["test_acc"].append(acc)
        history["test_loss"].append(loss)
        history["comm_cost"].append(cum_comm)
        history["wall_time"].append(time.time() - t0)

        if r % 10 == 0:
            print(f"  [FedAvg] Round {r:3d} | Acc {acc:.4f} | Loss {loss:.4f} "
                  f"| CommCost {cum_comm:.2e}")

    return history
