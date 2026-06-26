"""
FedProx — Li et al. (2020).
Adds proximal regularisation ||w - w_global||² to limit client drift.
Communication pattern is identical to FedAvg (all layers, every round).
"""

import copy, random, time
import numpy as np
import torch
import config
from train_utils import evaluate, local_train_fedprox, weighted_average


def run_fedprox(global_model, client_loaders, test_loader, device,
                mu=config.FEDPROX_MU, seed=config.SEED):
    rng = random.Random(seed)
    weights = np.array([len(cl.dataset) for cl in client_loaders], dtype=float)
    weights /= weights.sum()

    total_params = sum(p.numel() for p in global_model.parameters())
    cum_comm = 0.0
    t0 = time.time()

    history = {
        "round":     [],
        "test_acc":  [],
        "test_loss": [],
        "comm_cost": [],
        "wall_time": [],
    }

    for r in range(1, config.NUM_ROUNDS + 1):
        sampled   = rng.sample(range(config.NUM_CLIENTS), config.CLIENTS_PER_ROUND)
        global_sd = global_model.state_dict()
        local_sds, local_w = [], []

        for cid in sampled:
            local_model = copy.deepcopy(global_model).to(device)
            sd = local_train_fedprox(local_model, global_sd,
                                      client_loaders[cid], device, mu=mu)
            local_sds.append(sd)
            local_w.append(weights[cid])

        local_w = np.array(local_w)
        local_w /= local_w.sum()

        new_sd = weighted_average(local_sds, local_w)
        global_model.load_state_dict(new_sd)

        cum_comm += 2 * total_params * config.CLIENTS_PER_ROUND

        loss, acc = evaluate(global_model, test_loader, device)
        history["round"].append(r)
        history["test_acc"].append(acc)
        history["test_loss"].append(loss)
        history["comm_cost"].append(cum_comm)
        history["wall_time"].append(time.time() - t0)

        if r % 10 == 0:
            print(f"  [FedProx] Round {r:3d} | Acc {acc:.4f} | Loss {loss:.4f} "
                  f"| CommCost {cum_comm:.2e}")

    return history
