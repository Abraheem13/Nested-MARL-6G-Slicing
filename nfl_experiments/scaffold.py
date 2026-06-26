"""
SCAFFOLD — Karimireddy et al. (2020).
Uses control variates (c_i, c) to correct client drift.
Communication cost is 2× FedAvg (model + control variate per round).
"""

import copy, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from train_utils import evaluate, weighted_average


def _scaffold_local_train(model, global_sd, c_global, c_local,
                           loader, device, epochs=config.LOCAL_EPOCHS):
    """One round of SCAFFOLD local update."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.LOCAL_LR)
    criterion = nn.CrossEntropyLoss()

    # Flatten control variates for easy arithmetic
    c_g = {k: c_global[k].to(device) for k in c_global}
    c_l = {k: c_local[k].to(device)  for k in c_local}

    n_steps = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            # Correction step: g_i ← g_i - c_i + c
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad += c_g[name] - c_l[name]
            optimizer.step()
            n_steps += 1

    # Update local control variate
    # c_i^+ = c_i - c + (1/(K*lr)) * (x - y_i)
    new_c_local = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            new_c_local[name] = (
                c_l[name]
                - c_g[name]
                + (1.0 / (n_steps * config.LOCAL_LR))
                  * (global_sd[name].float().to(device) - param.float())
            ).cpu()

    return model.state_dict(), new_c_local


def run_scaffold(global_model, client_loaders, test_loader, device, seed=config.SEED):
    rng = random.Random(seed)
    weights = np.array([len(cl.dataset) for cl in client_loaders], dtype=float)
    weights /= weights.sum()

    total_params = sum(p.numel() for p in global_model.parameters())
    cum_comm = 0.0
    t0 = time.time()

    ## Initialise control variates — trainable params only (excludes BN buffers)
    param_keys = {k for k, _ in global_model.named_parameters()}
    global_sd  = global_model.state_dict()
    c_global   = {k: torch.zeros_like(v, dtype=torch.float32)
                  for k, v in global_sd.items() if k in param_keys}
    c_locals   = [{k: torch.zeros_like(v, dtype=torch.float32)
                   for k, v in global_sd.items() if k in param_keys}
                  for _ in range(config.NUM_CLIENTS)]

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
        local_sds, delta_c_list, local_w = [], [], []

        for cid in sampled:
            local_model = copy.deepcopy(global_model).to(device)
            new_sd, new_c_local = _scaffold_local_train(
                local_model, global_sd,
                c_global, c_locals[cid],
                client_loaders[cid], device)

            # Delta control variate
            delta_c = {k: new_c_local[k] - c_locals[cid][k]
                       for k in c_global if k in new_c_local}
            c_locals[cid] = new_c_local

            local_sds.append(new_sd)
            delta_c_list.append(delta_c)
            local_w.append(weights[cid])

        local_w = np.array(local_w)
        local_w /= local_w.sum()

        # Aggregate model
        new_sd = weighted_average(local_sds, local_w)
        global_model.load_state_dict(new_sd)

        # Update global control variate
        n = config.NUM_CLIENTS
        for k in c_global:
            c_global[k] += sum(dc[k] for dc in delta_c_list) / n

        # SCAFFOLD sends model + control variate (2× params)
        cum_comm += 4 * total_params * config.CLIENTS_PER_ROUND

        loss, acc = evaluate(global_model, test_loader, device)
        history["round"].append(r)
        history["test_acc"].append(acc)
        history["test_loss"].append(loss)
        history["comm_cost"].append(cum_comm)
        history["wall_time"].append(time.time() - t0)

        if r % 10 == 0:
            print(f"  [SCAFFOLD] Round {r:3d} | Acc {acc:.4f} | Loss {loss:.4f} "
                  f"| CommCost {cum_comm:.2e}")

    return history
