"""
Shared training utilities: local train, global evaluate, copy/add state dicts.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import config


def evaluate(model, loader, device):
    """Return (loss, accuracy) on the provided DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total


def local_train_fedavg(model, loader, device, epochs=config.LOCAL_EPOCHS):
    """Standard local SGD for FedAvg / FedProx-baseline."""
    model.train()
    optimizer = optim.SGD(model.parameters(),
                          lr=config.LOCAL_LR,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return model.state_dict()


def local_train_fedprox(model, global_sd, loader, device,
                        mu=config.FEDPROX_MU,
                        epochs=config.LOCAL_EPOCHS):
    """Local SGD with proximal term ||w - w_global||²."""
    global_params = {k: v.clone().to(device) for k, v in global_sd.items()}
    model.train()
    optimizer = optim.SGD(model.parameters(),
                          lr=config.LOCAL_LR,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            # Proximal regularisation
            for name, param in model.named_parameters():
                loss += (mu / 2) * torch.norm(param - global_params[name])**2
            loss.backward()
            optimizer.step()
    return model.state_dict()


def weighted_average(state_dicts, weights):
    """
    Federated weighted averaging of state dicts.
    weights must sum to 1.
    """
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        avg[key] = sum(w * sd[key].float()
                       for w, sd in zip(weights, state_dicts))
    return avg


def compute_delta(before_sd, after_sd):
    """Return Δ = after - before for every parameter."""
    delta = {}
    for k in before_sd:
        delta[k] = after_sd[k].float() - before_sd[k].float()
    return delta


def apply_delta(sd, delta, step=1.0):
    """sd = sd + step * delta  (in-place)."""
    new_sd = copy.deepcopy(sd)
    for k in new_sd:
        new_sd[k] = new_sd[k].float() + step * delta[k]
    return new_sd
