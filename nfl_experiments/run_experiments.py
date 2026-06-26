"""
Main experiment runner for Nested Federated Learning paper.

Usage:
    python run_experiments.py                    # all experiments
    python run_experiments.py --dataset MNIST    # single dataset
    python run_experiments.py --quick            # 20 rounds, fast check

Results saved to:  ./results/<dataset>_<alpha>.pkl
Figures saved to:  ./figures/
"""

import os, sys, copy, pickle, argparse, time
import numpy as np
import torch

import config
from models     import get_model
from data       import get_data_loaders, get_client_weights
from fedavg     import run_fedavg
from fedprox    import run_fedprox
from scaffold   import run_scaffold
from nfl        import run_nfl
from unlearning import run_unlearning_experiment
import plotting


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  Using CPU (no GPU detected)")
    return dev


def run_one_setting(dataset, alpha, device, quick=False, figures_dir=config.FIGURES_DIR):
    print(f"\n{'='*65}")
    print(f"  Dataset: {dataset}  |  Alpha: {alpha}"
          f"  ({'IID' if alpha > 100 else 'Non-IID'})")
    print(f"{'='*65}")

    if quick:
        config.NUM_ROUNDS    = 20
        config.LOCAL_EPOCHS  = 2
        print("  [QUICK MODE] 20 rounds, 2 local epochs")

    set_seed(config.SEED)

    # Data
    client_loaders_iid, test_loader_iid = get_data_loaders(
        dataset, config.DIRICHLET_ALPHA_IID)
    client_loaders, test_loader = get_data_loaders(dataset, alpha)

    histories = {}

    # ── FedAvg ──────────────────────────────────────────────────────────────
    print("\n[1/4] Running FedAvg...")
    set_seed(config.SEED)
    model = get_model(dataset).to(device)
    histories["FedAvg"] = run_fedavg(model, client_loaders, test_loader, device)

    # ── FedProx ─────────────────────────────────────────────────────────────
    print("\n[2/4] Running FedProx...")
    set_seed(config.SEED)
    model = get_model(dataset).to(device)
    histories["FedProx"] = run_fedprox(model, client_loaders, test_loader, device)

    # ── SCAFFOLD ────────────────────────────────────────────────────────────
    print("\n[3/4] Running SCAFFOLD...")
    set_seed(config.SEED)
    model = get_model(dataset).to(device)
    histories["SCAFFOLD"] = run_scaffold(model, client_loaders, test_loader, device)

    # ── NFL ─────────────────────────────────────────────────────────────────
    print("\n[4/4] Running NFL (proposed)...")
    set_seed(config.SEED)
    nfl_model = get_model(dataset).to(device)
    histories["NFL"] = run_nfl(nfl_model, client_loaders, test_loader, device)

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plotting.plot_accuracy_vs_rounds(histories, dataset, alpha, figures_dir)
    plotting.plot_accuracy_vs_comm(histories, dataset, alpha, figures_dir)
    plotting.plot_loss_curves(histories, dataset, alpha, figures_dir)
    plotting.plot_comm_overhead_bar(histories, dataset, figures_dir)

    return histories, nfl_model, client_loaders, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST","CIFAR10","both"],
                        default="both")
    parser.add_argument("--quick", action="store_true",
                        help="Run 20 rounds for fast sanity check")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    device = get_device()
    datasets = (["MNIST","CIFAR10"] if args.dataset == "both"
                else [args.dataset])

    all_results = {}
    final_acc_iid    = {ds: {} for ds in datasets}
    final_acc_noniid = {ds: {} for ds in datasets}

    for dataset in datasets:

        # ── Non-IID run (α=0.5) ─────────────────────────────────────────────
        histories_noniid, nfl_model, client_loaders, test_loader = \
            run_one_setting(dataset, config.DIRICHLET_ALPHA_NONIID,
                            device, quick=args.quick)

        for algo, h in histories_noniid.items():
            final_acc_noniid[dataset][algo] = h["test_acc"][-1]

        # ── IID run (α=1000) ────────────────────────────────────────────────
        histories_iid, _, _, _ = \
            run_one_setting(dataset, config.DIRICHLET_ALPHA_IID,
                            device, quick=args.quick)

        for algo, h in histories_iid.items():
            final_acc_iid[dataset][algo] = h["test_acc"][-1]

        # ── Heterogeneity robustness figure ─────────────────────────────────
        plotting.plot_heterogeneity_robustness(
            final_acc_iid[dataset],
            final_acc_noniid[dataset],
            dataset, config.FIGURES_DIR)

        # ── Unlearning experiment (uses the NFL model from non-IID run) ──────
        print(f"\n[Unlearning] Running unlearning experiment on {dataset}...")
        set_seed(config.SEED)
        unlearn_results, _ = run_unlearning_experiment(
            nfl_model, histories_noniid["NFL"],
            client_loaders, test_loader,
            device, target_client=0)
        plotting.plot_unlearning_results(unlearn_results, config.FIGURES_DIR)

        all_results[dataset] = {
            "non_iid": histories_noniid,
            "iid":     histories_iid,
            "unlearn": unlearn_results,
        }

        # Save raw results
        out_path = os.path.join(config.RESULTS_DIR, f"{dataset}_results.pkl")
        with open(out_path, "wb") as f:
            # Drop contribution_log tensors to keep file small
            save_results = {}
            for split, hists in [("non_iid", histories_noniid),
                                  ("iid",     histories_iid)]:
                save_results[split] = {}
                for algo, h in hists.items():
                    save_results[split][algo] = {
                        k: v for k, v in h.items()
                        if k != "contribution_log"}
            save_results["unlearn"] = unlearn_results
            pickle.dump(save_results, f)
        print(f"  Results saved → {out_path}")

    # ── Final summary table ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL SUMMARY")
    print("="*65)
    for dataset in datasets:
        print(f"\n  {dataset}")
        print(f"  {'Algorithm':<12} {'IID Acc':>10} {'NonIID Acc':>12} {'Drop':>8}")
        print(f"  {'-'*45}")
        for algo in ["FedAvg","FedProx","SCAFFOLD","NFL"]:
            iid_a = final_acc_iid[dataset].get(algo, 0)
            nni_a = final_acc_noniid[dataset].get(algo, 0)
            drop  = (iid_a - nni_a) * 100
            print(f"  {algo:<12} {iid_a:>9.4f}  {nni_a:>11.4f}  {drop:>7.2f}%")

    print("\nAll done. Check ./figures/ for plots and ./results/ for raw data.")


if __name__ == "__main__":
    main()
