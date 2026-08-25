"""
FedNoisyMAML: Federated Meta-Learning Under Heterogeneous Label Noise
Main experiment runner — multi-seed with statistical tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import copy
import torch
import numpy as np
from math import comb
from collections import defaultdict
from src.models import ClassifierMLP, ConvNet
from src.datasets import create_noisy_federation
from src.algorithms import FedMAML, FedAvg
import torch.nn.functional as F


DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def load_config(config_path=None):
    """Load config from YAML file, falling back to defaults."""
    config = {
        "n_clients": 10,
        "n_classes": 6,
        "n_rounds": 150,
        "noise_max": 0.5,
        "seeds": DEFAULT_SEEDS,
        "inner_steps": 3,
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "local_epochs": 3,
        "hidden_size": 64,
        "n_per_class": 120,
        "dataset": "gaussian",
        "dirichlet_alpha": 0.5,
        "device": "cpu",
        "output_dir": "./results",
    }
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            config.update(user_cfg)
    return config


def run_single(n_clients, n_classes, n_rounds, noise_max, seed,
               dataset="gaussian", device="cpu", cfg=None):
    """Run one seed. Returns per-client accuracies for both methods."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = cfg or {}

    noise_rates = np.linspace(0.0, noise_max, n_clients)

    if dataset == "gaussian":
        clients = create_noisy_federation(
            n_clients=n_clients, noise_rates=noise_rates,
            n_per_class=cfg.get("n_per_class", 120),
            n_classes=n_classes, seed=seed)
        model_maml = ClassifierMLP(
            input_dim=8, hidden_size=cfg.get("hidden_size", 64),
            n_classes=n_classes).to(device)
        model_avg = ClassifierMLP(
            input_dim=8, hidden_size=cfg.get("hidden_size", 64),
            n_classes=n_classes).to(device)
    else:
        from src.image_data import create_noisy_image_federation, DATASET_META
        clients, _ = create_noisy_image_federation(
            dataset_name=dataset, n_clients=n_clients,
            noise_rates=noise_rates,
            alpha=cfg.get("dirichlet_alpha", 0.5), seed=seed)
        meta = DATASET_META[dataset]
        n_classes = meta["n_classes"]
        model_maml = ConvNet(**meta).to(device)
        model_avg = ConvNet(**meta).to(device)

    fed_maml = FedMAML(model_maml, clients,
                       outer_lr=cfg.get("outer_lr", 0.001),
                       inner_lr=cfg.get("inner_lr", 0.01),
                       inner_steps=cfg.get("inner_steps", 3))
    fed_avg = FedAvg(model_avg, clients,
                     lr=cfg.get("outer_lr", 0.01),
                     local_epochs=cfg.get("local_epochs", 3))

    for rnd in range(n_rounds):
        fed_maml.train_round(n_support=15, n_query=15)
        fed_avg.train_round(n_samples=30)

    maml_accs = fed_maml.evaluate(n_test=40)
    avg_accs = fed_avg.evaluate(n_test=40)

    return noise_rates, maml_accs, avg_accs


def sign_test_pvalue(diffs):
    """Two-sided sign test p-value."""
    n_pos = int(np.sum(np.array(diffs) > 0))
    n_neg = int(np.sum(np.array(diffs) < 0))
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    min_sign = min(n_pos, n_neg)
    p = 2 * sum(comb(n, k) * 0.5**n for k in range(min_sign + 1))
    return min(p, 1.0)


def run_experiment(n_clients=None, n_classes=None, n_rounds=None, noise_max=None,
                   seed=None, dataset=None, device=None, config_path=None):
    """Multi-seed experiment comparing FedMAML vs FedAvg under label noise.

    Returns:
        True on success (for test compatibility)
    """
    cfg = load_config(config_path)
    n_clients = cfg["n_clients"] if n_clients is None else n_clients
    n_classes = cfg["n_classes"] if n_classes is None else n_classes
    n_rounds = cfg["n_rounds"] if n_rounds is None else n_rounds
    noise_max = cfg["noise_max"] if noise_max is None else noise_max
    seeds = [seed] if seed is not None else cfg["seeds"]
    dataset = dataset or cfg.get("dataset", "gaussian")
    device = device or cfg.get("device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    print(f"\n{'='*70}")
    print(f"  FedNoisyMAML | {n_clients} clients, {n_classes} classes, "
          f"noise 0-{noise_max:.0%}, {n_rounds} rounds")
    print(f"  Dataset: {dataset}  |  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*70}")

    noise_rates = np.linspace(0.0, noise_max, n_clients)

    # Collect per-seed results
    all_maml = []  # shape: (n_seeds, n_clients)
    all_avg = []

    for s in seeds:
        _, maml_accs, avg_accs = run_single(
            n_clients, n_classes, n_rounds, noise_max, s,
            dataset=dataset, device=device, cfg=cfg)
        all_maml.append(maml_accs)
        all_avg.append(avg_accs)
        print(f"  seed={s}: MAML={np.mean(maml_accs):.3f}  "
              f"FedAvg={np.mean(avg_accs):.3f}  "
              f"delta={np.mean(maml_accs)-np.mean(avg_accs):+.3f}")

    all_maml = np.array(all_maml)  # (n_seeds, n_clients)
    all_avg = np.array(all_avg)

    # ---- Table 1: Per noise-level accuracy (mean +- std across seeds) ----
    print(f"\n--- Table 1: Per-client accuracy (mean+-std over {len(seeds)} seeds) ---")
    print(f"{'Noise':>8} {'FedMAML':>16} {'FedAvg':>16} {'Delta':>12} {'p-val':>8}")
    print("-" * 66)

    for i in range(n_clients):
        maml_vals = all_maml[:, i]
        avg_vals = all_avg[:, i]
        delta = maml_vals - avg_vals
        p = sign_test_pvalue(delta)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"{noise_rates[i]:>8.1%} "
              f"{np.mean(maml_vals):.3f}+-{np.std(maml_vals):.3f}  "
              f"{np.mean(avg_vals):.3f}+-{np.std(avg_vals):.3f}  "
              f"{np.mean(delta):>+8.3f}  {p:>6.3f} {sig}")

    # Overall row
    maml_means = all_maml.mean(axis=1)  # per-seed global mean
    avg_means = all_avg.mean(axis=1)
    overall_delta = maml_means - avg_means
    p_overall = sign_test_pvalue(overall_delta)
    sig = "***" if p_overall < 0.01 else "**" if p_overall < 0.05 else "*" if p_overall < 0.1 else ""
    print("-" * 66)
    print(f"{'Overall':>8} "
          f"{np.mean(maml_means):.3f}+-{np.std(maml_means):.3f}  "
          f"{np.mean(avg_means):.3f}+-{np.std(avg_means):.3f}  "
          f"{np.mean(overall_delta):>+8.3f}  {p_overall:>6.3f} {sig}")

    # ---- Table 2: Low vs high noise robustness ----
    print(f"\n--- Table 2: Robustness (low vs high noise) ---")
    low_idx = [i for i in range(n_clients) if noise_rates[i] < 0.2]
    high_idx = [i for i in range(n_clients) if noise_rates[i] >= 0.2]

    for label, idx in [("Low (<20%)", low_idx), ("High (>=20%)", high_idx)]:
        if not idx:
            continue
        maml_group = all_maml[:, idx].mean(axis=1)  # per-seed
        avg_group = all_avg[:, idx].mean(axis=1)
        delta_group = maml_group - avg_group
        p = sign_test_pvalue(delta_group)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {label:>12}: MAML={np.mean(maml_group):.3f}+-{np.std(maml_group):.3f}  "
              f"FedAvg={np.mean(avg_group):.3f}+-{np.std(avg_group):.3f}  "
              f"delta={np.mean(delta_group):+.3f} p={p:.3f} {sig}")

    # ---- Table 3: Degradation slopes ----
    print(f"\n--- Table 3: Accuracy degradation slope (linear fit, acc vs noise) ---")
    maml_slopes = []
    avg_slopes = []
    for s in range(len(seeds)):
        ms = np.polyfit(noise_rates, all_maml[s], 1)[0]
        avs = np.polyfit(noise_rates, all_avg[s], 1)[0]
        maml_slopes.append(ms)
        avg_slopes.append(avs)

    print(f"  FedMAML slope: {np.mean(maml_slopes):.3f} +- {np.std(maml_slopes):.3f}")
    print(f"  FedAvg  slope: {np.mean(avg_slopes):.3f} +- {np.std(avg_slopes):.3f}")
    slope_diff = np.array(maml_slopes) - np.array(avg_slopes)
    print(f"  Diff (flatter=better): {np.mean(slope_diff):+.3f} "
          f"(MAML {'flatter' if np.mean(slope_diff) > 0 else 'steeper'})")

    # Save JSON results
    out_dir = cfg.get("output_dir", "./results")
    os.makedirs(out_dir, exist_ok=True)
    results_json = {
        "config": {"n_clients": n_clients, "n_classes": n_classes,
                   "n_rounds": n_rounds, "noise_max": noise_max,
                   "dataset": dataset, "seeds": seeds},
        "overall_maml": float(np.mean(maml_means)),
        "overall_fedavg": float(np.mean(avg_means)),
        "overall_delta": float(np.mean(overall_delta)),
        "overall_p": float(p_overall),
    }
    out_path = os.path.join(out_dir, f"fednoisy_maml_{dataset}_results.json")
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedNoisyMAML Experiment")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--n_clients", type=int, default=None)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--n_rounds", type=int, default=None)
    parser.add_argument("--noise_max", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: gaussian, cifar10, mnist")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu or cuda")
    args = parser.parse_args()
    run_experiment(n_clients=args.n_clients, n_classes=args.n_classes,
                   n_rounds=args.n_rounds, noise_max=args.noise_max,
                   dataset=args.dataset, device=args.device,
                   config_path=args.config)
