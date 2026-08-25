"""FairFedMeta experiment runner.

Usage:
    python run_experiment.py --config configs/default.yaml
    python run_experiment.py --config configs/default.yaml algorithm.name=fedavg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import copy
import logging
from pathlib import Path

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import seed_everything, load_config, get_device, setup_logging
from src.datasets.eicu import EICUFederatedLoader, FairFederatedDataset
from src.models.clinical import ClinicalLSTM, ClinicalGRU
from src.algorithms.fair_maml import FairMAML
from src.algorithms.fair_fedmaml import FairFedMAML
from src.algorithms.baselines import StandardFedAvg, PerFedAvg, FairFedAvg, AgnosticFairFL
from src.metrics.fairness import accuracy_parity, comprehensive_fairness_report

logger = logging.getLogger(__name__)


def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    if m["type"] == "lstm":
        return ClinicalLSTM(
            input_dim=m["input_dim"], hidden_dim=m["hidden_dim"],
            n_layers=m["n_layers"], n_classes=m["n_classes"], dropout=m["dropout"],
        )
    elif m["type"] == "gru":
        return ClinicalGRU(
            input_dim=m["input_dim"], hidden_dim=m["hidden_dim"],
            n_layers=m["n_layers"], n_classes=m["n_classes"], dropout=m["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {m['type']}")


def build_algorithm(cfg: dict, model: torch.nn.Module, device: torch.device | str = "cpu"):
    alg = cfg["algorithm"]
    name = alg["name"]
    device = str(device)

    if name == "fair_fedmaml":
        return FairFedMAML(
            model=model,
            n_groups=cfg["dataset"]["n_groups"],
            inner_lr=alg["inner_lr"],
            outer_lr=alg["outer_lr"],
            inner_steps=alg["inner_steps"],
            fairness_penalty=alg["fairness_penalty"],
            lambda_lr=alg["lambda_lr"],
            clients_per_round=alg["clients_per_round"],
            device=device,
        )
    elif name == "fedavg":
        return StandardFedAvg(model=model, lr=alg.get("inner_lr", 0.01))
    elif name == "per_fedavg":
        return PerFedAvg(
            model=model, inner_lr=alg.get("inner_lr", 0.01),
            outer_lr=alg.get("outer_lr", 0.001),
        )
    elif name == "fair_fedavg":
        return FairFedAvg(
            model=model, lr=alg.get("inner_lr", 0.01),
            n_groups=cfg["dataset"]["n_groups"],
        )
    elif name == "agnostic_fair":
        return AgnosticFairFL(
            model=model, lr=alg.get("inner_lr", 0.01),
            n_groups=cfg["dataset"]["n_groups"],
            lambda_fair=alg.get("lambda_fair", 0.5),
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def prepare_client_tasks(
    fed_dataset: FairFederatedDataset,
    client_ids: list[int],
    k_shot: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Create meta-learning tasks for selected clients.

    Includes both support and query group ids so fairness losses index the
    matching tensor (support for training, query for eval).
    """
    tasks = []
    for cid in client_ids:
        support, query = fed_dataset.get_group_stratified_split(cid, k_shot)
        tasks.append({
            "support_x": torch.as_tensor(support["features"], dtype=torch.float32, device=device),
            "support_y": torch.as_tensor(support["labels"], dtype=torch.long, device=device),
            "support_group_ids": torch.as_tensor(support["group_ids"], dtype=torch.long, device=device),
            "query_x": torch.as_tensor(query["features"], dtype=torch.float32, device=device),
            "query_y": torch.as_tensor(query["labels"], dtype=torch.long, device=device),
            "group_ids": torch.as_tensor(query["group_ids"], dtype=torch.long, device=device),
        })
    return tasks


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides: algorithm.name=fedavg."""
    for override in overrides:
        if "=" not in override:
            continue
        key, val = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Try to parse value
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
        d[keys[-1]] = val
    return cfg


def run_experiment(cfg: dict) -> dict:
    seed_everything(cfg["experiment"]["seed"])
    device_str = cfg["experiment"].get("device", "auto")
    device = get_device() if device_str == "auto" else torch.device(device_str)

    logger.info(f"Algorithm: {cfg['algorithm']['name']}, Device: {device}")

    # Load dataset
    dcfg = cfg["dataset"]
    loader = EICUFederatedLoader()
    features, labels, hospital_ids, group_ids, metadata = loader.load(
        data_dir=dcfg.get("data_path") or "./data",
        n_samples=dcfg.get("n_samples", 5000),
        n_hospitals=dcfg.get("n_hospitals", 20),
        n_groups=dcfg.get("n_groups", 5),
        n_features=dcfg.get("n_features", 15),
        seq_len=dcfg.get("seq_len", 48),
    )

    # Sync model input dim with loaded feature width
    cfg["model"]["input_dim"] = int(features.shape[-1]) if features.ndim == 2 else int(features.shape[-1])
    if features.ndim == 3:
        cfg["model"]["input_dim"] = int(features.shape[-1])

    fed_dataset = FairFederatedDataset(
        features, labels, hospital_ids, group_ids,
        metadata,
    )

    train_frac = float(dcfg.get("meta_train_ratio", 0.7))
    val_frac = max(0.0, min(0.2, 1.0 - train_frac))
    meta_train_ids, _, meta_test_ids = fed_dataset.create_meta_splits(
        train_frac=train_frac,
        val_frac=val_frac,
        seed=cfg["experiment"]["seed"],
    )
    logger.info(f"Meta-train clients: {len(meta_train_ids)}, Meta-test: {len(meta_test_ids)}")

    # Build model and algorithm
    model = build_model(cfg).to(device)
    algorithm = build_algorithm(cfg, model, device)

    k_shot = dcfg.get("k_shot", 10)
    n_rounds = cfg["training"]["n_rounds"]
    eval_every = cfg["training"]["eval_every"]

    results_history = []
    is_fedmaml = cfg["algorithm"]["name"] == "fair_fedmaml"

    for rnd in range(1, n_rounds + 1):
        # Select clients for this round
        n_clients = cfg["algorithm"].get("clients_per_round", 5)
        selected = np.random.choice(
            meta_train_ids, size=min(n_clients, len(meta_train_ids)), replace=False
        ).tolist()

        # Train
        if is_fedmaml:
            round_metrics = algorithm.train_round(fed_dataset, selected)
        else:
            tasks = prepare_client_tasks(fed_dataset, selected, k_shot, device)
            round_metrics = algorithm.train_round(tasks)

        # Evaluate
        if rnd % eval_every == 0 or rnd == n_rounds:
            test_tasks = prepare_client_tasks(fed_dataset, meta_test_ids, k_shot, device)

            eval_model = algorithm.global_model if hasattr(algorithm, "global_model") else model
            eval_model.eval()
            all_preds, all_labels, all_groups = [], [], []
            with torch.no_grad():
                for t in test_tasks:
                    logits = eval_model(t["query_x"])
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(t["query_y"].cpu().numpy())
                    all_groups.append(t["group_ids"].cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_groups = np.concatenate(all_groups)

            ap = accuracy_parity(all_preds, all_labels, all_groups, dcfg["n_groups"])
            eval_result = {
                "overall_accuracy": float((all_preds == all_labels).mean()),
                "worst_group_accuracy": ap["worst_group_accuracy"],
                "accuracy_gap": ap["accuracy_gap"],
            }

            eval_result["round"] = rnd
            results_history.append(eval_result)
            logger.info(
                f"Round {rnd}: acc={eval_result.get('overall_accuracy', 0):.4f}, "
                f"worst={eval_result.get('worst_group_accuracy', 0):.4f}, "
                f"gap={eval_result.get('accuracy_gap', 0):.4f}"
            )

    return {
        "config": cfg,
        "results_history": results_history,
        "final": results_history[-1] if results_history else {},
    }


def main():
    parser = argparse.ArgumentParser(description="FairFedMeta Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, unknown = parser.parse_known_args()

    setup_logging()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, unknown)

    results = run_experiment(cfg)

    # Save results
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg['experiment']['name']}_results.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
