"""FedMeta-Temporal+ experiment runner.

Usage:
    python run_experiment.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import seed_everything, load_config, get_device, setup_logging
from src.datasets.temporal_streams import DriftingStreamGenerator, FederatedTemporalDataset
from src.models.neural_cde import TemporalClassifier, SimpleClassifier
from src.algorithms.temporal_maml import TemporalMAML, EWCMAML, TemporalFedAvg

logger = logging.getLogger(__name__)


def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    if m["type"] == "neural_cde":
        return TemporalClassifier(
            input_dim=m["input_dim"], hidden_dim=m["hidden_dim"],
            n_classes=m["n_classes"], n_layers=m["n_layers"],
            dropout=m["dropout"],
        )
    elif m["type"] == "simple":
        return SimpleClassifier(
            input_dim=m["input_dim"], hidden_dim=m["hidden_dim"],
            n_classes=m["n_classes"],
        )
    else:
        raise ValueError(f"Unknown model: {m['type']}")


def build_algorithm(cfg: dict, model: torch.nn.Module, device: torch.device):
    """Instantiate the algorithm named in config (do not hard-code TemporalFedAvg)."""
    acfg = cfg["algorithm"]
    name = str(acfg.get("name", "temporal_fedavg")).lower().replace("-", "_")
    common = dict(
        model=model,
        inner_lr=acfg["inner_lr"],
        outer_lr=acfg["outer_lr"],
        inner_steps=acfg["inner_steps"],
        device=str(device),
    )

    if name in ("temporal_fedavg", "fedavg"):
        return TemporalFedAvg(
            temporal_decay=acfg.get("temporal_decay", 0.9),
            ewc_lambda=acfg.get("ewc_lambda", 0.5),
            clients_per_round=acfg.get("clients_per_round", 5),
            **common,
        )
    if name in ("temporal_maml", "maml"):
        return TemporalMAML(
            temporal_decay=acfg.get("temporal_decay", 0.9),
            clients_per_round=acfg.get("clients_per_round", 5),
            **common,
        )
    if name in ("ewc_maml", "ewc"):
        return EWCMAML(
            ewc_lambda=acfg.get("ewc_lambda", 0.5),
            clients_per_round=acfg.get("clients_per_round", 5),
            **common,
        )
    raise ValueError(
        f"Unknown algorithm: {acfg.get('name')}. "
        "Expected one of: temporal_fedavg, temporal_maml, ewc_maml"
    )


def run_experiment(cfg: dict) -> dict:
    seed_everything(cfg["experiment"]["seed"])
    device = get_device() if cfg["experiment"]["device"] == "auto" else torch.device(cfg["experiment"]["device"])

    # Generate drifting streams
    dcfg = cfg["dataset"]
    gen = DriftingStreamGenerator()
    data_info = gen.generate(
        n_clients=dcfg["n_clients"],
        n_samples_per_client=dcfg["n_samples_per_client"],
        n_features=dcfg["n_features"],
        n_classes=dcfg["n_classes"],
        drift_type=dcfg["drift_type"],
        n_drifts=dcfg["n_drifts"],
        seed=cfg["experiment"]["seed"],
    )
    fed_data = FederatedTemporalDataset(data_info)
    train_ids, test_ids = fed_data.get_train_test_split()
    logger.info(f"Train: {len(train_ids)} clients, Test: {len(test_ids)} clients, Drift: {dcfg['drift_type']}")

    model = build_model(cfg).to(device)
    acfg = cfg["algorithm"]
    tcfg = cfg["training"]

    alg = build_algorithm(cfg, model, device)
    if hasattr(alg, "init_drift_monitor"):
        alg.init_drift_monitor(fed_data.n_clients)
    logger.info(f"Algorithm: {acfg['name']}")

    results_history = []
    for rnd in range(1, tcfg["n_rounds"] + 1):
        round_metrics = alg.train_round(
            fed_data, train_ids,
            n_tasks=tcfg["n_tasks_per_client"],
            k_shot=tcfg["k_shot"],
        )

        if rnd % tcfg["eval_every"] == 0 or rnd == tcfg["n_rounds"]:
            eval_result = alg.evaluate(
                fed_data, test_ids,
                n_tasks=tcfg["n_tasks_per_client"],
                k_shot=tcfg["k_shot"],
            )
            eval_result["round"] = rnd
            eval_result.update(round_metrics)
            results_history.append(eval_result)
            logger.info(
                f"Round {rnd}: loss={round_metrics['avg_loss']:.4f}, "
                f"acc={eval_result['mean_accuracy']:.4f}, "
                f"drifts={round_metrics['n_drifts']}"
            )

    return {
        "config": cfg,
        "results_history": results_history,
        "final": results_history[-1] if results_history else {},
        "drift_summary": alg.drift_monitor.get_summary() if alg.drift_monitor else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    results = run_experiment(cfg)

    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = os.path.join(out_dir, f"{cfg['experiment']['name']}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
