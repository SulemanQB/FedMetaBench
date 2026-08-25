"""FedMeta-Env experiment runner.

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
from src.datasets.air_quality import BeijingAirQualityLoader, FederatedAirQuality
from src.models.env_models import EnvLSTM, EnvMLP
from src.algorithms.fed_env_maml import FedEnvMAML, FedAvgEnv, LocalOnly

logger = logging.getLogger(__name__)


def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    if m["type"] == "lstm":
        return EnvLSTM(
            input_dim=m["input_dim"], hidden_dim=m["hidden_dim"],
            n_layers=m["n_layers"], dropout=m["dropout"],
        )
    elif m["type"] == "mlp":
        return EnvMLP(
            input_dim=m["input_dim"],
            window_size=cfg["dataset"]["window_size"],
        )
    else:
        raise ValueError(f"Unknown model: {m['type']}")


def run_experiment(cfg: dict) -> dict:
    seed_everything(cfg["experiment"]["seed"])
    device = get_device() if cfg["experiment"]["device"] == "auto" else torch.device(cfg["experiment"]["device"])

    # Load data
    loader = BeijingAirQualityLoader()
    data_info = loader.load(
        data_dir=cfg["dataset"]["data_dir"],
        target_pollutant=cfg["dataset"]["target_pollutant"],
        window_size=cfg["dataset"]["window_size"],
    )
    fed_data = FederatedAirQuality(data_info, cfg["dataset"]["window_size"])
    logger.info(f"Loaded {fed_data.n_stations} stations, {data_info['n_features']} features")

    model = build_model(cfg).to(device)
    acfg = cfg["algorithm"]
    k_days = cfg["dataset"]["k_days"]
    n_rounds = cfg["training"]["n_rounds"]

    all_results = {}

    if cfg["evaluation"].get("loso", True):
        # LOSO evaluation
        station_ids = list(fed_data.station_ids)
        max_loso = cfg["evaluation"].get("max_loso_stations")
        if max_loso is not None:
            station_ids = station_ids[: int(max_loso)]
            logger.info(f"LOSO limited to first {len(station_ids)} stations (smoke/demo)")
        for test_station in station_ids:
            train_stations, _ = fed_data.get_loso_splits(test_station)
            logger.info(f"LOSO: held-out station {test_station}")

            if acfg["name"] == "fed_env_maml":
                import copy
                alg = FedEnvMAML(
                    copy.deepcopy(model), acfg["inner_lr"], acfg["outer_lr"],
                    acfg["inner_steps"], acfg["clients_per_round"],
                    acfg["local_meta_steps"], str(device),
                )
                for rnd in range(1, n_rounds + 1):
                    alg.train_round(fed_data, train_stations, k_days)

                k_list = cfg["evaluation"].get("k_days_list", [1, 3, 7, 14, 30])
                eval_results = alg.evaluate_loso(fed_data, test_station, k_list)
            elif acfg["name"] == "local_only":
                import copy
                local = LocalOnly(copy.deepcopy(model), acfg.get("inner_lr", 0.01), str(device))
                split = fed_data.get_k_shot_split(test_station, k_days)
                eval_results = {"k7": local.train_and_evaluate(
                    split["support_x"], split["support_y"],
                    split["query_x"], split["query_y"],
                )}
            else:
                import copy
                fedavg = FedAvgEnv(copy.deepcopy(model), acfg.get("inner_lr", 0.01), device=str(device))
                for rnd in range(1, n_rounds + 1):
                    fedavg.train_round(fed_data, train_stations, k_days)
                split = fed_data.get_k_shot_split(test_station, k_days)
                eval_results = {"k7": fedavg.evaluate(split["query_x"], split["query_y"])}

            all_results[f"station_{test_station}"] = eval_results

    return {"config": cfg, "loso_results": all_results}


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
