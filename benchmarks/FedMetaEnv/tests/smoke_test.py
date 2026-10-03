"""Smoke test for FedMetaEnv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_dataset():
    print("[1/5] Testing air quality dataset loader...")
    from src.datasets.air_quality import BeijingAirQualityLoader, FederatedAirQuality

    loader = BeijingAirQualityLoader()
    data_info = loader.load()
    assert data_info["n_stations"] == 12
    assert data_info["synthetic"] is True

    fed = FederatedAirQuality(data_info)
    assert fed.num_clients == 12

    split = fed.get_k_shot_split(0, k_days=7)
    assert "support_x" in split and "query_x" in split
    assert split["support_x"].dim() == 3  # (N, window, features)

    train, test = fed.get_loso_splits(0)
    assert len(train) == 11
    assert test == 0
    print("   PASSED")


def test_model():
    print("[2/5] Testing env models...")
    from src.models.env_models import EnvLSTM, EnvMLP

    x = torch.randn(8, 24, 17)

    lstm = EnvLSTM(input_dim=17, hidden_dim=32, n_layers=1)
    out = lstm(x)
    assert out.shape == (8,), f"LSTM output shape: {out.shape}"

    mlp = EnvMLP(input_dim=17, window_size=24)
    out = mlp(x)
    assert out.shape == (8,), f"MLP output shape: {out.shape}"
    print("   PASSED")


def test_env_maml():
    print("[3/5] Testing EnvMAML...")
    from src.models.env_models import EnvLSTM
    from src.algorithms.fed_env_maml import EnvMAML

    model = EnvLSTM(input_dim=17, hidden_dim=16, n_layers=1)
    maml = EnvMAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    task = {
        "support_x": torch.randn(20, 24, 17),
        "support_y": torch.randn(20),
        "query_x": torch.randn(10, 24, 17),
        "query_y": torch.randn(10),
    }
    metrics = maml.meta_train_step([task])
    assert "meta_loss" in metrics
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    print("   PASSED")


def test_fed_env_maml():
    print("[4/5] Testing FedEnvMAML...")
    from src.datasets.air_quality import BeijingAirQualityLoader, FederatedAirQuality
    from src.models.env_models import EnvLSTM
    from src.algorithms.fed_env_maml import FedEnvMAML

    loader = BeijingAirQualityLoader()
    data_info = loader.load()
    fed = FederatedAirQuality(data_info)

    model = EnvLSTM(input_dim=data_info["n_features"], hidden_dim=16, n_layers=1)
    train_stations, test_station = fed.get_loso_splits(0)

    alg = FedEnvMAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1,
                     clients_per_round=3, local_meta_steps=1)

    metrics = alg.train_round(fed, train_stations, k_days=3)
    assert "avg_loss" in metrics

    # Quick eval on test station
    eval_results = alg.evaluate_loso(fed, test_station, k_days_list=[1, 3])
    assert "k1" in eval_results
    assert "post_adapt_mse" in eval_results["k1"]
    print("   PASSED")


def test_metrics():
    print("[5/5] Testing env metrics...")
    from src.metrics.env_metrics import regression_metrics, k_shot_scaling_analysis, loso_summary

    preds = np.random.randn(100)
    targets = np.random.randn(100)
    rm = regression_metrics(preds, targets)
    assert "mse" in rm and "r2" in rm

    results_by_k = {1: {"rmse": 1.5}, 3: {"rmse": 1.2}, 7: {"rmse": 0.9}}
    ksa = k_shot_scaling_analysis(results_by_k)
    assert ksa["k_values"] == [1, 3, 7]

    station_results = {0: {"rmse": 1.0}, 1: {"rmse": 1.5}, 2: {"rmse": 0.8}}
    ls = loso_summary(station_results)
    assert "mean_rmse" in ls
    print("   PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("FedMetaEnv Smoke Test")
    print("=" * 50)
    test_dataset()
    test_model()
    test_env_maml()
    test_fed_env_maml()
    test_metrics()
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
