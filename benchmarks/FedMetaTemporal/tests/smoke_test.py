"""Smoke test for FedMetaTemporal."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_dataset():
    print("[1/6] Testing drifting stream generator...")
    from src.datasets.temporal_streams import DriftingStreamGenerator, FederatedTemporalDataset

    gen = DriftingStreamGenerator()
    data_info = gen.generate(
        n_clients=5, n_samples_per_client=500, n_features=10,
        n_classes=2, drift_type="sudden", n_drifts=2,
    )
    assert data_info["n_clients"] == 5
    assert data_info["n_features"] == 10

    fed = FederatedTemporalDataset(data_info)
    assert fed.num_clients == 5

    train_ids, test_ids = fed.get_train_test_split()
    assert len(train_ids) + len(test_ids) == 5

    tasks = fed.create_meta_tasks(train_ids[0], n_tasks=3, k_shot=10)
    assert len(tasks) == 3
    assert "support_x" in tasks[0]
    print("   PASSED")


def test_models():
    print("[2/6] Testing Neural CDE and SimpleClassifier...")
    from src.models.neural_cde import TemporalClassifier, SimpleClassifier

    x = torch.randn(8, 20)

    simple = SimpleClassifier(input_dim=20, hidden_dim=32, n_classes=2)
    out = simple(x)
    assert out.shape == (8, 2), f"Simple output shape: {out.shape}"

    # Neural CDE needs sequence input
    x_seq = torch.randn(8, 10, 20)  # (batch, seq_len, features)
    cde = TemporalClassifier(input_dim=20, hidden_dim=32, n_classes=2, n_layers=1)
    out = cde(x_seq)
    assert out.shape == (8, 2), f"CDE output shape: {out.shape}"
    print("   PASSED")


def test_drift_detection():
    print("[3/6] Testing drift detection...")
    from src.algorithms.drift_detection import PageHinkleyDetector, ClientDriftMonitor

    ph = PageHinkleyDetector(threshold=50.0)
    # Feed stable values — should not trigger drift quickly
    for _ in range(20):
        ph.update(1.0 + np.random.randn() * 0.01)
    # Just verify it runs without error

    monitor = ClientDriftMonitor(n_clients=3)
    for _ in range(10):
        for c in range(3):
            monitor.update(c, np.random.randn())
        monitor.advance_step()
    summary = monitor.get_summary()
    assert "total_drifts" in summary
    print("   PASSED")


def test_temporal_maml():
    print("[4/6] Testing TemporalMAML...")
    from src.models.neural_cde import SimpleClassifier
    from src.algorithms.temporal_maml import TemporalMAML

    model = SimpleClassifier(input_dim=10, hidden_dim=16, n_classes=2)
    maml = TemporalMAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    tasks = [{
        "support_x": torch.randn(15, 10),
        "support_y": torch.randint(0, 2, (15,)),
        "query_x": torch.randn(10, 10),
        "query_y": torch.randint(0, 2, (10,)),
    }]
    metrics = maml.meta_train_step(tasks)
    assert "meta_loss" in metrics
    assert "drift_detected" in metrics
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    print("   PASSED")


def test_ewc_maml():
    print("[5/6] Testing EWC-MAML...")
    from src.models.neural_cde import SimpleClassifier
    from src.algorithms.temporal_maml import EWCMAML

    model = SimpleClassifier(input_dim=10, hidden_dim=16, n_classes=2)
    ewc = EWCMAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1, ewc_lambda=1.0)

    tasks = [{
        "support_x": torch.randn(15, 10),
        "support_y": torch.randint(0, 2, (15,)),
        "query_x": torch.randn(10, 10),
        "query_y": torch.randint(0, 2, (10,)),
    }]

    # Train without Fisher first
    metrics = ewc.meta_train_step(tasks)
    assert "meta_loss" in metrics
    assert metrics["ewc_penalty"] == 0.0  # No Fisher yet

    # Compute Fisher and re-train
    ewc.compute_fisher(tasks)
    metrics2 = ewc.meta_train_step(tasks)
    assert metrics2["ewc_penalty"] >= 0.0
    print("   PASSED")


def test_temporal_fedavg():
    print("[6/6] Testing TemporalFedAvg end-to-end...")
    from src.datasets.temporal_streams import DriftingStreamGenerator, FederatedTemporalDataset
    from src.models.neural_cde import SimpleClassifier
    from src.algorithms.temporal_maml import TemporalFedAvg

    gen = DriftingStreamGenerator()
    data_info = gen.generate(n_clients=5, n_samples_per_client=400, n_features=10, drift_type="gradual")
    fed = FederatedTemporalDataset(data_info)
    train_ids, test_ids = fed.get_train_test_split()

    model = SimpleClassifier(input_dim=10, hidden_dim=16, n_classes=2)
    alg = TemporalFedAvg(
        model, inner_lr=0.01, outer_lr=0.001, inner_steps=1,
        clients_per_round=2,
    )
    alg.init_drift_monitor(fed.n_clients)

    # Train 2 rounds
    for _ in range(2):
        metrics = alg.train_round(fed, train_ids, n_tasks=2, k_shot=10)
    assert "avg_loss" in metrics

    # Evaluate
    eval_result = alg.evaluate(fed, test_ids, n_tasks=2, k_shot=10)
    assert "mean_accuracy" in eval_result
    print("   PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("FedMetaTemporal Smoke Test")
    print("=" * 50)
    test_dataset()
    test_models()
    test_drift_detection()
    test_temporal_maml()
    test_ewc_maml()
    test_temporal_fedavg()
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
