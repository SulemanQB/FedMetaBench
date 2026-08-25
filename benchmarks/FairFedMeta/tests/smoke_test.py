"""Smoke test for FairFedMeta — validates all components work end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

def test_dataset():
    print("[1/6] Testing eICU dataset loader...")
    from src.datasets.eicu import EICUFederatedLoader, FairFederatedDataset

    loader = EICUFederatedLoader()
    features, labels, hospital_ids, group_ids, metadata = loader.load()

    assert features.shape[1] == 48  # seq_len
    assert labels.shape[0] == features.shape[0]

    fed = FairFederatedDataset(features, labels, hospital_ids, group_ids, metadata)
    train_ids, val_ids, test_ids = fed.create_meta_splits()
    assert len(train_ids) + len(val_ids) + len(test_ids) == fed.num_clients

    support, query = fed.get_group_stratified_split(train_ids[0], k_shot=5)
    assert "features" in support and "features" in query
    print("   PASSED")


def test_model():
    print("[2/6] Testing clinical models...")
    from src.models.clinical import ClinicalLSTM, ClinicalGRU

    x = torch.randn(8, 48, 15)

    lstm = ClinicalLSTM(input_dim=15, hidden_dim=32, n_layers=1, n_classes=2)
    out = lstm(x)
    assert out.shape == (8, 2), f"LSTM output shape: {out.shape}"

    gru = ClinicalGRU(input_dim=15, hidden_dim=32, n_layers=1, n_classes=2)
    out = gru(x)
    assert out.shape == (8, 2), f"GRU output shape: {out.shape}"
    print("   PASSED")


def test_fair_maml():
    print("[3/6] Testing FairMAML...")
    from src.models.clinical import ClinicalLSTM
    from src.algorithms.fair_maml import FairMAML

    model = ClinicalLSTM(input_dim=15, hidden_dim=16, n_layers=1, n_classes=2)
    fair_maml = FairMAML(model=model, n_groups=3, inner_lr=0.01, outer_lr=0.001, inner_steps=1)

    # Create a synthetic task matching expected API
    task = {
        "support_features": torch.randn(20, 48, 15),
        "support_labels": torch.randint(0, 2, (20,)),
        "support_groups": torch.randint(0, 3, (20,)),
        "query_features": torch.randn(10, 48, 15),
        "query_labels": torch.randint(0, 2, (10,)),
        "query_groups": torch.randint(0, 3, (10,)),
    }

    metrics = fair_maml.outer_step([task])
    assert "meta_loss" in metrics, f"Expected 'meta_loss', got keys: {list(metrics.keys())}"
    print("   PASSED")


def test_fair_fedmaml():
    print("[4/6] Testing FairFedMAML...")
    from src.datasets.eicu import EICUFederatedLoader, FairFederatedDataset
    from src.algorithms.fair_fedmaml import FairFedMAML
    from src.models.clinical import ClinicalLSTM

    # Build a small dataset
    loader = EICUFederatedLoader()
    features, labels, hospital_ids, group_ids, metadata = loader.load()
    fed = FairFederatedDataset(features, labels, hospital_ids, group_ids, metadata)
    train_ids, _, _ = fed.create_meta_splits()

    model = ClinicalLSTM(input_dim=metadata["n_features"], hidden_dim=16, n_layers=1, n_classes=2)
    alg = FairFedMAML(
        model=model, n_groups=metadata["n_groups"],
        inner_lr=0.01, outer_lr=0.001, inner_steps=1,
        clients_per_round=2, local_meta_steps=1,
    )

    # Run one round with 2 hospitals
    selected = train_ids[:2]
    metrics = alg.train_round(fed, selected)
    assert "avg_loss" in metrics
    print("   PASSED")


def test_baselines():
    print("[5/6] Testing baselines...")
    from src.models.clinical import ClinicalLSTM
    from src.algorithms.baselines import StandardFedAvg, PerFedAvg, FairFedAvg

    model = ClinicalLSTM(input_dim=15, hidden_dim=16, n_layers=1, n_classes=2)

    # StandardFedAvg
    fedavg = StandardFedAvg(model=model, lr=0.01, local_epochs=1)
    tasks = [{
        "support_x": torch.randn(10, 48, 15),
        "support_y": torch.randint(0, 2, (10,)),
        "query_x": torch.randn(5, 48, 15),
        "query_y": torch.randint(0, 2, (5,)),
    }]
    result = fedavg.train_round(tasks)
    assert "avg_loss" in result

    # FairFedAvg
    import copy
    model2 = ClinicalLSTM(input_dim=15, hidden_dim=16, n_layers=1, n_classes=2)
    fair = FairFedAvg(model=model2, lr=0.01, n_groups=3, local_epochs=1)
    tasks2 = [{
        "support_x": torch.randn(10, 48, 15),
        "support_y": torch.randint(0, 2, (10,)),
        "query_x": torch.randn(5, 48, 15),
        "query_y": torch.randint(0, 2, (5,)),
        "group_ids": torch.randint(0, 3, (10,)),
    }]
    result2 = fair.train_round(tasks2)
    assert "n_clients" in result2
    print("   PASSED")


def test_fairness_metrics():
    print("[6/6] Testing fairness metrics...")
    from src.metrics.fairness import comprehensive_fairness_report

    preds = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    labels = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0])
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    report = comprehensive_fairness_report(preds, labels, groups, n_groups=3)
    assert "overall_accuracy" in report
    assert "fairness_summary" in report
    assert "dp_gap" in report["fairness_summary"]
    assert 0 <= report["overall_accuracy"] <= 1
    print("   PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("FairFedMeta Smoke Test")
    print("=" * 50)
    test_dataset()
    test_model()
    test_fair_maml()
    test_fair_fedmaml()
    test_baselines()
    test_fairness_metrics()
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
