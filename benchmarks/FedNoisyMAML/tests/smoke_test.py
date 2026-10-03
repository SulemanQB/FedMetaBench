"""
Smoke test for FedNoisyMAML PoC.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np


def test_data_generation():
    from src.datasets import generate_gaussian_data
    X, y = generate_gaussian_data(n_per_class=50, n_classes=4, seed=0)
    assert X.shape == (200, 8)
    assert y.shape == (200,)
    assert set(y.numpy().tolist()) == {0, 1, 2, 3}
    print("[PASS] Gaussian data generation")


def test_label_noise():
    from src.datasets import inject_label_noise
    labels = torch.zeros(100, dtype=torch.int64)
    noisy = inject_label_noise(labels, noise_rate=0.3, n_classes=4,
                               rng=np.random.RandomState(0))
    n_flipped = (labels != noisy).sum().item()
    assert 15 < n_flipped < 50  # ~30% flipped
    fully_noisy = inject_label_noise(
        labels, noise_rate=1.0, n_classes=4, rng=np.random.RandomState(1)
    )
    assert torch.all(fully_noisy != labels)
    print(f"[PASS] Label noise injection: {n_flipped}/100 flipped")


def test_noisy_client():
    from src.datasets import NoisyClient
    X = torch.randn(80, 8)
    y = torch.randint(0, 4, (80,))
    client = NoisyClient(0, X, y, noise_rate=0.2, n_classes=4, seed=0)
    xs, ys, xq, yq = client.sample_task(n_support=10, n_query=10)
    assert xs.shape == (10, 8)
    assert ys.shape == (10,)
    assert xq.shape == (10, 8)
    print("[PASS] NoisyClient sampling")


def test_federation():
    from src.datasets import create_noisy_federation
    clients = create_noisy_federation(n_clients=4, n_per_class=50, n_classes=4, seed=0)
    assert len(clients) == 4
    assert clients[0].noise_rate < clients[-1].noise_rate
    print("[PASS] Noisy federation creation")


def test_model():
    from src.models import ClassifierMLP
    model = ClassifierMLP(input_dim=8, hidden_size=32, n_classes=4)
    x = torch.randn(10, 8)
    out = model(x)
    assert out.shape == (10, 4)
    print("[PASS] ClassifierMLP forward pass")


def test_fedmaml():
    from src.datasets import create_noisy_federation
    from src.models import ClassifierMLP
    from src.algorithms import FedMAML

    torch.manual_seed(0)
    clients = create_noisy_federation(n_clients=3, n_per_class=50, n_classes=4, seed=0)
    model = ClassifierMLP(input_dim=8, hidden_size=32, n_classes=4)
    trainer = FedMAML(model, clients, outer_lr=0.001, inner_lr=0.01, inner_steps=2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    losses = []
    for _ in range(5):
        l = trainer.train_round(n_support=8, n_query=8)
        losses.append(l)
    assert len(losses) == 5
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))

    accs = trainer.evaluate(n_test=20)
    assert len(accs) == 3
    assert all(0 <= a <= 1 for a in accs)
    print(f"[PASS] FedMAML: 5 rounds, accs={[f'{a:.2f}' for a in accs]}")


def test_fedavg():
    from src.datasets import create_noisy_federation
    from src.models import ClassifierMLP
    from src.algorithms import FedAvg

    torch.manual_seed(0)
    clients = create_noisy_federation(n_clients=3, n_per_class=50, n_classes=4, seed=0)
    model = ClassifierMLP(input_dim=8, hidden_size=32, n_classes=4)
    trainer = FedAvg(model, clients, lr=0.01, local_epochs=2)

    losses = []
    for _ in range(5):
        l = trainer.train_round(n_samples=20)
        losses.append(l)
    assert len(losses) == 5

    accs = trainer.evaluate(n_test=20)
    assert len(accs) == 3
    assert all(0 <= a <= 1 for a in accs)
    print(f"[PASS] FedAvg:  5 rounds, accs={[f'{a:.2f}' for a in accs]}")


def test_end_to_end():
    from run_experiment import run_experiment
    result = run_experiment(n_clients=4, n_classes=4, n_rounds=10, seed=0)
    assert result is True
    print("[PASS] End-to-end experiment completes")


if __name__ == "__main__":
    print("=" * 50)
    print("FedNoisyMAML Smoke Tests")
    print("=" * 50)

    test_data_generation()
    test_label_noise()
    test_noisy_client()
    test_federation()
    test_model()
    test_fedmaml()
    test_fedavg()
    test_end_to_end()

    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)
