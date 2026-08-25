"""
Synthetic classification datasets with heterogeneous label noise per client.

Uses a 2D Gaussian mixture so results are fast and interpretable.
Each client gets a partition with a different label noise rate.
"""

import torch
import numpy as np


def generate_gaussian_data(n_per_class=200, n_classes=4, dim=8, seed=42,
                           separation=0.7, spread=1.3):
    """Generate a global Gaussian mixture classification dataset.
    
    Args:
        separation: distance of class centers from origin (lower = harder)
        spread: std of each cluster (higher = harder)
    """
    rng = np.random.RandomState(seed)

    # Generate random centers in dim-dimensional space
    centers = rng.randn(n_classes, dim).astype(np.float32) * separation

    X_all, y_all = [], []
    for cls_id, center in enumerate(centers):
        X = rng.randn(n_per_class, dim).astype(np.float32) * spread + center
        y = np.full(n_per_class, cls_id, dtype=np.int64)
        X_all.append(X)
        y_all.append(y)

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    perm = rng.permutation(len(X_all))
    return torch.tensor(X_all[perm]), torch.tensor(y_all[perm])


def inject_label_noise(labels, noise_rate, n_classes, rng):
    """Flip labels with probability noise_rate to a random other class."""
    noisy = labels.clone()
    n = len(labels)
    mask = torch.tensor(rng.rand(n) < noise_rate)
    n_flip = mask.sum().item()
    if n_flip > 0:
        random_labels = torch.tensor(rng.randint(0, n_classes, size=n_flip), dtype=labels.dtype)
        noisy[mask] = random_labels
    return noisy


class NoisyClient:
    """A federated client with a specific label noise rate."""

    def __init__(self, client_id, X, y, noise_rate, n_classes, seed=0,
                 test_fraction=0.2):
        self.client_id = client_id
        self.noise_rate = noise_rate
        self.n_classes = n_classes
        self.rng = np.random.RandomState(seed + client_id)

        # Train/test split
        n = len(X)
        perm = self.rng.permutation(n)
        n_test = max(int(n * test_fraction), 1)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        self.X = X[train_idx]
        self.y_clean = y[train_idx].clone()
        self.y_noisy = inject_label_noise(y[train_idx], noise_rate, n_classes, self.rng)
        self.n_corrupted = (self.y_clean != self.y_noisy).sum().item()

        # Held-out test data (always clean labels)
        self.X_test = X[test_idx]
        self.y_test = y[test_idx].clone()

    def sample_task(self, n_support=10, n_query=10):
        """Sample support/query split for MAML-style training."""
        n = len(self.X)
        perm = torch.randperm(n)
        s_idx = perm[:n_support]
        q_idx = perm[n_support:n_support + n_query]

        return (self.X[s_idx], self.y_noisy[s_idx],
                self.X[q_idx], self.y_clean[q_idx])  # query uses CLEAN labels

    def sample_task_all_noisy(self, n_support=10, n_query=10):
        """Both support and query use noisy labels."""
        n = len(self.X)
        perm = torch.randperm(n)
        s_idx = perm[:n_support]
        q_idx = perm[n_support:n_support + n_query]

        return (self.X[s_idx], self.y_noisy[s_idx],
                self.X[q_idx], self.y_noisy[q_idx])

    def get_clean_test(self, n_test=50):
        """Get clean-label test data from held-out test set."""
        n = min(n_test, len(self.X_test))
        perm = torch.randperm(len(self.X_test))[:n]
        return self.X_test[perm], self.y_test[perm]


def create_noisy_federation(n_clients=10, noise_rates=None,
                            n_per_class=200, n_classes=4, seed=42):
    """Create a federation with heterogeneous label noise."""
    if noise_rates is None:
        noise_rates = np.linspace(0.0, 0.4, n_clients)

    X, y = generate_gaussian_data(n_per_class=n_per_class, n_classes=n_classes, seed=seed)

    # Partition data among clients (equal split)
    n_total = len(X)
    chunk_size = n_total // n_clients
    clients = []

    for i in range(n_clients):
        start = i * chunk_size
        end = start + chunk_size
        noise_rate = noise_rates[i] if i < len(noise_rates) else noise_rates[-1]
        client = NoisyClient(i, X[start:end], y[start:end],
                             noise_rate, n_classes, seed=seed)
        clients.append(client)

    return clients
