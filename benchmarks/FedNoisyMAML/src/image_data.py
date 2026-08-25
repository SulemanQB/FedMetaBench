"""
Image dataset loaders for FedNoisyMAML experiments.
CIFAR-10 and MNIST with non-IID Dirichlet partitioning and per-client label noise.
"""

import os
import torch
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/sqamar/Data")

_dataset_cache = {}


def _load_dataset(name):
    """Load and normalize image dataset. Caches in memory after first load."""
    if name in _dataset_cache:
        return _dataset_cache[name]

    import torchvision
    root = os.path.join(DATA_ROOT, name)

    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        X_train = torch.tensor(trainset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        X_train = (X_train - mean) / std
        y_train = torch.tensor(trainset.targets, dtype=torch.long)
        X_test = torch.tensor(testset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        X_test = (X_test - mean) / std
        y_test = torch.tensor(testset.targets, dtype=torch.long)
    elif name == "mnist":
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True)
        X_train = trainset.data.unsqueeze(1).float() / 255.0
        X_train = (X_train - 0.1307) / 0.3081
        y_train = trainset.targets.long()
        X_test = testset.data.unsqueeze(1).float() / 255.0
        X_test = (X_test - 0.1307) / 0.3081
        y_test = testset.targets.long()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    _dataset_cache[name] = (X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test


def dirichlet_partition(labels, n_clients, alpha=0.5, seed=42):
    """Non-IID partition via Dirichlet distribution."""
    rng = np.random.RandomState(seed)
    n_classes = int(labels.max().item()) + 1
    class_idx = {c: np.where(labels.numpy() == c)[0] for c in range(n_classes)}

    client_idx = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = class_idx[c].copy()
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * n_clients)
        splits = np.split(idx, (np.cumsum(props)[:-1] * len(idx)).astype(int))
        for i in range(min(n_clients, len(splits))):
            client_idx[i].extend(splits[i].tolist())

    return [np.array(ci) for ci in client_idx]


DATASET_META = {
    "cifar10": {"in_channels": 3, "n_classes": 10, "img_size": 32},
    "mnist": {"in_channels": 1, "n_classes": 10, "img_size": 28},
}


class NoisyImageClient:
    """Federated client with image data and heterogeneous label noise."""

    def __init__(self, client_id, X, y, noise_rate, n_classes, seed=0,
                 test_fraction=0.2):
        self.client_id = client_id
        self.noise_rate = noise_rate
        self.n_classes = n_classes
        rng = np.random.RandomState(seed + client_id)

        n = len(X)
        perm = rng.permutation(n)
        n_test = max(int(n * test_fraction), 1)

        self.X = X[perm[n_test:]]
        self.y_clean = y[perm[n_test:]].clone()
        self.y_noisy = self.y_clean.clone()
        mask = torch.tensor(rng.rand(len(self.y_clean)) < noise_rate)
        n_flip = mask.sum().item()
        if n_flip > 0:
            self.y_noisy[mask] = torch.tensor(
                rng.randint(0, n_classes, size=n_flip), dtype=y.dtype)
        self.n_corrupted = (self.y_clean != self.y_noisy).sum().item()

        self.X_test = X[perm[:n_test]]
        self.y_test = y[perm[:n_test]].clone()

    def sample_task(self, n_support=10, n_query=10):
        """Support uses noisy labels, query uses clean labels."""
        perm = torch.randperm(len(self.X))
        s, q = perm[:n_support], perm[n_support:n_support + n_query]
        return self.X[s], self.y_noisy[s], self.X[q], self.y_clean[q]

    def sample_task_all_noisy(self, n_support=10, n_query=10):
        perm = torch.randperm(len(self.X))
        s, q = perm[:n_support], perm[n_support:n_support + n_query]
        return self.X[s], self.y_noisy[s], self.X[q], self.y_noisy[q]

    def get_clean_test(self, n_test=50):
        n = min(n_test, len(self.X_test))
        perm = torch.randperm(len(self.X_test))[:n]
        return self.X_test[perm], self.y_test[perm]


def create_noisy_image_federation(dataset_name="cifar10", n_clients=10,
                                  noise_rates=None, alpha=0.5, seed=42):
    """Create FL clients with non-IID image data and heterogeneous label noise."""
    X_train, y_train, _, _ = _load_dataset(dataset_name)
    n_classes = int(y_train.max().item()) + 1

    if noise_rates is None:
        noise_rates = np.linspace(0.0, 0.4, n_clients)

    parts = dirichlet_partition(y_train, n_clients, alpha=alpha, seed=seed)

    clients = []
    for i in range(n_clients):
        idx = parts[i]
        nr = noise_rates[i] if i < len(noise_rates) else noise_rates[-1]
        clients.append(NoisyImageClient(
            i, X_train[idx], y_train[idx], nr, n_classes, seed=seed))

    return clients, DATASET_META[dataset_name]
