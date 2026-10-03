"""Synthetic temporal streams with configurable concept drift."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalStreamDataset(Dataset):
    """A temporal data stream with concept drift."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        client_id: int,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.timestamps = torch.FloatTensor(timestamps)
        self.client_id = client_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.timestamps[idx]


class DriftingStreamGenerator:
    """Generate synthetic data streams with configurable concept drift.

    Drift types:
    - sudden: abrupt change at a point
    - gradual: smooth transition over a window
    - recurring: drift cycles back to earlier concept
    - incremental: slow continuous change
    """

    DRIFT_TYPES = ["sudden", "gradual", "recurring", "incremental"]

    def generate(
        self,
        n_clients: int = 10,
        n_samples_per_client: int = 2000,
        n_features: int = 20,
        n_classes: int = 2,
        drift_type: str = "sudden",
        n_drifts: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if n_drifts < 1:
            raise ValueError("n_drifts must be at least 1")
        if drift_type not in self.DRIFT_TYPES:
            raise ValueError(
                f"Unknown drift_type: {drift_type}. Expected one of {self.DRIFT_TYPES}"
            )
        rng = np.random.RandomState(seed)

        client_data = {}
        for c in range(n_clients):
            # Each client gets slightly different drift timing
            drift_offset = rng.uniform(-0.1, 0.1)

            features, labels, timestamps = self._generate_stream(
                n_samples_per_client, n_features, n_classes,
                drift_type, n_drifts, drift_offset,
                rng=np.random.RandomState(seed + c),
            )
            client_data[c] = {
                "features": features,
                "labels": labels,
                "timestamps": timestamps,
                "drift_type": drift_type,
                "n_drifts": n_drifts,
            }

        return {
            "client_data": client_data,
            "n_clients": n_clients,
            "n_features": n_features,
            "n_classes": n_classes,
            "drift_type": drift_type,
            "synthetic": True,
        }

    def _generate_stream(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int,
        drift_type: str,
        n_drifts: int,
        drift_offset: float,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps = np.linspace(0, 1, n_samples)
        features = rng.randn(n_samples, n_features).astype(np.float32)

        # Generate decision boundaries that change over time
        concepts = []
        for _ in range(n_drifts + 1):
            w = rng.randn(n_features)
            w = w / (np.linalg.norm(w) + 1e-8)
            concepts.append(w)

        # Drift points
        drift_points = np.linspace(0, 1, n_drifts + 2)[1:-1] + drift_offset
        drift_points = np.clip(drift_points, 0.05, 0.95)

        labels = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            t = timestamps[i]

            if drift_type == "sudden":
                concept_idx = np.searchsorted(drift_points, t)
                w = concepts[concept_idx]
            elif drift_type == "gradual":
                concept_idx = np.searchsorted(drift_points, t)
                if concept_idx > 0 and concept_idx <= len(drift_points):
                    dp = drift_points[concept_idx - 1]
                    width = 0.05
                    alpha = np.clip((t - dp) / width, 0, 1)
                    w = (1 - alpha) * concepts[concept_idx - 1] + alpha * concepts[concept_idx]
                else:
                    w = concepts[concept_idx]
            elif drift_type == "recurring":
                concept_idx = np.searchsorted(drift_points, t) % len(concepts)
                w = concepts[concept_idx]
            else:  # incremental
                progress = t
                idx = int(progress * (len(concepts) - 1))
                idx = min(idx, len(concepts) - 2)
                alpha = progress * (len(concepts) - 1) - idx
                w = (1 - alpha) * concepts[idx] + alpha * concepts[idx + 1]

            score = features[i] @ w
            if n_classes == 2:
                prob = 1 / (1 + np.exp(-score * 3))
                labels[i] = int(rng.rand() < prob)
            else:
                class_centers = np.linspace(-2.0, 2.0, n_classes)
                class_logits = -((score - class_centers) ** 2)
                class_logits -= class_logits.max()
                class_probs = np.exp(class_logits)
                class_probs /= class_probs.sum()
                labels[i] = rng.choice(n_classes, p=class_probs)

        return features, labels, timestamps

    def get_temporal_windows(
        self,
        client_data: dict,
        window_size: int = 200,
        stride: int = 100,
    ) -> list[dict[str, np.ndarray]]:
        """Split a client's stream into temporal windows for meta-tasks."""
        features = client_data["features"]
        labels = client_data["labels"]
        timestamps = client_data["timestamps"]
        n = len(labels)

        windows = []
        start = 0
        while start + window_size <= n:
            end = start + window_size
            windows.append({
                "features": features[start:end],
                "labels": labels[start:end],
                "timestamps": timestamps[start:end],
                "t_start": float(timestamps[start]),
                "t_end": float(timestamps[end - 1]),
            })
            start += stride

        return windows


class FederatedTemporalDataset:
    """Federated dataset with temporal concept drift.

    Supports creating meta-learning tasks from temporal windows.
    """

    def __init__(self, data_info: dict[str, Any]):
        self.client_data = data_info["client_data"]
        self.n_clients = data_info["n_clients"]
        self.n_features = data_info["n_features"]
        self.n_classes = data_info["n_classes"]
        self.client_ids = sorted(self.client_data.keys())

    @property
    def num_clients(self) -> int:
        return self.n_clients

    def get_client_stream(self, client_id: int) -> dict:
        return self.client_data[client_id]

    def get_temporal_task(
        self,
        client_id: int,
        time_start: float,
        time_end: float,
        k_shot: int = 20,
        seed: int = 42,
    ) -> dict[str, torch.Tensor]:
        """Create a K-shot task from a time window."""
        rng = np.random.RandomState(seed)
        data = self.client_data[client_id]
        ts = data["timestamps"]

        mask = (ts >= time_start) & (ts <= time_end)
        idx = np.where(mask)[0]

        if len(idx) < k_shot * 2:
            # Not enough data, use all and split
            rng.shuffle(idx)
            mid = len(idx) // 2
            s_idx, q_idx = idx[:mid], idx[mid:]
        else:
            rng.shuffle(idx)
            s_idx = idx[:k_shot]
            q_idx = idx[k_shot : k_shot * 2]

        return {
            "support_x": torch.FloatTensor(data["features"][s_idx]),
            "support_y": torch.LongTensor(data["labels"][s_idx]),
            "query_x": torch.FloatTensor(data["features"][q_idx]),
            "query_y": torch.LongTensor(data["labels"][q_idx]),
            "t_start": time_start,
            "t_end": time_end,
        }

    def create_meta_tasks(
        self,
        client_id: int,
        n_tasks: int = 10,
        k_shot: int = 20,
        window_frac: float = 0.1,
        seed: int = 42,
    ) -> list[dict[str, torch.Tensor]]:
        """Create multiple meta-tasks by sampling temporal windows."""
        rng = np.random.RandomState(seed)
        tasks = []
        for i in range(n_tasks):
            t_start = rng.uniform(0, 1 - window_frac)
            t_end = t_start + window_frac
            task = self.get_temporal_task(
                client_id, t_start, t_end, k_shot, seed=seed + i,
            )
            tasks.append(task)
        return tasks

    def get_train_test_split(
        self, test_frac: float = 0.2, seed: int = 42,
    ) -> tuple[list[int], list[int]]:
        """Split clients into train/test."""
        if self.n_clients < 2:
            raise ValueError("At least two clients are required for train/test evaluation")
        if not 0 < test_frac < 1:
            raise ValueError("test_frac must be between 0 and 1")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(self.n_clients)
        n_test = max(1, int(self.n_clients * test_frac))
        test_ids = [self.client_ids[i] for i in perm[:n_test]]
        train_ids = [self.client_ids[i] for i in perm[n_test:]]
        return train_ids, test_ids
