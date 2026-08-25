"""eICU dataset loader for FairFedMeta.

eICU Collaborative Research Database: 200,859 ICU stays across 208 hospitals.
Each hospital = natural federated client.
Sensitive attribute: ethnicity (used for fairness evaluation).

NOTE: Requires access to eICU via PhysioNet (credentialed access).
Provide preprocessed CSV/H5 file at data_dir/eicu_processed.h5

Expected format:
    features: (N, T, D) — temporal clinical features
    labels: (N,) — mortality/readmission outcome
    hospital_ids: (N,) — hospital ID for natural partitioning
    group_ids: (N,) — demographic group (ethnicity encoded)
    patient_ids: (N,) — unique patient identifiers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EICUDataset:
    """Wrapper for preprocessed eICU data."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        hospital_ids: np.ndarray,
        group_ids: np.ndarray,
    ):
        self.features = features  # (N, T, D) or (N, D)
        self.labels = labels      # (N,)
        self.hospital_ids = hospital_ids  # (N,)
        self.group_ids = group_ids  # (N,) — sensitive attribute

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EICUFederatedLoader:
    """Load eICU and partition by hospital (natural federation)."""

    # Demographic group mapping
    GROUP_NAMES = {0: "Caucasian", 1: "African American", 2: "Hispanic", 3: "Asian", 4: "Other"}

    def load(self, data_dir: str = "./data", **kwargs: Any):
        """Load preprocessed eICU data.

        Returns: (features, labels, hospital_ids, group_ids, metadata)
        """
        data_path = Path(data_dir) / "eicu_processed.h5"

        if data_path.exists():
            return self._load_h5(data_path)

        logger.warning(
            f"eICU data not found at {data_path}. "
            "Generating synthetic placeholder data for development."
        )
        return self._generate_synthetic(
            n_samples=int(kwargs.get("n_samples", 5000)),
            n_hospitals=int(kwargs.get("n_hospitals", 20)),
            n_groups=int(kwargs.get("n_groups", 5)),
            n_features=int(kwargs.get("n_features", 15)),
            seq_len=int(kwargs.get("seq_len", 48)),
        )

    def _load_h5(self, path: Path):
        """Load real eICU data from HDF5."""
        import h5py
        with h5py.File(path, "r") as f:
            features = np.array(f["features"])
            labels = np.array(f["labels"])
            hospital_ids = np.array(f["hospital_ids"])
            group_ids = np.array(f["group_ids"])

        metadata = {
            "n_samples": len(labels),
            "n_hospitals": len(np.unique(hospital_ids)),
            "n_groups": len(np.unique(group_ids)),
            "n_features": features.shape[-1],
            "prevalence": float(labels.mean()),
        }
        logger.info(f"eICU loaded: {metadata}")
        return features, labels, hospital_ids, group_ids, metadata

    def _generate_synthetic(self, n_samples: int = 5000, n_hospitals: int = 20,
                            n_groups: int = 5, n_features: int = 30, seq_len: int = 48):
        """Generate synthetic eICU-like data for development/testing."""
        rng = np.random.RandomState(42)

        # Assign hospitals and groups with realistic imbalance
        hospital_ids = rng.choice(n_hospitals, n_samples, p=_hospital_dist(n_hospitals, rng))
        group_ids = rng.choice(n_groups, n_samples, p=[0.65, 0.15, 0.10, 0.05, 0.05])

        # Generate features: each hospital has slightly different distribution
        features = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        for h in range(n_hospitals):
            mask = hospital_ids == h
            n_h = mask.sum()
            mu = rng.randn(n_features) * 0.3
            features[mask] = rng.randn(n_h, seq_len, n_features).astype(np.float32) + mu

        # Labels: mortality with hospital-specific and group-specific bias
        base_prob = 0.15
        logits = np.zeros(n_samples)
        for h in range(n_hospitals):
            mask = hospital_ids == h
            logits[mask] += rng.randn() * 0.3  # hospital effect
        for g in range(n_groups):
            mask = group_ids == g
            logits[mask] += rng.randn() * 0.2  # group-specific outcome disparity
        logits += features[:, -1, 0] * 0.5  # feature-driven component

        probs = 1 / (1 + np.exp(-logits + np.log(base_prob / (1 - base_prob))))
        labels = (rng.rand(n_samples) < probs).astype(np.int64)

        metadata = {
            "n_samples": n_samples,
            "n_hospitals": n_hospitals,
            "n_groups": n_groups,
            "n_features": n_features,
            "prevalence": float(labels.mean()),
            "synthetic": True,
        }
        logger.info(f"Synthetic eICU generated: {metadata}")
        return features, labels, hospital_ids, group_ids, metadata


def _hospital_dist(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Realistic hospital size distribution (log-normal)."""
    sizes = rng.lognormal(0, 1, n)
    return sizes / sizes.sum()


class FairFederatedDataset:
    """Federated dataset with fairness-aware group information.

    Extends the base federated concept with:
    - Per-client demographic group distributions
    - Group-stratified support/query splits for meta-learning
    - Client-group overlap statistics
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        hospital_ids: np.ndarray,
        group_ids: np.ndarray,
        metadata: dict,
    ):
        self.features = features
        self.labels = labels
        self.hospital_ids = hospital_ids
        self.group_ids = group_ids
        self.metadata = metadata
        self.n_groups = len(np.unique(group_ids))

        # Build per-hospital client data
        self.hospital_list = np.unique(hospital_ids)
        self.client_indices: dict[int, np.ndarray] = {}
        for h in self.hospital_list:
            self.client_indices[h] = np.where(hospital_ids == h)[0]

    @property
    def num_clients(self) -> int:
        return len(self.hospital_list)

    def get_client_data(self, hospital_id: int):
        """Get all data for a single hospital/client."""
        idx = self.client_indices[hospital_id]
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "group_ids": self.group_ids[idx],
            "indices": idx,
        }

    def get_client_group_distribution(self, hospital_id: int) -> np.ndarray:
        """Get demographic distribution for a client."""
        idx = self.client_indices[hospital_id]
        groups = self.group_ids[idx]
        dist = np.zeros(self.n_groups)
        for g in groups:
            dist[g] += 1
        return dist / max(dist.sum(), 1)

    def get_group_stratified_split(
        self, hospital_id: int, k_shot: int = 5, seed: int = 42,
    ) -> tuple[dict, dict]:
        """Split client data into support/query, stratified by group.

        Ensures each group is represented in support set for fair adaptation.
        Returns: (support_data, query_data) dicts with features/labels/group_ids.
        """
        rng = np.random.RandomState(seed)
        data = self.get_client_data(hospital_id)
        groups = data["group_ids"]
        unique_groups = np.unique(groups)

        support_mask = np.zeros(len(groups), dtype=bool)
        for g in unique_groups:
            g_mask = np.where(groups == g)[0]
            rng.shuffle(g_mask)
            k = min(k_shot, len(g_mask) - 1) if len(g_mask) > 1 else len(g_mask)
            support_mask[g_mask[:k]] = True

        query_mask = ~support_mask
        support = {k: v[support_mask] for k, v in data.items() if k != "indices"}
        query = {k: v[query_mask] for k, v in data.items() if k != "indices"}
        return support, query

    def create_meta_splits(
        self, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 42,
    ):
        """Split hospitals into meta-train/val/test."""
        rng = np.random.RandomState(seed)
        n = len(self.hospital_list)
        perm = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        self.meta_train_hospitals = self.hospital_list[perm[:n_train]].tolist()
        self.meta_val_hospitals = self.hospital_list[perm[n_train:n_train + n_val]].tolist()
        self.meta_test_hospitals = self.hospital_list[perm[n_train + n_val:]].tolist()
        return self.meta_train_hospitals, self.meta_val_hospitals, self.meta_test_hospitals

    def compute_fairness_statistics(self) -> dict:
        """Global fairness statistics across all clients."""
        group_rates = {}
        for g in range(self.n_groups):
            mask = self.group_ids == g
            if mask.sum() > 0:
                group_rates[g] = {
                    "count": int(mask.sum()),
                    "prevalence": float(self.labels[mask].mean()),
                    "fraction": float(mask.sum() / len(self.labels)),
                }
        return group_rates
