"""Air quality datasets for FedMetaEnv.

Supports Beijing Multi-Site Air Quality data and a synthetic station fallback.

Each station = federated client. Natural spatial heterogeneity.
Task: predict PM2.5 / AQI from meteorological + pollutant features.
Few-shot = K days of data for adaptation at a new/cold-start station.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Beijing 12 monitoring station names
BEIJING_STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi",
    "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
    "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
]

# Feature names for Beijing dataset
FEATURE_NAMES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "wd_sin", "wd_cos",  # wind direction encoded
    "hour_sin", "hour_cos", "month_sin", "month_cos",  # temporal features
]

TARGET_POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]


class AirQualityDataset(Dataset):
    """Single-station air quality dataset."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        station_id: int,
        window_size: int = 24,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.station_id = station_id
        self.window_size = window_size

    def __len__(self):
        return len(self.targets) - self.window_size + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return x, y


class BeijingAirQualityLoader:
    """Load Beijing Multi-Site Air Quality dataset.

    If real data is not available, generates synthetic data with spatial
    heterogeneity across 12 stations.
    """

    def load(
        self,
        data_dir: str = "./data",
        target_pollutant: str = "PM2.5",
        window_size: int = 24,
    ) -> dict[str, Any]:
        # Accept several common layouts for the UCI Beijing PRSA CSVs
        candidates = [
            Path(data_dir) / "beijing_air_quality",
            Path(data_dir) / "PRSA_Data_20130301-20170228",
            Path(data_dir),
        ]
        external_data_dir = os.environ.get("FEDMETAENV_DATA_DIR")
        if external_data_dir:
            candidates.append(Path(external_data_dir))
        data_path = None
        for cand in candidates:
            sample = cand / f"PRSA_Data_{BEIJING_STATIONS[0]}_20130301-20170228.csv"
            if sample.exists():
                data_path = cand
                break

        if data_path is not None:
            logger.info(f"Loading real Beijing air quality data from {data_path}")
            return self._load_real(data_path, target_pollutant, window_size)

        logger.warning(
            "Beijing data not found in %s. Generating synthetic data for development.",
            candidates[:3],
        )
        return self._generate_synthetic(target_pollutant, window_size)

    def _load_real(self, path: Path, target: str, window_size: int) -> dict:
        """Load real Beijing air quality CSV files."""
        import pandas as pd

        station_data = {}
        for i, station in enumerate(BEIJING_STATIONS):
            csv_path = path / f"PRSA_Data_{station}_20130301-20170228.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df = df.dropna()

            features = self._extract_features(df)
            target_col = df[target].values.astype(np.float32)

            station_data[i] = {
                "features": features,
                "targets": target_col,
                "station_name": station,
                "n_samples": len(features),
            }

        return {
            "station_data": station_data,
            "n_stations": len(station_data),
            "n_features": next(iter(station_data.values()))["features"].shape[1]
            if station_data else 17,
            "target_pollutant": target,
            "window_size": window_size,
            "synthetic": False,
        }

    def _extract_features(self, df) -> np.ndarray:
        """Extract and normalize features from DataFrame."""
        numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        feats = df[numeric_cols].values.astype(np.float32)

        # Encode wind direction
        wd_map = {"N": 0, "NE": 45, "E": 90, "SE": 135,
                   "S": 180, "SW": 225, "W": 270, "NW": 315}
        if "wd" in df.columns:
            angles = df["wd"].map(wd_map).fillna(0).values * np.pi / 180
            wd_sin = np.sin(angles).reshape(-1, 1)
            wd_cos = np.cos(angles).reshape(-1, 1)
        else:
            wd_sin = np.zeros((len(df), 1))
            wd_cos = np.zeros((len(df), 1))

        # Temporal features
        hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
        month = df["month"].values if "month" in df.columns else np.ones(len(df))
        hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
        hour_cos = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
        month_sin = np.sin(2 * np.pi * month / 12).reshape(-1, 1)
        month_cos = np.cos(2 * np.pi * month / 12).reshape(-1, 1)

        all_feats = np.hstack([feats, wd_sin, wd_cos, hour_sin, hour_cos, month_sin, month_cos])

        # Z-score normalize per feature
        mu = all_feats.mean(axis=0, keepdims=True)
        std = all_feats.std(axis=0, keepdims=True) + 1e-8
        return ((all_feats - mu) / std).astype(np.float32)

    def _generate_synthetic(
        self, target: str = "PM2.5", window_size: int = 24,
        n_hours: int = 8760,  # ~1 year
    ) -> dict[str, Any]:
        """Generate synthetic air quality data for 12 stations."""
        rng = np.random.RandomState(42)
        n_features = len(FEATURE_NAMES)
        n_stations = len(BEIJING_STATIONS)

        station_data = {}
        for i in range(n_stations):
            # Station-specific base pollution level (spatial heterogeneity)
            base_level = rng.uniform(40, 120)
            seasonal = 30 * np.sin(2 * np.pi * np.arange(n_hours) / (365 * 24))
            diurnal = 15 * np.sin(2 * np.pi * np.arange(n_hours) / 24 + rng.uniform(0, np.pi))
            noise = rng.randn(n_hours) * 20

            # Target: PM2.5-like values
            target_vals = np.maximum(base_level + seasonal + diurnal + noise, 0).astype(np.float32)

            # Features: correlated with target + station-specific offsets
            features = np.zeros((n_hours, n_features), dtype=np.float32)
            station_offset = rng.randn(n_features) * 0.3
            for j in range(n_features):
                correlation = rng.uniform(0.2, 0.8)
                features[:, j] = (
                    correlation * (target_vals - target_vals.mean()) / (target_vals.std() + 1e-8)
                    + (1 - correlation) * rng.randn(n_hours)
                    + station_offset[j]
                )

            # Normalize features
            mu = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - mu) / std

            # Normalize target
            target_mu = target_vals.mean()
            target_std = target_vals.std() + 1e-8
            target_vals = (target_vals - target_mu) / target_std

            station_data[i] = {
                "features": features,
                "targets": target_vals,
                "station_name": BEIJING_STATIONS[i],
                "n_samples": n_hours,
            }

        logger.info(f"Synthetic Beijing data: {n_stations} stations, {n_hours} hours each")
        return {
            "station_data": station_data,
            "n_stations": n_stations,
            "n_features": n_features,
            "target_pollutant": target,
            "window_size": window_size,
            "synthetic": True,
        }


class FederatedAirQuality:
    """Federated air quality dataset with LOSO evaluation and few-shot splits.

    Each station is a client. Supports:
    - Leave-One-Station-Out (LOSO) for generalization
    - K-shot adaptation (K days = K*24 hours of data)
    """

    def __init__(self, data_info: dict[str, Any], window_size: int = 24):
        self.station_data = data_info["station_data"]
        self.n_stations = data_info["n_stations"]
        if self.n_stations < 2:
            raise ValueError("At least two stations are required for LOSO evaluation")
        self.n_features = data_info["n_features"]
        self.window_size = window_size
        self.station_ids = sorted(self.station_data.keys())

    @property
    def num_clients(self) -> int:
        return self.n_stations

    def get_station_dataset(self, station_id: int) -> AirQualityDataset:
        """Get dataset for a single station."""
        d = self.station_data[station_id]
        return AirQualityDataset(
            d["features"], d["targets"], station_id, self.window_size
        )

    def get_loso_splits(self, held_out_station: int) -> tuple[list[int], int]:
        """Leave-One-Station-Out: return (train_stations, test_station)."""
        train = [s for s in self.station_ids if s != held_out_station]
        return train, held_out_station

    def get_k_shot_split(
        self, station_id: int, k_days: int = 7, seed: int = 42,
    ) -> dict[str, torch.Tensor]:
        """Get K-day support set and remaining query set for a station.

        K days = K * 24 hourly data points for few-shot adaptation.
        """
        d = self.station_data[station_id]
        features = d["features"]
        targets = d["targets"]
        n = len(targets) - self.window_size + 1

        k_hours = k_days * 24
        k_samples = min(k_hours, n // 2)

        # Use first k_samples as support (temporal order matters)
        support_indices = np.arange(k_samples)
        query_start = k_samples + self.window_size - 1
        query_indices = np.arange(query_start, n)

        def make_windows(indices):
            xs, ys = [], []
            for idx in indices:
                xs.append(features[idx : idx + self.window_size])
                ys.append(targets[idx + self.window_size - 1])
            return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

        sx, sy = make_windows(support_indices)
        qx, qy = make_windows(query_indices)

        return {
            "support_x": sx,
            "support_y": sy,
            "query_x": qx,
            "query_y": qy,
            "station_id": station_id,
            "k_days": k_days,
        }

    def get_all_loso_experiments(
        self, k_days: int = 7,
    ) -> list[dict]:
        """Generate LOSO experiments for all stations."""
        experiments = []
        for test_station in self.station_ids:
            train_stations, _ = self.get_loso_splits(test_station)
            experiments.append({
                "train_stations": train_stations,
                "test_station": test_station,
                "test_split": self.get_k_shot_split(test_station, k_days),
            })
        return experiments
