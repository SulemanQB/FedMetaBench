"""Drift detection for FedMeta-Temporal+.

Implements:
- Page-Hinkley (PH) test for drift detection
- ADWIN-style sliding window detector
- Per-client drift monitoring
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Any


class PageHinkleyDetector:
    """Page-Hinkley test for concept drift detection.

    Monitors a stream of loss values and signals drift when
    the cumulative deviation exceeds a threshold.
    """

    def __init__(
        self,
        delta: float = 0.005,  # minimum magnitude of change to detect
        threshold: float = 50.0,  # detection sensitivity
        alpha: float = 0.999,  # forgetting factor for running mean
    ):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = float("inf")
        self._drift_detected = False

    def update(self, value: float) -> bool:
        """Update with new observation. Returns True if drift detected."""
        self.n += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        self.sum += value - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)

        if self.sum - self.min_sum > self.threshold:
            self._drift_detected = True
            return True
        return False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected


class SlidingWindowDetector:
    """ADWIN-style sliding window drift detector.

    Compares statistics of recent window vs. older window.
    """

    def __init__(
        self,
        window_size: int = 100,
        significance: float = 2.0,
    ):
        self.window_size = window_size
        self.significance = significance
        self.values = deque(maxlen=window_size * 2)
        self._drift_detected = False

    def update(self, value: float) -> bool:
        self.values.append(value)

        if len(self.values) < self.window_size * 2:
            return False

        vals = list(self.values)
        mid = len(vals) // 2
        old_window = np.array(vals[:mid])
        new_window = np.array(vals[mid:])

        old_mean = old_window.mean()
        new_mean = new_window.mean()
        pooled_std = np.sqrt(
            (old_window.var() + new_window.var()) / 2 + 1e-8
        )

        z_score = abs(new_mean - old_mean) / pooled_std
        if z_score > self.significance:
            self._drift_detected = True
            return True
        return False

    def reset(self):
        self.values.clear()
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected


class ClientDriftMonitor:
    """Monitors drift across multiple federated clients.

    Tracks per-client drift status and triggers re-adaptation.
    """

    def __init__(
        self,
        n_clients: int,
        detector_type: str = "page_hinkley",
        **detector_kwargs,
    ):
        self.n_clients = n_clients
        if detector_type == "page_hinkley":
            self.detectors = {
                i: PageHinkleyDetector(**detector_kwargs)
                for i in range(n_clients)
            }
        else:
            self.detectors = {
                i: SlidingWindowDetector(**detector_kwargs)
                for i in range(n_clients)
            }
        self.drift_history: dict[int, list[int]] = {i: [] for i in range(n_clients)}
        self.step = 0

    def update(self, client_id: int, loss_value: float) -> bool:
        """Update drift detector for a client."""
        drift = self.detectors[client_id].update(loss_value)
        if drift:
            self.drift_history[client_id].append(self.step)
            self.detectors[client_id].reset()
        return drift

    def advance_step(self):
        self.step += 1

    def get_drifted_clients(self) -> list[int]:
        """Get clients that recently detected drift."""
        return [
            cid for cid, det in self.detectors.items()
            if det.drift_detected
        ]

    def get_summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "total_drifts": {
                cid: len(hist) for cid, hist in self.drift_history.items()
            },
            "drift_times": self.drift_history,
        }
