"""Environmental monitoring metrics for FedMeta-Env.

Regression metrics + station-level generalization analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Any


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Standard regression metrics."""
    residuals = predictions - targets
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = mse ** 0.5

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
    }


def k_shot_scaling_analysis(
    results_by_k: dict[int, dict[str, float]],
    metric: str = "rmse",
) -> dict[str, Any]:
    """Analyze how performance scales with K-shot (number of adaptation days)."""
    k_values = sorted(results_by_k.keys())
    metric_values = [results_by_k[k].get(metric, float("inf")) for k in k_values]

    # Relative improvement from K=1
    if metric_values[0] > 0:
        rel_improvements = [(metric_values[0] - v) / metric_values[0] for v in metric_values]
    else:
        rel_improvements = [0.0] * len(k_values)

    return {
        "k_values": k_values,
        "metric_values": metric_values,
        "relative_improvements": rel_improvements,
        "metric_name": metric,
    }


def loso_summary(
    station_results: dict[int, dict[str, float]],
    metric: str = "rmse",
) -> dict[str, float]:
    """Summarize Leave-One-Station-Out results across all stations."""
    values = [r[metric] for r in station_results.values() if metric in r]
    return {
        f"mean_{metric}": float(np.mean(values)),
        f"std_{metric}": float(np.std(values)),
        f"worst_{metric}": float(np.max(values)) if "error" in metric or metric in ("mse", "rmse", "mae") else float(np.min(values)),
        f"best_{metric}": float(np.min(values)) if "error" in metric or metric in ("mse", "rmse", "mae") else float(np.max(values)),
        "n_stations": len(values),
    }
