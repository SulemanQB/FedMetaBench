"""Temporal metrics for FedMetaTemporal.

Drift-aware evaluation metrics:
- Prequential accuracy (test-then-train)
- Post-drift recovery speed
- Adaptation efficiency over time
"""

from __future__ import annotations

import numpy as np
from typing import Any


def prequential_accuracy(
    predictions_over_time: list[np.ndarray],
    labels_over_time: list[np.ndarray],
    fading_factor: float = 0.99,
) -> dict[str, Any]:
    """Prequential (interleaved test-then-train) accuracy with fading factor."""
    weighted_correct = 0.0
    weighted_total = 0.0
    accuracy_curve = []

    for preds, labels in zip(predictions_over_time, labels_over_time):
        correct = (preds == labels).sum()
        total = len(labels)

        weighted_correct = fading_factor * weighted_correct + correct
        weighted_total = fading_factor * weighted_total + total

        acc = weighted_correct / (weighted_total + 1e-8)
        accuracy_curve.append(float(acc))

    return {
        "final_accuracy": accuracy_curve[-1] if accuracy_curve else 0.0,
        "accuracy_curve": accuracy_curve,
        "mean_accuracy": float(np.mean(accuracy_curve)) if accuracy_curve else 0.0,
    }


def post_drift_recovery(
    accuracy_curve: list[float],
    drift_points: list[int],
    recovery_threshold: float = 0.9,
) -> dict[str, Any]:
    """Measure how quickly accuracy recovers after drift events.

    Returns recovery steps (number of steps to reach threshold × pre-drift accuracy).
    """
    recoveries = []

    for dp in drift_points:
        if dp >= len(accuracy_curve) or dp == 0:
            continue

        pre_drift_acc = accuracy_curve[dp - 1]
        target = recovery_threshold * pre_drift_acc

        recovery_steps = None
        for t in range(dp, len(accuracy_curve)):
            if accuracy_curve[t] >= target:
                recovery_steps = t - dp
                break

        recoveries.append({
            "drift_point": dp,
            "pre_drift_accuracy": pre_drift_acc,
            "recovery_steps": recovery_steps,
            "recovered": recovery_steps is not None,
        })

    avg_recovery = np.mean([
        r["recovery_steps"] for r in recoveries
        if r["recovered"] and r["recovery_steps"] is not None
    ]) if recoveries else float("inf")

    return {
        "per_drift": recoveries,
        "avg_recovery_steps": float(avg_recovery),
        "recovery_rate": sum(1 for r in recoveries if r["recovered"]) / max(len(recoveries), 1),
    }


def temporal_generalization_gap(
    train_accs: list[float],
    test_accs: list[float],
) -> dict[str, float]:
    """Measure generalization gap over time (should remain small)."""
    gaps = [tr - te for tr, te in zip(train_accs, test_accs)]
    return {
        "mean_gap": float(np.mean(gaps)),
        "max_gap": float(np.max(gaps)) if gaps else 0.0,
        "gap_trend": float(np.polyfit(range(len(gaps)), gaps, 1)[0]) if len(gaps) > 1 else 0.0,
    }


def adaptation_efficiency(
    k_shot_results: dict[int, float],
) -> dict[str, float]:
    """How efficiently the model adapts with increasing K-shot data."""
    ks = sorted(k_shot_results.keys())
    accs = [k_shot_results[k] for k in ks]

    if len(ks) < 2:
        return {"efficiency": 0.0}

    # Compute area under K-shot curve (higher = better)
    auc = float(np.trapz(accs, ks))

    # Marginal improvement per additional shot
    marginal = []
    for i in range(1, len(ks)):
        dk = ks[i] - ks[i - 1]
        da = accs[i] - accs[i - 1]
        marginal.append(da / dk if dk > 0 else 0.0)

    return {
        "auc": auc,
        "marginal_improvements": marginal,
        "saturation_k": ks[np.argmin(marginal)] if marginal else ks[-1],
    }
