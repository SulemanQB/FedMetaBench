"""Fairness metrics for FairFedMeta evaluation.

Implements:
- Demographic Parity (DP)
- Equalized Odds (EO)
- Equal Opportunity (EOpp)
- Accuracy Parity
- Worst-group accuracy
- Fairness-accuracy tradeoff curves
"""

from __future__ import annotations

import numpy as np
from typing import Any


def demographic_parity(
    predictions: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int | None = None,
) -> dict[str, float]:
    """Demographic Parity: P(Y_hat=1 | G=g) should be equal across groups."""
    if n_groups is None:
        n_groups = len(np.unique(group_ids))

    group_rates = {}
    for g in range(n_groups):
        mask = group_ids == g
        if mask.sum() > 0:
            group_rates[g] = float(predictions[mask].mean())

    rates = list(group_rates.values())
    return {
        "group_positive_rates": group_rates,
        "dp_gap": max(rates) - min(rates) if len(rates) >= 2 else 0.0,
        "dp_ratio": min(rates) / max(rates) if max(rates) > 0 and len(rates) >= 2 else 1.0,
    }


def equalized_odds(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int | None = None,
) -> dict[str, float]:
    """Equalized Odds: P(Y_hat=1 | Y=y, G=g) should be equal across groups for both y=0,1."""
    if n_groups is None:
        n_groups = len(np.unique(group_ids))

    tpr_by_group = {}
    fpr_by_group = {}

    for g in range(n_groups):
        mask = group_ids == g
        if mask.sum() == 0:
            continue
        g_preds = predictions[mask]
        g_labels = labels[mask]

        pos_mask = g_labels == 1
        neg_mask = g_labels == 0

        tpr = float(g_preds[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
        fpr = float(g_preds[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
        tpr_by_group[g] = tpr
        fpr_by_group[g] = fpr

    tprs = list(tpr_by_group.values())
    fprs = list(fpr_by_group.values())

    return {
        "tpr_by_group": tpr_by_group,
        "fpr_by_group": fpr_by_group,
        "eo_tpr_gap": max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0,
        "eo_fpr_gap": max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0,
        "eo_gap": max(
            max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0,
            max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0,
        ),
    }


def equal_opportunity(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int | None = None,
) -> dict[str, float]:
    """Equal Opportunity: P(Y_hat=1 | Y=1, G=g) should be equal across groups."""
    if n_groups is None:
        n_groups = len(np.unique(group_ids))

    tpr_by_group = {}
    for g in range(n_groups):
        mask = (group_ids == g) & (labels == 1)
        if mask.sum() > 0:
            tpr_by_group[g] = float(predictions[mask].mean())

    tprs = list(tpr_by_group.values())
    return {
        "tpr_by_group": tpr_by_group,
        "eopp_gap": max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0,
    }


def accuracy_parity(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int | None = None,
) -> dict[str, float]:
    """Accuracy Parity: accuracy should be equal across groups."""
    if n_groups is None:
        n_groups = len(np.unique(group_ids))

    acc_by_group = {}
    for g in range(n_groups):
        mask = group_ids == g
        if mask.sum() > 0:
            acc_by_group[g] = float((predictions[mask] == labels[mask]).mean())

    accs = list(acc_by_group.values())
    return {
        "accuracy_by_group": acc_by_group,
        "accuracy_gap": max(accs) - min(accs) if len(accs) >= 2 else 0.0,
        "worst_group_accuracy": min(accs) if accs else 0.0,
        "best_group_accuracy": max(accs) if accs else 0.0,
        "accuracy_std": float(np.std(accs)) if accs else 0.0,
    }


def comprehensive_fairness_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int | None = None,
    group_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Combine the implemented parity and group-accuracy metrics."""
    dp = demographic_parity(predictions, group_ids, n_groups)
    eo = equalized_odds(predictions, labels, group_ids, n_groups)
    eopp = equal_opportunity(predictions, labels, group_ids, n_groups)
    ap = accuracy_parity(predictions, labels, group_ids, n_groups)

    overall_acc = float((predictions == labels).mean())

    report = {
        "overall_accuracy": overall_acc,
        "demographic_parity": dp,
        "equalized_odds": eo,
        "equal_opportunity": eopp,
        "accuracy_parity": ap,
        # Summary scores
        "fairness_summary": {
            "dp_gap": dp["dp_gap"],
            "eo_gap": eo["eo_gap"],
            "eopp_gap": eopp["eopp_gap"],
            "acc_gap": ap["accuracy_gap"],
            "worst_group_acc": ap["worst_group_accuracy"],
        },
    }

    if group_names:
        report["group_names"] = group_names

    return report


def fairness_accuracy_tradeoff(
    results_list: list[dict[str, float]],
    accuracy_key: str = "overall_accuracy",
    fairness_key: str = "accuracy_gap",
) -> dict[str, Any]:
    """Compute fairness-accuracy tradeoff curve from multiple experiment results.

    Returns Pareto-optimal points and area under tradeoff curve.
    """
    points = []
    for r in results_list:
        acc = r.get(accuracy_key, 0)
        fair = r.get(fairness_key, 0)
        points.append((acc, fair))

    points.sort(key=lambda x: x[0])
    accs = [p[0] for p in points]
    gaps = [p[1] for p in points]

    # Find Pareto front (higher acc, lower gap is better)
    pareto = []
    best_gap = float("inf")
    for acc, gap in sorted(points, key=lambda x: -x[0]):
        if gap <= best_gap:
            pareto.append((acc, gap))
            best_gap = gap

    return {
        "accuracies": accs,
        "fairness_gaps": gaps,
        "pareto_front": pareto,
        "n_experiments": len(results_list),
    }
