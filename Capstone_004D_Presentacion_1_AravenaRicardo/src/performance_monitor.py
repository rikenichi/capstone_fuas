from __future__ import annotations

from typing import Any


HIGHER_IS_BETTER = {
    "roc_auc",
    "balanced_accuracy",
    "recall",
    "precision",
    "f1",
    "accuracy",
    "average_precision",
}

LOWER_IS_BETTER = {
    "fnr",
}


def classify_degradation(
    degradation: float,
    watch_threshold: float = 0.02,
    high_threshold: float = 0.05,
) -> str:
    """
    Clasificación operativa del deterioro absoluto.
    """

    if degradation < watch_threshold:
        return "stable"

    if degradation < high_threshold:
        return "watch"

    return "high_degradation"


def metric_degradation(
    metric: str,
    baseline: float,
    current: float,
) -> float:
    """
    Devuelve deterioro positivo cuando el rendimiento empeora.

    Ejemplo:
    AUC 0.80 -> 0.78 = deterioro +0.02
    FNR 0.15 -> 0.17 = deterioro +0.02
    """

    if metric in HIGHER_IS_BETTER:
        return baseline - current

    if metric in LOWER_IS_BETTER:
        return current - baseline

    raise ValueError(
        f"No se conoce la dirección de la métrica: {metric}"
    )


def compare_performance(
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
    watch_threshold: float = 0.02,
    high_threshold: float = 0.05,
) -> dict[str, Any]:

    monitored = (
        HIGHER_IS_BETTER
        | LOWER_IS_BETTER
    )

    results = {}

    for metric in sorted(monitored):
        if metric not in baseline_metrics:
            raise ValueError(
                f"Falta {metric} en baseline"
            )

        if metric not in current_metrics:
            raise ValueError(
                f"Falta {metric} en current"
            )

        baseline = float(
            baseline_metrics[metric]
        )

        current = float(
            current_metrics[metric]
        )

        raw_delta = current - baseline

        degradation = metric_degradation(
            metric,
            baseline,
            current,
        )

        # Si mejora, no se considera deterioro.
        effective_degradation = max(
            0.0,
            degradation,
        )

        status = classify_degradation(
            effective_degradation,
            watch_threshold,
            high_threshold,
        )

        results[metric] = {
            "baseline": round(baseline, 6),
            "current": round(current, 6),
            "raw_delta": round(raw_delta, 6),
            "degradation": round(
                effective_degradation,
                6,
            ),
            "direction": (
                "higher_is_better"
                if metric in HIGHER_IS_BETTER
                else "lower_is_better"
            ),
            "status": status,
        }

    priority = {
        "stable": 0,
        "watch": 1,
        "high_degradation": 2,
    }

    worst_status = max(
        (
            item["status"]
            for item in results.values()
        ),
        key=lambda x: priority[x],
    )

    max_degradation = max(
        item["degradation"]
        for item in results.values()
    )

    return {
        "metrics": results,
        "watch_threshold": watch_threshold,
        "high_threshold": high_threshold,
        "max_degradation": round(
            max_degradation,
            6,
        ),
        "overall_status": worst_status,
    }
