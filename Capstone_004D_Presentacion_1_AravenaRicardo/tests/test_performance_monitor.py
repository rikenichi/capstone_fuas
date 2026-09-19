from performance_monitor import (
    classify_degradation,
    compare_performance,
    metric_degradation,
)


def test_auc_drop_is_degradation():
    degradation = metric_degradation(
        "roc_auc",
        0.80,
        0.78,
    )

    assert round(degradation, 2) == 0.02


def test_fnr_increase_is_degradation():
    degradation = metric_degradation(
        "fnr",
        0.15,
        0.18,
    )

    assert round(degradation, 2) == 0.03


def test_improvement_is_not_flagged():
    baseline = {
        "roc_auc": 0.80,
        "balanced_accuracy": 0.70,
        "recall": 0.80,
        "precision": 0.60,
        "f1": 0.68,
        "accuracy": 0.70,
        "average_precision": 0.74,
        "fnr": 0.20,
    }

    current = {
        "roc_auc": 0.82,
        "balanced_accuracy": 0.72,
        "recall": 0.82,
        "precision": 0.62,
        "f1": 0.70,
        "accuracy": 0.72,
        "average_precision": 0.76,
        "fnr": 0.18,
    }

    report = compare_performance(
        baseline,
        current,
    )

    assert report["overall_status"] == "stable"
    assert report["max_degradation"] == 0.0


def test_watch_threshold():
    assert classify_degradation(
        0.03
    ) == "watch"


def test_high_degradation_threshold():
    assert classify_degradation(
        0.06
    ) == "high_degradation"
