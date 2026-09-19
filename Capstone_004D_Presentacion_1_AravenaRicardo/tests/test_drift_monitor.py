import pandas as pd

from drift_monitor import (
    classify_psi,
    compare_feature_distributions,
    psi_categorical,
    psi_numeric,
)


def test_identical_numeric_has_low_psi():
    baseline = pd.Series(
        list(range(100)) * 5
    )

    current = baseline.copy()

    psi = psi_numeric(
        baseline,
        current,
    )

    assert psi < 0.01


def test_shifted_numeric_detects_drift():
    baseline = pd.Series(
        range(1000)
    )

    current = pd.Series(
        range(1000, 2000)
    )

    psi = psi_numeric(
        baseline,
        current,
    )

    assert psi >= 0.25


def test_categorical_drift():
    baseline = pd.Series(
        [1] * 90 + [2] * 10
    )

    current = pd.Series(
        [1] * 10 + [2] * 90
    )

    psi = psi_categorical(
        baseline,
        current,
    )

    assert psi >= 0.25


def test_compare_feature_distributions():
    baseline = pd.DataFrame({
        "NEM": [500, 550, 600, 650] * 20,
        "QUINTIL_SE4": [1, 2, 3, 4] * 20,
        "COD_DEPE": [1, 2, 3, 4] * 20,
        "EDAD": [18, 19, 20, 21] * 20,
        "GENERO": [1, 2, 1, 2] * 20,
        "NACIONALIDAD": [0, 0, 0, 1] * 20,
    })

    current = baseline.copy()

    report = compare_feature_distributions(
        baseline,
        current,
    )

    assert report["overall_status"] == "stable"
    assert report["max_psi"] < 0.10
    assert len(report["features"]) == 6


def test_psi_status_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "watch"
    assert classify_psi(0.30) == "high_drift"
