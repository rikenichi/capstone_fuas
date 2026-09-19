from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "NEM",
    "QUINTIL_SE4",
    "COD_DEPE",
    "EDAD",
    "GENERO",
    "NACIONALIDAD",
]


def classify_psi(psi: float) -> str:
    """
    Clasificación operativa heurística.
    """
    if psi < 0.10:
        return "stable"

    if psi < 0.25:
        return "watch"

    return "high_drift"


def _safe_distribution(counts: pd.Series, epsilon=1e-6):
    proportions = counts.astype(float)

    total = proportions.sum()

    if total == 0:
        return proportions

    proportions = proportions / total
    proportions = proportions.clip(lower=epsilon)

    return proportions / proportions.sum()


def psi_categorical(
    baseline: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-6,
) -> float:

    base = (
        baseline
        .fillna("__MISSING__")
        .astype(str)
    )

    curr = (
        current
        .fillna("__MISSING__")
        .astype(str)
    )

    categories = sorted(
        set(base.unique())
        | set(curr.unique())
    )

    base_counts = (
        base.value_counts()
        .reindex(categories, fill_value=0)
    )

    curr_counts = (
        curr.value_counts()
        .reindex(categories, fill_value=0)
    )

    base_pct = _safe_distribution(
        base_counts,
        epsilon,
    )

    curr_pct = _safe_distribution(
        curr_counts,
        epsilon,
    )

    psi = (
        (curr_pct - base_pct)
        * np.log(curr_pct / base_pct)
    ).sum()

    return float(psi)


def psi_numeric(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:

    base = pd.to_numeric(
        baseline,
        errors="coerce",
    ).dropna()

    curr = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if len(base) == 0 or len(curr) == 0:
        return math.nan

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = np.unique(
        base.quantile(quantiles).values
    )

    if len(edges) < 3:
        # Si prácticamente no hay variación,
        # tratar como categórica.
        return psi_categorical(
            baseline,
            current,
            epsilon,
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    base_bins = pd.cut(
        base,
        bins=edges,
        include_lowest=True,
    )

    curr_bins = pd.cut(
        curr,
        bins=edges,
        include_lowest=True,
    )

    categories = base_bins.cat.categories

    base_counts = (
        base_bins.value_counts(sort=False)
        .reindex(categories, fill_value=0)
    )

    curr_counts = (
        curr_bins.value_counts(sort=False)
        .reindex(categories, fill_value=0)
    )

    base_pct = _safe_distribution(
        base_counts,
        epsilon,
    )

    curr_pct = _safe_distribution(
        curr_counts,
        epsilon,
    )

    psi = (
        (curr_pct - base_pct)
        * np.log(curr_pct / base_pct)
    ).sum()

    return float(psi)


def compare_feature_distributions(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features=None,
) -> dict[str, Any]:

    features = features or DEFAULT_FEATURES

    categorical = {
        "QUINTIL_SE4",
        "COD_DEPE",
        "GENERO",
        "NACIONALIDAD",
    }

    results = {}

    for feature in features:
        if feature not in baseline_df.columns:
            raise ValueError(
                f"Falta {feature} en baseline"
            )

        if feature not in current_df.columns:
            raise ValueError(
                f"Falta {feature} en current"
            )

        if feature in categorical:
            psi = psi_categorical(
                baseline_df[feature],
                current_df[feature],
            )

            feature_type = "categorical"

        else:
            psi = psi_numeric(
                baseline_df[feature],
                current_df[feature],
            )

            feature_type = "numeric"

        results[feature] = {
            "type": feature_type,
            "psi": round(float(psi), 6),
            "status": classify_psi(float(psi)),
            "baseline_missing_rate": round(
                float(
                    baseline_df[feature]
                    .isna()
                    .mean()
                ),
                6,
            ),
            "current_missing_rate": round(
                float(
                    current_df[feature]
                    .isna()
                    .mean()
                ),
                6,
            ),
        }

    max_psi = max(
        value["psi"]
        for value in results.values()
    )

    return {
        "baseline_rows": int(len(baseline_df)),
        "current_rows": int(len(current_df)),
        "features": results,
        "max_psi": max_psi,
        "overall_status": classify_psi(max_psi),
    }
