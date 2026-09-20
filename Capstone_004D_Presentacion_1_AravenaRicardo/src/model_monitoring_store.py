from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONITORING_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
    / "model_monitoring.json"
)


@lru_cache(maxsize=1)
def get_monitoring_report() -> dict:
    """
    Carga el artefacto versionado de monitoreo.

    Se cachea durante la vida del proceso porque el
    archivo no cambia dentro de un mismo despliegue.
    """

    if not MONITORING_PATH.exists():
        raise FileNotFoundError(
            f"No existe el artefacto de monitoreo: "
            f"{MONITORING_PATH}"
        )

    with MONITORING_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    required = {
        "model_version",
        "baseline_year",
        "monitoring_year",
        "data_drift",
        "performance",
        "overall_status",
        "alerts",
    }

    missing = required - set(report)

    if missing:
        raise ValueError(
            "Artefacto de monitoreo incompleto. "
            f"Faltan: {sorted(missing)}"
        )

    return report
