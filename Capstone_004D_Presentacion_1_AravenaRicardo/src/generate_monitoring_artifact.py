from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_monitoring import build_model_monitoring_report


def load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_drift_report(report: dict) -> None:
    required = {
        "overall_status",
        "max_psi",
        "features",
    }

    missing = required - set(report)

    if missing:
        raise ValueError(
            "Reporte de drift incompleto. "
            f"Faltan: {sorted(missing)}"
        )


def validate_performance_report(report: dict) -> None:
    required = {
        "overall_status",
        "max_degradation",
        "metrics",
    }

    missing = required - set(report)

    if missing:
        raise ValueError(
            "Reporte de performance incompleto. "
            f"Faltan: {sorted(missing)}"
        )


def generate_monitoring_artifact(
    drift_report: dict,
    performance_report: dict,
    output_path: str | Path,
    model_version: str = "rf-2024-v1.0.0",
) -> dict:

    validate_drift_report(
        drift_report
    )

    validate_performance_report(
        performance_report
    )

    report = build_model_monitoring_report(
        data_drift_report=drift_report,
        performance_report=performance_report,
        model_version=model_version,
    )

    report["note"] = (
        "Los estados corresponden a criterios "
        "operativos heurísticos de monitoreo."
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


def generate_from_files(
    drift_path: str | Path,
    performance_path: str | Path,
    output_path: str | Path,
    model_version: str = "rf-2024-v1.0.0",
) -> dict:

    drift_report = load_json(
        drift_path
    )

    performance_report = load_json(
        performance_path
    )

    return generate_monitoring_artifact(
        drift_report=drift_report,
        performance_report=performance_report,
        output_path=output_path,
        model_version=model_version,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera el artefacto unificado "
            "de monitoreo del modelo."
        )
    )

    parser.add_argument(
        "--drift",
        required=True,
        help="Ruta al reporte JSON de data drift",
    )

    parser.add_argument(
        "--performance",
        required=True,
        help="Ruta al reporte JSON de performance",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida model_monitoring.json",
    )

    parser.add_argument(
        "--model-version",
        default="rf-2024-v1.0.0",
    )

    args = parser.parse_args()

    report = generate_from_files(
        drift_path=args.drift,
        performance_path=args.performance,
        output_path=args.output,
        model_version=args.model_version,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
