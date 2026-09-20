import json

import pytest

from generate_monitoring_artifact import (
    generate_monitoring_artifact,
    validate_drift_report,
    validate_performance_report,
)


def sample_drift():
    return {
        "overall_status": "stable",
        "max_psi": 0.08,
        "features": {
            "NEM": {
                "psi": 0.01,
                "status": "stable",
            }
        },
    }


def sample_performance():
    return {
        "overall_status": "watch",
        "max_degradation": 0.03,
        "metrics": {
            "average_precision": {
                "degradation": 0.03,
                "status": "watch",
            }
        },
    }


def test_generator_creates_json(tmp_path):
    output = (
        tmp_path
        / "model_monitoring.json"
    )

    report = generate_monitoring_artifact(
        drift_report=sample_drift(),
        performance_report=sample_performance(),
        output_path=output,
    )

    assert output.exists()
    assert report["overall_status"] == "watch"

    stored = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert stored["overall_status"] == "watch"
    assert stored["model_version"] == "rf-2024-v1.0.0"


def test_generator_creates_alert():
    output = "test_monitoring_temp.json"

    try:
        report = generate_monitoring_artifact(
            drift_report=sample_drift(),
            performance_report=sample_performance(),
            output_path=output,
        )

        assert len(report["alerts"]) == 1
        assert (
            report["alerts"][0]["metric"]
            == "average_precision"
        )

    finally:
        from pathlib import Path

        Path(output).unlink(
            missing_ok=True
        )


def test_invalid_drift_report():
    with pytest.raises(ValueError):
        validate_drift_report({
            "overall_status": "stable"
        })


def test_invalid_performance_report():
    with pytest.raises(ValueError):
        validate_performance_report({
            "overall_status": "stable"
        })
