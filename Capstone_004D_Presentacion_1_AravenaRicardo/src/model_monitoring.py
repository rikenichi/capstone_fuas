from pathlib import Path
import json


STATUS_PRIORITY = {
    "stable": 0,
    "watch": 1,
    "high_drift": 2,
    "high_degradation": 2,
}


def _worst_status(*statuses):
    return max(
        statuses,
        key=lambda s: STATUS_PRIORITY.get(s, 0)
    )


def build_model_monitoring_report(
    data_drift_report: dict,
    performance_report: dict,
    model_version: str = "rf-2024-v1.0.0",
) -> dict:

    data_status = data_drift_report["overall_status"]
    performance_status = performance_report["overall_status"]

    overall_status = _worst_status(
        data_status,
        performance_status,
    )

    alerts = []

    for feature, info in data_drift_report["features"].items():
        if info["status"] != "stable":
            alerts.append({
                "type": "data_drift",
                "feature": feature,
                "psi": info["psi"],
                "status": info["status"],
            })

    for metric, info in performance_report["metrics"].items():
        if info["status"] != "stable":
            alerts.append({
                "type": "performance",
                "metric": metric,
                "degradation": info["degradation"],
                "status": info["status"],
            })

    return {
        "model_version": model_version,
        "baseline_year": 2024,
        "monitoring_year": 2025,
        "data_drift": {
            "status": data_status,
            "max_psi": data_drift_report["max_psi"],
        },
        "performance": {
            "status": performance_status,
            "max_degradation": performance_report["max_degradation"],
        },
        "overall_status": overall_status,
        "alerts": alerts,
    }
