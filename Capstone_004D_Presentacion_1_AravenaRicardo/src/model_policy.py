from __future__ import annotations

from typing import Any


def evaluate_model_policy(
    monitoring_report: dict[str, Any],
) -> dict[str, Any]:

    overall_status = monitoring_report.get(
        "overall_status",
        "unknown",
    )

    if overall_status == "stable":
        return {
            "decision": "continue",
            "severity": "normal",
            "allow_current_model": True,
            "manual_review_required": False,
            "automatic_retraining": False,
            "automatic_model_replacement": False,
            "action": (
                "Mantener el modelo actual y continuar "
                "el monitoreo periódico."
            ),
        }

    if overall_status == "watch":
        return {
            "decision": "continue_with_monitoring",
            "severity": "warning",
            "allow_current_model": True,
            "manual_review_required": False,
            "automatic_retraining": False,
            "automatic_model_replacement": False,
            "action": (
                "Mantener el modelo actual, aumentar la "
                "vigilancia y revisar nuevamente cuando "
                "exista un nuevo período de datos."
            ),
        }

    if overall_status in {
        "high_drift",
        "high_degradation",
    }:
        return {
            "decision": "manual_review_required",
            "severity": "critical",
            "allow_current_model": True,
            "manual_review_required": True,
            "automatic_retraining": False,
            "automatic_model_replacement": False,
            "action": (
                "Requerir revisión técnica antes de "
                "promover, reemplazar o reentrenar el modelo."
            ),
        }

    return {
        "decision": "manual_review_required",
        "severity": "unknown",
        "allow_current_model": False,
        "manual_review_required": True,
        "automatic_retraining": False,
        "automatic_model_replacement": False,
        "action": (
            "Revisar el estado del sistema antes de "
            "continuar con decisiones de modelo."
        ),
    }


def build_policy_report(
    monitoring_report: dict[str, Any],
) -> dict[str, Any]:

    policy = evaluate_model_policy(
        monitoring_report
    )

    return {
        "model_version": monitoring_report.get(
            "model_version"
        ),
        "baseline_year": monitoring_report.get(
            "baseline_year"
        ),
        "monitoring_year": monitoring_report.get(
            "monitoring_year"
        ),
        "monitoring_status": monitoring_report.get(
            "overall_status"
        ),
        "data_drift_status": (
            monitoring_report
            .get("data_drift", {})
            .get("status")
        ),
        "performance_status": (
            monitoring_report
            .get("performance", {})
            .get("status")
        ),
        "alerts": monitoring_report.get(
            "alerts",
            []
        ),
        "policy": policy,
    }
