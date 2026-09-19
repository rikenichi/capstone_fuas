from model_monitoring import build_model_monitoring_report


def test_monitoring_report_watch():
    data = {
        "overall_status": "stable",
        "max_psi": 0.09,
        "features": {
            "NEM": {
                "status": "stable",
                "psi": 0.01
            }
        }
    }

    performance = {
        "overall_status": "watch",
        "max_degradation": 0.03,
        "metrics": {
            "average_precision": {
                "status": "watch",
                "degradation": 0.03
            }
        }
    }

    report = build_model_monitoring_report(
        data,
        performance,
    )

    assert report["overall_status"] == "watch"
    assert len(report["alerts"]) == 1
    assert report["alerts"][0]["metric"] == "average_precision"


def test_monitoring_report_stable():
    data = {
        "overall_status": "stable",
        "max_psi": 0.01,
        "features": {
            "NEM": {
                "status": "stable",
                "psi": 0.01
            }
        }
    }

    performance = {
        "overall_status": "stable",
        "max_degradation": 0.01,
        "metrics": {
            "roc_auc": {
                "status": "stable",
                "degradation": 0.01
            }
        }
    }

    report = build_model_monitoring_report(
        data,
        performance,
    )

    assert report["overall_status"] == "stable"
    assert report["alerts"] == []
