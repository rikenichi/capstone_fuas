from model_monitoring_store import get_monitoring_report


def test_monitoring_artifact_loads():
    report = get_monitoring_report()

    assert report["model_version"] == "rf-2024-v1.0.0"
    assert report["baseline_year"] == 2024
    assert report["monitoring_year"] == 2025


def test_monitoring_artifact_status():
    report = get_monitoring_report()

    assert report["data_drift"]["status"] == "stable"
    assert report["performance"]["status"] == "watch"
    assert report["overall_status"] == "watch"


def test_monitoring_artifact_has_alert():
    report = get_monitoring_report()

    assert len(report["alerts"]) >= 1

    assert (
        report["alerts"][0]["metric"]
        == "average_precision"
    )
