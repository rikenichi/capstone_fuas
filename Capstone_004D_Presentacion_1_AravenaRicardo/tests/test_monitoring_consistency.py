import json

import pytest

from check_monitoring_consistency import (
    compare_monitoring_artifacts,
)


def test_equal_monitoring_artifacts(tmp_path):
    data = {
        "model_version": "rf-2024-v1.0.0",
        "overall_status": "watch",
    }

    expected = tmp_path / "expected.json"
    generated = tmp_path / "generated.json"

    expected.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    generated.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    compare_monitoring_artifacts(
        expected,
        generated,
    )


def test_different_monitoring_artifacts_fail(tmp_path):
    expected = tmp_path / "expected.json"
    generated = tmp_path / "generated.json"

    expected.write_text(
        json.dumps({
            "overall_status": "watch"
        }),
        encoding="utf-8",
    )

    generated.write_text(
        json.dumps({
            "overall_status": "stable"
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        compare_monitoring_artifacts(
            expected,
            generated,
        )
