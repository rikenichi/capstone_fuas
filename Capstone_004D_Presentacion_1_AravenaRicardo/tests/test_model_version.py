from model_metadata import get_model_metadata


def test_model_metadata_structure():
    metadata = get_model_metadata()

    assert metadata["model_version"] == "rf-2024-v1.0.0"
    assert metadata["training_year"] == 2024
    assert metadata["temporal_test_year"] == 2025
    assert metadata["threshold"] == 0.42
    assert metadata["model_type"] == "RandomForestClassifier"
    assert metadata["status"] == "frozen"


def test_model_metadata_features():
    metadata = get_model_metadata()

    assert metadata["features"] == [
        "NEM",
        "QUINTIL_SE4",
        "COD_DEPE",
        "EDAD",
        "GENERO",
        "NACIONALIDAD",
    ]


def test_model_artifact_integrity():
    metadata = get_model_metadata()

    assert metadata["artifact_available"] is True
    assert metadata["sha256"] is not None
    assert len(metadata["sha256"]) == 64
