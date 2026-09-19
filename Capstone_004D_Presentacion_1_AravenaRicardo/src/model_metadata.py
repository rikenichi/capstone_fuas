from pathlib import Path
from functools import lru_cache
import hashlib


MODEL_VERSION = "rf-2024-v1.0.0"

TRAINING_YEAR = 2024
TEMPORAL_TEST_YEAR = 2025
THRESHOLD = 0.42

FEATURES = [
    "NEM",
    "QUINTIL_SE4",
    "COD_DEPE",
    "EDAD",
    "GENERO",
    "NACIONALIDAD",
]

MODEL_FILENAME = "rf_frozen_2024.joblib"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME


def calculate_sha256(path: Path) -> str | None:
    if not path.exists():
        return None

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


@lru_cache(maxsize=1)
def get_model_metadata():
    model_exists = MODEL_PATH.exists()

    return {
        "model_version": MODEL_VERSION,
        "artifact": MODEL_FILENAME,
        "artifact_available": model_exists,
        "sha256": (
            calculate_sha256(MODEL_PATH)
            if model_exists
            else None
        ),
        "training_year": TRAINING_YEAR,
        "temporal_test_year": TEMPORAL_TEST_YEAR,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "model_type": "RandomForestClassifier",
        "status": "frozen",
    }
