from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def normalize(report: dict) -> dict:
    """
    Normaliza el contenido antes de comparar.

    Actualmente no eliminamos campos porque queremos
    consistencia exacta del artefacto operacional.
    """
    return report


def compare_monitoring_artifacts(
    expected_path: str | Path,
    generated_path: str | Path,
) -> None:

    expected = normalize(
        load_json(expected_path)
    )

    generated = normalize(
        load_json(generated_path)
    )

    if expected != generated:
        expected_text = json.dumps(
            expected,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        generated_text = json.dumps(
            generated,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        raise RuntimeError(
            "El artefacto de monitoreo versionado "
            "no coincide con el generado por CI.\n\n"
            "VERSIONADO:\n"
            f"{expected_text}\n\n"
            "GENERADO:\n"
            f"{generated_text}"
        )

    print(
        "OK: model_monitoring.json coincide "
        "con el artefacto regenerado."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expected",
        required=True,
    )

    parser.add_argument(
        "--generated",
        required=True,
    )

    args = parser.parse_args()

    compare_monitoring_artifacts(
        expected_path=args.expected,
        generated_path=args.generated,
    )


if __name__ == "__main__":
    main()
