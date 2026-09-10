"""Integración end-to-end 2024 del Capstone FUAS.

Flujo:
RAW 2024 -> ETL -> data quality -> EDA -> split train/validation ->
Random Forest congelado -> validación -> reentrenamiento sobre TODO 2024 ->
serialización del pipeline final para la futura prueba temporal 2025.

REGLA: este módulo no lee ni usa datos 2025.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import etl
import data_quality
import eda
from baseline_models import build_variant_preprocessor, feature_list, metrics_from_probs
from preprocessing import split_train_valid

YEAR = 2024
THRESHOLD = 0.42
FEATURES = feature_list(include_etnia=False)
RF_PARAMS = {
    "n_estimators": 160,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}
EXPECTED_VALIDATION = {
    "roc_auc": 0.8007,
    "recall": 0.8485,
    "fnr": 0.1515,
    "f1": 0.7195,
    "balanced_accuracy": 0.7169,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_raw_inputs(fuas: Path, asignaciones: Path) -> tuple[Path, Path]:
    """Copia/normaliza los dos RAW 2024 a las rutas esperadas por config.py."""
    fuas = Path(fuas)
    asignaciones = Path(asignaciones)
    if not fuas.exists():
        raise FileNotFoundError(fuas)
    if not asignaciones.exists():
        raise FileNotFoundError(asignaciones)
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    dst_f = cfg.FUAS_FILES[YEAR]
    dst_a = cfg.ASIG_FILES[YEAR]
    if fuas.resolve() != dst_f.resolve():
        shutil.copy2(fuas, dst_f)
    if asignaciones.resolve() != dst_a.resolve():
        shutil.copy2(asignaciones, dst_a)
    return dst_f, dst_a


def make_frozen_pipeline() -> Pipeline:
    pre = build_variant_preprocessor(
        group_cod_depe=False,  # configuración congelada: códigos 1..6
        include_etnia=False,
        scale_numeric=False,
    )
    rf = RandomForestClassifier(**RF_PARAMS)
    return Pipeline([("preprocess", pre), ("model", rf)])


def validate_reproduction(metrics: dict, tolerance: float = 0.01) -> dict:
    checks = {}
    for k, expected in EXPECTED_VALIDATION.items():
        actual = float(metrics[k])
        checks[k] = {
            "expected_approx": expected,
            "actual": actual,
            "abs_diff": abs(actual - expected),
            "within_tolerance": abs(actual - expected) <= tolerance,
        }
    checks["all_within_tolerance"] = all(v["within_tolerance"] for v in checks.values() if isinstance(v, dict))
    return checks


def _load_analytic_2024() -> pd.DataFrame:
    pq = cfg.DATA_ANALYTIC / "analytic_2024.parquet"
    try:
        if pq.exists():
            return pd.read_parquet(pq)
    except ImportError:
        pass
    pkl = cfg.DATA_ANALYTIC / "analytic_2024.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    raise FileNotFoundError("No se encontró analytic_2024.parquet ni analytic_2024.pkl")


def run(fuas: Path | None = None, asignaciones: Path | None = None, from_analytic: bool = False) -> dict:
    if from_analytic:
        # Modo de verificación rápida cuando el entorno no dispone de pyarrow.
        # Usa el analytic 2024 ya generado; NO sustituye la prueba RAW->ETL en Colab.
        raw_fuas = raw_asig = None
        etl_metrics = {
            "filas_fuente_fuas": None, "filas_post_filtro_fuas": None,
            "filas_analytic_tras_excluir_na": None, "target_pos_rate": None,
        }
        dq = {"alarmas": []}
        eda_report = {"_report_path": str(cfg.REPORTS_DIR / "eda_2024.json")}
        df = _load_analytic_2024()
    else:
        if (fuas is None) ^ (asignaciones is None):
            raise ValueError("Debes proporcionar ambos archivos RAW 2024 o ninguno.")
        if fuas is not None:
            raw_fuas, raw_asig = install_raw_inputs(Path(fuas), Path(asignaciones))
        else:
            raw_fuas, raw_asig = cfg.FUAS_FILES[YEAR], cfg.ASIG_FILES[YEAR]
        if not raw_fuas.exists() or not raw_asig.exists():
            raise FileNotFoundError(
                "Faltan RAW 2024. Usa --fuas y --asig o copia los archivos a data/raw/."
            )

        # 1) ETL 2024 exclusivamente
        etl_metrics = etl.build_year(YEAR, solo_fuas=True)

        # 2) Calidad de datos
        dq = data_quality.quality_report(YEAR, etl_metrics)
        if dq.get("alarmas"):
            raise RuntimeError(f"Alarmas de calidad: {dq['alarmas']}")

        # 3) EDA 2024 (reproduce artefactos corregidos)
        eda_report = eda.run_eda(YEAR)

        # 4) Validación interna reproducible 2024
        df = _load_analytic_2024()
    Xtr, Xva, ytr, yva = split_train_valid(df)
    pipe_val = make_frozen_pipeline()
    pipe_val.fit(Xtr[FEATURES], ytr)
    p = pipe_val.predict_proba(Xva[FEATURES])[:, 1]
    val_metrics = metrics_from_probs(yva, p, threshold=THRESHOLD)
    reproduction = validate_reproduction(val_metrics)

    # 5) Congelar configuración y reentrenar en TODO 2024.
    #    El threshold sigue siendo el seleccionado únicamente con validación 2024.
    X_full = df[FEATURES].copy()
    y_full = df[cfg.TARGET].astype(int)
    frozen = make_frozen_pipeline()
    frozen.fit(X_full, y_full)

    cfg.models_dir = getattr(cfg, "MODELS_DIR", cfg.ROOT / "models")
    models_dir = cfg.ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "rf_frozen_2024.joblib"
    joblib.dump(frozen, model_path, compress=3)

    metadata = {
        "status": "FROZEN_BEFORE_TEMPORAL_TEST",
        "training_year": YEAR,
        "holdout_year": 2025,
        "holdout_policy": "2025 no fue usado en ETL/EDA/preprocessing/tuning de esta integración.",
        "features": FEATURES,
        "etnia": "excluida",
        "cod_depe": "original_1_6",
        "threshold": THRESHOLD,
        "rf_params": RF_PARAMS,
        "validation_2024": val_metrics,
        "reproduction_checks": reproduction,
        "full_2024_rows": int(len(df)),
        "full_2024_target_positive_rate": float(y_full.mean()),
        "raw_sha256": ({
            "fuas_2024": sha256_file(raw_fuas),
            "asignaciones_2024": sha256_file(raw_asig),
        } if raw_fuas is not None else None),
        "analytic_2024_sha256": sha256_file(cfg.DATA_ANALYTIC / "analytic_2024.parquet"),
        "model_sha256": sha256_file(model_path),
        "next_step": "Evaluar una sola vez este pipeline congelado sobre 2025, sin retuning.",
    }
    metadata_path = cfg.REPORTS_DIR / "integration_2024_frozen.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "etl": {
            "filas_raw": etl_metrics.get("filas_fuente_fuas"),
            "filas_fuas": etl_metrics.get("filas_post_filtro_fuas"),
            "filas_analytic": etl_metrics.get("filas_analytic_tras_excluir_na"),
            "target_pos_rate_pct": etl_metrics.get("target_pos_rate"),
        },
        "data_quality_alarmas": dq.get("alarmas", []),
        "eda_report": str(cfg.REPORTS_DIR / "eda_2024.json"),
        "validation_2024": val_metrics,
        "reproduction_checks": reproduction,
        "frozen_model": str(model_path),
        "metadata": str(metadata_path),
    }
    out = cfg.REPORTS_DIR / "integration_2024_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Integración end-to-end SOLO 2024")
    ap.add_argument("--fuas", type=Path, help="POSTULACIONES_FUAS_2024_WEB.csv")
    ap.add_argument("--asig", type=Path, help="Asignacion 2024_PA_PUBL.csv")
    ap.add_argument("--from-analytic", action="store_true", help="Verificación rápida desde analytic_2024 sin ejecutar RAW->ETL")
    args = ap.parse_args()
    summary = run(args.fuas, args.asig, from_analytic=args.from_analytic)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
