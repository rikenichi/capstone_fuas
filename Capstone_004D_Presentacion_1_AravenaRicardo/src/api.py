
from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import inference


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts" / "reports"
DASHBOARD_DIR = ROOT / "artifacts" / "dashboard"
DASHBOARD_PATH = DASHBOARD_DIR / "dashboard_historico.json"

app = FastAPI(
    title="Capstone FUAS API",
    description=(
        "API para estimar la probabilidad de obtención de beneficios "
        "estatales a partir de antecedentes FUAS y explicar la "
        "estimación mediante SHAP."
    ),
    version="1.0.0",
)


class PerfilFUAS(BaseModel):
    """
    Único esquema admitido por la API.

    No se aceptan identificadores personales como
    RUT, RUN o MRUN ni campos adicionales.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    NEM: float = Field(
        ge=100,
        le=700,
    )

    QUINTIL_SE4: int = Field(
        ge=1,
        le=5,
    )

    COD_DEPE: int = Field(
        ge=1,
        le=6,
    )

    EDAD: int = Field(
        ge=10,
        le=130,
    )

    GENERO: int = Field(
        ge=1,
        le=2,
    )

    NACIONALIDAD: int = Field(
        ge=0,
        le=2,
    )


def _read_json(filename: str) -> dict | None:
    path = REPORTS_DIR / filename

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api-info")
def api_info():
    return {
        "servicio": "Capstone FUAS API",
        "version": "1.0.0",
        "estado": "operativo",
    }


@app.get("/")
def frontend_root():
    index_path = ROOT / "frontend_dist" / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend no disponible."
        )

    return FileResponse(index_path)


@app.get("/health")
def health():
    model_exists = inference.MODEL_PATH.exists()

    if not model_exists:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )

    return {
        "status": "ok",
        "modelo_disponible": True,
    }


@app.get("/model-info")
def model_info():
    metrics_2024 = _read_json(
        "metrics_frozen_2024.json"
    )

    temporal_2025 = _read_json(
        "temporal_test_2025.json"
    )

    return {
        "modelo": "RandomForestClassifier",
        "training_year": 2024,
        "temporal_test_year": 2025,
        "threshold": inference.THRESHOLD,
        "features": inference.FEATURES,
        "etnia_incluida": False,
        "metricas_2024": (
            metrics_2024.get("metricas")
            if metrics_2024
            else None
        ),
        "metricas_2025": (
            temporal_2025.get("metricas")
            if temporal_2025
            else None
        ),
        "advertencia": (
            "El modelo entrega una estimación orientativa. "
            "No corresponde a una decisión oficial de "
            "asignación de beneficios."
        ),
    }


@app.post("/predict")
def predict_endpoint(perfil: PerfilFUAS):
    try:
        return inference.predict(
            perfil.model_dump()
        )

    except inference.InferenceError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error interno durante la inferencia."
        )


@app.post("/explain")
def explain_endpoint(perfil: PerfilFUAS):
    try:
        return inference.explain(
            perfil.model_dump()
        )

    except inference.InferenceError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error interno durante la explicación."
        )



@app.get("/dashboard")
def dashboard_historico():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Dashboard histórico no disponible."
        )

    try:
        with DASHBOARD_PATH.open(
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="No fue posible cargar el dashboard histórico."
        )


# ============================================================
# FRONTEND REACT
# ============================================================

FRONTEND_DIR = ROOT / "frontend_dist"

if FRONTEND_DIR.exists():

    assets_dir = FRONTEND_DIR / "assets"

    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="assets"
        )

    @app.get("/app")
    def frontend_app():
        return FileResponse(
            FRONTEND_DIR / "index.html"
        )

