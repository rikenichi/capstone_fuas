
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rf_frozen_2024.joblib"

THRESHOLD = 0.42

FEATURES = [
    "NEM",
    "QUINTIL_SE4",
    "COD_DEPE",
    "EDAD",
    "GENERO",
    "NACIONALIDAD",
]

# Cache para no cargar el modelo/SHAP en cada petición
_MODEL = None
_EXPLAINER = None
_FEATURE_NAMES = None


class InferenceError(ValueError):
    pass


def load_model():
    global _MODEL

    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {MODEL_PATH}"
            )

        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL


def validate_input(data: dict) -> dict:
    missing = [f for f in FEATURES if f not in data]

    if missing:
        raise InferenceError(
            f"Faltan variables requeridas: {missing}"
        )

    try:
        cleaned = {
            "NEM": float(data["NEM"]),
            "QUINTIL_SE4": int(data["QUINTIL_SE4"]),
            "COD_DEPE": int(data["COD_DEPE"]),
            "EDAD": int(data["EDAD"]),
            "GENERO": int(data["GENERO"]),
            "NACIONALIDAD": int(data["NACIONALIDAD"]),
        }
    except (TypeError, ValueError):
        raise InferenceError(
            "Uno o más valores tienen un tipo inválido."
        )

    if not 100 <= cleaned["NEM"] <= 700:
        raise InferenceError(
            "NEM debe estar entre 100 y 700."
        )

    if cleaned["QUINTIL_SE4"] not in {1, 2, 3, 4, 5}:
        raise InferenceError(
            "QUINTIL_SE4 debe estar entre 1 y 5."
        )

    if cleaned["COD_DEPE"] not in {1, 2, 3, 4, 5, 6}:
        raise InferenceError(
            "COD_DEPE debe estar entre 1 y 6."
        )

    if not 10 <= cleaned["EDAD"] <= 130:
        raise InferenceError(
            "EDAD está fuera del rango admitido por el sistema."
        )

    if cleaned["GENERO"] not in {1, 2}:
        raise InferenceError(
            "GENERO debe corresponder a una categoría válida del modelo."
        )

    if cleaned["NACIONALIDAD"] not in {0, 1, 2}:
        raise InferenceError(
            "NACIONALIDAD debe corresponder a una categoría válida del modelo."
        )

    return cleaned


def _to_dataframe(cleaned: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [cleaned],
        columns=FEATURES
    )


def predict(data: dict, model=None) -> dict:
    cleaned = validate_input(data)

    if model is None:
        model = load_model()

    X = _to_dataframe(cleaned)

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    return {
        "probabilidad": probability,
        "probabilidad_pct": round(
            probability * 100, 2
        ),
        "clasificacion": prediction,
        "threshold": THRESHOLD,
        "sobre_umbral": bool(prediction),
        "mensaje": (
            "El perfil se encuentra sobre el umbral del modelo."
            if prediction == 1
            else
            "El perfil se encuentra bajo el umbral del modelo."
        ),
        "advertencia": (
            "Resultado orientativo basado en patrones históricos. "
            "No corresponde a una decisión oficial de asignación "
            "de beneficios."
        ),
    }


def _get_shap_components(model):
    global _EXPLAINER
    global _FEATURE_NAMES

    preprocess = model.named_steps["preprocess"]
    rf = model.named_steps["model"]

    if _EXPLAINER is None:
        _EXPLAINER = shap.TreeExplainer(rf)

    if _FEATURE_NAMES is None:
        _FEATURE_NAMES = list(
            preprocess.get_feature_names_out()
        )

    return preprocess, rf, _EXPLAINER, _FEATURE_NAMES


def _positive_class_shap(shap_values):
    """
    Compatibilidad con distintas versiones de SHAP.

    Casos posibles:
      - lista [clase0, clase1]
      - ndarray (n, features, clases)
      - ndarray (n, features)
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])[0]

    arr = np.asarray(shap_values)

    if arr.ndim == 3:
        return arr[0, :, 1]

    if arr.ndim == 2:
        return arr[0]

    raise RuntimeError(
        f"Formato SHAP no reconocido: {arr.shape}"
    )


def _positive_base_value(explainer) -> float:
    base = np.asarray(
        explainer.expected_value
    )

    if base.ndim == 0:
        return float(base)

    if len(base) >= 2:
        return float(base[1])

    return float(base[0])


def _original_feature_name(
    transformed_name: str
) -> str:
    """
    Reagrupa variables One-Hot a la variable original.
    """

    if transformed_name in {
        "NEM",
        "EDAD",
        "QUINTIL_SE4"
    }:
        return transformed_name

    if transformed_name.startswith(
        "COD_DEPE_"
    ):
        return "COD_DEPE"

    if transformed_name.startswith(
        "GENERO_"
    ):
        return "GENERO"

    if transformed_name.startswith(
        "NACIONALIDAD_"
    ):
        return "NACIONALIDAD"

    return transformed_name


def explain(
    data: dict,
    top_n: int = 6,
    model=None
) -> dict:

    cleaned = validate_input(data)

    if model is None:
        model = load_model()

    X = _to_dataframe(cleaned)

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    preprocess, rf, explainer, feature_names = (
        _get_shap_components(model)
    )

    X_trans = preprocess.transform(X)

    shap_values = explainer.shap_values(
        X_trans
    )

    sv = _positive_class_shap(
        shap_values
    )

    if len(sv) != len(feature_names):
        raise RuntimeError(
            "Número de valores SHAP no coincide "
            "con las variables transformadas."
        )

    # Agrupar dummies a las 6 variables originales
    grouped = {}

    for name, value in zip(
        feature_names,
        sv
    ):
        original = _original_feature_name(
            str(name)
        )

        grouped[original] = (
            grouped.get(original, 0.0)
            + float(value)
        )

    factores = []

    for variable, impacto in grouped.items():
        factores.append({
            "variable": variable,
            "valor": cleaned.get(variable),
            "impacto_shap": float(impacto),
            "impacto_abs": float(abs(impacto)),
            "direccion": (
                "aumenta"
                if impacto > 0
                else
                "disminuye"
                if impacto < 0
                else
                "neutral"
            ),
        })

    factores = sorted(
        factores,
        key=lambda x: x["impacto_abs"],
        reverse=True
    )[:top_n]

    base_value = _positive_base_value(
        explainer
    )

    shap_sum = float(
        base_value + np.sum(sv)
    )

    return {
        "probabilidad": probability,
        "probabilidad_pct": round(
            probability * 100, 2
        ),
        "threshold": THRESHOLD,
        "clasificacion": int(
            probability >= THRESHOLD
        ),
        "valor_base_shap": base_value,
        "salida_reconstruida_shap": shap_sum,
        "error_aditividad": float(
            abs(probability - shap_sum)
        ),
        "factores_principales": factores,
        "nota_shap": (
            "Los valores SHAP explican cómo el modelo "
            "llegó a esta estimación. No representan "
            "relaciones causales ni reglas oficiales "
            "de asignación."
        ),
    }


if __name__ == "__main__":

    ejemplo = {
        "NEM": 650,
        "QUINTIL_SE4": 2,
        "COD_DEPE": 3,
        "EDAD": 19,
        "GENERO": 2,
        "NACIONALIDAD": 0,
    }

    print("Predicción:")
    print(predict(ejemplo))

    print("\nExplicación:")
    print(explain(ejemplo))
