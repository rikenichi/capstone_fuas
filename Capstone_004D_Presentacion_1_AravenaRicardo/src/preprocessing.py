"""
preprocessing.py — Preparación para modelado supervisado (SOLO 2024).

Incluye:
  - Auditoría de valores implausibles en EDAD y NEM (Parte B), sin eliminar por
    defecto; reglas configurables en config.py y aplicables solo a TRAIN.
  - Split interno estratificado 80/20 de 2024 con semilla fija (Parte G).
  - Pipeline scikit-learn (ColumnTransformer) con numéricas passthrough/scaling,
    ordinal (QUINTIL_SE4) en su orden natural y nominales One-Hot con
    handle_unknown="ignore" y agrupación opcional de categorías raras (Parte D/E).

NO entrena modelos. NO lee 2025. NO aplica SMOTE ni resampling.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ============================================================================
# Carga (solo 2024) + verificación de columnas prohibidas
# ============================================================================
def load_2024() -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA_ANALYTIC / "analytic_2024.parquet")


def feature_columns() -> list[str]:
    return cfg.FEATURES_NUMERIC + cfg.FEATURES_ORDINAL + cfg.FEATURES_NOMINAL


def assert_no_forbidden(cols) -> None:
    bad = [c for c in cols if c in cfg.FORBIDDEN_FEATURES]
    if bad:
        raise ValueError(f"Variables prohibidas presentes como feature: {bad}")


# ============================================================================
# Parte B — Auditoría de valores implausibles (no elimina por defecto)
# ============================================================================
def audit_implausible(df: pd.DataFrame) -> dict:
    edad = pd.to_numeric(df["EDAD"], errors="coerce")
    nem = pd.to_numeric(df["NEM"], errors="coerce")
    rep = {
        "EDAD": {
            "menor_15": int((edad < 15).sum()),
            "mayor_igual_60": int((edad >= 60).sum()),
            "mayor_igual_80": int((edad >= 80).sum()),
            "mayor_igual_90": int((edad >= 90).sum()),
            "mayor_igual_100": int((edad >= 100).sum()),
            "implausible_regla": int(((edad < cfg.EDAD_MIN_PLAUSIBLE) |
                                      (edad > cfg.EDAD_MAX_PLAUSIBLE)).sum()),
            "max_observado": int(edad.max()),
            "min_observado": int(edad.min()),
        },
        "NEM": {
            "igual_100": int((nem == 100).sum()),
            "menor_300": int((nem < 300).sum()),
            "fuera_rango": int(((nem < cfg.NEM_MIN_PLAUSIBLE) |
                                (nem > cfg.NEM_MAX_PLAUSIBLE)).sum()),
            "centinela": int(nem.isin(cfg.NEM_CENTINELA).sum()),
            "max_observado": int(nem.max()),
            "min_observado": int(nem.min()),
        },
    }
    return rep


def apply_implausible_rule(df: pd.DataFrame, rule: str | None = None) -> pd.DataFrame:
    """
    Aplica la regla de valores implausibles. SOLO debe usarse sobre TRAIN.
      - 'flag'       : agrega columnas _EDAD_implausible / _NEM_implausible (no elimina)
      - 'drop_train' : elimina filas implausibles (solo para el set de entrenamiento)
      - 'clip'       : recorta a los límites plausibles
      - None         : usa cfg.REGLA_VALORES_IMPLAUSIBLES
    """
    rule = rule or cfg.REGLA_VALORES_IMPLAUSIBLES
    out = df.copy()
    edad = pd.to_numeric(out["EDAD"], errors="coerce")
    nem = pd.to_numeric(out["NEM"], errors="coerce")
    edad_bad = (edad < cfg.EDAD_MIN_PLAUSIBLE) | (edad > cfg.EDAD_MAX_PLAUSIBLE)
    nem_bad = (nem < cfg.NEM_MIN_PLAUSIBLE) | (nem > cfg.NEM_MAX_PLAUSIBLE) | nem.isin(cfg.NEM_CENTINELA)

    if rule == "flag":
        out["_EDAD_implausible"] = edad_bad.astype(int)
        out["_NEM_implausible"] = nem_bad.astype(int)
    elif rule == "drop_train":
        out = out[~(edad_bad | nem_bad)].copy()
    elif rule == "clip":
        out["EDAD"] = edad.clip(cfg.EDAD_MIN_PLAUSIBLE, cfg.EDAD_MAX_PLAUSIBLE)
        out["NEM"] = nem.clip(cfg.NEM_MIN_PLAUSIBLE, cfg.NEM_MAX_PLAUSIBLE)
    else:
        raise ValueError(f"Regla desconocida: {rule}")
    return out


# ============================================================================
# Parte E — Agrupación de categorías raras (regla reproducible, ajustada en train)
# ============================================================================
class CodDepeGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa COD_DEPE según el mapa documentado (config.COD_DEPE_GROUP_MAP):
      1,2,6 -> PUBLICO_MUNICIPAL ; 3,5 -> PARTICULAR_SUBVENCIONADO ; 4 -> PARTICULAR_PAGADO
    Determinista (no aprende nada), pero se implementa como transformer para
    vivir dentro del pipeline y no provocar leakage ni pasos manuales fuera.
    """
    def __init__(self, col="COD_DEPE", mapping=None):
        self.col = col
        self.mapping = mapping or cfg.COD_DEPE_GROUP_MAP

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # X puede ser DataFrame de una o varias columnas; operamos sobre self.col
        if isinstance(X, pd.DataFrame) and self.col in X.columns:
            X[self.col] = X[self.col].map(self.mapping).fillna("DESCONOCIDO")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa en 'OTRAS' las categorías con frecuencia < threshold, aprendiendo las
    frecuencias SOLO en fit (train). En transform aplica el mapa aprendido.
    Evita leakage: las categorías 'frecuentes' se fijan con train.
    """
    def __init__(self, cols, threshold=cfg.RARE_THRESHOLD, other_label=-999):
        self.cols = cols
        self.threshold = threshold
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cols) if not isinstance(X, pd.DataFrame) else X
        self.frequent_ = {}
        n = len(X)
        for c in self.cols:
            vc = X[c].value_counts(dropna=False)
            self.frequent_[c] = set(vc[vc / n >= self.threshold].index.tolist())
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for c in self.cols:
            freq = self.frequent_[c]
            X[c] = X[c].where(X[c].isin(freq), other=self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


# ============================================================================
# Parte C/D — Pipeline de preprocesamiento
# ============================================================================
def build_preprocessor(scale_numeric: bool = False,
                       group_cod_depe: bool = False,
                       group_rare_nominal: bool = True) -> ColumnTransformer:
    """
    Construye el ColumnTransformer.

      scale_numeric : True para modelos sensibles a escala (Reg. Logística, KNN,
                      SVM); False para árboles/ensembles (no lo necesitan).
      group_cod_depe: True aplica la agrupación documentada de COD_DEPE (alt. B,
                      PUBLICO/SUBVENCIONADO/PAGADO); False usa los códigos 1..6
                      (alt. A). AHORA sí tiene efecto real dentro del pipeline.
      group_rare_nominal: agrupa categorías raras SOLO en nominales autorizadas
                      (config.FEATURES_NOMINAL_AGRUPABLES). ETNIA queda excluida
                      mientras ETNIA_CONFIRMADA sea False.

    Numéricas: passthrough (o StandardScaler si scale_numeric).
    Ordinal (QUINTIL_SE4): passthrough conservando su orden natural 1..5.
    Nominales: (agrupación COD_DEPE ->) (agrupación rara ->) OneHotEncoder.
    """
    numeric = cfg.FEATURES_NUMERIC
    ordinal = cfg.FEATURES_ORDINAL
    nominal = list(cfg.FEATURES_NOMINAL)

    num_steps = []
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    # Solo se agrupan por frecuencia las nominales explícitamente autorizadas.
    # ETNIA NO entra aquí mientras su codificación no esté confirmada.
    if group_rare_nominal:
        agrupables = [c for c in cfg.FEATURES_NOMINAL_AGRUPABLES if c in nominal]
    else:
        agrupables = []

    nom_steps = []
    if group_cod_depe and "COD_DEPE" in nominal:
        nom_steps.append(("cod_depe_group", CodDepeGrouper(col="COD_DEPE")))
    if agrupables:
        nom_steps.append(("rare", RareCategoryGrouper(cols=agrupables)))
    nom_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)))
    nom_pipe = Pipeline(nom_steps)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("ord", "passthrough", ordinal),
            ("nom", nom_pipe, nominal),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# ============================================================================
# Parte G — Split interno estratificado (SOLO 2024)
# ============================================================================
def split_train_valid(df: pd.DataFrame | None = None):
    """
    Devuelve (X_train, X_valid, y_train, y_valid) usando SOLO 2024, estratificado
    por el target, con semilla fija. 2025 nunca se carga aquí.
    """
    if df is None:
        df = load_2024()
    feats = feature_columns()
    assert_no_forbidden(feats)
    X = df[feats].copy()
    y = df[cfg.TARGET].astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=cfg.VALID_SIZE, random_state=cfg.RANDOM_SEED, stratify=y
    )
    return X_tr, X_va, y_tr, y_va


if __name__ == "__main__":
    import json
    df = load_2024()
    print("Auditoría implausibles:")
    print(json.dumps(audit_implausible(df), indent=2, ensure_ascii=False))
    Xtr, Xva, ytr, yva = split_train_valid(df)
    print(f"\nSplit: train={len(Xtr)}, valid={len(Xva)}")
    print(f"pos_rate train={ytr.mean():.4f}, valid={yva.mean():.4f}")
    pre = build_preprocessor(scale_numeric=False, group_cod_depe=False)
    Z = pre.fit_transform(Xtr)
    print(f"Matriz transformada train: {Z.shape}")
