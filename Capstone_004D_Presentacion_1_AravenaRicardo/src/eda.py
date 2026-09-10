"""
eda.py — Funciones reutilizables para el análisis exploratorio (EDA).

Diseñado para operar SOLO sobre la capa analítica de un año (por defecto 2024).
No usa 2025 para ninguna decisión. No entrena modelos, no hace split, scaling,
encoding, SMOTE, tuning ni SHAP.

Genera:
  - figuras en artifacts/figures/eda_<año>/
  - un reporte JSON en artifacts/reports/eda_<año>.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# Predictores base (los únicos usados como features potenciales).
PREDICTORS = cfg.PREDICTORS  # NEM, QUINTIL_SE4, COD_DEPE, EDAD, GENERO, NACIONALIDAD, ETNIA
TARGET = cfg.TARGET
NUMERIC = ["NEM", "EDAD"]
CATEGORICAL = ["QUINTIL_SE4", "COD_DEPE", "GENERO", "NACIONALIDAD", "ETNIA"]

# Columnas de resultado: SOLO para análisis descriptivo del resultado, NUNCA
# como features. Se listan para excluirlas explícitamente del análisis predictor.
RESULT_ONLY = ["BENEFICIOS_ASIGNADOS_RAW", "BENEFICIOS_VALIDOS", "TIPO_BENEFICIO",
               "TIENE_BECA", "TIENE_GRATUIDAD", "TIENE_FSCU", "TIENE_OTRO"]

# Etiquetas legibles
COD_DEPE_LABELS = {
    1: "Municipal (1)", 2: "Municipal (2)", 6: "Municipal (6)",
    3: "Part. Subv. (3)", 5: "Part. Subv. (5)", 4: "Part. Pagado (4)",
}
COD_DEPE_GROUP = {
    1: "Municipal/Público", 2: "Municipal/Público", 6: "Municipal/Público",
    3: "Particular Subvencionado", 5: "Particular Subvencionado",
    4: "Particular Pagado",
}
GENERO_LABELS = {0: "Sin información", 1: "Hombre", 2: "Mujer"}
NACIONALIDAD_LABELS = {0: "Chileno", 1: "Extranjero", 2: "Nacionalizado"}
# La correspondencia semántica de ETNIA para el CSV 2024 NO está confirmada.
# Por trazabilidad, se muestran códigos neutrales en gráficos y reportes hasta
# contar con una fuente documental que valide la codificación observada.
ETNIA_LABELS = {i: f"Código {i}" for i in range(0, 100)}

RARE_THRESHOLD = 0.01  # <1% del total = categoría rara (métricas inestables)


def _fig_dir(year: int) -> Path:
    d = cfg.FIGURES_DIR / f"eda_{year}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_analytic(year: int = 2024) -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA_ANALYTIC / f"analytic_{year}.parquet")


# ============================================================================
# 1. Calidad y estructura
# ============================================================================
def structure_report(df: pd.DataFrame) -> dict:
    rep = {
        "n_filas": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "nulos": {c: int(df[c].isna().sum()) for c in df.columns},
        "valores_unicos": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    # dominios observados de predictores categóricos + target
    rep["dominios_observados"] = {
        c: sorted([x.item() if hasattr(x, "item") else x
                   for x in df[c].dropna().unique().tolist()])
        for c in CATEGORICAL + [TARGET] if c in df.columns
    }
    vc = df[TARGET].value_counts().to_dict()
    n = len(df)
    rep["balance_target"] = {
        "conteo": {int(k): int(v) for k, v in vc.items()},
        "proporcion": {int(k): round(v / n, 6) for k, v in vc.items()},
        "pos_rate_pct": round(100 * df[TARGET].mean(), 4),
    }
    return rep


# ============================================================================
# 2. Numéricas
# ============================================================================
def numeric_describe(df: pd.DataFrame, cols=NUMERIC) -> dict:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "n": int(s.notna().sum()),
            "media": round(float(s.mean()), 4),
            "mediana": round(float(s.median()), 4),
            "desv": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "q1": round(float(s.quantile(0.25)), 4),
            "q3": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
        }
    return out


def numeric_by_target(df: pd.DataFrame, cols=NUMERIC) -> dict:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        by = {}
        for k in (0, 1):
            sk = s[df[TARGET] == k]
            by[k] = {
                "media": round(float(sk.mean()), 4),
                "mediana": round(float(sk.median()), 4),
                "desv": round(float(sk.std()), 4),
            }
        out[c] = by
    return out


def outliers_iqr(df: pd.DataFrame, cols=NUMERIC) -> dict:
    """Detecta outliers por regla IQR (1.5*IQR). No los elimina."""
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        out[c] = {
            "limite_inferior": round(float(lo), 4),
            "limite_superior": round(float(hi), 4),
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(100 * mask.mean(), 4),
            "min_observado": round(float(s.min()), 4),
            "max_observado": round(float(s.max()), 4),
        }
    return out


# ============================================================================
# 3. Categóricas: frecuencias + tasa de beneficio
# ============================================================================
def categorical_report(df: pd.DataFrame, cols=CATEGORICAL) -> dict:
    out = {}
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        vc = df[c].value_counts(dropna=False)
        rate = df.groupby(c, dropna=False)[TARGET].mean()
        cats = {}
        for val in vc.index:
            key = "NA" if pd.isna(val) else int(val)
            cnt = int(vc[val])
            cats[key] = {
                "conteo": cnt,
                "pct_del_total": round(100 * cnt / n, 4),
                "tasa_beneficio_pct": (None if pd.isna(val)
                                       else round(100 * float(rate[val]), 4)),
                "categoria_rara": bool(cnt / n < RARE_THRESHOLD),
            }
        out[c] = cats
    return out


def rare_categories(cat_report: dict) -> dict:
    rare = {}
    for c, cats in cat_report.items():
        r = {k: v for k, v in cats.items() if v.get("categoria_rara")}
        if r:
            rare[c] = list(r.keys())
    return rare


# ============================================================================
# 4. NEM por rangos + fuerza de separación
# ============================================================================
def nem_by_ranges(df: pd.DataFrame) -> dict:
    s = pd.to_numeric(df["NEM"], errors="coerce")
    bins = [0, 400, 450, 500, 550, 600, 650, 700, 750, 1000]
    labels = ["<400", "400-449", "450-499", "500-549", "550-599",
              "600-649", "650-699", "700-749", "750+"]
    cut = pd.cut(s, bins=bins, labels=labels, right=False)
    grp = df.assign(_r=cut).groupby("_r", observed=True)[TARGET]
    out = {}
    for r, sub in grp:
        out[str(r)] = {
            "conteo": int(sub.count()),
            "tasa_beneficio_pct": round(100 * float(sub.mean()), 4),
        }
    return out


def separation_strength(df: pd.DataFrame, col: str) -> dict:
    """
    Fuerza de separación de una numérica entre clases:
      - diferencia de medias estandarizada (Cohen's d)
    No es selección de features; es señal descriptiva.
    """
    s = pd.to_numeric(df[col], errors="coerce")
    a = s[df[TARGET] == 0].dropna()
    b = s[df[TARGET] == 1].dropna()
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    return {
        "media_clase_0": round(float(a.mean()), 4),
        "media_clase_1": round(float(b.mean()), 4),
        "cohens_d": round(float(d), 4),
    }


# ============================================================================
# 5. Correlaciones (solo donde tiene sentido)
# ============================================================================
def correlations(df: pd.DataFrame) -> dict:
    """
    - Pearson y Spearman entre numéricas reales (NEM, EDAD).
    - QUINTIL_SE4 es ordinal: se reporta Spearman con NEM/EDAD/target, con
      advertencia. Los demás códigos categóricos NO se tratan como continuos.
    """
    out = {"advertencia": (
        "COD_DEPE, GENERO, NACIONALIDAD y ETNIA son categóricos nominales: "
        "sus códigos NO se interpretan como continuos. QUINTIL_SE4 es ordinal "
        "y se usa solo con Spearman. Ninguna correlación implica causalidad."
    )}
    num = df[["NEM", "EDAD"]].apply(pd.to_numeric, errors="coerce")
    num["OBTUVO_BENEFICIO"] = df[TARGET]
    num["QUINTIL_SE4"] = pd.to_numeric(df["QUINTIL_SE4"], errors="coerce")
    out["pearson_num"] = num[["NEM", "EDAD", "OBTUVO_BENEFICIO"]].corr(
        method="pearson").round(4).to_dict()
    out["spearman_ordinal"] = num[["NEM", "EDAD", "QUINTIL_SE4",
                                   "OBTUVO_BENEFICIO"]].corr(
        method="spearman").round(4).to_dict()
    return out


# ============================================================================
# 6. Figuras
# ============================================================================
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_figures(df: pd.DataFrame, year: int = 2024) -> list[str]:
    d = _fig_dir(year)
    paths = []
    C0, C1, ACC = "#1C3A5E", "#0E7C86", "#F2A65A"

    # target
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df[TARGET].value_counts().sort_index()
    ax.bar(["No (0)", "Sí (1)"], vc.values, color=[C0, C1])
    ax.set_title(f"Distribución de OBTUVO_BENEFICIO ({year})")
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n{100*v/len(df):.1f}%", ha="center", va="bottom", fontsize=9)
    p = d / "target_distribucion.png"; _save(fig, p); paths.append(str(p))

    # histograma NEM
    nem = pd.to_numeric(df["NEM"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nem, bins=60, color=C1, alpha=0.85)
    ax.set_title(f"Distribución de NEM ({year})"); ax.set_xlabel("NEM")
    p = d / "nem_histograma.png"; _save(fig, p); paths.append(str(p))

    # boxplot NEM por target
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [pd.to_numeric(df[df[TARGET]==k]["NEM"], errors="coerce").dropna() for k in (0,1)]
    bp = ax.boxplot(data, labels=["No (0)", "Sí (1)"], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [C0, C1]): patch.set_facecolor(c)
    ax.set_title(f"NEM por OBTUVO_BENEFICIO ({year})"); ax.set_ylabel("NEM")
    p = d / "nem_boxplot_por_target.png"; _save(fig, p); paths.append(str(p))

    # distribución de edad
    edad = pd.to_numeric(df["EDAD"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(edad[edad <= 60], bins=45, color=C0, alpha=0.85)
    ax.set_title(f"Distribución de EDAD ({year}, ≤60)"); ax.set_xlabel("Edad")
    p = d / "edad_distribucion.png"; _save(fig, p); paths.append(str(p))

    # tasa por quintil
    def _rate_plot(col, labels_map, title, fname, group=None):
        tmp = df.copy()
        if group:
            tmp["_g"] = tmp[col].map(group)
            gcol = "_g"
        else:
            gcol = col
        rate = tmp.groupby(gcol)[TARGET].mean() * 100
        cnt = tmp.groupby(gcol)[TARGET].size()
        rate = rate.sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        xlabels = [labels_map.get(int(i), str(i)) if not group else str(i)
                   for i in rate.index] if labels_map else [str(i) for i in rate.index]
        ax.bar(range(len(rate)), rate.values, color=C1)
        ax.set_xticks(range(len(rate))); ax.set_xticklabels(xlabels, rotation=30, ha="right")
        ax.set_ylabel("Tasa de beneficio (%)"); ax.set_title(title)
        ymax = float(rate.max()) if len(rate) else 100.0
        ax.set_ylim(0, max(100.0, ymax * 1.12))
        for i, (v, c) in enumerate(zip(rate.values, cnt.values)):
            ax.text(i, v, f"{v:.1f}%\nn={c:,}", ha="center", va="bottom", fontsize=7)
        p = d / fname; _save(fig, p); return str(p)

    paths.append(_rate_plot("QUINTIL_SE4", None, f"Tasa de beneficio por quintil ({year})",
                            "tasa_por_quintil.png"))
    paths.append(_rate_plot("COD_DEPE", None, f"Tasa de beneficio por dependencia ({year})",
                            "tasa_por_dependencia.png", group=COD_DEPE_GROUP))
    paths.append(_rate_plot("GENERO", GENERO_LABELS, f"Tasa de beneficio por género ({year})",
                            "tasa_por_genero.png"))
    paths.append(_rate_plot("NACIONALIDAD", NACIONALIDAD_LABELS,
                            f"Tasa de beneficio por nacionalidad ({year})",
                            "tasa_por_nacionalidad.png"))
    paths.append(_rate_plot("ETNIA", ETNIA_LABELS,
                            f"Tasa de beneficio por código de ETNIA ({year})",
                            "tasa_por_etnia.png"))
    return paths


# ============================================================================
# 7. Reporte completo
# ============================================================================
def run_eda(year: int = 2024) -> dict:
    if year != 2024:
        raise ValueError("Esta etapa de EDA es solo para 2024. 2025 está reservado.")
    df = load_analytic(year)

    cat = categorical_report(df)
    rep = {
        "anio": year,
        "estructura": structure_report(df),
        "numericas_descriptivo": numeric_describe(df),
        "numericas_por_target": numeric_by_target(df),
        "outliers": outliers_iqr(df),
        "categoricas": cat,
        "categorias_raras": rare_categories(cat),
        "nem_por_rangos": nem_by_ranges(df),
        "separacion_NEM": separation_strength(df, "NEM"),
        "separacion_EDAD": separation_strength(df, "EDAD"),
        "correlaciones": correlations(df),
        "advertencias_metodologicas": {
            "ETNIA": (
                "La correspondencia semántica de los códigos de ETNIA del CSV 2024 "
                "no está confirmada documentalmente. Se reportan únicamente códigos "
                "neutrales y no se interpretan como nombres de pueblos o etnias."
            )
        },
    }
    figs = make_figures(df, year)
    rep["figuras"] = figs

    out = cfg.REPORTS_DIR / f"eda_{year}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    rep["_report_path"] = str(out)
    return rep


if __name__ == "__main__":
    r = run_eda(2024)
    print(json.dumps({k: v for k, v in r.items() if k != "figuras"},
                     ensure_ascii=False, indent=2, default=str)[:4000])
