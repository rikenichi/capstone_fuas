"""
data_quality.py — Controles de calidad sobre las capas del ETL.

Valida dominios, nulos, duplicados, ausencia de MRUN en analytic, distribución
del target y tasa de match, y escribe un reporte JSON por año en
artifacts/reports/data_quality_<año>.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def check_domains(df: pd.DataFrame) -> dict:
    """Cuenta violaciones de dominio por variable (ignora nulos)."""
    violations = {}
    for col, allowed in cfg.DOMAINS.items():
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        bad = (~non_null.isin(allowed)).sum()
        violations[col] = int(bad)
    return violations


def quality_report(year: int, etl_metrics: dict) -> dict:
    """
    Construye el reporte de calidad de un año a partir de la capa analytic
    y las métricas del ETL. Devuelve el dict (y lo escribe a JSON).
    """
    analytic_path = cfg.DATA_ANALYTIC / f"analytic_{year}.parquet"
    df = pd.read_parquet(analytic_path)

    rep = {
        "anio": year,
        "filas_raw": etl_metrics.get("filas_fuente_fuas"),
        "filas_post_filtro_fuas": etl_metrics.get("filas_post_filtro_fuas"),
        "mrun_unicos_fuas": etl_metrics.get("mrun_unicos_fuas"),
        "mrun_unicos_asignaciones_validas": etl_metrics.get("mrun_unicos_asignaciones_validas"),
        "mrun_match_beneficio_valido": etl_metrics.get("mrun_match_beneficio_valido"),
        "mrun_match_solo_desconocido": etl_metrics.get("mrun_match_solo_desconocido"),
        "mrun_match_asignaciones_cualquiera": etl_metrics.get("mrun_match_asignaciones"),
        "mrun_sin_registro_asignacion": etl_metrics.get("mrun_sin_registro_asignacion"),
        "pct_match_beneficio_valido": etl_metrics.get("pct_match_beneficio_valido"),
        "target_raw_dist_tres_estados": etl_metrics.get("target_raw_dist"),
        "target_dist": etl_metrics.get("target_dist"),
        "target_pos_rate_pct": etl_metrics.get("target_pos_rate"),
        "mrun_excluidos_solo_desconocido": etl_metrics.get("mrun_excluidos_solo_desconocido"),
        "pct_excluidos_solo_desconocido": etl_metrics.get("pct_excluidos_solo_desconocido"),
        "filas_analytic_tras_excluir_na": etl_metrics.get("filas_analytic_tras_excluir_na"),
        "siglas_desconocidas": etl_metrics.get("valores_desconocidos"),
        "siglas_desconocidas_ya_observadas": etl_metrics.get("valores_desconocidos_ya_observados"),
        "siglas_desconocidas_nuevas": etl_metrics.get("valores_desconocidos_nuevos"),
        "nulos_por_predictor": {
            c: int(df[c].isna().sum()) for c in cfg.PREDICTORS if c in df.columns
        },
        "conversiones_fallidas": {
            c: v["perdidos_por_conversion"]
            for c, v in etl_metrics.get("conversiones", {}).items()
        },
        "violaciones_dominio": check_domains(df),
        "mrun_en_analytic": cfg.KEY in df.columns,
        "filas_con_perfil_identico": int(df.duplicated().sum()),
        "beneficios_desconocidos": etl_metrics.get("valores_desconocidos"),
        "mrun_beneficios_multiples": etl_metrics.get("mrun_beneficios_multiples"),
        "encoding_fuas": etl_metrics.get("encoding_fuas"),
        "encoding_asig": etl_metrics.get("encoding_asig"),
    }

    # Señales de alarma explícitas
    alarms = []
    if rep["mrun_en_analytic"]:
        alarms.append("MRUN presente en capa analítica (debe eliminarse)")
    for col, bad in rep["violaciones_dominio"].items():
        if bad > 0:
            alarms.append(f"{bad} valores fuera de dominio en {col}")
    if rep["target_dist"] and set(rep["target_dist"].keys()) - {0, 1}:
        alarms.append("target fuera de {0,1}")
    rep["alarmas"] = alarms

    out = cfg.REPORTS_DIR / f"data_quality_{year}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    rep["_report_path"] = str(out)
    return rep


if __name__ == "__main__":
    import etl
    metrics = etl.run_all()
    for year in (2024, 2025):
        r = quality_report(year, metrics[year])
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
