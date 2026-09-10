"""
etl.py — Pipeline ETL del Capstone FUAS.

Flujo por año (2024 y 2025 por separado, cruce temporal del MISMO año):

    RAW (CSV Mineduc)
      -> lectura robusta de encoding + validación de mojibake
      -> normalización de nombres de columnas (por nombre, nunca por posición)
      -> STAGING: FUAS (base) LEFT JOIN Asignaciones agregadas por MRUN
                  (conserva MRUN para trazabilidad local; va en .gitignore)
      -> ANALYTIC: sin MRUN, con predictores + target + columnas de análisis

Principios:
  - Nada de conversiones silenciosas: cada casteo se registra.
  - El merge no puede multiplicar filas (aserciones duras).
  - Si algo contradice los supuestos, se detiene y se muestra evidencia.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ============================================================================
# 1. Lectura robusta de encoding
# ============================================================================
def _has_mojibake(df: pd.DataFrame) -> bool:
    """True si el texto muestra señales de mojibake en columnas de texto."""
    text_cols = [c for c in df.columns if df[c].dtype == object]
    sample = []
    for c in text_cols:
        vals = df[c].dropna().astype(str).head(2000).tolist()
        sample.extend(vals)
    blob = " ".join(sample)
    # Señal fuerte de corrupción: secuencias tipo 'RegiÃ³n'
    for bad in cfg.MOJIBAKE_BAD_SIGNS:
        if bad in blob:
            return True
    return False


def _has_expected_text(df: pd.DataFrame) -> bool:
    """True si aparece al menos una palabra canario bien formada."""
    text_cols = [c for c in df.columns if df[c].dtype == object]
    blob_parts = []
    for c in text_cols:
        blob_parts.extend(df[c].dropna().astype(str).head(2000).tolist())
    blob = " ".join(blob_parts)
    return any(canary in blob for canary in cfg.MOJIBAKE_CANARIES)


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Lee un CSV probando encodings en orden. Un encoding se acepta solo si:
      - se puede leer sin excepción, y
      - no deja mojibake, y
      - (si hay columnas de texto con canarios esperables) muestra texto correcto.
    Devuelve (df, encoding_usado). Lanza RuntimeError si ninguno sirve.
    """
    errors = {}
    fallback = None
    for enc in cfg.ENCODINGS_TRY:
        try:
            df = pd.read_csv(path, sep=cfg.CSV_SEP, dtype=str, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            errors[enc] = f"decode error: {e}"
            continue
        except Exception as e:  # noqa: BLE001
            errors[enc] = f"read error: {e}"
            continue

        if _has_mojibake(df):
            errors[enc] = "leyó pero con mojibake"
            # guardamos como último recurso por si nada mejor aparece
            if fallback is None:
                fallback = (df, enc)
            continue

        # Sin mojibake. Si además hay texto esperado, es claramente válido.
        if _has_expected_text(df) or not any(
            df[c].dtype == object for c in df.columns
        ):
            return df, enc
        # Sin mojibake y sin canarios detectables: aceptable.
        return df, enc

    if fallback is not None:
        raise RuntimeError(
            f"Ningún encoding limpio para {path.name}. "
            f"Intentos: {errors}. Se leyó con mojibake usando "
            f"'{fallback[1]}' — revisar archivo."
        )
    raise RuntimeError(f"No se pudo leer {path.name}. Intentos: {errors}")


# ============================================================================
# 2. Normalización de nombres de columnas
# ============================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas conocidas (p. ej. FEC_NAC_ALU -> FEC_NAC)."""
    return df.rename(columns={k: v for k, v in cfg.COLUMN_RENAME.items()
                              if k in df.columns})


# ============================================================================
# 3. Casteo explícito con registro (sin conversiones silenciosas)
# ============================================================================
def cast_numeric_tracked(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Convierte columnas a numérico registrando, por columna:
      no_nulo_antes, no_nulo_despues, perdidos_por_conversion.
    """
    report = {}
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        before = out[c].notna().sum()
        converted = pd.to_numeric(out[c], errors="coerce")
        after = converted.notna().sum()
        report[c] = {
            "no_nulo_antes": int(before),
            "no_nulo_despues": int(after),
            "perdidos_por_conversion": int(before - after),
        }
        out[c] = converted
    return out, report


# ============================================================================
# 4. Agregación de Asignaciones por MRUN + clasificación de beneficios
# ============================================================================
def _clasifica_lista(siglas: list[str]) -> dict:
    """
    Dada la lista de siglas válidas de un MRUN, deriva flags y tipo.
    Usa el catálogo explícito de becas (no la primera letra). Se mantiene como
    referencia/uso en pruebas; la agregación productiva es vectorizada.
    """
    s = set(siglas)
    tiene_grat = bool(s & cfg.BENEFICIO_GRATUIDAD)
    tiene_fscu = bool(s & cfg.BENEFICIO_FSCU)
    tiene_beca = bool(s & cfg.BENEFICIOS_BECA)
    tiene_otro = bool(s & cfg.BENEFICIO_OTRO_EXPLICITO)
    if tiene_grat:
        tipo = "GRATUIDAD"
    elif tiene_fscu:
        tipo = "FSCU"
    elif tiene_beca:
        tipo = "BECA"
    elif tiene_otro:
        tipo = "OTRO"
    else:
        tipo = "SIN_BENEFICIO"
    return {
        "TIENE_GRATUIDAD": int(tiene_grat),
        "TIENE_BECA": int(tiene_beca),
        "TIENE_FSCU": int(tiene_fscu),
        "TIENE_OTRO": int(tiene_otro),
        "TIPO_BENEFICIO": tipo,
    }


def aggregate_asignaciones(df_asig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Colapsa Asignaciones a 1 fila por MRUN.
      - Normaliza BENEFICIO_BECA_FSCU (strip + upper).
      - Separa tres clases de sigla:
          * válida       -> catálogo reconocido
          * desconocida  -> cualquier sigla NO vacía y NO catalogada
                            (regla general: incluye BSA/BPDT y cualquier
                             sigla nueva que aparezca en el futuro)
          * vacía        -> sin beneficio registrado en esa fila
      - Construye BENEFICIOS_VALIDOS (solo válidas) y BENEFICIOS_ASIGNADOS_RAW
        (todas, incl. desconocidas), flags, TIPO_BENEFICIO.
      - Marca TIENE_VALIDO y SOLO_DESCONOCIDO (para target de tres estados).
    Devuelve (asig_agg, meta). asig_agg tiene MRUN único.
    """
    import numpy as np
    meta = {}
    a = df_asig.copy()
    a["MRUN"] = a["MRUN"].astype(str).str.strip()
    b_norm = a["BENEFICIO_BECA_FSCU"].fillna("").astype(str).str.strip().str.upper()

    total = len(a)
    es_valido = b_norm.isin(cfg.CATALOGO_BENEFICIOS)
    es_vacio = (b_norm == "")
    # Regla general: desconocida = cualquier sigla no vacía que no esté en el
    # catálogo. Así una sigla nueva no catalogada recibe el mismo tratamiento
    # NA que BSA/BPDT, en vez de contarse como ausencia de beneficio.
    es_desconocido = ~es_valido & ~es_vacio

    # Distinción informativa: desconocidas ya observadas vs. nuevas (no vistas).
    es_desc_conocido = b_norm.isin(cfg.BENEFICIOS_DESCONOCIDOS_EN_REVISION)
    es_desc_nuevo = es_desconocido & ~es_desc_conocido

    meta["filas_asig_raw"] = int(total)
    meta["filas_beneficio_valido"] = int(es_valido.sum())
    meta["filas_beneficio_desconocido"] = int(es_desconocido.sum())
    meta["filas_beneficio_vacio"] = int(es_vacio.sum())
    meta["valores_desconocidos"] = sorted(set(b_norm[es_desconocido].unique()))
    meta["valores_desconocidos_ya_observados"] = sorted(
        set(b_norm[es_desc_conocido].unique()))
    meta["valores_desconocidos_nuevos"] = sorted(
        set(b_norm[es_desc_nuevo].unique()))
    if meta["valores_desconocidos_nuevos"]:
        meta["ALERTA_siglas_nuevas"] = (
            "Aparecieron siglas no catalogadas ni previamente observadas: "
            f"{meta['valores_desconocidos_nuevos']}. Se tratan como NA "
            "(en revisión); resolver su significado antes de modelar."
        )

    a = a.assign(_BEN=b_norm, _valido=es_valido, _desc=es_desconocido)

    # --- Lista RAW por MRUN: TODAS las siglas no vacías (válidas + desconocidas) ---
    # Cumple la decisión de no perder información antes de derivar categorías.
    a_nonempty = a[~es_vacio][["MRUN", "_BEN"]].drop_duplicates()
    a_nonempty = a_nonempty.sort_values(["MRUN", "_BEN"])
    raw_list = (a_nonempty.groupby("MRUN", sort=False)["_BEN"]
                .apply(lambda s: "|".join(s))
                .rename("BENEFICIOS_ASIGNADOS_RAW"))

    # --- MRUN con al menos una sigla válida ---
    a_valid = a[es_valido][["MRUN", "_BEN"]].drop_duplicates().copy()
    a_valid["_is_grat"] = a_valid["_BEN"].isin(cfg.BENEFICIO_GRATUIDAD)
    a_valid["_is_fscu"] = a_valid["_BEN"].isin(cfg.BENEFICIO_FSCU)
    # Beca por catálogo explícito (no por primera letra): incluye becas cuya
    # sigla no empieza con B (RETTIG, REUBICACION, TITULAR, TRASPASO, CONTINUIDAD).
    a_valid["_is_beca"] = a_valid["_BEN"].isin(cfg.BENEFICIOS_BECA)
    a_valid["_is_otro"] = a_valid["_BEN"].isin(cfg.BENEFICIO_OTRO_EXPLICITO)
    a_valid = a_valid.sort_values(["MRUN", "_BEN"])
    gb = a_valid.groupby("MRUN", sort=False)
    agg = gb.agg(
        BENEFICIOS_VALIDOS=("_BEN", lambda s: "|".join(s)),
        TIENE_GRATUIDAD=("_is_grat", "max"),
        TIENE_FSCU=("_is_fscu", "max"),
        TIENE_BECA=("_is_beca", "max"),
        TIENE_OTRO=("_is_otro", "max"),
    ).reset_index()
    for c in ["TIENE_GRATUIDAD", "TIENE_FSCU", "TIENE_BECA", "TIENE_OTRO"]:
        agg[c] = agg[c].astype(int)
    conds = [agg["TIENE_GRATUIDAD"] == 1, agg["TIENE_FSCU"] == 1,
             agg["TIENE_BECA"] == 1, agg["TIENE_OTRO"] == 1]
    agg["TIPO_BENEFICIO"] = np.select(conds, ["GRATUIDAD", "FSCU", "BECA", "OTRO"],
                                      default="SIN_BENEFICIO")
    agg["TIENE_VALIDO"] = 1

    # --- MRUN con alguna sigla desconocida ---
    mrun_con_desc = set(a[es_desconocido]["MRUN"].unique())
    mrun_con_valido = set(agg["MRUN"].unique())
    # MRUN cuyo ÚNICO aporte es desconocido (no tienen ninguna válida)
    solo_desconocido = mrun_con_desc - mrun_con_valido
    meta["mrun_con_desconocido"] = int(len(mrun_con_desc))
    meta["mrun_solo_desconocido"] = int(len(solo_desconocido))

    if solo_desconocido:
        extra = pd.DataFrame({"MRUN": sorted(solo_desconocido)})
        extra["BENEFICIOS_VALIDOS"] = ""
        for c in ["TIENE_GRATUIDAD", "TIENE_FSCU", "TIENE_BECA", "TIENE_OTRO"]:
            extra[c] = 0
        extra["TIPO_BENEFICIO"] = "BENEFICIO_DESCONOCIDO"
        extra["TIENE_VALIDO"] = 0
        agg = pd.concat([agg, extra], ignore_index=True)

    agg["SOLO_DESCONOCIDO"] = (agg["TIENE_VALIDO"] == 0).astype(int)

    # Adjuntar la lista RAW (válidas + desconocidas) a cada MRUN
    agg = agg.merge(raw_list, on="MRUN", how="left")
    agg["BENEFICIOS_ASIGNADOS_RAW"] = agg["BENEFICIOS_ASIGNADOS_RAW"].fillna("")

    meta["mrun_unicos_asignaciones_validas"] = int(
        agg.loc[agg["TIENE_VALIDO"] == 1, "MRUN"].nunique())
    assert agg["MRUN"].is_unique, "asig_agg MRUN no único"
    return agg, meta


# ============================================================================
# 5. Pipeline por año
# ============================================================================
def build_year(year: int, solo_fuas: bool = True) -> dict:
    """
    Construye staging + analytic para un año y devuelve un dict de métricas.
    """
    m = {"anio": year}

    # ---- lectura FUAS ----
    df_fuas, enc_fuas = read_csv_robust(cfg.FUAS_FILES[year])
    df_fuas = normalize_columns(df_fuas)
    m["encoding_fuas"] = enc_fuas
    m["filas_fuente_fuas"] = int(len(df_fuas))

    # ---- filtro de población ----
    if "PROCESO" not in df_fuas.columns:
        raise RuntimeError(f"FUAS {year}: falta columna PROCESO")
    m["proceso_valores"] = df_fuas["PROCESO"].value_counts(dropna=False).to_dict()
    if solo_fuas:
        df_fuas = df_fuas[df_fuas["PROCESO"] == cfg.PROCESO_MODELO].copy()
    m["filas_post_filtro_fuas"] = int(len(df_fuas))

    df_fuas["MRUN"] = df_fuas["MRUN"].astype(str).str.strip()
    m["mrun_unicos_fuas"] = int(df_fuas["MRUN"].nunique())
    if df_fuas["MRUN"].duplicated().any():
        raise RuntimeError(
            f"FUAS {year}: MRUN duplicado en base (no esperado). "
            f"Duplicados: {int(df_fuas['MRUN'].duplicated().sum())}"
        )

    # ---- lectura + agregación Asignaciones ----
    df_asig, enc_asig = read_csv_robust(cfg.ASIG_FILES[year])
    df_asig = normalize_columns(df_asig)
    m["encoding_asig"] = enc_asig
    asig_agg, meta_asig = aggregate_asignaciones(df_asig)
    m.update(meta_asig)

    # Aserción: MRUN único en asignaciones agregadas
    assert asig_agg["MRUN"].is_unique, "asig_agg tiene MRUN no único"

    # ---- merge (FUAS base, LEFT JOIN) ----
    df_merged = df_fuas.merge(asig_agg, on="MRUN", how="left", validate="one_to_one")
    m["filas_post_merge"] = int(len(df_merged))
    # Aserción dura: el merge no multiplica filas
    assert len(df_merged) == len(df_fuas), (
        f"Merge multiplicó filas: {len(df_merged)} != {len(df_fuas)}"
    )

    tiene_valido_s = df_merged["TIENE_VALIDO"].fillna(0).astype(int)
    solo_desc_s = df_merged["SOLO_DESCONOCIDO"].fillna(0).astype(int)
    # Métricas de match desglosadas para no confundir "aparece en Asignaciones"
    # con "tiene beneficio válido":
    m["mrun_match_beneficio_valido"] = int((tiene_valido_s == 1).sum())
    m["mrun_match_solo_desconocido"] = int(((tiene_valido_s == 0) & (solo_desc_s == 1)).sum())
    m["mrun_sin_registro_asignacion"] = int(((tiene_valido_s == 0) & (solo_desc_s == 0)).sum())
    # "aparece en Asignaciones" = tiene válido O solo desconocido
    m["mrun_match_asignaciones"] = int(m["mrun_match_beneficio_valido"]
                                       + m["mrun_match_solo_desconocido"])
    m["pct_match_beneficio_valido"] = round(100 * (tiene_valido_s == 1).mean(), 4)

    # ---- construcción del target de TRES estados (en staging) ----
    #   1  -> al menos un beneficio reconocido (TIENE_VALIDO == 1)
    #   0  -> no aparece con ningún beneficio (sin registro válido en Asignaciones)
    #   NA -> aparece SOLO con sigla(s) desconocida(s)
    target_raw = pd.Series(0, index=df_merged.index, dtype="float")
    target_raw[tiene_valido_s == 1] = 1.0
    target_raw[(tiene_valido_s == 0) & (solo_desc_s == 1)] = float("nan")
    df_merged["OBTUVO_BENEFICIO_RAW"] = target_raw

    m["target_raw_dist"] = {
        "1_beneficio_valido": int((target_raw == 1).sum()),
        "0_sin_registro_valido": int((target_raw == 0).sum()),
        "NA_solo_desconocido": int(target_raw.isna().sum()),
    }
    m["mrun_excluidos_solo_desconocido"] = int(target_raw.isna().sum())
    m["pct_excluidos_solo_desconocido"] = round(
        100 * target_raw.isna().mean(), 4)

    # Rellenar flags/listas para los no-match (sin beneficio)
    for flag in ["TIENE_GRATUIDAD", "TIENE_BECA", "TIENE_FSCU", "TIENE_OTRO"]:
        df_merged[flag] = df_merged[flag].fillna(0).astype(int)
    df_merged["TIPO_BENEFICIO"] = df_merged["TIPO_BENEFICIO"].fillna("SIN_BENEFICIO")
    for col in ["BENEFICIOS_VALIDOS", "BENEFICIOS_ASIGNADOS_RAW"]:
        df_merged[col] = df_merged[col].fillna("")
    df_merged["SOLO_DESCONOCIDO"] = solo_desc_s

    # ---- casteo explícito con registro ----
    df_merged, cast_report = cast_numeric_tracked(df_merged, cfg.CAST_NUMERIC)
    m["conversiones"] = cast_report

    # ---- STAGING (conserva MRUN y el target de tres estados) ----
    staging_path = cfg.DATA_STAGING / f"staging_{year}.parquet"
    df_merged.to_parquet(staging_path, index=False)
    m["staging_path"] = str(staging_path)

    # ---- ANALYTIC (sin MRUN, sin NA de target) ----
    # Se excluyen los MRUN cuyo único registro es una sigla desconocida:
    # su target es NA (no resuelto) y NO se fuerza a 0.
    df_resolved = df_merged[df_merged["OBTUVO_BENEFICIO_RAW"].notna()].copy()
    df_resolved[cfg.TARGET] = df_resolved["OBTUVO_BENEFICIO_RAW"].astype(int)
    m["filas_analytic_tras_excluir_na"] = int(len(df_resolved))

    tgt_counts = df_resolved[cfg.TARGET].value_counts().to_dict()
    m["target_dist"] = {int(k): int(v) for k, v in tgt_counts.items()}
    m["target_pos_rate"] = round(100 * df_resolved[cfg.TARGET].mean(), 4)

    keep = [c for c in cfg.PREDICTORS if c in df_resolved.columns]
    keep += [cfg.TARGET]
    keep += [c for c in cfg.ANALYSIS_ONLY if c in df_resolved.columns]
    df_analytic = df_resolved[keep].copy()
    assert cfg.KEY not in df_analytic.columns, "MRUN presente en capa analítica"
    assert df_analytic[cfg.TARGET].isna().sum() == 0, "target con NA en analytic"
    analytic_path = cfg.DATA_ANALYTIC / f"analytic_{year}.parquet"
    df_analytic.to_parquet(analytic_path, index=False)
    m["analytic_path"] = str(analytic_path)
    m["analytic_shape"] = list(df_analytic.shape)

    # ---- nulos por predictor (en analytic) ----
    m["nulos_predictores"] = {
        c: int(df_analytic[c].isna().sum())
        for c in cfg.PREDICTORS if c in df_analytic.columns
    }

    # beneficios múltiples (cuántos MRUN con >1 sigla válida)
    multi = (df_merged["BENEFICIOS_VALIDOS"].str.contains(r"\|", na=False)).sum()
    m["mrun_beneficios_multiples"] = int(multi)

    return m


def run_all(solo_fuas: bool = True) -> dict:
    results = {}
    for year in (2024, 2025):
        results[year] = build_year(year, solo_fuas=solo_fuas)
    return results


if __name__ == "__main__":
    res = run_all()
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
