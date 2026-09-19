from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Esquemas críticos del pipeline del modelo
# ============================================================

FUAS_REQUIRED_COMMON = {
    "MRUN",
    "PROCESO",
    "NEM",
    "QUINTIL_SE4",
    "COD_DEPE",
    "GENERO",
    "NACIONALIDAD",
}

FUAS_DATE_ALIASES = {
    # Esquemas confirmados contra los CSV reales MINEDUC.
    # Se mantienen alias históricos para tolerar variantes publicadas.
    2024: {"FEC_NAC", "FEC_NAC_ALU"},
    2025: {"FEC_NAC", "FEC_NAC_ALU"},
}

ASIGNACIONES_REQUIRED = {
    "MRUN",
    "BENEFICIO_BECA_FSCU",
}

VALID_DOMAINS = {
    "QUINTIL_SE4": {1, 2, 3, 4, 5},
    "COD_DEPE": {1, 2, 3, 4, 5, 6},
    "GENERO": {0, 1, 2},
    "NACIONALIDAD": {0, 1, 2},
}


class SchemaValidationError(Exception):
    """Error de validación estructural de datos."""


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    copia = df.copy()
    copia.columns = [
        str(col).strip().upper()
        for col in copia.columns
    ]
    return copia


def _valores_numericos_no_nulos(serie: pd.Series) -> set:
    valores = pd.to_numeric(
        serie,
        errors="coerce"
    ).dropna()

    return set(valores.astype(int).unique().tolist())


def validate_fuas(
    df: pd.DataFrame,
    year: int,
) -> dict[str, Any]:
    """
    Valida estructura y dominios críticos de un archivo FUAS.

    No modifica el dataframe original.
    """

    df = _normalizar_columnas(df)

    errors = []
    warnings = []

    columns = set(df.columns)

    # --------------------------------------------------------
    # Columnas obligatorias comunes
    # --------------------------------------------------------
    missing_common = sorted(
        FUAS_REQUIRED_COMMON - columns
    )

    if missing_common:
        errors.append({
            "type": "missing_columns",
            "columns": missing_common,
        })

    # --------------------------------------------------------
    # Fecha de nacimiento por año
    # --------------------------------------------------------
    expected_date_columns = FUAS_DATE_ALIASES.get(year)

    if expected_date_columns:
        if not (columns & expected_date_columns):
            errors.append({
                "type": "missing_birth_date_column",
                "expected_any_of": sorted(
                    expected_date_columns
                ),
            })

    # --------------------------------------------------------
    # Validar población FUAS
    # --------------------------------------------------------
    if "PROCESO" in df.columns:
        procesos = set(
            df["PROCESO"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
        )

        if "FUAS" not in procesos:
            errors.append({
                "type": "missing_required_process",
                "expected": "FUAS",
                "observed_sample": sorted(procesos)[:20],
            })

    # --------------------------------------------------------
    # Dominios categóricos
    # --------------------------------------------------------
    for column, valid_values in VALID_DOMAINS.items():
        if column not in df.columns:
            continue

        observed = _valores_numericos_no_nulos(
            df[column]
        )

        invalid = sorted(
            observed - valid_values
        )

        if invalid:
            errors.append({
                "type": "invalid_domain",
                "column": column,
                "invalid_values": invalid[:20],
            })

    # --------------------------------------------------------
    # NEM
    # --------------------------------------------------------
    if "NEM" in df.columns:
        nem = pd.to_numeric(
            df["NEM"],
            errors="coerce"
        ).dropna()

        outside_range = nem[
            (nem < 100) | (nem > 700)
        ]

        if len(outside_range) > 0:
            warnings.append({
                "type": "nem_outside_expected_range",
                "count": int(len(outside_range)),
            })

    # --------------------------------------------------------
    # MRUN
    # --------------------------------------------------------
    if "MRUN" in df.columns:
        null_mrun = int(
            df["MRUN"].isna().sum()
        )

        if null_mrun > 0:
            warnings.append({
                "type": "null_mrun",
                "count": null_mrun,
            })

        duplicated = int(
            df["MRUN"]
            .dropna()
            .duplicated()
            .sum()
        )

        if duplicated > 0:
            warnings.append({
                "type": "duplicated_mrun",
                "count": duplicated,
            })

    return {
        "dataset": "FUAS",
        "year": year,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_asignaciones(
    df: pd.DataFrame,
    year: int,
) -> dict[str, Any]:
    """
    Valida estructura crítica de archivos de asignaciones.
    """

    df = _normalizar_columnas(df)

    errors = []
    warnings = []

    columns = set(df.columns)

    missing = sorted(
        ASIGNACIONES_REQUIRED - columns
    )

    if missing:
        errors.append({
            "type": "missing_columns",
            "columns": missing,
        })

    if "MRUN" in df.columns:
        null_mrun = int(
            df["MRUN"].isna().sum()
        )

        if null_mrun > 0:
            warnings.append({
                "type": "null_mrun",
                "count": null_mrun,
            })

    if "BENEFICIO_BECA_FSCU" in df.columns:
        empty = int(
            df["BENEFICIO_BECA_FSCU"]
            .isna()
            .sum()
        )

        if empty > 0:
            warnings.append({
                "type": "empty_benefit_code",
                "count": empty,
            })

    return {
        "dataset": "ASIGNACIONES",
        "year": year,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_dataframe(
    df: pd.DataFrame,
    dataset_type: str,
    year: int,
) -> dict[str, Any]:

    dataset_type = dataset_type.strip().upper()

    if dataset_type == "FUAS":
        return validate_fuas(
            df=df,
            year=year,
        )

    if dataset_type in {
        "ASIGNACIONES",
        "ASIGNACION",
    }:
        return validate_asignaciones(
            df=df,
            year=year,
        )

    raise ValueError(
        f"dataset_type no soportado: {dataset_type}"
    )


def assert_valid_schema(
    df: pd.DataFrame,
    dataset_type: str,
    year: int,
) -> dict[str, Any]:
    """
    Interrumpe el pipeline si existe una incompatibilidad
    estructural crítica.
    """

    report = validate_dataframe(
        df=df,
        dataset_type=dataset_type,
        year=year,
    )

    if not report["valid"]:
        raise SchemaValidationError(
            f"Esquema inválido para "
            f"{dataset_type} {year}: "
            f"{report['errors']}"
        )

    return report


def read_csv_with_encoding_fallback(
    path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Lectura robusta para las codificaciones observadas
    en archivos MINEDUC.
    """

    path = Path(path)

    encodings = [
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                **kwargs,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error
