import pandas as pd
import pytest

from schema_validation import (
    SchemaValidationError,
    assert_valid_schema,
    validate_dataframe,
)


def make_valid_fuas_2024():
    return pd.DataFrame({
        "MRUN": [1, 2],
        "PROCESO": ["FUAS", "FUAS"],
        "NEM": [600, 650],
        "QUINTIL_SE4": [1, 2],
        "COD_DEPE": [2, 3],
        "GENERO": [1, 2],
        "NACIONALIDAD": [0, 0],
        "FEC_NAC_ALU": [
            "2005-01-01",
            "2004-01-01",
        ],
    })


def make_valid_fuas_2025():
    return pd.DataFrame({
        "MRUN": [1, 2],
        "PROCESO": ["FUAS", "FUAS"],
        "NEM": [600, 650],
        "QUINTIL_SE4": [1, 2],
        "COD_DEPE": [2, 3],
        "GENERO": [1, 2],
        "NACIONALIDAD": [0, 0],
        "FEC_NAC": [
            "2005-01-01",
            "2004-01-01",
        ],
    })


def test_valid_fuas_2024():
    report = validate_dataframe(
        make_valid_fuas_2024(),
        "FUAS",
        2024,
    )

    assert report["valid"] is True
    assert report["errors"] == []


def test_valid_fuas_2025():
    report = validate_dataframe(
        make_valid_fuas_2025(),
        "FUAS",
        2025,
    )

    assert report["valid"] is True


def test_missing_required_column():
    df = make_valid_fuas_2024().drop(
        columns=["NEM"]
    )

    report = validate_dataframe(
        df,
        "FUAS",
        2024,
    )

    assert report["valid"] is False

    assert any(
        error["type"] == "missing_columns"
        for error in report["errors"]
    )


def test_wrong_birth_date_column_2024():
    df = make_valid_fuas_2024().rename(
        columns={
            "FEC_NAC_ALU": "FEC_NAC"
        }
    )

    report = validate_dataframe(
        df,
        "FUAS",
        2024,
    )

    assert report["valid"] is False


def test_invalid_quintile():
    df = make_valid_fuas_2024()
    df.loc[0, "QUINTIL_SE4"] = 9

    report = validate_dataframe(
        df,
        "FUAS",
        2024,
    )

    assert report["valid"] is False

    assert any(
        error["type"] == "invalid_domain"
        for error in report["errors"]
    )


def test_assert_valid_schema_raises():
    df = make_valid_fuas_2024().drop(
        columns=["MRUN"]
    )

    with pytest.raises(
        SchemaValidationError
    ):
        assert_valid_schema(
            df,
            "FUAS",
            2024,
        )


def test_valid_asignaciones():
    df = pd.DataFrame({
        "MRUN": [1, 2],
        "BENEFICIO_BECA_FSCU": [
            "GRATUIDAD",
            "BGM",
        ],
    })

    report = validate_dataframe(
        df,
        "ASIGNACIONES",
        2025,
    )

    assert report["valid"] is True
