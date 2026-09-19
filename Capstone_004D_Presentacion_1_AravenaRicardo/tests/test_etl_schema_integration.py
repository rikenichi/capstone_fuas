import pandas as pd
import pytest

from schema_validation import (
    SchemaValidationError,
    assert_valid_schema,
)


def test_etl_stops_before_processing_invalid_schema():
    """
    Simula un archivo FUAS cuyo esquema cambió antes
    de entrar al procesamiento ETL.
    """

    df = pd.DataFrame({
        "MRUN": [1],
        "PROCESO": ["FUAS"],
        "NEM": [650],

        # Cambio inesperado de esquema:
        "QUINTIL_NUEVO": [2],

        "COD_DEPE": [3],
        "GENERO": [2],
        "NACIONALIDAD": [0],
        "FEC_NAC": ["2005-01-01"],
    })

    with pytest.raises(
        SchemaValidationError
    ):
        assert_valid_schema(
            df,
            "FUAS",
            2025,
        )
