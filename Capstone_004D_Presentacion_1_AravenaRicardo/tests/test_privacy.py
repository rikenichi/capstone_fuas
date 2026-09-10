
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


PERFIL = {
    "NEM": 650,
    "QUINTIL_SE4": 2,
    "COD_DEPE": 3,
    "EDAD": 19,
    "GENERO": 2,
    "NACIONALIDAD": 0,
}


def test_predict_no_reexpone_perfil():
    response = client.post(
        "/predict",
        json=PERFIL,
    )

    assert response.status_code == 200

    body = response.json()

    assert "features_utilizadas" not in body


def test_predict_no_contiene_identificadores():
    response = client.post(
        "/predict",
        json=PERFIL,
    )

    assert response.status_code == 200

    contenido = str(response.json()).upper()

    assert "MRUN" not in contenido
    assert "RUT" not in contenido
    assert "RUN" not in contenido


def test_openapi_no_define_identificadores():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    perfil = (
        schema["components"]
        ["schemas"]
        ["PerfilFUAS"]
    )

    campos = set(
        perfil["properties"].keys()
    )

    assert "RUT" not in campos
    assert "RUN" not in campos
    assert "MRUN" not in campos

    assert perfil.get(
        "additionalProperties"
    ) is False
