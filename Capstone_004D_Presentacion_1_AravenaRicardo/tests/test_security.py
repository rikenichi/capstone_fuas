
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


PERFIL_VALIDO = {
    "NEM": 650,
    "QUINTIL_SE4": 2,
    "COD_DEPE": 3,
    "EDAD": 19,
    "GENERO": 2,
    "NACIONALIDAD": 0,
}


def test_rechaza_mrun():
    payload = {
        **PERFIL_VALIDO,
        "MRUN": "123456",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_rechaza_rut():
    payload = {
        **PERFIL_VALIDO,
        "RUT": "12.345.678-9",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_rechaza_campos_desconocidos():
    payload = {
        **PERFIL_VALIDO,
        "variable_inventada": "abc",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_rechaza_nem_fuera_de_rango():
    payload = {
        **PERFIL_VALIDO,
        "NEM": 900,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_rechaza_quintil_invalido():
    payload = {
        **PERFIL_VALIDO,
        "QUINTIL_SE4": 6,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_no_devuelve_identificadores():
    response = client.post(
        "/predict",
        json=PERFIL_VALIDO,
    )

    assert response.status_code == 200

    body = response.json()

    texto = str(body).upper()

    assert "MRUN" not in texto
    assert "RUT" not in texto
