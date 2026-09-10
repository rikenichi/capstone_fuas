
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
import api


client = TestClient(api.app)


PERFIL_VALIDO = {
    "NEM": 650,
    "QUINTIL_SE4": 2,
    "COD_DEPE": 3,
    "EDAD": 19,
    "GENERO": 2,
    "NACIONALIDAD": 0,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    response = client.post("/predict", json=PERFIL_VALIDO)

    assert response.status_code == 200

    data = response.json()

    assert abs(data["probabilidad_pct"] - 74.98) < 0.01
    assert data["threshold"] == 0.42
    assert data["clasificacion"] == 1


def test_explain():
    response = client.post("/explain", json=PERFIL_VALIDO)

    assert response.status_code == 200

    data = response.json()

    assert "factores_principales" in data
    assert len(data["factores_principales"]) > 0
    assert data["error_aditividad"] < 1e-10


def test_nem_invalido():
    perfil = PERFIL_VALIDO.copy()
    perfil["NEM"] = 900

    response = client.post("/predict", json=perfil)

    assert response.status_code == 422


def test_quintil_invalido():
    perfil = PERFIL_VALIDO.copy()
    perfil["QUINTIL_SE4"] = 9

    response = client.post("/predict", json=perfil)

    assert response.status_code == 422


def test_campo_faltante():
    perfil = PERFIL_VALIDO.copy()
    perfil.pop("NACIONALIDAD")

    response = client.post("/predict", json=perfil)

    assert response.status_code == 422
