
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_dashboard_endpoint():
    response = client.get("/dashboard")

    assert response.status_code == 200

    body = response.json()

    assert "metadata" in body
    assert "resumen_anual" in body
    assert "por_genero" in body
    assert "por_quintil" in body
    assert "por_edad" in body
    assert "cobertura" in body

    assert len(body["resumen_anual"]) == 18

    anios = {
        item["anio"]
        for item in body["resumen_anual"]
    }

    assert 2008 in anios
    assert 2025 in anios
