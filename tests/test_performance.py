from time import perf_counter

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
PROPERTY = {
    "area": 1500,
    "bedrooms": 2,
    "bathrooms": 2,
    "age": 10,
    "location": "Bengaluru",
    "property_type": "Apartment",
    "floor": 5,
    "facing": "East",
    "amenities_score": 70,
    "distance_to_center": 8,
}


def test_batch_prediction_performance_smoke(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    started_at = perf_counter()
    response = client.post("/api/v1/batch", json=[PROPERTY] * 100)
    elapsed_seconds = perf_counter() - started_at

    assert response.status_code == 200
    assert response.json()["count"] == 100
    assert elapsed_seconds < 15
