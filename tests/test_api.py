import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)


def test_predict_endpoint():
    response = client.post(
        "/api/v1/predict",
        json={
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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_price"] > 0
    assert payload["currency"] == "INR"
    assert payload["confidence_interval"]["lower_bound"] < payload["confidence_interval"]["upper_bound"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_property_is_rejected():
    response = client.post(
        "/api/v1/predict",
        json={
            "area": -1,
            "bedrooms": 2,
            "bathrooms": 2,
            "age": 10,
            "location": "Bengaluru",
            "property_type": "Apartment",
            "floor": 5,
        },
    )
    assert response.status_code == 422


def test_prediction_is_recorded_for_business_dashboard():
    test_predict_endpoint()

    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction_count"] > 0
    assert payload["recent_predictions"][0]["predicted_price"] > 0
    assert payload["daily_activity"]


def test_model_registry_endpoint_includes_versions():
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["versions"]
    assert payload["versions"][-1]["production"] is True


def test_prediction_api_key_is_optional_but_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    response = client.post(
        "/api/v1/predict",
        json={
            "area": 1500,
            "bedrooms": 2,
            "bathrooms": 2,
            "age": 10,
            "location": "Bengaluru",
            "property_type": "Apartment",
            "floor": 5,
        },
    )
    assert response.status_code == 401

    authorized_response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": "test-secret"},
        json={
            "area": 1500,
            "bedrooms": 2,
            "bathrooms": 2,
            "age": 10,
            "location": "Bengaluru",
            "property_type": "Apartment",
            "floor": 5,
        },
    )
    assert authorized_response.status_code == 200
