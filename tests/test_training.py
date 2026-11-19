from __future__ import annotations

import pandas as pd
import pytest

from app.data.generate_data import generate_real_estate_dataset
from app.data.ingest import validate_dataset
from app.models.pipeline import train_model


def test_training_pipeline(tmp_path):
    df = generate_real_estate_dataset(500)
    model_path = tmp_path / "test_model.pkl"
    model, metrics = train_model(df, model_path=model_path)

    assert model is not None
    assert metrics["r2_score"] > 0.7
    assert metrics["mae"] > 0
    assert metrics["validation_rows"] == 100
    assert metrics["selected_model"] in {"xgboost", "random_forest", "neural_network"}
    assert len(metrics["candidate_metrics"]) == 3
    assert metrics["interval_radius"] > 0
    assert (tmp_path / "model_registry.json").exists()
    registry = (tmp_path / "model_registry.json").read_text(encoding="utf-8")
    assert "dataset_sha256" in registry
    assert list((tmp_path / "datasets").glob("dataset-*.csv"))

    prediction = model.predict(pd.DataFrame([{
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
    }]))[0]

    assert prediction > 0


def test_dataset_validation_rejects_missing_values():
    df = generate_real_estate_dataset(20)
    df.loc[0, "price"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(df)
