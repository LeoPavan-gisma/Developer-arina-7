from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.config import MODEL_PATH, MODEL_VERSION
from app.api.storage import get_dashboard_analytics, initialize_database, record_prediction
from app.models.pipeline import ensure_model, get_model_metadata, get_model_registry, predict_price

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = FastAPI(title="Real Estate Price Predictor", version="1.0.0")
logger = logging.getLogger("real_estate_api")

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction failures")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

MODEL = ensure_model(MODEL_PATH)
MODEL_METADATA = get_model_metadata(MODEL_PATH)
initialize_database()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected_api_key = os.getenv("API_KEY")
    if expected_api_key and (
        x_api_key is None or not secrets.compare_digest(x_api_key, expected_api_key)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class PropertyRequest(BaseModel):
    area: float = Field(..., gt=0, description="Area in square feet")
    bedrooms: int = Field(..., ge=1)
    bathrooms: int = Field(..., ge=1)
    age: int = Field(..., ge=0)
    location: str = Field(..., min_length=1)
    property_type: str = Field(..., min_length=1)
    floor: int = Field(..., ge=1)
    facing: str = Field(default="East", min_length=1)
    amenities_score: int = Field(default=70, ge=0, le=100)
    distance_to_center: float = Field(default=8.0, gt=0)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "model_loaded": True}


@app.get("/api/v1/health")
def api_health() -> dict[str, Any]:
    return health()


@app.post("/api/v1/predict")
@PREDICTION_LATENCY.time()
def predict(request: PropertyRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
    prediction_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        prediction = predict_price(request.model_dump(), model=MODEL)
        interval_radius = float(MODEL_METADATA.get("interval_radius", 0.0))
        PREDICTION_COUNTER.inc()
        record_prediction(
            prediction_id=prediction_id,
            predicted_price=prediction,
            location=request.location,
            property_type=request.property_type,
            area=request.area,
            model_version=str(MODEL_METADATA.get("version", MODEL_VERSION)),
        )
        return {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "predicted_price": round(float(prediction), 2),
            "currency": "INR",
            "confidence_interval": {
                "lower_bound": round(max(0.0, float(prediction - interval_radius)), 2),
                "upper_bound": round(float(prediction + interval_radius), 2),
            },
            "model_version": str(MODEL_METADATA.get("version", MODEL_VERSION)),
            "metadata": {
                "area_sqft": request.area,
                "location": request.location,
                "property_type": request.property_type,
            },
        }
    except Exception as exc:  # pragma: no cover - defensive path
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed for %s", prediction_id)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/api/v1/batch")
def batch_predict(
    records: list[PropertyRequest],
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    if len(records) > 1000:
        raise HTTPException(status_code=422, detail="A batch may contain at most 1000 properties")
    predictions = []
    for record in records:
        prediction = predict_price(record.model_dump(), model=MODEL)
        prediction_id = str(uuid4())
        record_prediction(
            prediction_id=prediction_id,
            predicted_price=prediction,
            location=record.location,
            property_type=record.property_type,
            area=record.area,
            model_version=str(MODEL_METADATA.get("version", MODEL_VERSION)),
        )
        PREDICTION_COUNTER.inc()
        predictions.append({
            "prediction_id": prediction_id,
            "location": record.location,
            "predicted_price": round(float(prediction), 2),
        })
    return {"count": len(predictions), "predictions": predictions}


@app.get("/api/v1/metrics")
def service_metrics() -> dict[str, Any]:
    analytics = get_dashboard_analytics()
    return {
        "status": "healthy",
        "metrics": {
            "predictions_total": analytics["prediction_count"],
            "model_version": MODEL_METADATA.get("version", MODEL_VERSION),
            "validation_mae": MODEL_METADATA.get("metrics", {}).get("mae"),
            "validation_r2": MODEL_METADATA.get("metrics", {}).get("r2_score"),
        },
    }


@app.get("/api/v1/model")
def model_details() -> dict[str, Any]:
    return {
        "version": MODEL_METADATA.get("version", MODEL_VERSION),
        "model_type": MODEL_METADATA.get("model_type", "unknown"),
        "registered_at": MODEL_METADATA.get("registered_at"),
        "dataset_sha256": MODEL_METADATA.get("dataset_sha256"),
        "metrics": MODEL_METADATA.get("metrics", {}),
        "versions": get_model_registry(MODEL_PATH),
    }


@app.get("/api/v1/dashboard")
def dashboard() -> dict[str, Any]:
    return get_dashboard_analytics()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
