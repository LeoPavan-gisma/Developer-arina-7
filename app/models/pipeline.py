from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import ARTIFACTS_DIR, MODEL_PATH
from app.data.ingest import FEATURE_COLUMNS, load_training_data, validate_dataset


def build_regressor(model_name: str):
    if model_name == "xgboost" and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )

    if model_name == "xgboost":
        raise ImportError("XGBoost was requested but is not installed")

    if model_name == "neural_network":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            early_stopping=True,
            max_iter=300,
            random_state=42,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model candidate: {model_name}")


def build_pipeline(model_name: str = "xgboost") -> Any:
    numeric_features = [
        "area",
        "bedrooms",
        "bathrooms",
        "age",
        "floor",
        "amenities_score",
        "distance_to_center",
    ]
    categorical_features = ["location", "property_type", "facing"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_regressor(model_name)),
        ]
    )
    if model_name == "neural_network":
        return TransformedTargetRegressor(
            regressor=pipeline,
            transformer=StandardScaler(),
        )
    return pipeline


def _dataset_digest(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()


def _model_registry_path(model_path: Path) -> Path:
    return model_path.parent / "model_registry.json"


def train_model(
    df: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    model_candidates: tuple[str, ...] = ("xgboost", "random_forest", "neural_network"),
) -> tuple[Any, dict[str, Any]]:
    model_path = Path(model_path)
    model_path.parent.mkdir(exist_ok=True, parents=True)
    df = validate_dataset(df)
    invalid_candidates = set(model_candidates) - {"xgboost", "random_forest", "neural_network"}
    if invalid_candidates:
        raise ValueError(f"Unknown model candidates: {', '.join(sorted(invalid_candidates))}")

    X = df[FEATURE_COLUMNS]
    y = df["price"]
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    candidate_results: list[tuple[float, str, Any, dict[str, float]]] = []

    for candidate in model_candidates:
        try:
            pipeline = build_pipeline(candidate)
        except ImportError:
            if candidate == "xgboost":
                continue
            raise
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_validation)
        metrics = {
            "mae": float(mean_absolute_error(y_validation, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_validation, predictions))),
            "r2_score": float(r2_score(y_validation, predictions)),
        }
        candidate_results.append((metrics["r2_score"], candidate, pipeline, metrics))

    if not candidate_results:
        raise RuntimeError("No available model candidates could be trained")
    _, selected_model, pipeline, selected_metrics = max(candidate_results, key=lambda item: item[0])

    validation_errors = np.abs(y_validation.to_numpy() - pipeline.predict(X_validation))
    interval_radius = float(np.quantile(validation_errors, 0.9))
    metrics: dict[str, Any] = {
        **selected_metrics,
        "validation_rows": int(len(y_validation)),
        "training_rows": int(len(y_train)),
        "selected_model": selected_model,
        "interval_radius": interval_radius,
        "candidate_metrics": {
            name: candidate_metrics
            for _, name, _, candidate_metrics in candidate_results
        },
    }

    temp_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(pipeline, temp_model_path)
    temp_model_path.replace(model_path)

    registry_path = _model_registry_path(model_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": []}
    version = len(registry["models"]) + 1
    archived_model_path = None
    if model_path.resolve() == MODEL_PATH.resolve():
        archived_dir = ARTIFACTS_DIR / "registry"
        archived_dir.mkdir(parents=True, exist_ok=True)
        archived_model_path = archived_dir / f"model-v{version:03d}.joblib"
        shutil.copy2(model_path, archived_model_path)
    dataset_sha256 = _dataset_digest(df)
    dataset_snapshot_dir = model_path.parent / "datasets"
    dataset_snapshot_dir.mkdir(parents=True, exist_ok=True)
    dataset_snapshot = dataset_snapshot_dir / f"dataset-{dataset_sha256[:16]}.csv"
    if not dataset_snapshot.exists():
        df.to_csv(dataset_snapshot, index=False)
    registry_entry = {
        "version": version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_type": selected_model,
        "model_path": str((archived_model_path or model_path).resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_sha256,
        "dataset_snapshot": str(dataset_snapshot.resolve()),
        "metrics": metrics,
        "interval_radius": interval_radius,
        "production": True,
    }
    for entry in registry["models"]:
        entry["production"] = False
    registry["models"].append(registry_entry)
    temp_registry_path = registry_path.with_suffix(".json.tmp")
    temp_registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    temp_registry_path.replace(registry_path)

    if archived_model_path is not None:
        archived_dir = archived_model_path.parent
        (archived_dir / f"model-v{version:03d}.json").write_text(
            json.dumps(registry_entry, indent=2), encoding="utf-8"
        )
    return pipeline, metrics


def load_model(model_path: str | Path = MODEL_PATH) -> Any:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def ensure_model(model_path: str | Path = MODEL_PATH) -> Any:
    model_path = Path(model_path)
    if not model_path.exists() or not _model_registry_path(model_path).exists():
        df = load_training_data()
        train_model(df, model_path=model_path)
    return load_model(model_path)


def get_model_metadata(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    registry_path = _model_registry_path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found at {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    production_models = [item for item in registry.get("models", []) if item.get("production")]
    if not production_models:
        raise ValueError(f"No production model is registered in {registry_path}")
    return production_models[-1]


def get_model_registry(model_path: str | Path = MODEL_PATH) -> list[dict[str, Any]]:
    registry_path = _model_registry_path(Path(model_path))
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found at {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return [
        {
            "version": entry["version"],
            "registered_at": entry["registered_at"],
            "model_type": entry["model_type"],
            "dataset_sha256": entry["dataset_sha256"],
            "metrics": entry["metrics"],
            "production": entry["production"],
        }
        for entry in registry.get("models", [])
    ]


def predict_price(record: dict[str, Any], model: Any = None) -> float:
    if model is None:
        model = ensure_model()

    row = pd.DataFrame([record])
    return float(model.predict(row)[0])


if __name__ == "__main__":
    dataset = load_training_data()
    _, metrics = train_model(dataset)
    print(metrics)
