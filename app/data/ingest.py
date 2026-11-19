from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATASET_PATH
from app.data.generate_data import generate_real_estate_dataset

FEATURE_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "age",
    "location",
    "property_type",
    "floor",
    "facing",
    "amenities_score",
    "distance_to_center",
]
NUMERIC_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "age",
    "floor",
    "amenities_score",
    "distance_to_center",
    "price",
]


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = FEATURE_COLUMNS + ["price"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
    if len(df) < 10:
        raise ValueError("Dataset must contain at least 10 rows")
    if df[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")

    validated = df.copy()
    for column in NUMERIC_COLUMNS:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset column '{column}' must contain numeric values") from exc
        if not np.isfinite(validated[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Dataset column '{column}' contains non-finite values")

    positive_columns = ["area", "bedrooms", "bathrooms", "floor", "distance_to_center", "price"]
    if (validated[positive_columns] <= 0).any().any():
        raise ValueError(f"Dataset values must be positive for: {', '.join(positive_columns)}")
    if (validated[["age", "amenities_score"]] < 0).any().any():
        raise ValueError("Age and amenities_score cannot be negative")
    for column in ["location", "property_type", "facing"]:
        if validated[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset column '{column}' cannot contain blank values")
    return validated


def load_training_data(dataset_path: str | Path | None = None) -> pd.DataFrame:
    configured_path = dataset_path or os.getenv("DATA_SOURCE_PATH") or DATASET_PATH
    source_path = Path(configured_path)
    if not source_path.exists():
        if dataset_path is not None or os.getenv("DATA_SOURCE_PATH"):
            raise FileNotFoundError(f"Configured training data file does not exist: {source_path}")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        generate_real_estate_dataset(output_path=source_path)
    return validate_dataset(pd.read_csv(source_path))
