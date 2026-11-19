from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DATABASE_PATH


def _connect(database_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path: str | Path = DATABASE_PATH) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                location TEXT NOT NULL,
                property_type TEXT NOT NULL,
                area REAL NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )


def record_prediction(
    prediction_id: str,
    predicted_price: float,
    location: str,
    property_type: str,
    area: float,
    model_version: str,
    database_path: str | Path = DATABASE_PATH,
) -> None:
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO predictions
                (prediction_id, created_at, predicted_price, location, property_type, area, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                datetime.now(timezone.utc).isoformat(),
                predicted_price,
                location,
                property_type,
                area,
                model_version,
            ),
        )


def get_dashboard_analytics(database_path: str | Path = DATABASE_PATH) -> dict[str, Any]:
    with _connect(database_path) as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS prediction_count,
                   COALESCE(AVG(predicted_price), 0) AS average_price,
                   COALESCE(MIN(predicted_price), 0) AS minimum_price,
                   COALESCE(MAX(predicted_price), 0) AS maximum_price
            FROM predictions
            """
        ).fetchone()
        by_location = connection.execute(
            """
            SELECT location, COUNT(*) AS prediction_count,
                   ROUND(AVG(predicted_price), 2) AS average_price
            FROM predictions
            GROUP BY location
            ORDER BY prediction_count DESC, location
            """
        ).fetchall()
        daily_activity = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS date,
                   COUNT(*) AS prediction_count,
                   ROUND(AVG(predicted_price), 2) AS average_price
            FROM predictions
            GROUP BY substr(created_at, 1, 10)
            ORDER BY date DESC
            LIMIT 30
            """
        ).fetchall()
        recent = connection.execute(
            """
            SELECT prediction_id, created_at, predicted_price, location, property_type, area, model_version
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()

    return {
        "prediction_count": int(totals["prediction_count"]),
        "average_price": float(totals["average_price"]),
        "minimum_price": float(totals["minimum_price"]),
        "maximum_price": float(totals["maximum_price"]),
        "by_location": [dict(row) for row in by_location],
        "daily_activity": [dict(row) for row in reversed(daily_activity)],
        "recent_predictions": [dict(row) for row in recent],
    }
