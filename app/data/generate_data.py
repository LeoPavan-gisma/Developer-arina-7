from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_real_estate_dataset(n_samples: int = 2500, output_path: str | Path | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    locations = ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Delhi", "Mumbai"]
    property_types = ["Apartment", "Villa", "Studio", "Townhouse"]
    facing_options = ["East", "West", "North", "South"]

    area = rng.normal(1650, 550, n_samples).clip(400, 5000)
    bedrooms = rng.integers(1, 6, size=n_samples)
    bathrooms = rng.integers(1, 5, size=n_samples)
    age = rng.integers(0, 36, size=n_samples)
    floor_number = rng.integers(1, 25, size=n_samples)
    amenities_score = rng.integers(20, 100, size=n_samples)
    distance_to_center = rng.normal(9, 5, size=n_samples).clip(1, 30)
    location = rng.choice(locations, size=n_samples)
    property_type = rng.choice(property_types, size=n_samples)
    facing = rng.choice(facing_options, size=n_samples)

    location_multiplier = {
        "Bengaluru": 1.18,
        "Hyderabad": 1.10,
        "Pune": 1.05,
        "Chennai": 1.00,
        "Delhi": 1.22,
        "Mumbai": 1.35,
    }
    property_multiplier = {
        "Apartment": 1.00,
        "Villa": 1.28,
        "Studio": 0.74,
        "Townhouse": 1.16,
    }
    facing_multiplier = {
        "East": 1.06,
        "West": 1.04,
        "North": 1.02,
        "South": 1.00,
    }

    base_price = (
        area * 2900
        + bedrooms * 360000
        + bathrooms * 260000
        + floor_number * 18000
        + amenities_score * 6800
        - age * 11000
        - distance_to_center * 15000
    )

    price = (
        base_price
        * np.array([location_multiplier[item] for item in location])
        * np.array([property_multiplier[item] for item in property_type])
        * np.array([facing_multiplier[item] for item in facing])
    )

    df = pd.DataFrame(
        {
            "area": area.astype(float),
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms.astype(int),
            "age": age.astype(int),
            "location": location,
            "property_type": property_type,
            "floor": floor_number.astype(int),
            "facing": facing,
            "amenities_score": amenities_score.astype(int),
            "distance_to_center": distance_to_center.astype(float),
            "price": price.astype(float),
        }
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    generate_real_estate_dataset(output_path="data/real_estate_dataset.csv")
