from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "production_model.pkl"
DATASET_PATH = BASE_DIR / "data" / "real_estate_dataset.csv"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(ARTIFACTS_DIR / "predictions.db")))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
