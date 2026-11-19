# Real Estate Price Prediction System

A runnable data-science capstone with a reproducible data pipeline, model comparison and registry, prediction API, interactive business dashboard, local Docker deployment, monitoring, and automated tests.

> **Important:** The included data is synthetic and generated for demonstration. Model quality and prediction ranges are not evidence of real-world property valuation accuracy. Connect a licensed, representative data source and validate the model before using it for business decisions.

## What is included

- **Data:** deterministic synthetic data generator; the pipeline creates the dataset if it is missing.
- **Machine learning:** preprocessing pipeline and validation-based comparison of XGBoost, Random Forest, and an MLP neural network. Reports validation MAE, RMSE, and R².
- **Versioning:** each training run records a model version, model digest, dataset SHA-256, dataset snapshot, selected estimator, validation metrics, and estimated 90th-percentile absolute validation error. Earlier model binaries are retained under `artifacts/registry/`.
- **API:** FastAPI prediction, batch, health, model-info, dashboard analytics, JSON service metrics, Prometheus metrics, and interactive OpenAPI docs.
- **App and analytics:** Streamlit prediction form plus prediction history and breakdown by location, stored in SQLite.
- **Operations:** Docker Compose, persistent local artifacts, Prometheus scrape configuration and alert rules, provisioned Grafana dashboard, optional API-key protection.
- **Automation:** GitHub Actions tests on pushes and pull requests, plus scheduled model refreshes that publish versioned artifacts.

## Run locally (Windows PowerShell)

Python 3.9 was used for local validation. Newer Python versions have not been verified; the machine's system Python 3.14 was not compatible with the selected dependency wheels.

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional: generate sample data explicitly; API startup does this if needed.
python -m app.data.generate_data
python -m app.models.pipeline

# Start the API (keep this terminal open).
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal, activate the same environment and start the UI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

- Streamlit app: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- Prometheus metrics: <http://localhost:8000/metrics>

The generated dataset, model, model registry, snapshots, and prediction database are stored under `data/` and `artifacts/`. These runtime files are excluded from Git.

## Run with Docker Compose

Install Docker Desktop, then from the project directory:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Services:

| Service | URL | Notes |
|---|---|---|
| Streamlit | <http://localhost:8501> | Prediction form and business dashboard |
| FastAPI | <http://localhost:8000/docs> | API docs and endpoints |
| Prometheus | <http://localhost:9090> | Scraped API metrics and alert rules |
| Grafana | <http://localhost:3000> | Provisioned dashboard; local default login is `admin` / `admin` |

For a shared or production environment, set unique `API_KEY`, `GRAFANA_USER`, and `GRAFANA_PASSWORD` values in `.env`, restrict exposed ports, and use a managed database and secret store. The default Grafana credentials are only suitable for a local demo.

## API

Example request:

```powershell
$body = @{
  area = 1600
  bedrooms = 3
  bathrooms = 2
  age = 12
  location = "Bengaluru"
  property_type = "Apartment"
  floor = 5
  facing = "East"
  amenities_score = 75
  distance_to_center = 8
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/api/v1/predict -Method Post `
  -ContentType "application/json" -Body $body
```

When `API_KEY` is non-empty, prediction and batch endpoints require an `X-API-Key` header. The Streamlit UI reads the same `API_KEY` environment variable.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/api/v1/health` | Liveness and model-loaded status |
| POST | `/api/v1/predict` | Predict one property's price |
| POST | `/api/v1/batch` | Predict up to 1,000 properties |
| GET | `/api/v1/model` | Active version, data hash, and validation metrics |
| GET | `/api/v1/dashboard` | Prediction totals, location summary, and recent history |
| GET | `/api/v1/metrics` | JSON service/model metrics |
| GET | `/metrics` | Prometheus exposition format |

The prediction range uses the selected model's 90th-percentile absolute error on its held-out validation split. It is an empirical error range, not a guarantee or a formal prediction interval.

## Train, test, and retrain

```powershell
python -m app.data.generate_data
python -m app.models.pipeline
python -m pytest -q
```

Training requires at least 10 rows and a `price` target. It uses a reproducible 80/20 train/validation split, compares all available candidate models, and registers the best validation R². If XGBoost is unavailable, the Random Forest and neural-network candidates can still be compared.

To train with your own CSV, include the required feature columns shown in `app/data/ingest.py` plus the `price` target, then set `DATA_SOURCE_PATH` to its path before training. The ingestion step validates required columns, missing/non-numeric values, and basic ranges. Use only a licensed, appropriately anonymized dataset; predictions and performance are only as trustworthy as that source.

The weekly GitHub Actions workflow trains on the included synthetic data and uploads model artifacts to the workflow run. It does **not** deploy a model automatically. For production, connect an approved licensed data source, define quality/accuracy release gates, and add an explicit deployment stage for your hosting platform.

## Tests

```powershell
python -m pytest -q
```

The tests cover model training and version metadata, API predictions, input validation, optional API-key protection, analytics persistence, and a 100-item batch latency smoke check.

## Project layout

```text
app/
  api/          FastAPI routes and SQLite prediction history
  data/         Synthetic dataset generator
  frontend/     Streamlit prediction and dashboard UI
  models/       Training, model comparison, and local model registry
monitoring/     Prometheus alert rules and Grafana provisioning/dashboard
tests/          Focused unit and API tests
.github/        CI and scheduled model refresh workflow
```
