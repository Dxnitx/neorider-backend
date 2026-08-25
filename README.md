# NeoRider Backend

FastAPI backend for NeoRider's helmet and chest IMU safety pipeline. The application performs inference with the checked-in MMC1 artifacts; it does not retrain the model at runtime.

## Requirements

- Python 3.11 (recommended; use a compatible 3.11+ release)
- Firebase project with Firestore enabled
- Firebase service-account JSON stored outside version control
- Packages in `requirements.txt`, including FastAPI, Uvicorn, NumPy, pandas, SciPy, scikit-learn, LightGBM, Firebase Admin, Pydantic, and joblib
- XGBoost and pyserial support the existing offline training comparison script and sensor recording utility

## Setup

From the repository root on Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `FIREBASE_CREDENTIALS_PATH` in `.env` to a Firebase service-account JSON file. Never commit that file or `.env`.

## Run

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Interactive API documentation is at `http://localhost:8002/docs`.

## Important Endpoints

- `POST /sensor/live` — ingest a helmet or chest reading and run rolling dual-device inference when ready
- `POST /predict` — run manual prediction for a complete sensor window
- `GET /ml/model-status` — validate the active model and artifact metadata
- `/docs` — OpenAPI/Swagger UI

## ML Architecture

```text
Helmet IMU
  -> 20-sample window
  -> 112 features
  -> scaler
  -> LightGBM

Chest IMU
  -> 20-sample window
  -> 112 features
  -> scaler
  -> LightGBM

Helmet + Chest predictions
  -> fusion
  -> final safety state
```

Helmet and chest streams are buffered and inferred independently. Rolling inference uses a stride of 2 paired samples. The application then fuses their predictions and applies the existing stationary, impact, confidence, and consecutive-window confirmation rules.

The required active artifacts are:

- `app/ml/mmc1/model.pkl`
- `app/ml/mmc1/scaler.pkl`
- `app/ml/mmc1/feature_columns.pkl`
- `app/ml/mmc1/class_mapping.pkl`
- `app/ml/mmc1/class_mapping.json`

Keep these files together and do not change their feature order independently.

## Safety States

- `SAFE` — no current elevated safety condition
- `RISK` — risky riding behavior detected
- `ACCIDENT_PENDING` — an accident prediction awaits the required gates and/or consecutive qualifying windows
- `ACCIDENT_CONFIRMED` — accident confirmation rules have been satisfied
- `UNKNOWN` — no recognized final state is available

`ACCIDENT_CONFIRMED` is the state expected to trigger physical safety feedback in the Flutter/ESP32 flow.

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `FIREBASE_CREDENTIALS_PATH` | Yes | Firebase service-account JSON path; relative paths resolve from the process working directory |
| `APP_ENV` | No | Environment label; defaults to `development` |
| `APP_VERSION` | No | Application version label; defaults to `1.0.0` |
| `TESTING` | Tests only | Set to `true` to skip Firebase initialization during application lifespan tests |
| `PORT` | Deployment | Port supplied by the hosting platform to the Uvicorn start command |
| `NEORIDER_BASE_URL` | Utility only | Overrides the local URL used by `test_live_window.py` |

Use `.env.example` as the template. Store production values in the hosting platform's secret/environment configuration.

## Testing

```powershell
$env:TESTING = "true"
python -m pytest -q
```

## Deployment

Install `requirements.txt`, provide the Firebase credential securely, set `FIREBASE_CREDENTIALS_PATH`, and use the platform's dynamic port:

```sh
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

No platform-specific Python launcher is required. Run from the repository root so `.env` and any relative Firebase credential path resolve consistently.
