# NeoRider Backend

An IoT-based intelligent motorcycle rider safety and training system backend built with FastAPI and Firebase.

## Overview

NeoRider collects real-time sensor data from smart helmets and chest-mounted IMUs to analyze rider behavior and provide safety insights. This backend API handles sensor data ingestion, ride session management, and ML-based behavior classification.

## Prerequisites

- Python 3.11+
- Firebase project with Firestore enabled
- Service account key JSON file from Firebase

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd neorider-backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update `FIREBASE_CREDENTIALS_PATH` to point to your Firebase service account JSON file
   - Adjust other settings as needed

5. Ensure your Firebase service account has Firestore access.

## Running the Server

Start the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/sensor/reading` | Submit sensor reading |
| GET | `/sensor/readings/{ride_id}` | Get sensor readings for a ride |
| POST | `/predict` | Generate behavior prediction |
| GET | `/predict/history/{ride_id}` | Get prediction history for a ride |
| POST | `/rides` | Create new ride session |
| GET | `/rides/{ride_id}` | Get ride details |
| PATCH | `/rides/{ride_id}/end` | End ride session |
| GET | `/rides` | List all rides |

## ML Integration

The current implementation includes a placeholder ML function that returns hardcoded predictions. In Phase 3, this will be replaced with a trained XGBoost model.

The feature extractor computes 27 statistical features (mean, std, max) from 9 sensor channels over a 20-reading window, ready for model input.

## Example Requests

### Submit Sensor Reading
```bash
curl -X POST "http://localhost:8000/sensor/reading" \
  -H "Content-Type: application/json" \
  -d '{
    "ride_id": "ride_123",
    "device": "helmet",
    "timestamp": "2023-01-01T12:00:00Z",
    "accel_x": 1.2,
    "accel_y": -0.5,
    "accel_z": 9.8,
    "gyro_x": 0.1,
    "gyro_y": -0.2,
    "gyro_z": 0.05,
    "pitch": 15.0,
    "roll": 5.0,
    "yaw": 90.0
  }'
```

### Generate Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ride_id": "ride_123",
    "sensor_window": [
      {
        "ride_id": "ride_123",
        "device": "helmet",
        "timestamp": "2023-01-01T12:00:00Z",
        "accel_x": 1.2,
        "accel_y": -0.5,
        "accel_z": 9.8,
        "gyro_x": 0.1,
        "gyro_y": -0.2,
        "gyro_z": 0.05,
        "pitch": 15.0,
        "roll": 5.0,
        "yaw": 90.0
      },
      // ... 19 more readings
    ]
  }'
```

## Testing

Run tests with pytest:
```bash
pytest
```

## Project Structure

```
neorider-backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── firebase.py          # Firebase setup
│   ├── routers/             # API endpoints
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   └── utils/               # Utilities
├── tests/                   # Test files
├── .env.example             # Environment template
├── requirements.txt         # Dependencies
└── README.md
```