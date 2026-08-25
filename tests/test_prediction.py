import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.prediction import PredictionRequest
from app.models.sensor import SensorReading

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_firebase():
    with patch('app.firebase.get_firestore_client') as mock_db, \
         patch('app.routers.prediction.PredictionService') as mock_service_class:
        mock_db_instance = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_class.return_value = mock_service_instance
        yield mock_service_instance

def test_create_prediction_valid(mock_firebase):
    """Test POST /predict with a valid 20-step dual-IMU window."""
    helmet_readings = [
        SensorReading(
            ride_id="test_ride_1",
            device="helmet",
            timestamp=f"2023-01-01T00:00:{i:02d}Z",
            accel_x=1.0 + i * 0.1,
            accel_y=2.0 + i * 0.1,
            accel_z=3.0 + i * 0.1,
            gyro_x=0.1 + i * 0.01,
            gyro_y=0.2 + i * 0.01,
            gyro_z=0.3 + i * 0.01,
            pitch=10.0 + i * 0.5,
            roll=20.0 + i * 0.5,
            yaw=30.0 + i * 0.5
        ) for i in range(20)
    ]
    chest_readings = [
        reading.model_copy(update={"device": "chest"})
        for reading in helmet_readings
    ]
    sensor_readings = [
        reading
        for pair in zip(helmet_readings, chest_readings)
        for reading in pair
    ]
    
    request = PredictionRequest(
        ride_id="test_ride_1",
        sensor_window=sensor_readings
    )
    
    mock_firebase.create_prediction.return_value = {
        "ride_id": "test_ride_1",
        "prediction": "safe_riding",
        "confidence": 0.91,
        "timestamp": "2023-01-01T00:00:00Z",
        "model_used": "LightGBM_tuned_real_dataset"
    }
    
    response = client.post("/predict", json=request.model_dump(mode="json"))
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "safe_riding"
    assert data["confidence"] == 0.91
    assert data["model_used"] == "LightGBM_tuned_real_dataset"

def test_create_prediction_invalid_window_size():
    """Test POST /predict with fewer than 20 dual-IMU time steps."""
    sensor_readings = [
        SensorReading(
            ride_id="test_ride_1",
            device="helmet",
            timestamp="2023-01-01T00:00:00Z",
            accel_x=1.0,
            accel_y=2.0,
            accel_z=3.0,
            gyro_x=0.1,
            gyro_y=0.2,
            gyro_z=0.3,
            pitch=10.0,
            roll=20.0,
            yaw=30.0
        ) for _ in range(10)  # Only 10 readings
    ]
    
    payload = {
        "ride_id": "test_ride_1",
        "sensor_window": [
            reading.model_dump(mode="json") for reading in sensor_readings
        ],
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
