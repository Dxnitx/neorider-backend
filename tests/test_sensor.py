import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.sensor import SensorReading

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_firebase():
    with patch('app.firebase.get_firestore_client') as mock_db, \
         patch('app.routers.sensor.SensorService') as mock_service_class:
        mock_db_instance = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_class.return_value = mock_service_instance
        yield mock_service_instance

def test_create_sensor_reading_valid(mock_firebase):
    """Test POST /sensor/reading with valid data."""
    reading = SensorReading(
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
    )
    
    mock_firebase.create_reading.return_value = {
        "id": "test_id",
        **reading.model_dump(mode="json")
    }
    
    response = client.post("/sensor/reading", json=reading.model_dump(mode="json"))
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["ride_id"] == "test_ride_1"

def test_create_sensor_reading_missing_fields():
    """Test POST /sensor/reading with missing fields."""
    invalid_data = {
        "ride_id": "test_ride_1",
        "device": "helmet",
        # Missing other fields
    }
    
    response = client.post("/sensor/reading", json=invalid_data)
    assert response.status_code == 422
