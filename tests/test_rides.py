import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.ride import RideCreate

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_firebase():
    with patch('app.firebase.get_firestore_client') as mock_db, \
         patch('app.routers.rides.RideService') as mock_service_class:
        mock_db_instance = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_class.return_value = mock_service_instance
        yield mock_service_instance

def test_create_ride(mock_firebase):
    """Test POST /rides."""
    ride = RideCreate(rider_name="Test Rider")
    
    mock_firebase.create_ride.return_value = {
        "id": "test_ride_id",
        "rider_name": "Test Rider",
        "start_time": "2023-01-01T00:00:00Z",
        "end_time": None,
        "status": "active",
        "total_readings": 0,
        "summary": {
            "safe_count": 0,
            "risky_count": 0,
            "accident_alerts": 0
        }
    }
    
    response = client.post("/rides", json=ride.dict())
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["rider_name"] == "Test Rider"
    assert data["status"] == "active"

def test_end_ride(mock_firebase):
    """Test PATCH /rides/{ride_id}/end."""
    # Mock create first
    mock_firebase.create_ride.return_value = {
        "id": "test_ride_id",
        "rider_name": "Test Rider",
        "start_time": "2023-01-01T00:00:00Z",
        "end_time": None,
        "status": "active",
        "total_readings": 0,
        "summary": {
            "safe_count": 0,
            "risky_count": 0,
            "accident_alerts": 0
        }
    }
    
    # Create ride
    ride = RideCreate(rider_name="Test Rider")
    create_response = client.post("/rides", json=ride.dict())
    assert create_response.status_code == 201
    ride_data = create_response.json()
    ride_id = ride_data["id"]
    
    # Mock end
    mock_firebase.end_ride.return_value = {
        "id": ride_id,
        "rider_name": "Test Rider",
        "start_time": "2023-01-01T00:00:00Z",
        "end_time": "2023-01-01T01:00:00Z",
        "status": "completed",
        "total_readings": 0,
        "summary": {
            "safe_count": 0,
            "risky_count": 0,
            "accident_alerts": 0
        }
    }
    
    # End ride
    end_response = client.patch(f"/rides/{ride_id}/end")
    assert end_response.status_code == 200
    end_data = end_response.json()
    assert end_data["status"] == "completed"
    assert end_data["end_time"] is not None

def test_health_check():
    """Test GET /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data