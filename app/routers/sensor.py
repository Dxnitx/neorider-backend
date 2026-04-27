from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.sensor import SensorReading
from app.services.sensor_service import SensorService
from app.firebase import get_firestore_client

router = APIRouter()

@router.post("/reading", status_code=201)
async def create_sensor_reading(
    reading: SensorReading,
    db=Depends(get_firestore_client)
):
    """Create a new sensor reading."""
    service = SensorService(db)
    try:
        result = service.create_reading(reading)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save sensor reading: {str(e)}")

@router.get("/readings/{ride_id}")
async def get_sensor_readings(
    ride_id: str,
    db=Depends(get_firestore_client)
) -> List[SensorReading]:
    """Get all sensor readings for a ride."""
    service = SensorService(db)
    try:
        readings = service.get_readings_by_ride(ride_id)
        return readings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sensor readings: {str(e)}")