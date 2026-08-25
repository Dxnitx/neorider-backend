# app/routers/sensor.py

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from typing import List

from app.models.sensor import SensorReading
from app.models.live_prediction import LivePredictionResponse

from app.services.sensor_service import SensorService
from app.services.live_prediction_service import LivePredictionService

from app.firebase import get_firestore_client


router = APIRouter()


@router.post("/reading", status_code=201)
async def create_sensor_reading(
    reading: SensorReading,
    db=Depends(get_firestore_client),
):
    """
    Save one sensor reading without running prediction.
    """

    service = SensorService(db)

    try:
        result = service.create_reading(reading)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save sensor reading: {str(e)}",
        )


@router.post(
    "/live",
    response_model=LivePredictionResponse,
)
async def process_live_sensor_reading(
    reading: SensorReading,
    background_tasks: BackgroundTasks,
    db=Depends(get_firestore_client),
):
    """
    Receive one live reading.

    The backend collects the latest 20 readings
    and automatically runs a prediction.
    """

    service = LivePredictionService(db)

    try:
        return service.process_reading(
            reading,
            defer_persistence=background_tasks.add_task,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process live reading: {str(e)}",
        )


@router.get("/live/buffer/{ride_id}")
async def get_live_buffer_status(
    ride_id: str,
    db=Depends(get_firestore_client),
):
    """
    Check Helmet + Chest live buffer status.
    """

    service = LivePredictionService(db)

    try:
        status = service.get_buffer_status(ride_id)

        return {
            "ride_id": ride_id,
            "helmet_buffer_size": status["helmet"],
            "chest_buffer_size": status["chest"],
            "paired_buffer_size": status["paired"],
            "required_readings_per_device": 20,
            "ready_for_prediction": (
                status["helmet"] >= 20 and status["chest"] >= 20
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve buffer status: {str(e)}",
        )


@router.delete("/live/buffer/{ride_id}")
async def clear_live_buffer(
    ride_id: str,
    db=Depends(get_firestore_client),
):
    """
    Clear the live buffer for a ride.
    """

    service = LivePredictionService(db)

    try:
        cleared = service.clear_buffer(ride_id)

        return {
            "ride_id": ride_id,
            "buffer_cleared": cleared,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear live buffer: {str(e)}",
        )


@router.get("/readings/{ride_id}")
async def get_sensor_readings(
    ride_id: str,
    db=Depends(get_firestore_client),
) -> List[SensorReading]:
    """
    Get all sensor readings for a ride.
    """

    service = SensorService(db)

    try:
        readings = service.get_readings_by_ride(ride_id)
        return readings

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sensor readings: {str(e)}",
        )
