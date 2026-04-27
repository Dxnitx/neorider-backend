from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class SensorReading(BaseModel):
    """Model for sensor reading data."""
    ride_id: str
    device: Literal["helmet", "chest"]
    timestamp: str  # ISO 8601 string
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    pitch: float
    roll: float
    yaw: float

class SensorReadingResponse(SensorReading):
    """Response model for sensor reading with ID."""
    id: str