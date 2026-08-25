from pydantic import BaseModel
from datetime import datetime
from typing import Literal


class SensorReading(BaseModel):
    ride_id: str
    device: Literal["helmet", "chest"]

    # Firestore timestamp
    timestamp: datetime

    accel_x: float
    accel_y: float
    accel_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0

    pitch: float
    roll: float
    yaw: float


class SensorReadingResponse(SensorReading):
    id: str
