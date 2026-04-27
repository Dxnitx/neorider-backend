from fastapi import APIRouter
from pydantic import BaseModel
from app.firebase.firebase_config import db
from datetime import datetime

router = APIRouter()

class TelemetryData(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    pitch: float
    roll: float
    yaw: float
    chest_tilt: float

@router.post("/telemetry/{ride_id}")
def add_telemetry(ride_id: int, data: TelemetryData):
    telemetry = {
        "ride_id": ride_id,
        "accel_x": data.accel_x,
        "accel_y": data.accel_y,
        "accel_z": data.accel_z,
        "gyro_x": data.gyro_x,
        "gyro_y": data.gyro_y,
        "gyro_z": data.gyro_z,
        "pitch": data.pitch,
        "roll": data.roll,
        "yaw": data.yaw,
        "chest_tilt": data.chest_tilt,
        "timestamp": datetime.utcnow().isoformat()
    }

    db.collection("telemetry").add(telemetry)

    return {
        "message": "Telemetry saved to Firebase",
        "ride_id": ride_id
    }