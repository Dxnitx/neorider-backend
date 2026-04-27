from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.telemetry import Telemetry
from app.models.alert import Alert
from app.services.ml_service import predict_behavior

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

telemetry_buffer = defaultdict(list)
WINDOW_SIZE = 20


@router.post("/{ride_id}")
def add_telemetry(
    ride_id: int,
    accel_x: float,
    accel_y: float,
    accel_z: float,
    gyro_x: float,
    gyro_y: float,
    gyro_z: float,
    pitch: float,
    roll: float,
    yaw: float,
    chest_tilt: float,
    db: Session = Depends(get_db)
):
    # Save raw telemetry to DB
    telemetry = Telemetry(
        ride_id=ride_id,
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        pitch=pitch,
        roll=roll,
        yaw=yaw,
        chest_tilt=chest_tilt
    )

    db.add(telemetry)
    db.flush()

    # Add current point to in-memory buffer
    point = {
        "acc_x": accel_x,
        "acc_y": accel_y,
        "acc_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
    }

    telemetry_buffer[ride_id].append(point)

    # Keep only latest WINDOW_SIZE points
    if len(telemetry_buffer[ride_id]) > WINDOW_SIZE:
        telemetry_buffer[ride_id] = telemetry_buffer[ride_id][-WINDOW_SIZE:]

    prediction = None
    alert = None

    # Predict only when enough points are collected
    if len(telemetry_buffer[ride_id]) == WINDOW_SIZE:
        prediction = predict_behavior(telemetry_buffer[ride_id])

        if prediction == "possible_accident":
            alert = Alert(
                ride_id=ride_id,
                alert_type="possible_accident",
                severity="critical",
                message="Possible accident detected!"
            )
        elif prediction == "unstable_movement":
            alert = Alert(
                ride_id=ride_id,
                alert_type="unstable_movement",
                severity="medium",
                message="Unstable movement detected"
            )
        elif prediction == "sudden_motion":
            alert = Alert(
                ride_id=ride_id,
                alert_type="sudden_motion",
                severity="high",
                message="Sudden motion detected"
            )

        if alert:
            db.add(alert)

    db.commit()

    return {
        "message": "Telemetry added successfully",
        "buffer_size": len(telemetry_buffer[ride_id]),
        "window_size": WINDOW_SIZE,
        "prediction": prediction,
    }