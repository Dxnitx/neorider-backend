from datetime import datetime, timezone

import pytest

from app.models.sensor import SensorReading
from app.services.mmc1_model_service import model_status, predict_safety


def _window():
    return [SensorReading(
        ride_id="test", device="helmet", timestamp=datetime.now(timezone.utc),
        accel_x=i * 0.01, accel_y=i * 0.02, accel_z=9.81,
        gyro_x=i * 0.1, gyro_y=i * 0.05, gyro_z=i * 0.02,
        pitch=0, roll=0, yaw=0,
    ) for i in range(20)]


def test_mmc1_artifacts_and_inference():
    status = model_status()
    assert status["features"] == status["model_features"] == status["scaler_features"] == 112
    result = predict_safety(_window())
    assert result["safety_class"] in {"SAFE", "RISK", "ACCIDENT"}
    assert set(result["probabilities"]) == {"SAFE", "RISK", "ACCIDENT"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=0.001)
    assert result["max_abs_scaled_feature"] >= 0
    assert len(result["largest_scaled_features"]) == 5
