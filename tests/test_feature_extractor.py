import pytest

from app.models.sensor import SensorReading
from app.utils.feature_extractor import extract_features


def test_extract_features_matches_lightgbm_training_shape():
    helmet_readings = [
        SensorReading(
            ride_id="test_ride_1",
            device="helmet",
            timestamp=f"2023-01-01T00:00:{i:02d}Z",
            accel_x=1.0 + i,
            accel_y=2.0 + i,
            accel_z=3.0 + i,
            gyro_x=0.1 + i,
            gyro_y=0.2 + i,
            gyro_z=0.3 + i,
            pitch=10.0 + i,
            roll=20.0 + i,
            yaw=30.0 + i,
        )
        for i in range(20)
    ]
    features = extract_features(helmet_readings)

    assert len(features) == 112
    assert features["ax_mean"] == 10.5
    assert features["ax_range"] == 19.0
    assert features["rx_mean"] == pytest.approx(9.6)
    assert features["ax_energy"] == sum((1.0 + i) ** 2 for i in range(20))
    assert "acc_jerk_energy" in features


def test_extract_features_processes_chest_independently_and_rejects_mixing():
    helmet = [
        SensorReading(
            ride_id="test", device="helmet", timestamp=f"2023-01-01T00:00:{i:02d}Z",
            accel_x=i, accel_y=i + 1, accel_z=i + 2,
            gyro_x=i + 3, gyro_y=i + 4, gyro_z=i + 5,
            pitch=999, roll=999, yaw=999,
        )
        for i in range(20)
    ]
    chest = [reading.model_copy(update={"device": "chest", "accel_x": -reading.accel_x}) for reading in helmet]

    assert len(extract_features(chest)) == 112
    assert extract_features(chest)["ax_mean"] == -extract_features(helmet)["ax_mean"]
    with pytest.raises(ValueError, match="exactly 20|cannot mix"):
        extract_features(helmet + chest)
