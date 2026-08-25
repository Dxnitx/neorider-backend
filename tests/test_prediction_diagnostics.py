import logging

import pytest

from app.models.sensor import SensorReading
from app.utils.prediction_diagnostics import (
    build_combined_window,
    log_window_diagnostics,
    vector_stats,
)


def _reading(device: str, accel_x: float) -> SensorReading:
    return SensorReading(
        ride_id="diagnostic-test",
        device=device,
        timestamp="2026-01-01T00:00:00Z",
        accel_x=accel_x,
        accel_y=0,
        accel_z=0,
        gyro_x=accel_x,
        gyro_y=0,
        gyro_z=0,
        pitch=accel_x,
        roll=accel_x,
        yaw=accel_x,
    )


def test_model_input_uses_helmet_only():
    combined = build_combined_window([_reading("helmet", 8)], [_reading("chest", -8)])
    assert combined[0]["accel_x"] == 8
    assert combined[0]["gyro_x"] == 8


def test_diagnostics_does_not_average_opposing_signals(caplog):
    helmet = [_reading("helmet", value) for value in ([8, -8] * 10)]
    chest = [_reading("chest", -value) for value in ([8, -8] * 10)]
    with caplog.at_level(logging.WARNING, logger="neorider.ml_diagnostics"):
        result = log_window_diagnostics(
            [reading for pair in zip(helmet, chest) for reading in pair]
        )
    assert result["combined"]["accel_magnitude"]["max"] == 8


def test_vector_stats_reports_invalid_values():
    stats = vector_stats([1.0, float("nan"), float("inf"), -1.0])
    assert stats["count"] == 4
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1
    assert stats["mean"] == pytest.approx(0.0)
