from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.models.sensor import SensorReading
from app.services.live_prediction_service import LivePredictionService


def _reading(device: str, sequence: int) -> SensorReading:
    return SensorReading(
        ride_id="dual-test", device=device, timestamp=datetime.now(timezone.utc),
        accel_x=sequence, accel_y=sequence + 1, accel_z=sequence + 2,
        gyro_x=sequence + 3, gyro_y=sequence + 4, gyro_z=sequence + 5,
        pitch=999, roll=999, yaw=999,
    )


def _result(label: str, probabilities: dict[str, float]) -> dict:
    return {
        "safety_class": label, "prediction": "high_risk", "model_used": "test",
        "probabilities": probabilities, "feature_count": 112,
        "sample_rate_hz": 20,
    }


def test_live_inference_uses_two_independent_windows_and_severity_fusion():
    db = MagicMock()
    service = LivePredictionService(db)
    service.clear_buffer("dual-test")
    service.sensor_service.create_reading = MagicMock()

    helmet_result = _result("RISK", {"SAFE": 0.1, "RISK": 0.8, "ACCIDENT": 0.1})
    chest_result = _result("ACCIDENT", {"SAFE": 0.1, "RISK": 0.2, "ACCIDENT": 0.7})
    with patch(
        "app.services.live_prediction_service.predict_safety",
        side_effect=[helmet_result, chest_result, helmet_result, chest_result],
    ) as predict:
        for sequence in range(20):
            service.process_reading(_reading("helmet", sequence))
            response = service.process_reading(_reading("chest", sequence))

        service.process_reading(_reading("helmet", 20))
        after_one_pair = service.process_reading(_reading("chest", 20))
        state_during_collecting = service._final_states["dual-test"]
        streak_during_collecting = service._accident_streaks["dual-test"]
        service.process_reading(_reading("helmet", 21))
        after_two_pairs = service.process_reading(_reading("chest", 21))

    assert [item.device for item in predict.call_args_list[0].args[0]] == ["helmet"] * 20
    assert [item.device for item in predict.call_args_list[1].args[0]] == ["chest"] * 20
    assert response.helmet_prediction == "RISK"
    assert response.chest_prediction == "ACCIDENT"
    assert response.fused_prediction == "ACCIDENT"
    assert response.fused_confidence == 0.7
    assert response.final_state == "ACCIDENT_PENDING"
    assert after_one_pair.status == "collecting"
    assert after_one_pair.final_state is None
    assert state_during_collecting == "ACCIDENT_PENDING"
    assert streak_during_collecting == 1
    assert after_two_pairs.status == "predicted"
    assert after_two_pairs.prediction_stride == 2
    assert after_two_pairs.helmet_buffer_size == 20
    assert after_two_pairs.chest_buffer_size == 20
    assert len(predict.call_args_list) == 4
    assert predict.call_args_list[2].args[0][0].accel_x == 2
    assert predict.call_args_list[2].args[0][-1].accel_x == 21


def test_accident_requires_three_qualifying_windows_and_safe_resets_streak():
    service = LivePredictionService(MagicMock())
    service.clear_buffer("confirmation-test")

    first = service._classify_runtime("confirmation-test", "ACCIDENT", 0.9, True)
    reset = service._classify_runtime("confirmation-test", "SAFE", 0.99, True)
    second_first = service._classify_runtime("confirmation-test", "ACCIDENT", 0.9, True)
    second = service._classify_runtime("confirmation-test", "ACCIDENT", 0.9, True)
    third = service._classify_runtime("confirmation-test", "ACCIDENT", 0.9, True)

    assert first[:2] == ("ACCIDENT_PENDING", 1)
    assert reset[:2] == ("SAFE", 0)
    assert second_first[:2] == ("ACCIDENT_PENDING", 1)
    assert second[:2] == ("ACCIDENT_PENDING", 2)
    assert third[:2] == ("ACCIDENT_CONFIRMED", 3)


def test_accident_without_impact_never_confirms():
    service = LivePredictionService(MagicMock())
    service.clear_buffer("no-impact-test")

    states = [
        service._classify_runtime("no-impact-test", "ACCIDENT", 0.95, False)
        for _ in range(3)
    ]

    assert all(state[0] == "ACCIDENT_PENDING" for state in states)
    assert states[-1][1] == 0


def test_stationary_risk_and_accident_are_safe_and_reset_streak():
    service = LivePredictionService(MagicMock())
    service.clear_buffer("stationary-test")
    service._classify_runtime("stationary-test", "ACCIDENT", 0.95, True)

    risk = service._classify_runtime(
        "stationary-test", "RISK", 1.0, False, both_stationary=True
    )
    accident = service._classify_runtime(
        "stationary-test", "ACCIDENT", 1.0, False, both_stationary=True
    )

    assert risk[:2] == ("SAFE", 0)
    assert accident[:2] == ("SAFE", 0)


def test_meaningful_movement_preserves_risk_state():
    service = LivePredictionService(MagicMock())
    service.clear_buffer("moving-risk-test")
    assert service._classify_runtime(
        "moving-risk-test", "RISK", 0.9, False, both_stationary=False
    )[0] == "RISK"


def test_synchronized_low_motion_does_not_pass_impact_gate():
    helmet = [_reading("helmet", 0).model_copy(update={
        "accel_x": 0.1, "accel_y": 0.1, "accel_z": 9.81,
        "gyro_x": 0.1, "gyro_y": 0.1, "gyro_z": 0.1,
    }) for _ in range(20)]
    chest = [item.model_copy(update={"device": "chest"}) for item in helmet]

    helmet_metrics = LivePredictionService._motion_metrics(helmet)
    chest_metrics = LivePredictionService._motion_metrics(chest)

    assert helmet_metrics["accel_delta"] == 0
    assert chest_metrics["accel_delta"] == 0
    assert helmet_metrics["max_gyro"] < 3.5
    assert LivePredictionService._is_stationary(helmet_metrics)
    assert LivePredictionService._is_stationary(chest_metrics)
