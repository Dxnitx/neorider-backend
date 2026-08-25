from collections import defaultdict, deque
from datetime import datetime, timezone
import logging
import math
import time
from threading import Lock
from typing import Deque, Dict, Tuple

from firebase_admin import firestore

from app.models.sensor import SensorReading
from app.models.prediction import PredictionResponse
from app.models.live_prediction import LivePredictionResponse
from app.services.mmc1_model_service import predict_safety
from app.services.sensor_service import SensorService


# Number of synchronized time steps required per device.
WINDOW_SIZE = 20
PREDICTION_STRIDE = 2
SEVERITY = {"SAFE": 0, "RISK": 1, "ACCIDENT": 2}
# Live/training acceleration is in g and gyroscope data is in degrees/second.
# Initial prototype tuning values; they must
# be calibrated against labelled NeoRider hardware recordings before deployment.
ACCIDENT_MIN_CONFIDENCE = 0.70
ACCIDENT_CONFIRM_WINDOWS = 3
IMPACT_ACCEL_DELTA_THRESHOLD = 0.60
IMPACT_GYRO_THRESHOLD = 200.0
STATIONARY_ACCEL_STD_THRESHOLD = 0.03
STATIONARY_ACCEL_RANGE_THRESHOLD = 0.10
STATIONARY_GYRO_MEAN_THRESHOLD = 25.0
logger = logging.getLogger("neorider.live_prediction")


class LivePredictionService:
    """Buffer helmet and chest readings independently for each ride."""

    _buffers: Dict[str, Dict[str, Deque[SensorReading]]] = defaultdict(
        lambda: {
            "helmet": deque(maxlen=WINDOW_SIZE),
            "chest": deque(maxlen=WINDOW_SIZE),
        }
    )
    _reading_counters: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"helmet": 0, "chest": 0}
    )
    _last_prediction_counts: Dict[str, Tuple[int, int]] = {}
    _last_counted_pair_total: Dict[str, int] = {}
    _new_pairs_since_prediction: Dict[str, int] = defaultdict(int)
    _accident_streaks: Dict[str, int] = defaultdict(int)
    _final_states: Dict[str, str] = {}
    _lock = Lock()

    def __init__(self, db: firestore.Client):
        self.db = db
        self.sensor_service = SensorService(db)

    def process_reading(self, reading: SensorReading, defer_persistence=None) -> LivePredictionResponse:
        """Save one reading and predict once both devices have a full window."""
        try:
            self._persist(
                "sensor_reading", self.sensor_service.create_reading, reading,
                defer_persistence=defer_persistence,
            )

            with self._lock:
                ride_id = reading.ride_id
                device = reading.device
                ride_buffers = self._buffers[ride_id]
                ride_buffers[device].append(reading)
                self._reading_counters[ride_id][device] += 1

                helmet_size = len(ride_buffers["helmet"])
                chest_size = len(ride_buffers["chest"])
                paired_size = min(helmet_size, chest_size)
                helmet_total = self._reading_counters[ride_id]["helmet"]
                chest_total = self._reading_counters[ride_id]["chest"]
                paired_total = min(helmet_total, chest_total)
                last_counted = self._last_counted_pair_total.get(ride_id, 0)
                if paired_total > last_counted:
                    self._new_pairs_since_prediction[ride_id] += paired_total - last_counted
                    self._last_counted_pair_total[ride_id] = paired_total

                if helmet_size < WINDOW_SIZE or chest_size < WINDOW_SIZE:
                    return LivePredictionResponse(
                        ride_id=ride_id,
                        status="collecting",
                        buffer_size=paired_size,
                        helmet_buffer_size=helmet_size,
                        chest_buffer_size=chest_size,
                        required_readings=WINDOW_SIZE,
                        paired_buffer_size=paired_size,
                        prediction_stride=PREDICTION_STRIDE,
                        prediction=None,
                    )

                last_helmet, last_chest = self._last_prediction_counts.get(
                    ride_id, (0, 0)
                )
                first_prediction = ride_id not in self._last_prediction_counts
                enough_new_pairs = (
                    first_prediction
                    or self._new_pairs_since_prediction[ride_id] >= PREDICTION_STRIDE
                )
                if (
                    helmet_total <= last_helmet
                    or chest_total <= last_chest
                    or not enough_new_pairs
                ):
                    return LivePredictionResponse(
                        ride_id=ride_id,
                        status="collecting",
                        buffer_size=paired_size,
                        helmet_buffer_size=helmet_size,
                        chest_buffer_size=chest_size,
                        required_readings=WINDOW_SIZE,
                        paired_buffer_size=paired_size,
                        prediction_stride=PREDICTION_STRIDE,
                        prediction=None,
                    )

                helmet_window = list(ride_buffers["helmet"])[-WINDOW_SIZE:]
                chest_window = list(ride_buffers["chest"])[-WINDOW_SIZE:]
                previous_counts = (last_helmet, last_chest)
                current_counts = (helmet_total, chest_total)
                self._last_prediction_counts[ride_id] = current_counts
                previous_new_pairs = self._new_pairs_since_prediction[ride_id]
                self._new_pairs_since_prediction[ride_id] = 0

            try:
                prediction_started = time.perf_counter()
                helmet_result = predict_safety(helmet_window)
                chest_result = predict_safety(chest_window)
                fused_result = max(
                    (helmet_result, chest_result),
                    key=lambda result: SEVERITY[result["safety_class"]],
                )
                fused_prediction = fused_result["safety_class"]
                fused_confidence = fused_result["probabilities"][fused_prediction]
                motion_metrics = {
                    "helmet": self._motion_metrics(helmet_window),
                    "chest": self._motion_metrics(chest_window),
                }
                impact_gate_passed = any(
                    metrics["accel_delta"] >= IMPACT_ACCEL_DELTA_THRESHOLD
                    or metrics["max_gyro"] >= IMPACT_GYRO_THRESHOLD
                    for metrics in motion_metrics.values()
                )
                is_helmet_stationary = self._is_stationary(motion_metrics["helmet"])
                is_chest_stationary = self._is_stationary(motion_metrics["chest"])
                both_stationary = is_helmet_stationary and is_chest_stationary
                if both_stationary:
                    impact_gate_passed = False
                logger.info(
                    "[MOTION_GATE] helmet_acc_std=%.4f helmet_acc_range=%.4f "
                    "helmet_gyro_mean=%.3f chest_acc_std=%.4f chest_acc_range=%.4f "
                    "chest_gyro_mean=%.3f helmet_stationary=%s chest_stationary=%s "
                    "both_stationary=%s",
                    motion_metrics["helmet"]["accel_std"],
                    motion_metrics["helmet"]["accel_delta"],
                    motion_metrics["helmet"]["gyro_mean"],
                    motion_metrics["chest"]["accel_std"],
                    motion_metrics["chest"]["accel_delta"],
                    motion_metrics["chest"]["gyro_mean"],
                    is_helmet_stationary, is_chest_stationary, both_stationary,
                )
                logger.info(
                    "[IMPACT_GATE] helmet_acc_delta=%.3f helmet_max_gyro=%.3f "
                    "chest_acc_delta=%.3f chest_max_gyro=%.3f passed=%s",
                    motion_metrics["helmet"]["accel_delta"],
                    motion_metrics["helmet"]["max_gyro"],
                    motion_metrics["chest"]["accel_delta"],
                    motion_metrics["chest"]["max_gyro"], impact_gate_passed,
                )
                final_state, accident_streak, previous_state = self._classify_runtime(
                    ride_id, fused_prediction, fused_confidence,
                    impact_gate_passed, both_stationary,
                )
                logger.info(
                    "[SAFETY] model=%s confidence=%.4f stationary=%s impact_gate=%s "
                    "accident_streak=%d/%d final_state=%s",
                    fused_prediction, fused_confidence, both_stationary,
                    impact_gate_passed,
                    accident_streak, ACCIDENT_CONFIRM_WINDOWS, final_state,
                )
                if previous_state != final_state:
                    logger.info(
                        "[PREDICTION][STATE] ride_id=%s old=%s new=%s",
                        ride_id, previous_state, final_state,
                    )
                legacy_prediction = PredictionResponse(
                    ride_id=ride_id,
                    safety_class=fused_prediction,
                    class_id=SEVERITY[fused_prediction],
                    prediction=fused_result["prediction"],
                    confidence=fused_confidence,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_used=fused_result["model_used"],
                    probabilities=fused_result["probabilities"],
                    window_size=WINDOW_SIZE,
                    sample_rate_hz=fused_result["sample_rate_hz"],
                    feature_count=fused_result["feature_count"],
                )
                persistence_data = {
                    "ride_id": ride_id,
                    "helmet_prediction": helmet_result["safety_class"],
                    "helmet_probabilities": helmet_result["probabilities"],
                    "chest_prediction": chest_result["safety_class"],
                    "chest_probabilities": chest_result["probabilities"],
                    "fused_prediction": fused_prediction,
                    "fused_confidence": fused_confidence,
                    "final_state": final_state,
                    "impact_gate_passed": impact_gate_passed,
                    "is_helmet_stationary": is_helmet_stationary,
                    "is_chest_stationary": is_chest_stationary,
                    "both_stationary": both_stationary,
                    "accident_streak": accident_streak,
                    "motion_metrics": motion_metrics,
                    # Keep legacy history consumers working from the fused result.
                    "safety_class": fused_prediction,
                    "class_id": SEVERITY[fused_prediction],
                    "prediction": fused_result["prediction"],
                    "confidence": fused_confidence,
                    "model_used": fused_result["model_used"],
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
                self._persist(
                    "prediction", self.db.collection("predictions").document().set,
                    persistence_data, defer_persistence=defer_persistence,
                )

                helmet_raw = helmet_window[-1]
                chest_raw = chest_window[-1]
                logger.debug(
                    "Dual MMC1 inference | Helmet raw: ax=%s ay=%s az=%s gyro_x=%s gyro_y=%s gyro_z=%s | "
                    "Chest raw: ax=%s ay=%s az=%s gyro_x=%s gyro_y=%s gyro_z=%s | "
                    "Helmet: prediction=%s SAFE=%s RISK=%s ACCIDENT=%s feature_count=%d | "
                    "Chest: prediction=%s SAFE=%s RISK=%s ACCIDENT=%s feature_count=%d | "
                    "Fused: prediction=%s confidence=%s",
                    helmet_raw.accel_x, helmet_raw.accel_y, helmet_raw.accel_z,
                    helmet_raw.gyro_x, helmet_raw.gyro_y, helmet_raw.gyro_z,
                    chest_raw.accel_x, chest_raw.accel_y, chest_raw.accel_z,
                    chest_raw.gyro_x, chest_raw.gyro_y, chest_raw.gyro_z,
                    helmet_result["safety_class"],
                    helmet_result["probabilities"]["SAFE"],
                    helmet_result["probabilities"]["RISK"],
                    helmet_result["probabilities"]["ACCIDENT"],
                    helmet_result["feature_count"],
                    chest_result["safety_class"],
                    chest_result["probabilities"]["SAFE"],
                    chest_result["probabilities"]["RISK"],
                    chest_result["probabilities"]["ACCIDENT"],
                    chest_result["feature_count"],
                    fused_prediction, fused_confidence,
                )
                timings = {
                    key: helmet_result.get("timings_ms", {}).get(key, 0.0)
                    + chest_result.get("timings_ms", {}).get(key, 0.0)
                    for key in ("feature", "scaler", "inference")
                }
                prediction_latency_ms = (time.perf_counter() - prediction_started) * 1000
                logger.info(
                    "[NeoRider LATENCY] ride_id=%s feature=%.2fms scaler=%.2fms "
                    "inference=%.2fms db=deferred total=%.2fms helmet_buffer=%d "
                    "chest_buffer=%d paired_buffer=%d prediction_stride=%d "
                    "prediction=%s confidence=%.4f",
                    ride_id, timings["feature"], timings["scaler"], timings["inference"],
                    prediction_latency_ms, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE,
                    PREDICTION_STRIDE, fused_prediction, fused_confidence,
                )
            except Exception:
                with self._lock:
                    if self._last_prediction_counts.get(ride_id) == current_counts:
                        self._last_prediction_counts[ride_id] = previous_counts
                        self._new_pairs_since_prediction[ride_id] = previous_new_pairs
                raise

            return LivePredictionResponse(
                ride_id=ride_id,
                status="predicted",
                buffer_size=WINDOW_SIZE,
                helmet_buffer_size=WINDOW_SIZE,
                chest_buffer_size=WINDOW_SIZE,
                required_readings=WINDOW_SIZE,
                paired_buffer_size=WINDOW_SIZE,
                prediction_stride=PREDICTION_STRIDE,
                prediction_latency_ms=round(prediction_latency_ms, 2),
                prediction=legacy_prediction,
                helmet_prediction=helmet_result["safety_class"],
                helmet_probabilities=helmet_result["probabilities"],
                chest_prediction=chest_result["safety_class"],
                chest_probabilities=chest_result["probabilities"],
                fused_prediction=fused_prediction,
                fused_confidence=fused_confidence,
                final_state=final_state,
                impact_gate_passed=impact_gate_passed,
                is_helmet_stationary=is_helmet_stationary,
                is_chest_stationary=is_chest_stationary,
                both_stationary=both_stationary,
                accident_streak=accident_streak,
                accident_confirm_windows=ACCIDENT_CONFIRM_WINDOWS,
                motion_metrics=motion_metrics,
            )
        except Exception as exc:
            raise RuntimeError(f"Live prediction processing failed: {exc}") from exc

    def get_buffer_status(self, ride_id: str) -> dict:
        with self._lock:
            ride_buffers = self._buffers.get(ride_id)
            if ride_buffers is None:
                return {"helmet": 0, "chest": 0, "paired": 0}
            helmet_size = len(ride_buffers["helmet"])
            chest_size = len(ride_buffers["chest"])
            return {
                "helmet": helmet_size,
                "chest": chest_size,
                "paired": min(helmet_size, chest_size),
            }

    def get_buffer_size(self, ride_id: str) -> int:
        return self.get_buffer_status(ride_id)["paired"]

    def clear_buffer(self, ride_id: str) -> bool:
        with self._lock:
            existed = False
            for mapping in (
                self._buffers,
                self._reading_counters,
                self._last_prediction_counts,
                self._last_counted_pair_total,
                self._new_pairs_since_prediction,
                self._accident_streaks,
                self._final_states,
            ):
                if ride_id in mapping:
                    del mapping[ride_id]
                    existed = True
            return existed

    @staticmethod
    def _motion_metrics(window) -> Dict[str, float]:
        acc_magnitudes = [
            math.sqrt(item.accel_x ** 2 + item.accel_y ** 2 + item.accel_z ** 2)
            for item in window
        ]
        gyro_magnitudes = [
            math.sqrt(item.gyro_x ** 2 + item.gyro_y ** 2 + item.gyro_z ** 2)
            for item in window
        ]
        return {
            "max_accel": max(acc_magnitudes),
            "min_accel": min(acc_magnitudes),
            "accel_delta": max(acc_magnitudes) - min(acc_magnitudes),
            "accel_std": LivePredictionService._population_std(acc_magnitudes),
            "gyro_mean": sum(gyro_magnitudes) / len(gyro_magnitudes),
            "max_gyro": max(gyro_magnitudes),
            "accel_x_std": LivePredictionService._population_std([item.accel_x for item in window]),
            "accel_y_std": LivePredictionService._population_std([item.accel_y for item in window]),
            "accel_z_std": LivePredictionService._population_std([item.accel_z for item in window]),
        }

    @staticmethod
    def _population_std(values) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _is_stationary(metrics: Dict[str, float]) -> bool:
        return (
            metrics["accel_std"] <= STATIONARY_ACCEL_STD_THRESHOLD
            and metrics["accel_delta"] <= STATIONARY_ACCEL_RANGE_THRESHOLD
            and metrics["gyro_mean"] <= STATIONARY_GYRO_MEAN_THRESHOLD
        )

    @classmethod
    def _classify_runtime(
        cls, ride_id: str, model_prediction: str, confidence: float,
        impact_gate_passed: bool, both_stationary: bool = False,
    ) -> Tuple[str, int, str | None]:
        """Apply confidence, physical-motion, and persistence confirmation."""
        with cls._lock:
            if both_stationary:
                cls._accident_streaks[ride_id] = 0
                final_state = "SAFE"
            elif model_prediction == "ACCIDENT":
                qualifies = (
                    confidence >= ACCIDENT_MIN_CONFIDENCE and impact_gate_passed
                )
                cls._accident_streaks[ride_id] = (
                    cls._accident_streaks[ride_id] + 1 if qualifies else 0
                )
                final_state = (
                    "ACCIDENT_CONFIRMED"
                    if cls._accident_streaks[ride_id] >= ACCIDENT_CONFIRM_WINDOWS
                    else "ACCIDENT_PENDING"
                )
            elif model_prediction in ("SAFE", "RISK"):
                cls._accident_streaks[ride_id] = 0
                final_state = model_prediction
            else:
                logger.warning("Unknown fused prediction %r", model_prediction)
                cls._accident_streaks[ride_id] = 0
                final_state = cls._final_states.get(ride_id, "UNKNOWN")
            accident_streak = cls._accident_streaks[ride_id]
            previous_state = cls._final_states.get(ride_id)
            cls._final_states[ride_id] = final_state
            return final_state, accident_streak, previous_state

    @staticmethod
    def _persist(operation, callback, *args, defer_persistence=None) -> None:
        def measured_callback():
            started = time.perf_counter()
            try:
                callback(*args)
            finally:
                logger.info(
                    "[NeoRider LATENCY] db_operation=%s db=%.2fms",
                    operation, (time.perf_counter() - started) * 1000,
                )

        if defer_persistence is None:
            measured_callback()
        else:
            defer_persistence(measured_callback)
