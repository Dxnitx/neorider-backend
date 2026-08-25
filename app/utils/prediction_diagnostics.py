"""Diagnostics for the dual-IMU inference pipeline.

This module observes inference values only. It deliberately does not alter the
sensor window, extracted features, scaled values, or model decision.
"""

import logging
from typing import Dict, Iterable, List

import numpy as np

from app.utils.feature_extractor import _get_value


logger = logging.getLogger("neorider.ml_diagnostics")

VECTOR_FIELDS = {
    "accel_magnitude": ("accel_x", "accel_y", "accel_z"),
    "gyro_magnitude": ("gyro_x", "gyro_y", "gyro_z"),
}
ANGLE_FIELDS = ("pitch", "roll", "yaw")
CANCELLATION_RATIO = 0.60


def _values(readings: Iterable, fields: Iterable[str]) -> np.ndarray:
    return np.asarray(
        [[_get_value(reading, (field,)) for field in fields] for reading in readings],
        dtype=np.float64,
    )


def summarize_window(readings: List) -> Dict[str, Dict[str, float]]:
    """Summarize motion magnitudes and Euler angles for one sensor window."""
    summary: Dict[str, Dict[str, float]] = {}
    for name, fields in VECTOR_FIELDS.items():
        magnitude = np.linalg.norm(_values(readings, fields), axis=1)
        summary[name] = _array_stats(magnitude, include_range=False)
    for field in ANGLE_FIELDS:
        values = _values(readings, (field,)).reshape(-1)
        summary[field] = _array_stats(values, include_range=True)
    return summary


def build_combined_window(helmet_window: List, chest_window: List) -> List[dict]:
    """Compatibility helper returning Helmet values only for MMC1 diagnostics."""
    fields = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "pitch", "roll", "yaw")
    return [{field: _get_value(helmet, (field,)) for field in fields} for helmet in helmet_window]


def _array_stats(values: np.ndarray, include_range: bool = False) -> Dict[str, float]:
    result = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    if include_range:
        result["range"] = result["max"] - result["min"]
    return result


def vector_stats(values) -> Dict[str, float | int]:
    """Return finite-value diagnostics without repairing the supplied vector."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
    }


def log_window_diagnostics(sensor_window: List) -> dict:
    helmet = [item for item in sensor_window if _get_device(item) == "helmet"]
    chest = [item for item in sensor_window if _get_device(item) == "chest"]
    combined = build_combined_window(helmet, chest)
    summaries = {
        "helmet": summarize_window(helmet),
        "chest": summarize_window(chest),
        "combined": summarize_window(combined),
    }

    logger.info("========== RAW WINDOW SUMMARY ==========")
    _log_motion_summary("HELMET", summaries["helmet"])
    _log_motion_summary("CHEST", summaries["chest"])
    logger.info("========== MMC1 HELMET MODEL INPUT ==========")
    _log_motion_summary("HELMET", summaries["combined"])

    cancellations = []
    comparisons = (
        ("accel_magnitude", "max"),
        ("gyro_magnitude", "max"),
        ("pitch", "range"),
        ("roll", "range"),
        ("yaw", "range"),
    )
    for metric, stat in comparisons:
        raw_peak = max(summaries["helmet"][metric][stat], summaries["chest"][metric][stat])
        combined_value = summaries["combined"][metric][stat]
        if raw_peak > 1e-9 and combined_value < raw_peak * CANCELLATION_RATIO:
            cancellations.append(
                f"{metric}.{stat} combined={combined_value:.6g}, raw_peak={raw_peak:.6g}"
            )
    if cancellations:
        logger.warning("DUAL-IMU SIGNAL CANCELLATION DETECTED: %s", "; ".join(cancellations))
    return {**summaries, "cancellation_detected": bool(cancellations)}


def _get_device(reading) -> str | None:
    return reading.get("device") if isinstance(reading, dict) else getattr(reading, "device", None)


def _log_motion_summary(name: str, summary: dict) -> None:
    logger.info(
        "%s | accel magnitude min=%.6g max=%.6g mean=%.6g std=%.6g | "
        "gyro magnitude min=%.6g max=%.6g mean=%.6g std=%.6g | "
        "pitch min=%.6g max=%.6g range=%.6g | roll min=%.6g max=%.6g range=%.6g | "
        "yaw min=%.6g max=%.6g range=%.6g",
        name,
        *[summary["accel_magnitude"][key] for key in ("min", "max", "mean", "std")],
        *[summary["gyro_magnitude"][key] for key in ("min", "max", "mean", "std")],
        *[summary[field][key] for field in ANGLE_FIELDS for key in ("min", "max", "range")],
    )
