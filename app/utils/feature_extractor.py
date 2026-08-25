"""Exact feature extraction used by the MMC1 classifier."""

from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import kurtosis, skew

WINDOW_SIZE = 20
SAMPLE_RATE_HZ = 20
DT = 1.0 / SAMPLE_RATE_HZ
SIGNAL_COLS = ("ax", "ay", "az", "rx", "ry", "rz", "acc_mag", "rot_mag")


def _get_value(reading, field_names: Iterable[str]) -> float:
    for field_name in field_names:
        value = reading.get(field_name) if isinstance(reading, dict) else getattr(reading, field_name, None)
        if value is not None:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Non-finite sensor value in {field_name}")
            return numeric
    raise ValueError(f"Missing sensor field; expected one of {tuple(field_names)}")


def _statistics(prefix: str, values: np.ndarray) -> Dict[str, float]:
    standard_deviation = float(values.std())
    minimum, maximum = float(values.min()), float(values.max())
    constant = standard_deviation <= 1e-9
    result = {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": standard_deviation,
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_range": maximum - minimum,
        f"{prefix}_rms": np.sqrt(np.mean(values ** 2)),
        f"{prefix}_median": np.median(values),
        f"{prefix}_iqr": np.percentile(values, 75) - np.percentile(values, 25),
        f"{prefix}_skew": 0.0 if constant else skew(values),
        f"{prefix}_kurtosis": 0.0 if constant else kurtosis(values),
        f"{prefix}_energy": np.sum(values ** 2),
        f"{prefix}_abs_max": np.max(np.abs(values)),
        f"{prefix}_abs_mean": np.mean(np.abs(values)),
    }
    return {name: float(value) for name, value in result.items()}


def _device(reading) -> str | None:
    return reading.get("device") if isinstance(reading, dict) else getattr(reading, "device", None)


def extract_features(sensor_window: List) -> Dict[str, float]:
    """Build the exact 112 MMC1 features from one 20-sample IMU window."""
    if sensor_window is None:
        raise ValueError("sensor_window cannot be None")
    if len(sensor_window) != WINDOW_SIZE:
        raise ValueError(f"MMC1 requires exactly 20 readings from one device; received {len(sensor_window)}")
    devices = {_device(reading) for reading in sensor_window} - {None}
    if len(devices) > 1:
        raise ValueError("MMC1 feature extraction cannot mix Helmet and Chest readings")

    signals = {name: [] for name in SIGNAL_COLS}
    for reading in sensor_window:
        ax = _get_value(reading, ("accel_x", "ax")); ay = _get_value(reading, ("accel_y", "ay")); az = _get_value(reading, ("accel_z", "az"))
        rx = _get_value(reading, ("gyro_x", "rx")); ry = _get_value(reading, ("gyro_y", "ry")); rz = _get_value(reading, ("gyro_z", "rz"))
        row = {"ax": ax, "ay": ay, "az": az, "rx": rx, "ry": ry, "rz": rz,
               "acc_mag": np.sqrt(ax ** 2 + ay ** 2 + az ** 2),
               "rot_mag": np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)}
        for name, value in row.items():
            signals[name].append(value)

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in signals.items()}
    features: Dict[str, float] = {}
    for name in SIGNAL_COLS:
        features.update(_statistics(name, arrays[name]))
    for prefix, signal_name in (("acc", "acc_mag"), ("rot", "rot_mag")):
        jerk = np.diff(arrays[signal_name]) / DT
        features.update({f"{prefix}_jerk_mean": float(jerk.mean()),
                         f"{prefix}_jerk_std": float(jerk.std()),
                         f"{prefix}_jerk_max_abs": float(np.max(np.abs(jerk))),
                         f"{prefix}_jerk_energy": float(np.sum(jerk ** 2))})
    if len(features) != 112 or not all(np.isfinite(value) for value in features.values()):
        raise RuntimeError("MMC1 feature extraction did not produce 112 finite features")
    return features
