"""Loading, validation, and inference for the MMC1 LightGBM model."""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.utils.feature_extractor import SAMPLE_RATE_HZ, WINDOW_SIZE, extract_features

APP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_DIR / "ml" / "mmc1"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
CLASS_MAPPING_PKL_PATH = MODEL_DIR / "class_mapping.pkl"
MODEL_USED = "LightGBM_MMC1_SAFE_RISK_ACCIDENT"
EXPECTED_FEATURE_COUNT = 112
LEGACY_LABELS = {"SAFE": "safe_riding", "RISK": "high_risk", "ACCIDENT": "possible_accident"}

logger = logging.getLogger("neorider.mmc1")
model = scaler = None
feature_columns = []
id_to_label = {}


def load_model():
    global model, scaler, feature_columns, id_to_label
    if model is not None:
        return model, scaler, feature_columns
    paths = (
        MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH,
        CLASS_MAPPING_PKL_PATH, CLASS_MAPPING_PATH,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Missing MMC1 artifact(s): " + ", ".join(missing))

    logger.info("Loading MMC1 model, scaler, feature metadata, and class mapping once")
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
    with CLASS_MAPPING_PATH.open(encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)
    pickled_mapping = joblib.load(CLASS_MAPPING_PKL_PATH)
    if pickled_mapping != mapping:
        raise RuntimeError("MMC1 class_mapping.pkl and class_mapping.json do not match")
    counts = (len(loaded_columns), int(getattr(loaded_model, "n_features_in_", -1)), int(getattr(loaded_scaler, "n_features_in_", -1)))
    if counts != (EXPECTED_FEATURE_COUNT,) * 3:
        raise RuntimeError(f"MMC1 feature-count mismatch: feature_columns={counts[0]}, model={counts[1]}, scaler={counts[2]}; expected 112")
    loaded_labels = {str(key): int(value) for key, value in mapping["label_to_id"].items()}
    if loaded_labels != {"SAFE": 0, "RISK": 1, "ACCIDENT": 2}:
        raise RuntimeError(f"Unexpected MMC1 class mapping: {loaded_labels}")
    if list(getattr(loaded_model, "classes_", [])) != [0, 1, 2]:
        raise RuntimeError(f"Unexpected MMC1 model classes: {loaded_model.classes_}")
    if mapping.get("window_size") != WINDOW_SIZE or mapping.get("sample_rate_hz") != SAMPLE_RATE_HZ:
        raise RuntimeError("MMC1 artifact sampling metadata does not match live inference")
    if mapping.get("signal_columns") != ["ax", "ay", "az", "rx", "ry", "rz", "acc_mag", "rot_mag"]:
        raise RuntimeError("MMC1 artifact signal columns do not match live inference")

    model, scaler, feature_columns = loaded_model, loaded_scaler, loaded_columns
    id_to_label = {int(key): str(value) for key, value in mapping["id_to_label"].items()}
    return model, scaler, feature_columns


def predict_safety(sensor_readings) -> dict:
    loaded_model, loaded_scaler, columns = load_model()
    feature_started = time.perf_counter()
    features = extract_features(sensor_readings)
    missing = [name for name in columns if name not in features]
    extra = [name for name in features if name not in columns]
    if missing or extra:
        raise RuntimeError(f"MMC1 feature schema mismatch: missing={missing}, extra={extra}")
    frame = pd.DataFrame([features]).reindex(columns=columns)
    values = frame.to_numpy(dtype=np.float64)
    if frame.shape != (1, EXPECTED_FEATURE_COUNT):
        raise RuntimeError(f"MMC1 input shape is {frame.shape}, expected (1, 112)")
    if not np.isfinite(values).all():
        raise RuntimeError("MMC1 input contains NaN or Inf")
    feature_ms = (time.perf_counter() - feature_started) * 1000

    scaler_started = time.perf_counter()
    scaled = loaded_scaler.transform(values)
    scaled_frame = pd.DataFrame(scaled, columns=columns)
    scaled_pairs = sorted(
        zip(columns, scaled[0]), key=lambda item: abs(item[1]), reverse=True
    )
    largest_scaled_features = [
        {"feature": name, "value": round(float(value), 4)}
        for name, value in scaled_pairs[:5]
    ]
    max_abs_scaled_feature = abs(float(scaled_pairs[0][1]))
    if max_abs_scaled_feature >= 6.0:
        logger.warning(
            "[MODEL_OOD] max_abs_scaled_feature=%.3f largest=%s",
            max_abs_scaled_feature, largest_scaled_features,
        )
    scaler_ms = (time.perf_counter() - scaler_started) * 1000
    inference_started = time.perf_counter()
    raw_probabilities = loaded_model.predict_proba(scaled_frame)[0]
    class_id = int(loaded_model.classes_[int(np.argmax(raw_probabilities))])
    inference_ms = (time.perf_counter() - inference_started) * 1000
    probabilities = {id_to_label[int(model_class)]: float(probability) for model_class, probability in zip(loaded_model.classes_, raw_probabilities)}
    safety_class = id_to_label[class_id]
    return {
        "safety_class": safety_class,
        "class_id": class_id,
        "confidence": round(probabilities[safety_class], 4),
        "probabilities": {label: round(probabilities.get(label, 0.0), 4) for label in ("SAFE", "RISK", "ACCIDENT")},
        "prediction": LEGACY_LABELS[safety_class],
        "model_used": MODEL_USED,
        "window_size": WINDOW_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "max_abs_scaled_feature": round(max_abs_scaled_feature, 4),
        "largest_scaled_features": largest_scaled_features,
        "timings_ms": {
            "feature": feature_ms,
            "scaler": scaler_ms,
            "inference": inference_ms,
        },
    }


def model_status() -> dict:
    loaded_model, loaded_scaler, columns = load_model()
    return {"loaded": True, "model": "LightGBM", "classes": {"SAFE": 0, "RISK": 1, "ACCIDENT": 2},
            "window_size": WINDOW_SIZE, "sample_rate_hz": SAMPLE_RATE_HZ, "features": len(columns),
            "model_features": int(loaded_model.n_features_in_), "scaler_features": int(loaded_scaler.n_features_in_)}


def log_startup_status() -> None:
    loaded_model, loaded_scaler, columns = load_model()
    model_name = type(loaded_model).__name__
    logger.info(
        "NeoRider MMC1 startup | model=%s | feature_count=%d "
        "(model=%d scaler=%d columns=%d) | sample_rate=%d Hz | window_size=%d | class_mapping=%s",
        model_name, len(columns), loaded_model.n_features_in_, loaded_scaler.n_features_in_,
        len(columns), SAMPLE_RATE_HZ, WINDOW_SIZE, id_to_label,
    )
