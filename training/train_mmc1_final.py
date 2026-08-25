"""Train the final NeoRider 3-class MMC1 model from recording-level sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.feature_extractor import SIGNAL_COLS, extract_features


FS_TARGET = 20
WINDOW_SIZE = 20
WINDOW_STEP = 10
G_TO_MS2 = 9.80665
LABEL_TO_ID = {"SAFE": 0, "RISK": 1, "ACCIDENT": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
RAW_COLUMNS = ["time", "ax", "ay", "az", "rx", "ry", "rz"]
FALL_EVENT_RADIUS_SECONDS = 2.5
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-dir", type=Path, default=Path("safe_recordings"))
    parser.add_argument(
        "--mmc1-dir",
        type=Path,
        default=Path("data/mmc1/Corrected_DataSet_DataInBrief"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("app/ml/mmc1"))
    parser.add_argument("--report", type=Path, default=Path("training/final_training_report.json"))
    return parser.parse_args()


def file_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")


def validate_frame(frame: pd.DataFrame, path: Path, *, safe: bool) -> pd.DataFrame:
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    frame = frame[RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            rows = converted.index[converted.isna()].tolist()[:5]
            raise ValueError(f"{path}: non-numeric/NaN values in {column}, rows {rows}")
        frame[column] = converted.astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: infinite values found")
    if not np.all(np.diff(frame["time"].to_numpy()) > 0):
        raise ValueError(f"{path}: timestamps must be strictly monotonic")
    if len(frame) < WINDOW_SIZE:
        raise ValueError(f"{path}: {len(frame)} samples cannot form a {WINDOW_SIZE}-sample window")
    if safe:
        median_dt = float(np.median(np.diff(frame["time"])))
        if not np.isclose(median_dt, 1 / FS_TARGET, rtol=0.05, atol=0.002):
            raise ValueError(f"{path}: expected 20 Hz SAFE data, median dt is {median_dt:.6g}s")
    return frame


def read_safe(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"{path}: columns must be exactly {RAW_COLUMNS}; got {list(frame.columns)}")
    return validate_frame(frame, path, safe=True)


def read_mmc1(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", usecols=range(7), encoding="latin1")
    frame.columns = RAW_COLUMNS
    frame = validate_frame(frame, path, safe=False)
    # MMC1 is m/s^2; NeoRider/live IMUs are g. Train in live-device units.
    frame[["ax", "ay", "az"]] /= G_TO_MS2
    return frame


def resample_20_hz(frame: pd.DataFrame) -> pd.DataFrame:
    start, stop = float(frame.time.iloc[0]), float(frame.time.iloc[-1])
    target_time = np.arange(start, stop + 1e-9, 1 / FS_TARGET)
    values = {"time": target_time}
    for column in RAW_COLUMNS[1:]:
        values[column] = np.interp(target_time, frame.time.to_numpy(), frame[column].to_numpy())
    return pd.DataFrame(values)


def isolate_fall_event(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    acceleration = np.sqrt(np.square(frame[["ax", "ay", "az"]]).sum(axis=1))
    impact_time = float(frame.loc[int(acceleration.idxmax()), "time"])
    event = frame[
        frame.time.between(
            impact_time - FALL_EVENT_RADIUS_SECONDS,
            impact_time + FALL_EVENT_RADIUS_SECONDS,
            inclusive="both",
        )
    ].copy()
    if len(event) < WINDOW_SIZE:
        raise ValueError(f"Fall event at {impact_time}s is too short")
    return event, impact_time


def make_windows(frame: pd.DataFrame, label: str, recording_id: str) -> list[dict]:
    rows = frame[RAW_COLUMNS[1:]].to_dict("records")
    output = []
    for start in range(0, len(rows) - WINDOW_SIZE + 1, WINDOW_STEP):
        output.append(
            {
                "features": extract_features(rows[start:start + WINDOW_SIZE]),
                "label": LABEL_TO_ID[label],
                "label_name": label,
                "file_id": recording_id,
                "window_start": start,
            }
        )
    return output


def choose_group_split(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    best = None
    for seed in range(1000):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(metadata, metadata.label, metadata.file_id))
        if set(metadata.iloc[train_idx].label) != set(LABEL_TO_ID.values()):
            continue
        if set(metadata.iloc[test_idx].label) != set(LABEL_TO_ID.values()):
            continue
        score = abs(len(test_idx) / len(metadata) - 0.2)
        candidate = (score, seed, train_idx, test_idx)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not create a recording-level split containing every class")
    return best[2], best[3]


def metrics_for(name: str, model, x_test, y_test) -> dict:
    predicted = model.predict(x_test).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=[0, 1, 2], zero_division=0
    )
    macro = precision_recall_fscore_support(y_test, predicted, average="macro", zero_division=0)
    per_class = {
        ID_TO_LABEL[index]: {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index in range(3)
    }
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=[0, 1, 2], target_names=list(LABEL_TO_ID),
            zero_division=0,
        ),
    }


def counts_by_name(values) -> dict[str, int]:
    counts = Counter(ID_TO_LABEL[int(value)] for value in values)
    return {name: counts.get(name, 0) for name in LABEL_TO_ID}


def main() -> None:
    args = parse_args()
    safe_paths = sorted(args.safe_dir.glob("*.csv"))
    expected_safe = {
        f"safe_{activity}_{number}_{device}.csv"
        for activity, number in (("ride", "01"), ("ride", "02"), ("ride", "03"),
                                 ("turns", "01"), ("braking", "01"), ("acceleration", "01"))
        for device in ("helmet", "chest")
    }
    found_safe = {path.name for path in safe_paths}
    if found_safe != expected_safe:
        raise ValueError(
            f"SAFE recording set mismatch; missing={sorted(expected_safe - found_safe)}, "
            f"unexpected={sorted(found_safe - expected_safe)}"
        )

    risk_paths = sorted((args.mmc1_dir / "Extreme manoeuvres").rglob("*.csv"))
    fall_paths = sorted((args.mmc1_dir / "Falls scenarios").rglob("*.csv"))
    if len(risk_paths) != 6 or len(fall_paths) != 4:
        raise ValueError(f"Expected 6 RISK and 4 fall files; got {len(risk_paths)} and {len(fall_paths)}")

    raw_counts = Counter()
    windows = []
    fall_events = {}
    for label, paths in (("SAFE", safe_paths), ("RISK", risk_paths), ("ACCIDENT", fall_paths)):
        for path in paths:
            frame = read_safe(path) if label == "SAFE" else read_mmc1(path)
            if label == "ACCIDENT":
                frame, impact_time = isolate_fall_event(frame)
                fall_events[file_id(path)] = impact_time
            raw_counts[label] += len(frame)
            sampled = frame if label == "SAFE" else resample_20_hz(frame)
            recording_windows = make_windows(sampled, label, file_id(path))
            if not recording_windows:
                raise ValueError(f"{path}: no windows after preprocessing")
            windows.extend(recording_windows)

    feature_columns = list(windows[0]["features"])
    existing_columns = list(joblib.load(args.output_dir / "feature_columns.pkl"))
    if feature_columns != existing_columns or len(feature_columns) != 112:
        raise RuntimeError("Feature ordering differs from the deployed 112-feature schema")
    if any(list(window["features"]) != feature_columns for window in windows):
        raise RuntimeError("Inconsistent feature ordering between windows")

    x = pd.DataFrame([window["features"] for window in windows], columns=feature_columns)
    metadata = pd.DataFrame([{key: value for key, value in window.items() if key != "features"} for window in windows])
    train_idx, test_idx = choose_group_split(metadata)
    train_groups = set(metadata.iloc[train_idx].file_id)
    test_groups = set(metadata.iloc[test_idx].file_id)
    assert train_groups.isdisjoint(test_groups)

    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train = metadata.iloc[train_idx].label.to_numpy()
    y_test = metadata.iloc[test_idx].label.to_numpy()
    scaler = StandardScaler().fit(x_train.to_numpy())
    x_train_scaled = scaler.transform(x_train.to_numpy())
    x_test_scaled = scaler.transform(x_test.to_numpy())
    balanced_weights = compute_sample_weight("balanced", y_train)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=450, learning_rate=0.035, num_leaves=31, max_depth=-1,
            min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=450, learning_rate=0.035, max_depth=6, min_child_weight=2,
            subsample=0.9, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    results = {}
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(x_train_scaled, y_train, sample_weight=balanced_weights)
        else:
            model.fit(x_train_scaled, y_train)
        results[name] = metrics_for(name, model, x_test_scaled, y_test)

    viable = [
        name for name, result in results.items()
        if result["per_class"]["SAFE"]["recall"] >= 0.60
        and result["per_class"]["RISK"]["recall"] >= 0.60
    ]
    if not viable:
        raise RuntimeError("Every candidate has unusably low SAFE or RISK recall (<0.60)")
    winner_name = max(
        viable,
        key=lambda name: (
            results[name]["per_class"]["ACCIDENT"]["recall"],
            results[name]["macro_f1"],
            results[name]["per_class"]["RISK"]["recall"],
            results[name]["per_class"]["SAFE"]["recall"],
            # Operational tie-break only after all mandated quality criteria.
            {"LightGBM": 2, "XGBoost": 1, "RandomForest": 0}[name],
        ),
    )
    winner = models[winner_name]

    window_counts = counts_by_name(metadata.label)
    per_recording = metadata.groupby(["file_id", "label_name"]).size().astype(int).to_dict()
    report = {
        "safe_source_recordings": len(safe_paths),
        "raw_class_distribution": {name: int(raw_counts[name]) for name in LABEL_TO_ID},
        "window_class_distribution": window_counts,
        "window_count_per_recording": {
            file_name: count for (file_name, _), count in sorted(per_recording.items())
        },
        "train_distribution": counts_by_name(y_train),
        "test_distribution": counts_by_name(y_test),
        "split_strategy": "GroupShuffleSplit by whole recording/file_id; no file appears in both sets",
        "train_file_ids": sorted(train_groups),
        "test_file_ids": sorted(test_groups),
        "fall_impact_times_seconds": fall_events,
        "fall_event_radius_seconds": FALL_EVENT_RADIUS_SECONDS,
        "unit_normalization": "MMC1 acceleration divided by 9.80665 (m/s^2 to g); gyroscope unchanged in degrees/s",
        "models": results,
        "winning_model": winner_name,
    }

    mapping = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "sample_rate_hz": FS_TARGET,
        "winning_model": winner_name,
        "test_metrics": results[winner_name],
        "signal_columns": list(SIGNAL_COLS),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, args.output_dir / "model.pkl")
    joblib.dump(scaler, args.output_dir / "scaler.pkl")
    joblib.dump(feature_columns, args.output_dir / "feature_columns.pkl")
    joblib.dump(mapping, args.output_dir / "class_mapping.pkl")
    (args.output_dir / "class_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    exported_model = joblib.load(args.output_dir / "model.pkl")
    exported_scaler = joblib.load(args.output_dir / "scaler.pkl")
    exported_columns = joblib.load(args.output_dir / "feature_columns.pkl")
    dimensions = {
        "model": int(exported_model.n_features_in_),
        "scaler": int(exported_scaler.n_features_in_),
        "feature_columns": len(exported_columns),
    }
    if set(dimensions.values()) != {112}:
        raise RuntimeError(f"Exported artifact dimension mismatch: {dimensions}")
    report["artifact_dimensions"] = dimensions
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nDATASET AUDIT")
    print("Raw samples:", report["raw_class_distribution"])
    print("Windows:", window_counts)
    print("Windows per recording:", json.dumps(report["window_count_per_recording"], indent=2))
    print("Train:", report["train_distribution"], "files:", report["train_file_ids"])
    print("Test:", report["test_distribution"], "files:", report["test_file_ids"])
    print("Split:", report["split_strategy"])
    comparison = pd.DataFrame([
        {
            "model": name, "accuracy": result["accuracy"],
            "macro_precision": result["macro_precision"], "macro_recall": result["macro_recall"],
            "macro_f1": result["macro_f1"],
            **{f"{label}_recall": result["per_class"][label]["recall"] for label in LABEL_TO_ID},
        }
        for name, result in results.items()
    ])
    print("\nMODEL COMPARISON\n", comparison.to_string(index=False))
    for name, result in results.items():
        print(f"\n{name}\n{result['classification_report']}")
        print("Confusion matrix [SAFE, RISK, ACCIDENT]:", result["confusion_matrix"])
    print("\nWinner:", winner_name)
    print("Artifact dimensions:", dimensions)


if __name__ == "__main__":
    main()
