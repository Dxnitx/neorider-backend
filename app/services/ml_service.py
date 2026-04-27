import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"

model = joblib.load(ML_DIR / "neorider_xgb_model.pkl")
encoder = joblib.load(ML_DIR / "label_encoder.pkl")
feature_columns = joblib.load(ML_DIR / "feature_columns.pkl")


def extract_features(data_list):
    df = pd.DataFrame(data_list)

    features = {
        "acc_x_mean": df["acc_x"].mean(),
        "acc_x_std": df["acc_x"].std(),
        "acc_x_min": df["acc_x"].min(),
        "acc_x_max": df["acc_x"].max(),

        "acc_y_mean": df["acc_y"].mean(),
        "acc_y_std": df["acc_y"].std(),
        "acc_y_min": df["acc_y"].min(),
        "acc_y_max": df["acc_y"].max(),

        "acc_z_mean": df["acc_z"].mean(),
        "acc_z_std": df["acc_z"].std(),
        "acc_z_min": df["acc_z"].min(),
        "acc_z_max": df["acc_z"].max(),

        "gyro_x_mean": df["gyro_x"].mean(),
        "gyro_x_std": df["gyro_x"].std(),
        "gyro_x_min": df["gyro_x"].min(),
        "gyro_x_max": df["gyro_x"].max(),

        "gyro_y_mean": df["gyro_y"].mean(),
        "gyro_y_std": df["gyro_y"].std(),
        "gyro_y_min": df["gyro_y"].min(),
        "gyro_y_max": df["gyro_y"].max(),

        "gyro_z_mean": df["gyro_z"].mean(),
        "gyro_z_std": df["gyro_z"].std(),
        "gyro_z_min": df["gyro_z"].min(),
        "gyro_z_max": df["gyro_z"].max(),

        "pitch_mean": df["pitch"].mean(),
        "pitch_std": df["pitch"].std(),

        "roll_mean": df["roll"].mean(),
        "roll_std": df["roll"].std(),

        "yaw_mean": df["yaw"].mean(),
        "yaw_std": df["yaw"].std(),
    }

    feature_df = pd.DataFrame([features])

    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df = feature_df[feature_columns].fillna(0.0)
    return feature_df


def predict_behavior(data_list):
    feature_df = extract_features(data_list)
    pred = model.predict(feature_df)
    label = encoder.inverse_transform(pred)[0]
    return label