import time
from datetime import datetime, timezone

import os

import requests


BASE_URL = os.getenv("NEORIDER_BASE_URL", "http://127.0.0.1:8002")
RIDE_ID = "06"
WINDOW_SIZE = 20
# The first prediction may include a cold model/scaler load.
REQUEST_TIMEOUT = 60


def get_buffer_status() -> dict:
    response = requests.get(
        f"{BASE_URL}/sensor/live/buffer/{RIDE_ID}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def send_reading(device: str, sequence: int) -> dict:
    # Deterministic, slightly changing values exercise the real feature pipeline.
    offset = sequence * 0.01
    payload = {
        "ride_id": RIDE_ID,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accel_x": 0.12 + offset,
        "accel_y": 0.04 + offset,
        "accel_z": 9.81 + offset,
        "gyro_x": 0.01 + offset,
        "gyro_y": 0.02 + offset,
        "gyro_z": 0.03 + offset,
        "mag_x": 0.0,
        "mag_y": 0.0,
        "mag_z": 0.0,
        "pitch": 2.0 + offset,
        "roll": 1.0 + offset,
        "yaw": 15.0 + offset,
    }
    response = requests.post(
        f"{BASE_URL}/sensor/live",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    status = get_buffer_status()
    helmet_count = status["helmet_buffer_size"]
    chest_count = status["chest_buffer_size"]

    if helmet_count != chest_count:
        raise RuntimeError(
            "Helmet and chest buffers are not aligned: "
            f"helmet={helmet_count}, chest={chest_count}"
        )

    print(
        f"Starting ride {RIDE_ID} at helmet={helmet_count}, "
        f"chest={chest_count}, paired={status['paired_buffer_size']}"
    )

    for sequence in range(helmet_count + 1, WINDOW_SIZE + 1):
        send_reading("helmet", sequence)
        time.sleep(0.05)
        result = send_reading("chest", sequence)

        print(
            f"\nPAIR {sequence}/{WINDOW_SIZE}"
            f" | Helmet: {result.get('helmet_buffer_size')}"
            f" | Chest: {result.get('chest_buffer_size')}"
            f" | Paired: {result.get('buffer_size')}"
            f" | Status: {result.get('status')}\n"
        )

        if result.get("status") == "predicted":
            print("=" * 60)
            print("ML PREDICTION RECEIVED")
            print("=" * 60)
            print(result.get("prediction"))
            print("=" * 60)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
