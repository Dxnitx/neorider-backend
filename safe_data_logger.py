"""Record NeoRider SAFE IMU samples from an ESP32 serial connection.

Expected serial format (comma-separated):
    time,device,ax,ay,az,rx,ry,rz

Example:
    12.450,helmet,0.01,-0.02,1.00,0.40,-0.20,0.10
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

PORT = "COM5"
BAUD_RATE = 115200
OUTPUT_DIR = Path("safe_recordings")
CSV_HEADER = ["time", "ax", "ay", "az", "rx", "ry", "rz"]


def parse_sensor_line(line: str) -> tuple[str, list[float]] | None:
    """Parse one ESP32 line and reject headers, diagnostics, or malformed data."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 8:
        return None

    device = parts[1].lower()
    if device not in {"helmet", "chest"}:
        return None

    try:
        row = [float(parts[index]) for index in (0, 2, 3, 4, 5, 6, 7)]
    except ValueError:
        return None
    return device, row


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save Helmet and Chest SAFE samples into separate CSV files."
    )
    parser.add_argument("--port", default=PORT, help=f"ESP32 serial port (default: {PORT})")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help=f"Serial baud rate (default: {BAUD_RATE})")
    parser.add_argument(
        "--name",
        default=f"safe_ride_{datetime.now():%Y%m%d_%H%M%S}",
        help="Recording name used as the CSV filename prefix",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    try:
        import serial
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyserial is required. Install it with: python -m pip install pyserial"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    helmet_path = args.output_dir / f"{args.name}_helmet.csv"
    chest_path = args.output_dir / f"{args.name}_chest.csv"
    helmet_count = 0
    chest_count = 0
    samples_since_flush = 0

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser, \
                helmet_path.open("w", newline="", encoding="utf-8") as helmet_file, \
                chest_path.open("w", newline="", encoding="utf-8") as chest_file:
            helmet_writer = csv.writer(helmet_file)
            chest_writer = csv.writer(chest_file)
            helmet_writer.writerow(CSV_HEADER)
            chest_writer.writerow(CSV_HEADER)

            print("Recording SAFE data. Press Ctrl+C to stop.")
            while True:
                raw_line = ser.readline()
                if not raw_line:
                    continue
                parsed = parse_sensor_line(raw_line.decode("utf-8", errors="ignore").strip())
                if parsed is None:
                    continue

                device, row = parsed
                if device == "helmet":
                    helmet_writer.writerow(row)
                    helmet_count += 1
                else:
                    chest_writer.writerow(row)
                    chest_count += 1

                samples_since_flush += 1
                if samples_since_flush >= 100:
                    helmet_file.flush()
                    chest_file.flush()
                    samples_since_flush = 0
                    print(
                        f"Helmet: {helmet_count:6d} | Chest: {chest_count:6d}",
                        end="\r",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\n\nCollection stopped.")
    except serial.SerialException as exc:
        raise SystemExit(f"Could not use serial port {args.port}: {exc}") from exc

    print(f"\nSaved:\n{helmet_path.resolve()}\n{chest_path.resolve()}")
    print(f"\nHelmet samples: {helmet_count}\nChest samples : {chest_count}")


if __name__ == "__main__":
    main()
