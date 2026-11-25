from __future__ import annotations

import argparse
import csv
from pathlib import Path
import queue
import re
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial
from serial.tools import list_ports

RAW_LINE = re.compile(
    r"^RAW,(\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)$"
)
SAMPLE_RATE_HZ = 100


def find_port(requested_port: str) -> str:
    if requested_port.lower() != "auto":
        return requested_port

    ports = list(list_ports.comports())
    if len(ports) == 1:
        return ports[0].device
    likely = [
        port for port in ports
        if any(word in port.description.lower() for word in ("esp", "cp210", "ch340", "usb serial"))
    ]
    if len(likely) == 1:
        return likely[0].device
    available = ", ".join(f"{port.device} ({port.description})" for port in ports) or "none"
    raise SystemExit(f"Could not choose a port automatically. Available ports: {available}")


def next_filename(out_dir: Path, label: str) -> Path:
    idx = 1
    while True:
        candidate = out_dir / f"{label}.{idx:02d}.csv"
        if not candidate.exists():
            return candidate
        idx += 1


def serial_worker(port: str, data_queue: queue.Queue, stop_event: threading.Event) -> None:
    try:
        with serial.Serial(port, 115200, timeout=0.05) as device:
            time.sleep(0.5)
            device.reset_input_buffer()
            while not stop_event.is_set():
                raw = device.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                match = RAW_LINE.fullmatch(raw)
                if not match:
                    continue
                ts = int(match.group(1))
                ax = float(match.group(2))
                ay = float(match.group(3))
                az = float(match.group(4))
                data_queue.put((ts, ax, ay, az))
    except serial.SerialException as error:
        print(f"\nSerial error on {port}: {error}")
        stop_event.set()


def record_clip(port: str, out_dir: Path, label: str, duration_sec: int) -> None:
    target_samples = duration_sec * SAMPLE_RATE_HZ
    out_file = next_filename(out_dir, label)

    data_queue: queue.Queue[tuple[int, float, float, float]] = queue.Queue()
    stop_event = threading.Event()
    recorded: list[tuple[int, float, float, float]] = []
    base_ts: int | None = None

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Recording [{label}] — {duration_sec}s ({target_samples} samples)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_ylim(-2.1, 2.1)
    ax.grid(alpha=0.25)

    line_x, = ax.plot([], [], label="accX", lw=1)
    line_y, = ax.plot([], [], label="accY", lw=1)
    line_z, = ax.plot([], [], label="accZ", lw=1)
    ax.legend(loc="upper right")
    fig.tight_layout()

    def update(_frame):
        nonlocal base_ts
        while not data_queue.empty() and len(recorded) < target_samples:
            raw_ts, x, y, z = data_queue.get_nowait()
            if base_ts is None:
                base_ts = raw_ts
            recorded.append((raw_ts - base_ts, x, y, z))

        if recorded:
            window = recorded[-300:]
            t = [r[0] / 1000.0 for r in window]
            line_x.set_data(t, [r[1] for r in window])
            line_y.set_data(t, [r[2] for r in window])
            line_z.set_data(t, [r[3] for r in window])
            ax.set_xlim(max(0.0, t[-1] - 3.0), max(3.0, t[-1]))

        if len(recorded) >= target_samples:
            stop_event.set()
            plt.close(fig)

        return [line_x, line_y, line_z]

    worker = threading.Thread(target=serial_worker, args=(port, data_queue, stop_event), daemon=True)
    worker.start()

    ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    stop_event.set()

    if len(recorded) < 100:
        print("Recording cancelled before 1 second of data was captured.\n")
        return

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "accX", "accY", "accZ"])
        for ts, x, y, z in recorded:
            writer.writerow([ts, f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])

    print(f"Saved {len(recorded)} samples ({len(recorded)/100:.1f}s) -> {out_file}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record 100Hz MPU6050 CSVs for Edge Impulse")
    parser.add_argument("--port", default="auto", help="COM port (default: auto)")
    parser.add_argument("--dir", default="ei_dataset", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    port = find_port(args.port)
    print(f"Using ESP32 on {port}. Output folder: {out_dir.resolve()}\n")

    while True:
        label = input("Label to record (NORMAL / HIGH / CRITICAL) or 'q' to quit: ").strip().upper()
        if label == "Q":
            break
        if label not in {"NORMAL", "HIGH", "CRITICAL"}:
            print("Please enter NORMAL, HIGH, or CRITICAL.")
            continue

        sec_str = input("Duration in seconds [default 20]: ").strip()
        duration = int(sec_str) if sec_str.isdigit() else 20
        print("Start inducing vibration—recording starts now...")
        record_clip(port, out_dir, label, duration)


if __name__ == "__main__":
    main()