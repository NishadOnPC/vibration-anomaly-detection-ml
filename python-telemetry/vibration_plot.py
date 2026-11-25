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


# Thresholds tuned for vibration after gravity removal
RMS_HIGH = 0.0150
RMS_CRITICAL = 0.0400
P2P_HIGH = 0.0400
P2P_CRITICAL = 0.1200

LINE = re.compile(r"^LIVE,(\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),(NORMAL|HIGH|CRITICAL)$")
COLORS = {"NORMAL": "#2ca02c", "HIGH": "#ff9900", "CRITICAL": "#d62728"}


def classify(rms: float, peak_to_peak: float) -> str:
    if rms >= RMS_CRITICAL or peak_to_peak >= P2P_CRITICAL:
        return "CRITICAL"
    if rms >= RMS_HIGH or peak_to_peak >= P2P_HIGH:
        return "HIGH"
    return "NORMAL"


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


def initialise_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["Time", "RMS", "PeakToPeak", "Status", "Source"])


def append_row(path: Path, elapsed_ms: int, rms: float, peak_to_peak: float, status: str) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([elapsed_ms, f"{rms:.5f}", f"{peak_to_peak:.5f}", status, "LIVE_MPU6050"])


def serial_worker(port: str, data_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Reads serial data in a background thread so the Matplotlib UI never freezes."""
    try:
        with serial.Serial(port, 115200, timeout=0.05) as device:
            time.sleep(0.5)
            device.reset_input_buffer()
            while not stop_event.is_set():
                raw = device.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                match = LINE.fullmatch(raw)
                if not match:
                    print(f"Ignoring: {raw}")
                    continue

                elapsed_ms = int(match.group(1))
                rms = float(match.group(2))
                peak_to_peak = float(match.group(3))
                status = classify(rms, peak_to_peak)

                data_queue.put((elapsed_ms, rms, peak_to_peak, status))
    except serial.SerialException as error:
        print(f"\nSerial error on {port}: {error}")
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically log real MPU6050 readings from ESP32")
    parser.add_argument("--port", default="auto", help="COM port from Arduino IDE, e.g. COM3 (default: auto)")
    parser.add_argument("--output", default="live_mpu6050_log.csv", help="CSV file for live physical readings")
    args = parser.parse_args()

    output_path = Path(args.output)
    port = find_port(args.port)
    initialise_csv(output_path)

    data_queue: queue.Queue[tuple[int, float, float, str]] = queue.Queue()
    stop_event = threading.Event()
    rows: list[tuple[int, float, float, str]] = []

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    figure.suptitle("ANOMALY DETECTOR — live ESP32 data")
    axes[0].set_title("Live physical MPU6050 data received from ESP32")
    axes[1].set_xlabel("ESP32 time (seconds)")

    lines = {}
    scatters = {}
    for ax_idx, (axis, label, high, critical) in enumerate((
        (axes[0], "RMS (g)", RMS_HIGH, RMS_CRITICAL),
        (axes[1], "Peak-to-Peak (g)", P2P_HIGH, P2P_CRITICAL),
    )):
        # Continuous connecting line
        (lines[ax_idx],) = axis.plot([], [], color="#1f77b4", linewidth=1.6, alpha=0.85, zorder=2)

        # Colored status points on top of the line
        for status in ("NORMAL", "HIGH", "CRITICAL"):
            scatters[(ax_idx, status)] = axis.scatter(
                [], [], color=COLORS[status], label=status, s=36, zorder=3
            )

        axis.axhline(high, color=COLORS["HIGH"], linestyle="--", linewidth=1, label="HIGH threshold")
        axis.axhline(critical, color=COLORS["CRITICAL"], linestyle="--", linewidth=1, label="CRITICAL threshold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    def update_plot(_frame):
        updated = False
        while not data_queue.empty():
            row = data_queue.get_nowait()
            rows.append(row)
            append_row(output_path, *row)
            print(f"LIVE: RMS={row[1]:.5f} g, P2P={row[2]:.5f} g, {row[3]}")
            updated = True

        if not updated:
            return []

        times = [r[0] / 1000.0 for r in rows]
        for ax_idx, val_idx in ((0, 1), (1, 2)):
            # Update the continuous line
            values = [r[val_idx] for r in rows]
            lines[ax_idx].set_data(times, values)

            # Update the colored status dots
            for status in ("NORMAL", "HIGH", "CRITICAL"):
                pts = [(times[i], rows[i][val_idx]) for i, r in enumerate(rows) if r[3] == status]
                if pts:
                    scatters[(ax_idx, status)].set_offsets(pts)

            axes[ax_idx].relim()
            axes[ax_idx].autoscale_view()

        return [*lines.values(), *scatters.values()]

    print(f"Connecting to {port} at 115200 baud...")
    print(f"Logging real MPU6050 data to: {output_path.resolve()}")
    print("Keep this window open. Close the graph window or press Ctrl+C to stop.")

    worker = threading.Thread(target=serial_worker, args=(port, data_queue, stop_event), daemon=True)
    worker.start()

    ani = animation.FuncAnimation(figure, update_plot, interval=100, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\nLive logging stopped. Your CSV was saved.")


if __name__ == "__main__":
    main()