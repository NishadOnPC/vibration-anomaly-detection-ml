# Vibration & Anomaly Detection (V.A.D) with ESP32

A real-time vibration monitoring system built using an **ESP32**, an **MPU6050** accelerometer, an **OLED display**, and a **TinyML** model trained on Edge Impulse. It detects abnormal vibrations directly on the microcontroller and sends live data to a PC for plotting and CSV logging.

---

## How It Works

1. **Read Vibration Data:** The ESP32 reads X, Y, and Z acceleration values from the MPU6050 sensor 100 times per second (100 Hz).
2. **Calculate Magnitude & Run ML:** It combines the 3 axes into overall vibration magnitude and runs the Edge Impulse model locally on the ESP32—no internet or cloud needed.
3. **Classify Vibration State:** The model predicts one of three states:
   * **`NORMAL`**: Safe, steady operation.
   * **`HIGH`**: Increased vibration that needs attention.
   * **`CRITICAL`**: Heavy vibration or sudden shock/fault.
4. **Show & Log Results:** The predicted state is printed on the 0.96" OLED screen and sent over USB serial to Python scripts on a PC for live graphing and saving data.

---

## Hardware & Wiring

Both the MPU6050 sensor and the OLED screen use **I2C communication**, so they share the exact same two data pins on the ESP32:

| Component | Pin on Component | ESP32 Pin |
| :--- | :--- | :--- |
| **MPU6050 Sensor** | VCC / GND | 3.3V / GND |
| **MPU6050 Sensor** | SDA / SCL | GPIO 21 / GPIO 22 |
| **OLED Display (128x64)** | VCC / GND | 3.3V / GND |
| **OLED Display (128x64)** | SDA / SCL | GPIO 21 / GPIO 22 |

---

## Project Folders

* **`firmware/src/Sketch.ino`**: Main Arduino C++ code running on the ESP32.
* **`firmware/lib/`**: The exported Edge Impulse machine learning library (`V.A.D_-_ML_inferencing`).
* **`python-telemetry/collect_data.py`**: Reads raw sensor values from USB serial and saves them to CSV files for training.
* **`python-telemetry/vibration_plot.py`**: Plots live vibration waveforms in real time on your PC.

---

## How to Run

### 1. Upload Code to ESP32
1. Open **Arduino IDE** and install the ESP32 board package.
2. Go to **Sketch → Include Library → Add .ZIP Library** and select `firmware/lib/ei-v.a.d-ml-arduino-1.0.1-impulse-#1.zip`.
3. Open `firmware/src/Sketch.ino`, pick your ESP32 board and USB port (`/dev/ttyUSB0` on Linux or `COM3` on Windows), and click **Upload**.

### 2. Run Python Live Grapher
```bash
cd python-telemetry
pip install -r requirements.txt
python3 vibration_plot.py --port /dev/ttyUSB0