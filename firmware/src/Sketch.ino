#include <Wire.h>
#include <math.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <V.A.D_-_ML_inferencing.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

const int MPU_ADDR = 0x68;
const int BATCH_SIZE = 100;
const unsigned long SAMPLE_INTERVAL_US = 10000; // 100 Hz

float mag[BATCH_SIZE];
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
size_t feature_ix = 0;
int sample_idx = 0;
float sum_mag = 0.0;

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

void updateOLED(float rms, float p2p, const char* status, float confidence) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("EDGE AI (");
  display.print(confidence * 100.0f, 0);
  display.println("%)");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setCursor(0, 16);
  display.print("RMS: ");
  display.print(rms, 4);
  display.println(" g");

  display.setCursor(0, 28);
  display.print("P2P: ");
  display.print(p2p, 4);
  display.println(" g");

  display.setTextSize(2);
  display.setCursor(0, 44);
  display.print(status);

  display.display();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(); // SDA = GPIO 21, SCL = GPIO 22
  Wire.setClock(400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 allocation failed");
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 24);
    display.println("Starting Edge AI...");
    display.display();
  }

  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  // Set Accelerometer range to +/- 2g (matches training data)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Enable ~21 Hz Digital Low-Pass Filter (matches training data)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A);
  Wire.write(0x04);
  Wire.endTransmission(true);
}

void loop() {
  static unsigned long next_sample_us = micros();

  if ((long)(micros() - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_INTERVAL_US;

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true);

    int16_t rawX = (Wire.read() << 8) | Wire.read();
    int16_t rawY = (Wire.read() << 8) | Wire.read();
    int16_t rawZ = (Wire.read() << 8) | Wire.read();

    float ax = rawX / 16384.0;
    float ay = rawY / 16384.0;
    float az = rawZ / 16384.0;

    // Feed accX, accY, accZ into Edge Impulse buffer
    if (feature_ix + 2 < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
      features[feature_ix++] = ax;
      features[feature_ix++] = ay;
      features[feature_ix++] = az;
    }

    float m = sqrt(ax * ax + ay * ay + az * az);
    mag[sample_idx] = m;
    sum_mag += m;
    sample_idx++;

    // Every 100 samples (1 second), run Edge Impulse inference
    if (sample_idx >= BATCH_SIZE) {
      float mean_g = sum_mag / BATCH_SIZE;
      float sq_sum = 0.0;
      float min_v = 999.0;
      float max_v = -999.0;

      for (int i = 0; i < BATCH_SIZE; i++) {
        float vib = mag[i] - mean_g;
        sq_sum += vib * vib;
        if (vib < min_v) min_v = vib;
        if (vib > max_v) max_v = vib;
      }

      float rms = sqrt(sq_sum / BATCH_SIZE);
      float p2p = max_v - min_v;

      signal_t signal;
      signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
      signal.get_data = &raw_feature_get_data;

      ei_impulse_result_t result = { 0 };
      EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

      const char* status = "ERR";
      float top_conf = 0.0f;

      if (res == EI_IMPULSE_OK) {
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
          if (result.classification[ix].value > top_conf) {
            top_conf = result.classification[ix].value;
            status = result.classification[ix].label;
          }
        }
      }

      updateOLED(rms, p2p, status, top_conf);

      Serial.print("LIVE,");
      Serial.print(millis());
      Serial.print(",");
      Serial.print(rms, 5);
      Serial.print(",");
      Serial.print(p2p, 5);
      Serial.print(",");
      Serial.print(status);
      Serial.print(",");
      Serial.println(top_conf, 3);

      sample_idx = 0;
      sum_mag = 0.0;
      feature_ix = 0;
      next_sample_us = micros() + SAMPLE_INTERVAL_US;
    }
  }
}