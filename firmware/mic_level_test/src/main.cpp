// INMP441 sound-level test for the ESP32 DevKit.
//
// Wiring (mic -> ESP32):
//   VDD -> 3V3   (NEVER VIN/5V)
//   GND -> GND
//   L/R -> GND   (selects the left channel)
//   WS  -> D25
//   SCK -> D32
//   SD  -> D33
//
// What you should see in the Serial Monitor (115200 baud):
//   a quiet room around -60 to -45 dBFS, a clap jumping to around -20 dBFS or louder.
// Both channels are printed so a wiring problem is easy to spot:
//   - the mic should show up on ONE channel (usually L) and the other stays near -120
//   - both stuck at -120 means no data: check SD -> D33, SCK -> D32, WS -> D25 and power
//   - a value that never changes, even when you clap, usually means a bad solder joint

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

static const int PIN_SCK = 32;
static const int PIN_WS = 25;
static const int PIN_SD = 33;

static const i2s_port_t I2S_PORT = I2S_NUM_0;
static const int SAMPLE_RATE = 16000;
static const int FRAMES = 512;              // stereo frames per read (32 ms)
static int32_t buffer[FRAMES * 2];          // interleaved left/right

void setupI2S() {
  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = SAMPLE_RATE;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;   // INMP441 sends 24-bit data in 32-bit slots
  config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;   // read both so we can see which one is live
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 8;
  config.dma_buf_len = 256;
  config.use_apll = false;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = PIN_SCK;
  pins.ws_io_num = PIN_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_SD;

  esp_err_t err = i2s_driver_install(I2S_PORT, &config, 0, nullptr);
  if (err == ESP_OK) err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    Serial.printf("I2S setup failed: %d\n", err);
    while (true) delay(1000);
  }
}

float toDbfs(double sumSquares, int count) {
  if (count == 0 || sumSquares <= 0) return -120.0f;
  double rms = sqrt(sumSquares / count) / 8388608.0;  // 24-bit full scale
  if (rms < 1e-6) return -120.0f;
  return 20.0f * log10f((float)rms);
}

void printBar(float db) {
  int len = (int)((db + 80.0f) / 2.0f);  // -80 dBFS = empty, 0 dBFS = 40 chars
  if (len < 0) len = 0;
  if (len > 40) len = 40;
  for (int i = 0; i < len; i++) Serial.print('#');
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\nINMP441 sound-level test. Clap or talk near the mic.");
  setupI2S();
  // The mic needs a moment to settle; throw away the first reads.
  size_t bytesRead;
  for (int i = 0; i < 10; i++) i2s_read(I2S_PORT, buffer, sizeof(buffer), &bytesRead, portMAX_DELAY);
}

void loop() {
  static float peakL = -120, peakR = -120;
  static uint32_t lastPrint = 0;

  size_t bytesRead = 0;
  i2s_read(I2S_PORT, buffer, sizeof(buffer), &bytesRead, portMAX_DELAY);
  int frames = bytesRead / (2 * sizeof(int32_t));

  // Remove the DC offset per channel, then measure loudness.
  double meanL = 0, meanR = 0;
  for (int i = 0; i < frames; i++) {
    meanR += buffer[2 * i] >> 8;       // ESP32 puts the right slot first
    meanL += buffer[2 * i + 1] >> 8;
  }
  if (frames > 0) { meanL /= frames; meanR /= frames; }

  double sqL = 0, sqR = 0;
  for (int i = 0; i < frames; i++) {
    double r = (buffer[2 * i] >> 8) - meanR;
    double l = (buffer[2 * i + 1] >> 8) - meanL;
    sqL += l * l;
    sqR += r * r;
  }
  float dbL = toDbfs(sqL, frames);
  float dbR = toDbfs(sqR, frames);
  if (dbL > peakL) peakL = dbL;
  if (dbR > peakR) peakR = dbR;

  if (millis() - lastPrint >= 150) {
    lastPrint = millis();
    Serial.printf("L %6.1f dBFS  R %6.1f dBFS  |", peakL, peakR);
    printBar(peakL > peakR ? peakL : peakR);
    Serial.println();
    peakL = peakR = -120;
  }
}
