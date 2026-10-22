// Mic node: ESP32 DevKit + INMP441.
//
// Listens continuously at 16 kHz. When a 32 ms block gets loud enough, it sends a
// 1.5 s WAV clip (0.5 s before the trigger + 1.0 s after) to the Pi over MQTT:
//   home/audio/<NODE_NAME>/clip     the WAV file (16 kHz, 16-bit, mono)
//   home/status/<NODE_NAME>         heartbeat JSON every 30 s; "offline" as the last will
//
// The trigger is adaptive: it tracks the room's background level (noise floor) and
// fires at TRIGGER_MARGIN_DB above it, never below MIN_TRIGGER_DBFS. The Pi decides
// whether a clip is glass; this node only decides "loud".
//
// Wiring (mic -> ESP32): VDD->3V3 (never VIN), GND->GND, L/R->GND, WS->D25, SCK->D32, SD->D33

#include <Arduino.h>
#include <ESPmDNS.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <driver/i2s.h>
#include <math.h>
#include <string.h>

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Copy include/secrets.example.h to include/secrets.h and fill in your WiFi details"
#endif

#ifndef NODE_NAME
#define NODE_NAME "mic_node"
#endif
#ifndef MQTT_PORT
#define MQTT_PORT 1883
#endif
#ifndef TOPIC_PREFIX
#define TOPIC_PREFIX "home"
#endif

// ---------------------------------------------------------------- settings

static const int PIN_SCK = 32;
static const int PIN_WS = 25;
static const int PIN_SD = 33;

static const i2s_port_t I2S_PORT = I2S_NUM_0;
static const int SAMPLE_RATE = 16000;
static const int FRAMES = 512;                       // one block = 32 ms
static const int PRE_SAMPLES = SAMPLE_RATE / 2;      // 0.5 s before the trigger
static const int POST_SAMPLES = SAMPLE_RATE;         // 1.0 s after
static const int CLIP_SAMPLES = PRE_SAMPLES + POST_SAMPLES;

static const float TRIGGER_MARGIN_DB = 18.0f;        // fire this far above the noise floor...
static const float MIN_TRIGGER_DBFS = -40.0f;        // ...but never below this level
static const float INITIAL_FLOOR_DBFS = -50.0f;      // your quiet room measured about -48 to -51
static const uint32_t WARMUP_MS = 3000;              // learn the room before triggering
static const int MAX_CLIPS_PER_MINUTE = 12;          // stop a noisy room from flooding the Pi
static const uint32_t HEARTBEAT_MS = 30000;
static const uint32_t SEND_GIVE_UP_MS = 15000;       // drop a clip if MQTT stays down this long
static const uint32_t LOG_EVERY_MS = 5000;

// ---------------------------------------------------------------- shared state

static int32_t i2sBuf[FRAMES * 2];                   // interleaved right/left slots
static int16_t ring[PRE_SAMPLES];                    // rolling pre-trigger audio
static int ringPos = 0;
static int16_t clip[CLIP_SAMPLES];
static volatile int clipFill = 0;

enum CaptureState : uint8_t { IDLE, CAPTURING, READY };
static volatile CaptureState captureState = IDLE;
static volatile uint32_t readySinceMs = 0;

static volatile float noiseFloorDb = INITIAL_FLOOR_DBFS;
static volatile float lastLevelDb = -120.0f;
static volatile float peakLevelDb = -120.0f;         // loudest block since the last log line
static volatile float lastTriggerDb = 0.0f;
static volatile uint32_t clipsTriggered = 0;
static volatile uint32_t clipsRateLimited = 0;
static uint32_t clipsSent = 0;
static uint32_t clipsDropped = 0;

static uint32_t recentClipMs[MAX_CLIPS_PER_MINUTE] = {0};
static int recentClipIdx = 0;

// ---------------------------------------------------------------- network state

static WiFiClient wifiClient;
static PubSubClient mqtt(wifiClient);
static char clipTopic[64];
static char statusTopic[64];
static IPAddress brokerIp;
static bool brokerResolved = false;
static bool mdnsStarted = false;
static int connectFailures = 0;

// ---------------------------------------------------------------- audio

static float triggerThresholdDb() {
  float t = noiseFloorDb + TRIGGER_MARGIN_DB;
  return t > MIN_TRIGGER_DBFS ? t : MIN_TRIGGER_DBFS;
}

static bool rateLimitAllows(uint32_t now) {
  uint32_t oldest = recentClipMs[recentClipIdx];
  if (oldest != 0 && now - oldest < 60000) return false;
  recentClipMs[recentClipIdx] = now;
  recentClipIdx = (recentClipIdx + 1) % MAX_CLIPS_PER_MINUTE;
  return true;
}

static void setupI2S() {
  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = SAMPLE_RATE;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;   // 24-bit data in 32-bit slots
  config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;   // same as the test sketch that worked
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

// Runs forever on core 1: read audio, track the noise floor, capture clips.
static void audioTask(void *) {
  float prevX = 0.0f, prevY = 0.0f;                  // DC-removal filter state
  const uint32_t startMs = millis();

  for (;;) {
    size_t bytesRead = 0;
    if (i2s_read(I2S_PORT, i2sBuf, sizeof(i2sBuf), &bytesRead, portMAX_DELAY) != ESP_OK) continue;
    int frames = bytesRead / (2 * sizeof(int32_t));
    if (frames == 0) continue;

    CaptureState state = captureState;
    double sumSquares = 0.0;

    for (int i = 0; i < frames; i++) {
      // The mic is on the LEFT channel (the odd slots), as the test sketch showed.
      float x = (float)(i2sBuf[2 * i + 1] >> 8);      // 24-bit signed sample
      float y = x - prevX + 0.995f * prevY;            // remove DC offset
      prevX = x;
      prevY = y;
      sumSquares += (double)y * y;

      float v = y / 256.0f;                            // 24-bit -> 16-bit
      if (v > 32767.0f) v = 32767.0f;
      if (v < -32768.0f) v = -32768.0f;
      int16_t sample = (int16_t)v;

      ring[ringPos] = sample;
      ringPos = (ringPos + 1) % PRE_SAMPLES;
      if (state == CAPTURING && clipFill < CLIP_SAMPLES) clip[clipFill++] = sample;
    }

    double rms = sqrt(sumSquares / frames) / 8388608.0;
    float db = rms > 1e-6 ? 20.0f * log10f((float)rms) : -120.0f;
    lastLevelDb = db;
    if (db > peakLevelDb) peakLevelDb = db;

    if (state == CAPTURING && clipFill >= CLIP_SAMPLES) {
      readySinceMs = millis();
      captureState = READY;                            // the main loop sends it
      continue;
    }
    if (state != IDLE) continue;

    float threshold = triggerThresholdDb();
    bool warmedUp = millis() - startMs > WARMUP_MS;

    if (warmedUp && db >= threshold) {
      if (!rateLimitAllows(millis())) {
        clipsRateLimited++;
        continue;
      }
      // Copy the last 0.5 s (including this loud block) as the start of the clip.
      for (int k = 0; k < PRE_SAMPLES; k++) clip[k] = ring[(ringPos + k) % PRE_SAMPLES];
      clipFill = PRE_SAMPLES;
      lastTriggerDb = db;
      clipsTriggered++;
      captureState = CAPTURING;
    } else {
      // Learn the background level: follow quiet moments quickly, loud ones slowly.
      float nf = noiseFloorDb;
      nf += (db < nf ? 0.2f : 0.005f) * (db - nf);
      noiseFloorDb = nf;
    }
  }
}

// ---------------------------------------------------------------- WAV + MQTT

static void put16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xFF; }

static void makeWavHeader(uint8_t *h, uint32_t dataBytes) {
  memcpy(h, "RIFF", 4);
  put32(h + 4, 36 + dataBytes);
  memcpy(h + 8, "WAVEfmt ", 8);
  put32(h + 16, 16);                  // fmt chunk size
  put16(h + 20, 1);                   // PCM
  put16(h + 22, 1);                   // mono
  put32(h + 24, SAMPLE_RATE);
  put32(h + 28, SAMPLE_RATE * 2);     // bytes per second
  put16(h + 32, 2);                   // bytes per frame
  put16(h + 34, 16);                  // bits per sample
  memcpy(h + 36, "data", 4);
  put32(h + 40, dataBytes);
}

static bool sendClip() {
  const uint32_t dataBytes = CLIP_SAMPLES * sizeof(int16_t);
  uint8_t header[44];
  makeWavHeader(header, dataBytes);

  // Stream the clip in pieces so no 48 KB MQTT buffer is needed.
  if (!mqtt.beginPublish(clipTopic, sizeof(header) + dataBytes, false)) return false;
  if (mqtt.write(header, sizeof(header)) != sizeof(header)) {
    wifiClient.stop();
    return false;
  }
  const uint8_t *p = (const uint8_t *)clip;   // ESP32 is little-endian, like WAV
  size_t left = dataBytes;
  while (left > 0) {
    size_t n = left > 1024 ? 1024 : left;
    size_t written = mqtt.write(p, n);
    if (written == 0) {
      wifiClient.stop();                       // half-sent packet: drop the connection to resync
      return false;
    }
    p += written;
    left -= written;
  }
  return mqtt.endPublish() == 1;
}

static bool resolveBroker() {
  if (brokerResolved) return true;
  IPAddress ip;
  if (ip.fromString(MQTT_HOST)) {
    brokerIp = ip;
    brokerResolved = true;
    return true;
  }

  char host[64];
  strncpy(host, MQTT_HOST, sizeof(host) - 1);
  host[sizeof(host) - 1] = '\0';
  char *suffix = strstr(host, ".local");
  if (suffix != nullptr) {
    *suffix = '\0';
    if (!mdnsStarted) mdnsStarted = MDNS.begin(NODE_NAME);
    ip = MDNS.queryHost(host, 2000);
  } else if (!WiFi.hostByName(MQTT_HOST, ip)) {
    ip = IPAddress();
  }

  if (ip[0] == 0) {
    Serial.printf("MQTT: can't find %s on the network yet\n", MQTT_HOST);
    return false;
  }
  brokerIp = ip;
  brokerResolved = true;
  Serial.printf("MQTT: %s is %s\n", MQTT_HOST, ip.toString().c_str());
  return true;
}

static void publishStatus() {
  char json[256];
  snprintf(json, sizeof(json),
           "{\"status\":\"online\",\"rssi\":%d,\"uptime_s\":%lu,\"noise_floor_dbfs\":%.1f,"
           "\"threshold_dbfs\":%.1f,\"clips_sent\":%lu,\"clips_dropped\":%lu,"
           "\"clips_rate_limited\":%lu,\"free_heap\":%lu}",
           WiFi.RSSI(), (unsigned long)(millis() / 1000), (float)noiseFloorDb, triggerThresholdDb(),
           (unsigned long)clipsSent, (unsigned long)clipsDropped,
           (unsigned long)clipsRateLimited, (unsigned long)ESP.getFreeHeap());
  mqtt.publish(statusTopic, json);
}

static void maintainWifi() {
  static bool wasConnected = false;
  static uint32_t lastAttemptMs = 0;
  bool connected = WiFi.status() == WL_CONNECTED;

  if (connected && !wasConnected) {
    Serial.printf("WiFi: connected, IP %s, signal %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else if (!connected && wasConnected) {
    Serial.println("WiFi: lost connection");
    lastAttemptMs = millis();
  }
  wasConnected = connected;

  if (!connected && millis() - lastAttemptMs > 20000) {
    lastAttemptMs = millis();
    Serial.printf("WiFi: connecting to \"%s\"...\n", WIFI_SSID);
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}

static void maintainMqtt() {
  static uint32_t lastAttemptMs = 0;
  if (mqtt.connected() || WiFi.status() != WL_CONNECTED) return;
  if (millis() - lastAttemptMs < 5000) return;
  lastAttemptMs = millis();

  if (!resolveBroker()) return;
  mqtt.setServer(brokerIp, MQTT_PORT);

  char clientId[48];
  snprintf(clientId, sizeof(clientId), "mic-%s-%06lx", NODE_NAME,
           (unsigned long)(ESP.getEfuseMac() & 0xFFFFFF));
  if (mqtt.connect(clientId, statusTopic, 0, false, "offline")) {
    Serial.printf("MQTT: connected to %s:%d\n", brokerIp.toString().c_str(), MQTT_PORT);
    connectFailures = 0;
    publishStatus();
  } else {
    Serial.printf("MQTT: connect failed (state %d)\n", mqtt.state());
    if (++connectFailures >= 3) {
      brokerResolved = false;          // the Pi's address may have changed; look it up again
      connectFailures = 0;
    }
  }
}

// ---------------------------------------------------------------- setup / loop

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.printf("\nMic node \"%s\" starting\n", NODE_NAME);

  snprintf(clipTopic, sizeof(clipTopic), "%s/audio/%s/clip", TOPIC_PREFIX, NODE_NAME);
  snprintf(statusTopic, sizeof(statusTopic), "%s/status/%s", TOPIC_PREFIX, NODE_NAME);

  setupI2S();
  size_t bytesRead;
  for (int i = 0; i < 10; i++) i2s_read(I2S_PORT, i2sBuf, sizeof(i2sBuf), &bytesRead, portMAX_DELAY);
  xTaskCreatePinnedToCore(audioTask, "audio", 8192, nullptr, 5, nullptr, 1);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                // wall-powered: keep the radio awake for fast alerts
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("WiFi: connecting to \"%s\"...\n", WIFI_SSID);

  mqtt.setBufferSize(512);             // status JSON; clips are streamed separately
  mqtt.setKeepAlive(30);
  mqtt.setSocketTimeout(10);
}

void loop() {
  static uint32_t lastHeartbeatMs = 0;
  static uint32_t lastLogMs = 0;
  static uint32_t reportedTriggers = 0;

  maintainWifi();
  maintainMqtt();
  mqtt.loop();

  if (clipsTriggered != reportedTriggers) {
    reportedTriggers = clipsTriggered;
    Serial.printf("TRIGGER at %.1f dBFS (threshold %.1f), capturing clip\n",
                  (float)lastTriggerDb, triggerThresholdDb());
  }

  if (captureState == READY) {
    if (mqtt.connected()) {
      uint32_t t0 = millis();
      if (sendClip()) {
        clipsSent++;
        Serial.printf("Clip sent (%lu ms)\n", (unsigned long)(millis() - t0));
      } else {
        clipsDropped++;
        Serial.println("Clip send failed");
      }
      clipFill = 0;
      captureState = IDLE;
    } else if (millis() - readySinceMs > SEND_GIVE_UP_MS) {
      clipsDropped++;
      Serial.println("Clip dropped: MQTT not connected");
      clipFill = 0;
      captureState = IDLE;
    }
  }

  if (mqtt.connected() && millis() - lastHeartbeatMs > HEARTBEAT_MS) {
    lastHeartbeatMs = millis();
    publishStatus();
  }

  if (millis() - lastLogMs > LOG_EVERY_MS) {
    lastLogMs = millis();
    Serial.printf("peak %6.1f dBFS | floor %6.1f | trigger at %6.1f | wifi %s | mqtt %s | sent %lu\n",
                  (float)peakLevelDb, (float)noiseFloorDb, triggerThresholdDb(),
                  WiFi.status() == WL_CONNECTED ? "ok" : "--", mqtt.connected() ? "ok" : "--",
                  (unsigned long)clipsSent);
    peakLevelDb = -120.0f;
  }

  delay(5);
}
