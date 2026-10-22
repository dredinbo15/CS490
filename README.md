# Pi home security AI

Glass-break detection and temperature monitoring on a Raspberry Pi 4. Three ESP32 + INMP441
mic nodes send short clips over WiFi; two Govee thermometers broadcast over Bluetooth. The Pi
classifies every clip with YAMNet (offline), logs everything to SQLite and pushes alerts to
your phone.

```
firmware/mic_level_test/   ESP32 sketch: prints mic loudness (test your soldering + wiring)
pi_service/                everything that runs on the Pi
  hub/service.py           main service: MQTT -> YAMNet -> SQLite -> alerts
  hub/govee.py             Govee Bluetooth reader -> MQTT
  hub/alerts.py            alert rules (glass fusion, temperature, offline nodes)
  hub/classifier.py        YAMNet wrapper (+ optional trained head, + dummy for testing)
  hub/messages.py          the MQTT topic/payload contract every node follows
  config.yaml              all settings and thresholds
  scripts/setup_pi.sh      one-time Pi setup
  tests/                   unit tests (python -m unittest discover -s tests)
simulator/simulate.py      fake sensors for testing without hardware
```

## MQTT contract

| Topic | Payload | Sent by |
|---|---|---|
| `home/audio/<node>/clip` | WAV file, 16 kHz, 16-bit, mono | mic nodes |
| `home/temp/<sensor>` | `{"temp_c": 21.4, "humidity": 45.2, "battery": 87}` | Govee reader |
| `home/status/<node>` | heartbeat JSON, or `offline` as the MQTT last will | mic nodes |
| `home/alerts` | alert JSON | the hub |

Any future node hardware (for example the battery-powered nRF52840 version) only has to
follow this table; nothing on the Pi changes.

## 1. Test the mic you just soldered

1. Wire it: VDD → 3V3, GND → GND, L/R → GND, WS → D25, SCK → D32, SD → D33. **Never VIN.**
2. Open `firmware/mic_level_test` in VS Code with the PlatformIO extension, plug in the ESP32,
   click **Upload**, then open the **Serial Monitor**.
3. A quiet room reads about -60 to -45 dBFS; a clap should jump to about -20 or louder.
   The sketch prints both channels; the mic should appear on one (usually L).
   Both stuck at -120 = no data: recheck SD, SCK, WS and power, then the solder joints.

## 1b. Flash the real mic node firmware

`firmware/mic_node` listens all the time and sends a 1.5 s clip (0.5 s before the loud
moment + 1 s after) to the Pi whenever a sound rises 18 dB above the room's background
level (never below -40 dBFS). It reads the LEFT channel, as the test sketch showed.

1. Copy `firmware/mic_node/include/secrets.example.h` to `secrets.h` in the same folder and
   fill in your 2.4 GHz WiFi name, password and the Pi's address (`security-hub.local` or an IP).
2. Open the `firmware/mic_node` folder in VS Code (File → Open Folder).
3. Pick the node in the PlatformIO environment selector on the bottom bar
   (`env:kitchen`, `env:living_room` or `env:bedroom`; rename them in `platformio.ini`).
4. Upload and Monitor. Every 5 s it prints the loudest level, the learned noise floor and the
   trigger level; a clap prints `TRIGGER ...`, then `Clip sent` once the Pi is reachable.

Until the Pi is running, the node still detects and logs triggers, then drops the clip after
15 s with `Clip dropped: MQTT not connected`. That's expected.

## 2. Set up the Pi

Use 64-bit Raspberry Pi OS. Copy this project to the Pi (or `git clone` it), then:

```bash
bash pi_service/scripts/setup_pi.sh
cd pi_service
.venv/bin/python -m hub.govee --discover     # lists your Govees and their addresses
nano config.yaml                              # add the addresses under govee.sensors
sudo systemctl start home-hub home-govee
journalctl -u home-hub -u home-govee -f       # watch it work
```

For phone alerts, install the **ntfy** app, subscribe to a long random topic name, and put
that name in `notify.ntfy_topic`.

## 3. Test without hardware (laptop or Pi)

```bash
# terminal 1 (from pi_service/)
python -m hub.service
# terminal 2 (from the project root)
python simulator/simulate.py temps
python simulator/simulate.py glass --wav some_glass_break.wav   # 2 mics -> high-confidence alert
python simulator/simulate.py freeze                              # freeze + rapid-drop alerts
```

Real glass-break WAVs: the ESC-50 dataset has a "glass breaking" class. The simulator's
built-in synthetic crash tests the plumbing only; YAMNet won't call it glass.

If `ai-edge-litert` won't install on your laptop, set `classifier.backend: dummy` in a
`config.local.yaml` there. The dummy treats any loud clip as glass so you can test the
pipeline; use the real model on the Pi.

## Tuning notes

- `classifier.glass_threshold` (default 0.30) is a starting guess. Play real glass clips and
  your false-alarm sounds (dishes, door slams), look at the `glass_score` column in
  `data/hub.db` (`sound_events` table), and set the threshold between them.
- Every received clip is saved under `data/clips/<date>/`. Those are recordings from your real
  mics in your real enclosures, which makes them ideal training data for Phase 2.
- Govee byte formats follow the community-documented layouts (H5072/H5074/H5075/H5101/H5102).
  `--discover` prints the raw bytes; if a reading doesn't match the sensor's screen, send them
  over and the decoder can be adjusted.
- Mosquitto is opened to the home network without a password for the proof of concept
  (see `setup_pi.sh`). Add a password file before exposing anything beyond your LAN.
