"""Fake sensors for testing the hub on your laptop before the hardware is ready.

Run from the project root (Mosquitto must be running):

  python simulator/simulate.py temps                     # 2 fake thermometers, readings every 5 s
  python simulator/simulate.py clip --node kitchen --wav glass.wav
  python simulator/simulate.py glass --wav glass.wav     # same clip from 2 mics = high-confidence alert
  python simulator/simulate.py freeze                    # temperature falls fast -> freeze + rapid-drop alerts
  python simulator/simulate.py demo --wav-dir test_clips # everything at once, random clips

Without a --wav file, a synthetic "crash" sound is generated. YAMNet will NOT call it
glass (it isn't real glass); use real glass-break WAVs (ESC-50 has them) to test the AI.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pi_service"))

from hub import messages  # noqa: E402
from hub.audio import encode_wav, load_clip  # noqa: E402

PREFIX = "home"


def connect(host, port):
    import paho.mqtt.client as mqtt
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="simulator")
    client.connect(host, port)
    client.loop_start()
    return client


def synthetic_crash(seconds=1.5, rate=16000):
    """A sharp noise burst with ringing tones. Exercises the pipeline, not the AI."""
    t = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng()
    noise = rng.normal(0, 0.6, len(t)) * np.exp(-t * 12)
    ring = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * 6) * 0.15 for f in (3100, 4700, 6200))
    clip = np.concatenate([np.zeros(int(0.2 * rate)), noise + ring])
    return encode_wav(clip, rate)


def read_wav(path):
    """Load any WAV and re-encode it as 16 kHz mono 16-bit, like the ESP32 sends."""
    return encode_wav(load_clip(Path(path).read_bytes()))


def send_clip(client, node, wav):
    client.publish(messages.audio_topic(PREFIX, node), wav, qos=1)
    print(f"sent {len(wav) / 1024:.0f} KB clip from {node}")


def heartbeat(client, nodes):
    for node in nodes:
        client.publish(messages.status_topic(PREFIX, node), '{"rssi": -55, "uptime_s": 0}', qos=1)


def cmd_temps(client, args):
    temps = {"living_room": 21.0, "basement": 17.0}
    print("Publishing fake temperatures every 5 s (Ctrl+C to stop)")
    while True:
        for sensor in temps:
            temps[sensor] += random.uniform(-0.15, 0.15)
            client.publish(messages.temp_topic(PREFIX, sensor),
                           messages.temp_payload(temps[sensor], random.uniform(40, 50), 90), qos=1)
        print("  " + "  ".join(f"{s} {t:.1f}°C" for s, t in temps.items()))
        time.sleep(5)


def cmd_clip(client, args):
    wav = read_wav(args.wav) if args.wav else synthetic_crash()
    send_clip(client, args.node, wav)


def cmd_glass(client, args):
    wav = read_wav(args.wav) if args.wav else synthetic_crash()
    send_clip(client, "kitchen", wav)
    time.sleep(0.3)
    send_clip(client, "living_room", wav)


def cmd_freeze(client, args):
    temp = 15.0
    print("Dropping basement temperature quickly (Ctrl+C to stop)")
    while temp > 2:
        client.publish(messages.temp_topic(PREFIX, "basement"), messages.temp_payload(temp, 50, 90), qos=1)
        print(f"  basement {temp:.1f}°C")
        temp -= 1.0
        time.sleep(3)


def cmd_demo(client, args):
    wavs = sorted(Path(args.wav_dir).glob("*.wav")) if args.wav_dir else []
    nodes = ["kitchen", "living_room", "bedroom"]
    temps = {"living_room": 21.0, "basement": 17.0}
    print(f"Demo running with {len(wavs) or 'synthetic'} clips (Ctrl+C to stop)")
    tick = 0
    while True:
        heartbeat(client, nodes)
        for sensor in temps:
            temps[sensor] += random.uniform(-0.15, 0.15)
            client.publish(messages.temp_topic(PREFIX, sensor),
                           messages.temp_payload(temps[sensor], 45, 90), qos=1)
        if tick % 3 == 0:
            node = random.choice(nodes)
            if wavs:
                path = random.choice(wavs)
                print(f"{node} <- {path.name}")
                send_clip(client, node, read_wav(path))
            else:
                send_clip(client, node, synthetic_crash())
        tick += 1
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Fake sensors for the Pi hub")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("temps")
    p = sub.add_parser("clip")
    p.add_argument("--node", default="kitchen")
    p.add_argument("--wav")
    p = sub.add_parser("glass")
    p.add_argument("--wav")
    sub.add_parser("freeze")
    p = sub.add_parser("demo")
    p.add_argument("--wav-dir")
    args = parser.parse_args()

    client = connect(args.host, args.port)
    try:
        {"temps": cmd_temps, "clip": cmd_clip, "glass": cmd_glass,
         "freeze": cmd_freeze, "demo": cmd_demo}[args.command](client, args)
        time.sleep(1)  # let the last messages go out
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
