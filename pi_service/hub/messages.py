"""The MQTT message contract shared by every node, the simulator and the hub.

Any sensor hardware (ESP32 today, an nRF52840 bridge later) only has to follow
these topics and payloads; nothing on the Pi side needs to change.

Topics (prefix defaults to "home"):
  <prefix>/audio/<node>/clip   payload: a complete WAV file (16 kHz, 16-bit, mono)
  <prefix>/temp/<sensor>       payload: JSON {"temp_c": 21.4, "humidity": 45.2, "battery": 87}
  <prefix>/status/<node>       payload: JSON {"rssi": -60, "uptime_s": 1234} or the text "offline"
  <prefix>/alerts              published BY the hub: JSON alert (see alerts.Alert)
"""

import json
import time


def audio_topic(prefix, node):
    return f"{prefix}/audio/{node}/clip"


def temp_topic(prefix, sensor):
    return f"{prefix}/temp/{sensor}"


def status_topic(prefix, node):
    return f"{prefix}/status/{node}"


def alerts_topic(prefix):
    return f"{prefix}/alerts"


def subscriptions(prefix):
    return [f"{prefix}/audio/+/clip", f"{prefix}/temp/+", f"{prefix}/status/+"]


def parse_topic(prefix, topic):
    """Return (kind, name) for a known topic, or (None, None).

    kind is "audio", "temp" or "status"; name is the node or sensor name.
    """
    parts = topic.split("/")
    if len(parts) < 3 or parts[0] != prefix:
        return None, None
    kind = parts[1]
    if kind == "audio" and len(parts) == 4 and parts[3] == "clip":
        return "audio", parts[2]
    if kind in ("temp", "status") and len(parts) == 3:
        return kind, parts[2]
    return None, None


def temp_payload(temp_c, humidity=None, battery=None, ts=None):
    return json.dumps({
        "temp_c": round(float(temp_c), 2),
        "humidity": None if humidity is None else round(float(humidity), 1),
        "battery": battery,
        "ts": ts if ts is not None else time.time(),
    })


def parse_temp_payload(payload):
    data = json.loads(payload)
    if "temp_c" not in data:
        raise ValueError("temperature payload missing temp_c")
    return data
