"""Reads Govee Bluetooth thermometers and publishes their readings to MQTT.

Govee sensors broadcast temperature/humidity in their Bluetooth advertisements on
stock firmware, so the Pi only has to listen. No pairing, no flashing.

  python -m hub.govee --discover   # list nearby Govees for 20 s (find their addresses)
  python -m hub.govee              # run continuously, publishing to MQTT

The byte formats below follow the community-documented Govee formats (the same
ones Home Assistant uses). If a reading looks wrong, run --discover, which also
prints the raw bytes, and compare with the sensor's screen.
"""

import argparse
import asyncio
import logging
import time

from . import messages
from .config import load_config

log = logging.getLogger("govee")

GOVEE_COMPANY_ID = 0xEC88   # H5072, H5074, H5075
GOVEE_ALT_COMPANY_ID = 0x0001  # H5101, H5102, H5177 and similar


def _decode_packed(three_bytes):
    """H5072/H5075/H5101/H5102 pack temperature and humidity into 3 bytes."""
    packed = int.from_bytes(three_bytes, "big")
    sign = 1
    if packed & 0x800000:  # top bit set = below 0 °C
        packed &= 0x7FFFFF
        sign = -1
    temp_c = sign * (packed // 1000) / 10.0
    humidity = (packed % 1000) / 10.0
    return temp_c, humidity


def decode(name, manufacturer_data):
    """Decode one advertisement. Returns dict(model, temp_c, humidity, battery) or None.

    manufacturer_data is bleak's dict {company_id: bytes} (company id already removed).
    """
    name = name or ""

    data = manufacturer_data.get(GOVEE_COMPANY_ID)
    if data is not None:
        # H5074: little-endian signed int16 temperature and uint16 humidity, in hundredths.
        if ("H5074" in name or len(data) == 7) and len(data) >= 6:
            temp_c = int.from_bytes(data[1:3], "little", signed=True) / 100.0
            humidity = int.from_bytes(data[3:5], "little") / 100.0
            return {"model": "H5074", "temp_c": temp_c, "humidity": humidity, "battery": data[5]}
        # H5072 / H5075: one padding byte, 3 packed bytes, battery.
        if len(data) >= 5:
            temp_c, humidity = _decode_packed(data[1:4])
            model = "H5072" if "H5072" in name else "H5075"
            return {"model": model, "temp_c": temp_c, "humidity": humidity, "battery": data[4]}

    data = manufacturer_data.get(GOVEE_ALT_COMPANY_ID)
    if data is not None and name.startswith(("GVH51", "GVH5177", "GVH5174")) and len(data) >= 6:
        temp_c, humidity = _decode_packed(data[2:5])
        return {"model": name[2:7], "temp_c": temp_c, "humidity": humidity, "battery": data[5]}

    return None


def plausible(reading):
    return -40.0 <= reading["temp_c"] <= 85.0 and 0.0 <= reading["humidity"] <= 100.0


class Publisher:
    """Turns advertisements into MQTT messages, rate-limited per sensor."""

    def __init__(self, cfg, publish):
        self.prefix = cfg["mqtt"].get("topic_prefix", "home")
        g = cfg.get("govee", {})
        self.every_s = float(g.get("publish_every_s", 30))
        self.names = {addr.upper(): room for addr, room in (g.get("sensors") or {}).items()}
        self.publish = publish
        self.last = {}

    def sensor_name(self, address, model):
        address = address.upper()
        return self.names.get(address) or f"govee_{model.lower()}_{address.replace(':', '')[-4:].lower()}"

    def handle(self, address, name, manufacturer_data, now=None):
        reading = decode(name, manufacturer_data)
        if reading is None or not plausible(reading):
            return None
        now = now if now is not None else time.time()
        sensor = self.sensor_name(address, reading["model"])
        previous = self.last.get(sensor)
        if previous is not None and now - previous < self.every_s:
            return None
        self.last[sensor] = now
        payload = messages.temp_payload(reading["temp_c"], reading["humidity"], reading["battery"], now)
        self.publish(messages.temp_topic(self.prefix, sensor), payload)
        log.info("%s (%s): %.1f °C, %.1f%% RH, battery %s%%",
                 sensor, reading["model"], reading["temp_c"], reading["humidity"], reading["battery"])
        return sensor


async def discover(seconds=20):
    from bleak import BleakScanner

    seen = {}

    def callback(device, adv):
        reading = decode(adv.local_name or device.name, adv.manufacturer_data)
        if reading is None:
            return
        raw = {hex(k): v.hex() for k, v in adv.manufacturer_data.items()}
        if device.address not in seen:
            print(f"\nFound {adv.local_name or device.name}  address {device.address}")
        seen[device.address] = reading
        print(f"  {device.address}: {reading['temp_c']:.1f} °C  {reading['humidity']:.1f}% RH  "
              f"battery {reading['battery']}%  raw {raw}")

    print(f"Scanning for Govee sensors for {seconds} seconds...")
    async with BleakScanner(detection_callback=callback):
        await asyncio.sleep(seconds)
    if not seen:
        print("\nNo Govee sensors found. Check that Bluetooth is on (bluetoothctl power on) "
              "and the sensors are within a few meters.")
    else:
        print("\nAdd these to the govee.sensors section of your config, e.g.:")
        for addr in seen:
            print(f'    "{addr}": room_name')


async def run(cfg):
    import paho.mqtt.client as mqtt
    from bleak import BleakScanner

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="govee-reader")
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect_async(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]))
    client.loop_start()

    pub = Publisher(cfg, lambda topic, payload: client.publish(topic, payload, qos=1))

    def callback(device, adv):
        try:
            pub.handle(device.address, adv.local_name or device.name, adv.manufacturer_data)
        except Exception:
            log.exception("Failed to handle advertisement from %s", device.address)

    log.info("Listening for Govee sensors...")
    async with BleakScanner(detection_callback=callback):
        await asyncio.Event().wait()  # run forever


def main():
    parser = argparse.ArgumentParser(description="Govee Bluetooth thermometer reader")
    parser.add_argument("--discover", action="store_true", help="list nearby Govee sensors and exit")
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--config")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.discover:
        asyncio.run(discover(args.seconds))
    else:
        asyncio.run(run(load_config(args.config)))


if __name__ == "__main__":
    main()
