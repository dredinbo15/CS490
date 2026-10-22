"""The hub service: listens on MQTT, classifies clips, logs everything and raises alerts.

Run it from the pi_service folder:
    python -m hub.service
"""

import argparse
import logging
import queue
import signal
import threading
import time

from . import messages
from .alerts import GlassFusion, NodeWatchdog, TempRules
from .audio import load_clip, rms_dbfs
from .classifier import make_classifier
from .config import load_config
from .notify import Notifier
from .storage import Storage

log = logging.getLogger("hub")


class Hub:
    """All the decision-making, separate from MQTT so it can be tested directly."""

    def __init__(self, cfg, classifier, storage, notifier, publish=None, clock=time.time):
        self.cfg = cfg
        self.prefix = cfg["mqtt"].get("topic_prefix", "home")
        self.classifier = classifier
        self.storage = storage
        self.notifier = notifier
        self.publish = publish        # callable(topic, payload) or None
        self.clock = clock
        self.save_clips = cfg["storage"].get("save_clips", True)

        a = cfg["alerts"]
        self.fusion = GlassFusion(a["multi_mic_window_s"], a["glass_cooldown_s"])
        self.temps = TempRules(a["temp_units"], a["freeze_below"], a["overheat_above"],
                               a["rapid_change"], a["rapid_window_s"], a["temp_cooldown_s"])
        self.watchdog = NodeWatchdog(a["node_offline_after_s"])

    def handle(self, topic, payload, received_ts=None):
        now = received_ts if received_ts is not None else self.clock()
        kind, name = messages.parse_topic(self.prefix, topic)
        try:
            if kind == "audio":
                self.handle_audio(name, payload, now)
            elif kind == "temp":
                self.handle_temp(name, payload, now)
            elif kind == "status":
                self.handle_status(name, payload, now)
        except Exception:
            log.exception("Failed to handle message on %s", topic)

    def handle_audio(self, node, wav_bytes, now):
        self._emit(self.watchdog.seen(node, now))
        samples = load_clip(wav_bytes)
        loudness = rms_dbfs(samples)

        start = time.perf_counter()
        result = self.classifier.classify(samples)
        inference_ms = (time.perf_counter() - start) * 1000

        clip_path = self.storage.save_clip(node, wav_bytes, now) if self.save_clips else None
        self.storage.add_sound_event(now, node, result, loudness, clip_path, inference_ms)
        log.info("%s: %.2fs clip, %.0f dBFS, top '%s' %.2f, glass %.2f%s (%.0f ms)",
                 node, len(samples) / 16000, loudness, result.label, result.top[0][1] if result.top else 0,
                 result.glass_score, " GLASS" if result.is_glass else "", inference_ms)

        if result.is_glass:
            self._emit(self.fusion.detection(node, result.glass_score, now))
        return result

    def handle_temp(self, sensor, payload, now):
        data = messages.parse_temp_payload(payload)
        self._emit(self.watchdog.seen(sensor, now))
        temp_c = float(data["temp_c"])
        self.storage.add_temp(now, sensor, temp_c, data.get("humidity"), data.get("battery"))
        log.debug("%s: %s", sensor, self.temps.fmt(temp_c))
        for alert in self.temps.reading(sensor, temp_c, now):
            self._emit(alert)

    def handle_status(self, node, payload, now):
        text = payload.decode("utf-8", "replace").strip() if isinstance(payload, bytes) else str(payload)
        if text.lower() == "offline":
            self._emit(self.watchdog.went_offline(node, now))
        else:
            self._emit(self.watchdog.seen(node, now))

    def tick(self):
        """Call regularly (every few seconds) to catch silent nodes."""
        for alert in self.watchdog.check(self.clock()):
            self._emit(alert)

    def _emit(self, alert):
        if alert is None:
            return
        self.storage.add_alert(alert)
        self.notifier.send(alert)
        if self.publish:
            self.publish(messages.alerts_topic(self.prefix), alert.to_json())


def run(cfg):
    import paho.mqtt.client as mqtt  # imported here so the Hub logic is testable without paho

    storage = Storage(cfg["storage"]["database"], cfg["storage"].get("clips_dir"))
    classifier = make_classifier(cfg)
    notifier = Notifier(cfg)
    prefix = cfg["mqtt"].get("topic_prefix", "home")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                         client_id=cfg["mqtt"].get("client_id", "pi-hub"))
    hub = Hub(cfg, classifier, storage, notifier,
              publish=lambda topic, payload: client.publish(topic, payload, qos=1))

    # Clips are processed on a worker thread so inference never blocks the MQTT connection.
    work = queue.Queue(maxsize=200)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect failed: %s", reason_code)
            return
        for topic in messages.subscriptions(prefix):
            client.subscribe(topic, qos=1)
        log.info("Connected to MQTT %s:%s, listening on %s/#",
                 cfg["mqtt"]["host"], cfg["mqtt"]["port"], prefix)

    def on_disconnect(client, userdata, flags, reason_code, properties):
        log.warning("MQTT disconnected (%s); reconnecting automatically", reason_code)

    def on_message(client, userdata, msg):
        try:
            work.put_nowait((msg.topic, msg.payload, time.time()))
        except queue.Full:
            log.error("Work queue full; dropping message on %s", msg.topic)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect_async(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]))
    client.loop_start()

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def worker():
        while not stop.is_set():
            try:
                topic, payload, ts = work.get(timeout=1)
            except queue.Empty:
                continue
            hub.handle(topic, payload, ts)

    threading.Thread(target=worker, name="hub-worker", daemon=True).start()
    log.info("Hub running. Press Ctrl+C to stop.")
    while not stop.is_set():
        hub.tick()
        stop.wait(10)

    log.info("Shutting down")
    client.loop_stop()
    client.disconnect()
    storage.close()


def main():
    parser = argparse.ArgumentParser(description="Pi home security hub")
    parser.add_argument("--config", help="path to a config file (default: config.local.yaml or config.yaml)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
