"""Unit tests. Run from the pi_service folder:  python -m unittest discover -s tests -v"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hub import classifier as clf
from hub import messages
from hub.alerts import GlassFusion, NodeWatchdog, TempRules, f_to_c
from hub.audio import decode_wav, encode_wav, load_clip, rms_dbfs, to_16k
from hub.govee import Publisher, decode
from hub.service import Hub
from hub.storage import Storage


def base_cfg(tmp):
    return {
        "mqtt": {"host": "localhost", "port": 1883, "topic_prefix": "home"},
        "storage": {"database": str(Path(tmp) / "hub.db"), "save_clips": True,
                    "clips_dir": str(Path(tmp) / "clips")},
        "classifier": {"backend": "dummy", "glass_threshold": 0.3,
                       "glass_classes": ["Glass", "Shatter"]},
        "alerts": {"multi_mic_window_s": 2.0, "glass_cooldown_s": 30, "node_offline_after_s": 120,
                   "temp_units": "F", "freeze_below": 45, "overheat_above": 95,
                   "rapid_change": 5, "rapid_window_s": 300, "temp_cooldown_s": 1800},
        "notify": {"ntfy_topic": ""},
        "govee": {"publish_every_s": 30, "sensors": {"A4:C1:38:12:34:56": "living_room"}},
    }


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, alert):
        self.sent.append(alert)


# ---------------------------------------------------------------- alerts

class GlassFusionTests(unittest.TestCase):
    def test_single_mic_is_alert(self):
        f = GlassFusion(window_s=2, cooldown_s=30)
        a = f.detection("kitchen", 0.8, now=100)
        self.assertEqual((a.kind, a.level, a.nodes), ("glass", "alert", ["kitchen"]))

    def test_two_mics_within_window_is_high(self):
        f = GlassFusion(window_s=2, cooldown_s=30)
        f.detection("kitchen", 0.8, now=100)
        a = f.detection("living_room", 0.7, now=101.5)
        self.assertEqual(a.level, "high")
        self.assertEqual(a.nodes, ["kitchen", "living_room"])

    def test_two_mics_outside_window_are_separate(self):
        f = GlassFusion(window_s=2, cooldown_s=30)
        f.detection("kitchen", 0.8, now=100)
        a = f.detection("living_room", 0.7, now=105)
        self.assertEqual(a.level, "alert")

    def test_cooldown_suppresses_repeats(self):
        f = GlassFusion(window_s=2, cooldown_s=30)
        self.assertIsNotNone(f.detection("kitchen", 0.8, now=100))
        self.assertIsNone(f.detection("kitchen", 0.8, now=110))
        self.assertIsNotNone(f.detection("kitchen", 0.8, now=131))

    def test_same_mic_twice_is_not_high(self):
        f = GlassFusion(window_s=2, cooldown_s=0)
        f.detection("kitchen", 0.8, now=100)
        a = f.detection("kitchen", 0.8, now=100.5)
        self.assertEqual(a.level, "alert")


class TempRuleTests(unittest.TestCase):
    def rules(self):
        return TempRules("F", freeze_below=45, overheat_above=95, rapid_change=5,
                         rapid_window_s=300, cooldown_s=1800)

    def test_normal_reading_no_alert(self):
        self.assertEqual(self.rules().reading("living_room", f_to_c(70), now=0), [])

    def test_freeze(self):
        alerts = self.rules().reading("basement", f_to_c(40), now=0)
        self.assertEqual([a.kind for a in alerts], ["freeze"])
        self.assertIn("40.0°F", alerts[0].message)

    def test_overheat(self):
        alerts = self.rules().reading("attic", f_to_c(100), now=0)
        self.assertEqual([a.kind for a in alerts], ["overheat"])

    def test_rapid_rise_uses_degree_difference(self):
        r = self.rules()
        r.reading("kitchen", f_to_c(70), now=0)
        self.assertEqual(r.reading("kitchen", f_to_c(74), now=60), [])   # 4 °F: no alert
        alerts = r.reading("kitchen", f_to_c(75.5), now=120)              # 5.5 °F: alert
        self.assertEqual([a.kind for a in alerts], ["rapid_rise"])

    def test_rapid_change_outside_window_ignored(self):
        r = self.rules()
        r.reading("kitchen", f_to_c(70), now=0)
        self.assertEqual(r.reading("kitchen", f_to_c(76), now=400), [])

    def test_rapid_drop(self):
        r = self.rules()
        r.reading("bedroom", f_to_c(70), now=0)
        alerts = r.reading("bedroom", f_to_c(63), now=200)
        self.assertEqual([a.kind for a in alerts], ["rapid_drop"])

    def test_cooldown(self):
        r = self.rules()
        self.assertEqual(len(r.reading("basement", f_to_c(40), now=0)), 1)
        self.assertEqual(len(r.reading("basement", f_to_c(40), now=60)), 0)

    def test_celsius_config(self):
        r = TempRules("C", freeze_below=5, overheat_above=35, rapid_change=3,
                      rapid_window_s=300, cooldown_s=0)
        self.assertEqual([a.kind for a in r.reading("x", 4.0, now=0)], ["freeze"])


class WatchdogTests(unittest.TestCase):
    def test_offline_then_online(self):
        w = NodeWatchdog(offline_after_s=120)
        self.assertIsNone(w.seen("kitchen", now=0))
        self.assertEqual(w.check(now=100), [])
        alerts = w.check(now=121)
        self.assertEqual([a.kind for a in alerts], ["node_offline"])
        self.assertEqual(w.check(now=200), [])   # reported once
        self.assertEqual(w.seen("kitchen", now=201).kind, "node_online")

    def test_last_will(self):
        w = NodeWatchdog()
        w.seen("kitchen", now=0)
        self.assertEqual(w.went_offline("kitchen", now=5).kind, "node_offline")
        self.assertIsNone(w.went_offline("kitchen", now=6))


# ---------------------------------------------------------------- audio

class AudioTests(unittest.TestCase):
    def test_wav_round_trip(self):
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        samples, rate = decode_wav(encode_wav(tone))
        self.assertEqual(rate, 16000)
        self.assertEqual(len(samples), 16000)
        self.assertLess(np.max(np.abs(samples - tone)), 1e-3)

    def test_resample_to_16k(self):
        out = to_16k(np.zeros(44100, dtype=np.float32), 44100)
        self.assertEqual(len(out), 16000)

    def test_load_clip_from_44k(self):
        tone = 0.3 * np.sin(2 * np.pi * 1000 * np.arange(44100) / 44100)
        self.assertEqual(len(load_clip(encode_wav(tone, 44100))), 16000)

    def test_loudness(self):
        self.assertAlmostEqual(rms_dbfs(np.full(100, 0.1)), -20.0, places=3)
        self.assertLess(rms_dbfs(np.zeros(100)), -100)


# ---------------------------------------------------------------- govee

class GoveeTests(unittest.TestCase):
    def test_h5075(self):
        packed = 215 * 1000 + 456                     # 21.5 °C, 45.6 %
        data = bytes([0x00]) + packed.to_bytes(3, "big") + bytes([100, 0x00])
        r = decode("GVH5075_1234", {0xEC88: data})
        self.assertEqual((r["model"], r["temp_c"], r["humidity"], r["battery"]), ("H5075", 21.5, 45.6, 100))

    def test_h5075_below_zero(self):
        packed = (52 * 1000 + 600) | 0x800000         # -5.2 °C, 60.0 %
        data = bytes([0x00]) + packed.to_bytes(3, "big") + bytes([80, 0x00])
        r = decode("GVH5075_1234", {0xEC88: data})
        self.assertEqual((r["temp_c"], r["humidity"]), (-5.2, 60.0))

    def test_h5074(self):
        data = bytes([0x00]) + (2150).to_bytes(2, "little", signed=True) + (4560).to_bytes(2, "little") + bytes([100, 0x02])
        r = decode("Govee_H5074_ABCD", {0xEC88: data})
        self.assertEqual((r["model"], r["temp_c"], r["humidity"], r["battery"]), ("H5074", 21.5, 45.6, 100))

    def test_h5102(self):
        packed = 233 * 1000 + 512
        data = bytes([0x01, 0x01]) + packed.to_bytes(3, "big") + bytes([97])
        r = decode("GVH5102_EEFF", {0x0001: data})
        self.assertEqual((r["temp_c"], r["humidity"], r["battery"]), (23.3, 51.2, 97))

    def test_non_govee_ignored(self):
        self.assertIsNone(decode("SomePhone", {0x004C: b"\x02\x15abc"}))

    def test_publisher_names_and_rate_limit(self):
        sent = []
        pub = Publisher(base_cfg("/tmp"), lambda t, p: sent.append((t, p)))
        packed = 215 * 1000 + 456
        data = {0xEC88: bytes([0]) + packed.to_bytes(3, "big") + bytes([100, 0])}
        self.assertEqual(pub.handle("a4:c1:38:12:34:56", "GVH5075_3456", data, now=0), "living_room")
        self.assertIsNone(pub.handle("a4:c1:38:12:34:56", "GVH5075_3456", data, now=10))
        self.assertEqual(pub.handle("A4:C1:38:00:AB:CD", "GVH5075_ABCD", data, now=10), "govee_h5075_abcd")
        self.assertEqual(sent[0][0], "home/temp/living_room")
        self.assertEqual(json.loads(sent[0][1])["temp_c"], 21.5)


# ---------------------------------------------------------------- messages

class MessageTests(unittest.TestCase):
    def test_parse_topics(self):
        self.assertEqual(messages.parse_topic("home", "home/audio/kitchen/clip"), ("audio", "kitchen"))
        self.assertEqual(messages.parse_topic("home", "home/temp/basement"), ("temp", "basement"))
        self.assertEqual(messages.parse_topic("home", "home/status/kitchen"), ("status", "kitchen"))
        self.assertEqual(messages.parse_topic("home", "other/temp/x"), (None, None))
        self.assertEqual(messages.parse_topic("home", "home/alerts"), (None, None))


# ---------------------------------------------------------------- YAMNet wrapper (fake interpreter)

def make_fake_interpreter(dynamic, glass_index, embeddings=False):
    class FakeInterpreter:
        calls = []

        def __init__(self, model_path):
            self.length = 15600

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            shape = np.array([1 if dynamic else 15600])
            sig = np.array([-1]) if dynamic else shape
            return [{"index": 0, "shape": shape, "shape_signature": sig}]

        def get_output_details(self):
            outs = [{"index": 1, "shape": np.array([1, 521])}]
            if embeddings:
                outs.append({"index": 2, "shape": np.array([1, 1024])})
            return outs

        def resize_tensor_input(self, index, shape):
            self.length = shape[0]

        def set_tensor(self, index, value):
            FakeInterpreter.calls.append(np.asarray(value).shape)
            self.loud = float(np.max(np.abs(value)))

        def invoke(self):
            pass

        def get_tensor(self, index):
            frames = 1 if not dynamic else max(1, self.length // 7680)
            if index == 1:
                s = np.full((frames, 521), 0.01, dtype=np.float32)
                s[:, 0] = 0.5                                   # "Speech"
                s[:, glass_index] = 0.9 if self.loud > 0.5 else 0.02
                return s
            return np.zeros((frames, 1024), dtype=np.float32)

    return FakeInterpreter


class YamnetWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cmap = Path(self.tmp.name) / "map.csv"
        with open(cmap, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "mid", "display_name"])
            for i in range(521):
                name = {0: "Speech", 435: "Glass", 437: "Shatter"}.get(i, f"Class {i}")
                w.writerow([i, f"/m/{i}", name])
        self.cfg = base_cfg(self.tmp.name)
        self.cfg["classifier"].update({"backend": "yamnet", "model_path": "x.tflite",
                                       "class_map_path": str(cmap)})
        self.original = clf._interpreter_class

    def tearDown(self):
        clf._interpreter_class = self.original
        self.tmp.cleanup()

    def run_case(self, dynamic):
        clf._interpreter_class = lambda: make_fake_interpreter(dynamic, glass_index=437)
        c = clf.make_classifier(self.cfg)
        quiet = c.classify(np.zeros(32000, dtype=np.float32))
        loud = np.zeros(32000, dtype=np.float32)
        loud[20000] = 0.9
        glassy = c.classify(loud)
        return c, quiet, glassy

    def test_fixed_input_windows_cover_clip(self):
        c, quiet, glassy = self.run_case(dynamic=False)
        self.assertFalse(quiet.is_glass)
        self.assertTrue(glassy.is_glass)          # glass found in a later window, not just the first
        self.assertEqual(glassy.top[0][0], "Shatter")
        self.assertAlmostEqual(glassy.glass_score, 0.9, places=5)

    def test_dynamic_input(self):
        c, quiet, glassy = self.run_case(dynamic=True)
        self.assertTrue(c.dynamic)
        self.assertTrue(glassy.is_glass)

    def test_missing_glass_classes_rejected(self):
        self.cfg["classifier"]["glass_classes"] = ["Nope"]
        clf._interpreter_class = lambda: make_fake_interpreter(False, 437)
        with self.assertRaises(ValueError):
            clf.make_classifier(self.cfg)


# ---------------------------------------------------------------- hub end to end

class HubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = base_cfg(self.tmp.name)
        self.storage = Storage(self.cfg["storage"]["database"], self.cfg["storage"]["clips_dir"])
        self.notifier = FakeNotifier()
        self.published = []
        self.hub = Hub(self.cfg, clf.make_classifier(self.cfg), self.storage, self.notifier,
                       publish=lambda t, p: self.published.append((t, p)), clock=lambda: 1000.0)

    def tearDown(self):
        self.storage.close()
        self.tmp.cleanup()

    def loud_clip(self):
        rng = np.random.default_rng(0)
        return encode_wav(rng.normal(0, 0.4, 16000))

    def test_loud_clips_from_two_mics_escalate(self):
        self.hub.handle("home/audio/kitchen/clip", self.loud_clip(), received_ts=1000)
        self.hub.handle("home/audio/living_room/clip", self.loud_clip(), received_ts=1000.4)
        levels = [a.level for a in self.notifier.sent]
        self.assertEqual(levels, ["alert", "high"])
        self.assertEqual(self.published[-1][0], "home/alerts")
        self.assertEqual(len(self.storage.recent("sound_events")), 2)
        self.assertEqual(len(self.storage.recent("alerts")), 2)
        saved = list(Path(self.cfg["storage"]["clips_dir"]).rglob("*.wav"))
        self.assertEqual(len(saved), 2)

    def test_quiet_clip_no_alert(self):
        self.hub.handle("home/audio/kitchen/clip", encode_wav(np.zeros(16000)), received_ts=1000)
        self.assertEqual(self.notifier.sent, [])
        self.assertEqual(self.storage.recent("sound_events")[0]["is_glass"], 0)

    def test_temperature_logged_and_freeze_alert(self):
        self.hub.handle("home/temp/basement", messages.temp_payload(f_to_c(40), 50, 90), received_ts=1000)
        self.assertEqual(self.notifier.sent[0].kind, "freeze")
        self.assertAlmostEqual(self.storage.recent("temps")[0]["temp_c"], f_to_c(40), places=2)

    def test_bad_payload_does_not_crash(self):
        self.hub.handle("home/temp/basement", b"not json", received_ts=1000)
        self.hub.handle("home/audio/kitchen/clip", b"not a wav", received_ts=1000)
        self.assertEqual(self.notifier.sent, [])

    def test_last_will_offline(self):
        self.hub.handle("home/status/kitchen", b'{"rssi": -50}', received_ts=1000)
        self.hub.handle("home/status/kitchen", b"offline", received_ts=1001)
        self.assertEqual(self.notifier.sent[-1].kind, "node_offline")


if __name__ == "__main__":
    unittest.main()
