"""SQLite logging for temperatures, sound events and alerts, plus clip saving."""

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS temps (
    ts REAL NOT NULL, sensor TEXT NOT NULL,
    temp_c REAL NOT NULL, humidity REAL, battery INTEGER
);
CREATE INDEX IF NOT EXISTS temps_sensor_ts ON temps(sensor, ts);

CREATE TABLE IF NOT EXISTS sound_events (
    ts REAL NOT NULL, node TEXT NOT NULL, label TEXT, glass_score REAL,
    is_glass INTEGER, source TEXT, top_labels TEXT, loudness_dbfs REAL,
    clip_path TEXT, inference_ms REAL
);
CREATE INDEX IF NOT EXISTS sound_node_ts ON sound_events(node, ts);

CREATE TABLE IF NOT EXISTS alerts (
    ts REAL NOT NULL, kind TEXT NOT NULL, level TEXT NOT NULL,
    message TEXT NOT NULL, nodes TEXT
);
"""


class Storage:
    def __init__(self, db_path, clips_dir=None):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.clips_dir = Path(clips_dir) if clips_dir else None

    def _write(self, sql, params):
        with self.lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def add_temp(self, ts, sensor, temp_c, humidity=None, battery=None):
        self._write("INSERT INTO temps VALUES (?,?,?,?,?)",
                    (ts, sensor, temp_c, humidity, battery))

    def add_sound_event(self, ts, node, result, loudness_dbfs, clip_path, inference_ms):
        self._write("INSERT INTO sound_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ts, node, result.label, result.glass_score, int(result.is_glass),
                     result.source, json.dumps(result.top), loudness_dbfs,
                     clip_path, inference_ms))

    def add_alert(self, alert):
        self._write("INSERT INTO alerts VALUES (?,?,?,?,?)",
                    (alert.ts, alert.kind, alert.level, alert.message, json.dumps(alert.nodes)))

    def save_clip(self, node, wav_bytes, ts=None):
        """Save the raw WAV under clips/<date>/<node>_<time>.wav and return its path."""
        if self.clips_dir is None:
            return None
        ts = ts or time.time()
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        stamp = time.strftime("%H%M%S", time.localtime(ts)) + f"_{int((ts % 1) * 1000):03d}"
        folder = self.clips_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{node}_{stamp}.wav"
        path.write_bytes(wav_bytes)
        return str(path)

    def recent(self, table, limit=20):
        if table not in ("temps", "sound_events", "alerts"):
            raise ValueError(table)
        with self.lock:
            cur = self.conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (limit,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def temps_since(self, sensor, since_ts):
        with self.lock:
            cur = self.conn.execute(
                "SELECT ts, temp_c, humidity FROM temps WHERE sensor=? AND ts>=? ORDER BY ts",
                (sensor, since_ts))
            return cur.fetchall()

    def close(self):
        with self.lock:
            self.conn.close()
