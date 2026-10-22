"""Alert rules. Pure logic with no I/O, so every rule is unit-testable.

Every method takes the current time `now` (seconds) as an argument instead of
reading the clock, which lets the tests replay any timeline.
"""

import json
from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Alert:
    kind: str            # glass | freeze | overheat | rapid_rise | rapid_drop | node_offline | node_online
    level: str           # info | warning | alert | high
    message: str
    ts: float
    nodes: list = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self))


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


class Cooldown:
    """Allows an event key at most once per `seconds`."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.last = {}

    def ready(self, key, now):
        prev = self.last.get(key)
        if prev is not None and now - prev < self.seconds:
            return False
        self.last[key] = now
        return True


class GlassFusion:
    """One mic hearing glass = alert. Two or more different mics within the window = high."""

    def __init__(self, window_s=2.0, cooldown_s=30.0):
        self.window_s = window_s
        self.recent = deque()  # (ts, node, score)
        self.cooldown = Cooldown(cooldown_s)

    def detection(self, node, score, now):
        self.recent.append((now, node, score))
        while self.recent and now - self.recent[0][0] > self.window_s:
            self.recent.popleft()

        nodes = sorted({n for _, n, _ in self.recent})
        if len(nodes) >= 2:
            if self.cooldown.ready("high", now):
                return Alert("glass", "high",
                             f"Glass break heard by {len(nodes)} mics: {', '.join(nodes)}",
                             now, nodes)
            return None
        if self.cooldown.ready(f"single:{node}", now):
            return Alert("glass", "alert",
                         f"Possible glass break near {node} (score {score:.2f})",
                         now, [node])
        return None


class TempRules:
    """Freeze, overheat and rapid-change rules per sensor. Thresholds are in the configured units."""

    def __init__(self, units="F", freeze_below=45, overheat_above=95,
                 rapid_change=5, rapid_window_s=300, cooldown_s=1800):
        self.units = units.upper()
        to_c = f_to_c if self.units == "F" else (lambda v: v)
        self.freeze_c = to_c(freeze_below)
        self.overheat_c = to_c(overheat_above)
        # A temperature *difference* converts by the scale factor only (no offset).
        self.rapid_c = rapid_change * 5.0 / 9.0 if self.units == "F" else rapid_change
        self.rapid_window_s = rapid_window_s
        self.history = {}  # sensor -> deque[(ts, temp_c)]
        self.cooldown = Cooldown(cooldown_s)

    def fmt(self, temp_c):
        if self.units == "F":
            return f"{c_to_f(temp_c):.1f}°F"
        return f"{temp_c:.1f}°C"

    def reading(self, sensor, temp_c, now):
        alerts = []
        hist = self.history.setdefault(sensor, deque())
        hist.append((now, temp_c))
        while hist and now - hist[0][0] > self.rapid_window_s:
            hist.popleft()

        if temp_c <= self.freeze_c and self.cooldown.ready((sensor, "freeze"), now):
            alerts.append(Alert("freeze", "warning",
                                f"{sensor} is cold: {self.fmt(temp_c)}", now, [sensor]))
        if temp_c >= self.overheat_c and self.cooldown.ready((sensor, "overheat"), now):
            alerts.append(Alert("overheat", "alert",
                                f"{sensor} is hot: {self.fmt(temp_c)}", now, [sensor]))

        if len(hist) >= 2:
            lowest = min(t for _, t in hist)
            highest = max(t for _, t in hist)
            minutes = self.rapid_window_s / 60
            if temp_c - lowest >= self.rapid_c and self.cooldown.ready((sensor, "rise"), now):
                alerts.append(Alert("rapid_rise", "alert",
                                    f"{sensor} rose to {self.fmt(temp_c)} from {self.fmt(lowest)} "
                                    f"within {minutes:.0f} min", now, [sensor]))
            if highest - temp_c >= self.rapid_c and self.cooldown.ready((sensor, "drop"), now):
                alerts.append(Alert("rapid_drop", "warning",
                                    f"{sensor} dropped to {self.fmt(temp_c)} from {self.fmt(highest)} "
                                    f"within {minutes:.0f} min (window open?)", now, [sensor]))
        return alerts


class NodeWatchdog:
    """Tracks when each node was last heard from and reports offline/online changes."""

    def __init__(self, offline_after_s=120):
        self.offline_after_s = offline_after_s
        self.last_seen = {}
        self.offline = set()

    def seen(self, node, now):
        self.last_seen[node] = now
        if node in self.offline:
            self.offline.discard(node)
            return Alert("node_online", "info", f"{node} is back online", now, [node])
        return None

    def went_offline(self, node, now):
        """The node said goodbye itself (MQTT last-will "offline" message)."""
        if node not in self.offline:
            self.offline.add(node)
            return Alert("node_offline", "warning", f"{node} went offline", now, [node])
        return None

    def check(self, now):
        alerts = []
        for node, seen in self.last_seen.items():
            if node not in self.offline and now - seen > self.offline_after_s:
                self.offline.add(node)
                alerts.append(Alert("node_offline", "warning",
                                    f"{node} silent for {now - seen:.0f} s", now, [node]))
        return alerts

    def status(self, now):
        return {node: {"last_seen": seen, "online": node not in self.offline}
                for node, seen in self.last_seen.items()}
