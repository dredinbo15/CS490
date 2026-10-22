"""Phone notifications through ntfy (free; install the ntfy app and subscribe to your topic)."""

import logging
import urllib.request

log = logging.getLogger(__name__)

PRIORITY = {"info": "low", "warning": "default", "alert": "high", "high": "urgent"}
TAGS = {
    "glass": "rotating_light", "freeze": "snowflake", "overheat": "fire",
    "rapid_rise": "fire", "rapid_drop": "wind_face",
    "node_offline": "electric_plug", "node_online": "white_check_mark",
}
TITLES = {
    "glass": "Glass break", "freeze": "Freeze warning", "overheat": "Overheat",
    "rapid_rise": "Rapid temperature rise", "rapid_drop": "Rapid temperature drop",
    "node_offline": "Sensor offline", "node_online": "Sensor back online",
}


class Notifier:
    def __init__(self, cfg):
        n = cfg.get("notify", {})
        self.topic = (n.get("ntfy_topic") or "").strip()
        self.server = (n.get("ntfy_server") or "https://ntfy.sh").rstrip("/")
        if not self.topic:
            log.info("No ntfy topic set; alerts are logged but not pushed to your phone")

    def send(self, alert):
        log.warning("ALERT [%s/%s] %s", alert.level, alert.kind, alert.message)
        if not self.topic:
            return False
        title = TITLES.get(alert.kind, alert.kind)
        if alert.kind == "glass" and alert.level == "high":
            title = "Glass break - HIGH confidence"
        req = urllib.request.Request(
            f"{self.server}/{self.topic}",
            data=alert.message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": PRIORITY.get(alert.level, "default"),
                "Tags": TAGS.get(alert.kind, "house"),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:  # never let a notification failure stop the hub
            log.error("ntfy send failed: %s", exc)
            return False
