"""Loads config.yaml (or config.local.yaml when present)."""

from pathlib import Path

import yaml

SERVICE_DIR = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Return the config as a dict.

    Relative paths inside the config (database, models, clips) are resolved
    against the pi_service folder so the service works from any directory.
    """
    if path is None:
        local = SERVICE_DIR / "config.local.yaml"
        path = local if local.exists() else SERVICE_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def resolve(section, key):
        value = cfg.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(SERVICE_DIR / value)

    resolve("storage", "database")
    resolve("storage", "clips_dir")
    resolve("classifier", "model_path")
    resolve("classifier", "class_map_path")
    resolve("classifier", "head_model_path")
    return cfg
