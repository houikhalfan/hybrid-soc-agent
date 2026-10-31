"""
Single entry point for reading config.yaml. Every module on both the
host side and the sensor side (sensor_agent/*) should import
`get_config()` from here rather than opening config.yaml itself or
hardcoding a network/path/threshold value.

Usage:
    from config_loader import get_config
    cfg = get_config()
    cfg.rules.port_scan.distinct_port_threshold
    cfg["rules"]["port_scan"]["distinct_port_threshold"]   # also works

Resolution order for the config file path:
    1. SOC_CONFIG_PATH environment variable, if set.
    2. config.yaml next to this file (repo root) — works for both the
       host checkout and a sensor deployment that copies the repo root
       (config.yaml, config_loader.py, schema.py) alongside sensor_agent/.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_ENV_VAR = "SOC_CONFIG_PATH"

_lock = threading.Lock()
_cached_config: "ConfigNode | None" = None
_cached_path: Path | None = None


class ConfigNode:
    """
    Thin read-only wrapper around a dict so nested config can be
    accessed either as attributes (cfg.rules.port_scan.x) or like a
    plain dict (cfg["rules"]["port_scan"]["x"]). Lists of scalars are
    left as plain lists; lists of dicts are wrapped recursively too.
    """

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._wrap(self._data[name])
        except KeyError as exc:
            raise AttributeError(
                f"config has no key '{name}' (available: {sorted(self._data.keys())})"
            ) from exc

    def __getitem__(self, name: str) -> Any:
        return self._wrap(self._data[name])

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __repr__(self) -> str:
        return f"ConfigNode({self._data!r})"

    def to_dict(self) -> dict:
        return self._data

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [ConfigNode(v) if isinstance(v, dict) else v for v in value]
        return value


def _resolve_path() -> Path:
    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _DEFAULT_CONFIG_PATH


def get_config(force_reload: bool = False) -> ConfigNode:
    """
    Returns the parsed config.yaml as a ConfigNode, cached after the
    first call. Pass force_reload=True to re-read the file (useful in
    tests, or after editing config.yaml during a long-running process).
    """
    global _cached_config, _cached_path

    path = _resolve_path()

    with _lock:
        if _cached_config is not None and not force_reload and path == _cached_path:
            return _cached_config

        if not path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {path} "
                f"(set {_ENV_VAR} to override the default location)"
            )

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"{path} did not parse to a mapping — check its contents")

        _cached_config = ConfigNode(raw)
        _cached_path = path
        return _cached_config


def reload_config() -> ConfigNode:
    """Convenience alias for get_config(force_reload=True)."""
    return get_config(force_reload=True)
