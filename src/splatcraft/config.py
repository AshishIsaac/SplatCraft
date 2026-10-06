"""YAML configuration: ``configs/default.yaml`` overlaid with a preset (smoke | quick | full | a path)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .paths import CONFIG_DIR

PRESETS = ("smoke", "quick", "full")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(preset: str | None = "quick", overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    if preset:
        p = Path(preset)
        if not p.is_file():
            p = CONFIG_DIR / f"{preset}.yaml"
        if not p.is_file():
            raise FileNotFoundError(f"unknown config preset {preset!r} (choose from {', '.join(PRESETS)} or a path)")
        cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
        cfg["preset"] = p.stem
    return _merge(cfg, overrides or {})


def set_dotted(cfg: dict, assignments: list[str]) -> dict:
    """Apply ``train.iterations=7000``-style overrides from the command line."""
    for a in assignments or []:
        key, _, val = a.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
