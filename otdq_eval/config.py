"""Configuration helpers for fixed OTDQ-Eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_OTDQ_CONFIG: Dict[str, Any] = {
    "min_size": 15,
    "n_q": 1000,
    "var_thresh": 1e-4,
    "flat_alpha": 5.0,
    "window_size": 15,
    "texture_pct": 40.0,
    "halo_pct": 90.0,
    "patch_window": 32,
    "resize_method": "LANCZOS",
}


OTDQ_PARAMETER_KEYS = {
    "min_size",
    "n_q",
    "var_thresh",
    "flat_alpha",
    "window_size",
    "texture_pct",
    "halo_pct",
    "patch_window",
}


def load_json_config(path: str | Path | None) -> Dict[str, Any]:
    """Load a JSON config file. Missing paths return an empty config."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")
    return data


def merged_otdq_config(path: str | Path | None = None, **overrides: Any) -> Dict[str, Any]:
    """Return default OTDQ settings updated by a JSON file and CLI overrides."""
    config = dict(DEFAULT_OTDQ_CONFIG)
    loaded = load_json_config(path)
    config.update(loaded.get("otdq", loaded))
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def otdq_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract arguments accepted by ``compute_otdq``."""
    return {key: config[key] for key in OTDQ_PARAMETER_KEYS if key in config}
