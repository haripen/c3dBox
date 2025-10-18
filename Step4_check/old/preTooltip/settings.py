
"""
c3dBox.Step4_check.settings

Simple settings manager backed by JSON with minimal schema validation.

Public API:
- load() -> dict
- get(path: str, default: Any | _MISSING = _MISSING) -> Any
- update_and_save(updates: dict[str, Any] | dict[str, dict[str, Any]]) -> dict
- reset_to_defaults() -> dict

Notes:
- `get` uses dotted paths, e.g. get("ik.marker_error_RMS_max")
- `update_and_save` accepts dotted keys or nested dicts (or a mix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# ----- Defaults (kept in code for reset) -----
DEFAULTS: Dict[str, Any] = {
    "ik": {"marker_error_RMS_max": 0.002, "marker_error_max_max": 0.004},
    "so": {
        "force_rms_max": 10,
        "force_max_max": 25,
        "moment_rms_max": 50,
        "moment_max_max": 75,
        "so_frames_tol": 10,
    },
    "ui": {"start_fullscreen": True, "ylim_margin_pct": 0.05},
}

# Path to the json next to this file
SETTINGS_PATH = Path(__file__).with_name("settings.json")

# Module-level cache
_CACHE: Dict[str, Any] | None = None

# Simple schema for minimal validation
_NUMBER = (int, float)
_SCHEMA: Dict[str, Dict[str, Any]] = {
    "ik": {
        "marker_error_RMS_max": {"type": _NUMBER, "check": lambda v: v >= 0},
        "marker_error_max_max": {"type": _NUMBER, "check": lambda v: v >= 0},
    },
    "so": {
        "force_rms_max": {"type": _NUMBER, "check": lambda v: v >= 0},
        "force_max_max": {"type": _NUMBER, "check": lambda v: v >= 0},
        "moment_rms_max": {"type": _NUMBER, "check": lambda v: v >= 0},
        "moment_max_max": {"type": _NUMBER, "check": lambda v: v >= 0},
        "so_frames_tol": {"type": int, "check": lambda v: v >= 0},
    },
    "ui": {
        "start_fullscreen": {"type": bool, "check": lambda v: isinstance(v, bool)},
        "ylim_margin_pct": {"type": _NUMBER, "check": lambda v: 0 <= v < 1},
    },
}

class _Missing:
    pass
_MISSING = _Missing()

def _deepcopy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(d))

def _validate(data: Dict[str, Any]) -> None:
    """Minimal schema & range/type validation. Raises ValueError with clear messages."""
    # Top-level keys
    expected_top = set(_SCHEMA.keys())
    found_top = set(data.keys())
    missing = expected_top - found_top
    unknown = found_top - expected_top
    if missing:
        raise ValueError(f"settings.json is missing top-level section(s): {sorted(missing)}")
    if unknown:
        raise ValueError(f"settings.json has unknown top-level section(s): {sorted(unknown)}")

    # Per-section fields
    for section, fields in _SCHEMA.items():
        sec = data.get(section, {})
        expected_fields = set(fields.keys())
        found_fields = set(sec.keys())
        miss = expected_fields - found_fields
        unk = found_fields - expected_fields
        if miss:
            raise ValueError(f"Section '{section}' missing field(s): {sorted(miss)}")
        if unk:
            raise ValueError(f"Section '{section}' has unknown field(s): {sorted(unk)}")

        # Type & simple range checks
        for key, rule in fields.items():
            val = sec[key]
            typ = rule["type"]
            if not isinstance(val, typ):
                # Accept JSON numbers that load as int for float fields (int is instance of int only).
                if typ is _NUMBER and isinstance(val, (int, float)):
                    pass
                else:
                    raise ValueError(f"Invalid type for '{section}.{key}': expected {typ}, got {type(val).__name__}")

            chk = rule.get("check")
            if chk and not chk(val):
                raise ValueError(f"Invalid value for '{section}.{key}': failed validation check (value={val!r})")

    # Cross-field consistency checks (minimal but helpful)
    if data["ik"]["marker_error_max_max"] < data["ik"]["marker_error_RMS_max"]:
        raise ValueError("ik.marker_error_max_max must be >= ik.marker_error_RMS_max")

    if data["so"]["force_max_max"] < data["so"]["force_rms_max"]:
        raise ValueError("so.force_max_max must be >= so.force_rms_max")

    if data["so"]["moment_max_max"] < data["so"]["moment_rms_max"]:
        raise ValueError("so.moment_max_max must be >= so.moment_rms_max")


def _load_from_disk() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(json.dumps(DEFAULTS, indent=2, ensure_ascii=False), encoding="utf-8")
        return _deepcopy_dict(DEFAULTS)

    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {SETTINGS_PATH.name} as JSON: {e}") from e

    _validate(data)
    return data


def _save_to_disk(data: Dict[str, Any]) -> None:
    _validate(data)  # validate before write
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load() -> Dict[str, Any]:
    """
    Load settings from disk (or defaults if the file doesn't exist), validate, and cache them.
    Returns a deep copy of the settings dict.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_from_disk()
    return _deepcopy_dict(_CACHE)


def _get_from_dict(d: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            if default is _MISSING:
                raise KeyError(f"Unknown settings key path: '{path}'")
            return default
    return cur


def get(path: str, default: Any | _Missing = _MISSING) -> Any:
    """
    Retrieve a value using a dotted path, e.g. get("ui.ylim_margin_pct").

    If `default` is not provided and the path does not exist, KeyError is raised.
    """
    data = load()
    return _get_from_dict(data, path, default)


def _set_in_dict(d: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    *parents, last = list(path)
    cur = d
    for p in parents:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[last] = value


def _merge_nested(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow+deep merge where dict values are merged recursively."""
    out = _deepcopy_dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_nested(out[k], v)
        else:
            out[k] = v
    return out


def update_and_save(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update settings with either:
      - dotted keys, e.g. {"ui.ylim_margin_pct": 0.08, "so.force_rms_max": 12}
      - nested dicts, e.g. {"ui": {"ylim_margin_pct": 0.08}}

    Unknown keys or invalid values raise ValueError.
    Returns the updated settings dict.
    """
    if not isinstance(updates, dict) or not updates:
        raise ValueError("updates must be a non-empty dict")

    current = load()
    candidate = _deepcopy_dict(current)

    dotted, nested = {}, {}
    for k, v in updates.items():
        if isinstance(k, str) and "." in k:
            dotted[k] = v
        else:
            nested[k] = v

    if nested:
        candidate = _merge_nested(candidate, nested)

    for path, val in dotted.items():
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid dotted key '{path}'; expected at least 'section.field'")
        _set_in_dict(candidate, parts, val)

    # Validate & save
    _save_to_disk(candidate)

    # Update cache
    global _CACHE
    _CACHE = _deepcopy_dict(candidate)
    return _deepcopy_dict(candidate)


def reset_to_defaults() -> Dict[str, Any]:
    """Overwrite the JSON file with DEFAULTS and refresh cache. Returns the defaults."""
    _save_to_disk(_deepcopy_dict(DEFAULTS))
    global _CACHE
    _CACHE = _deepcopy_dict(DEFAULTS)
    return _deepcopy_dict(DEFAULTS)


__all__ = ["load", "get", "update_and_save", "reset_to_defaults", "DEFAULTS", "SETTINGS_PATH"]

# ----- Compatibility adapter for code expecting a Settings class -----
# Lets existing code `from .settings import Settings` keep working.
import sys as _sys
_settings_mod = _sys.modules[__name__]

class Settings:
    @staticmethod
    def load():
        # Return an object with .get/.update_and_save/.reset_to_defaults
        return Settings

    @staticmethod
    def get(section: str, key: str | None = None, default=_MISSING):
        path = section if key is None else f"{section}.{key}"
        try:
            return _settings_mod.get(path, default)
        except KeyError:
            # Fall back to default if provided
            if default is not _MISSING:
                return default
            raise

    @staticmethod
    def update_and_save(updates: dict):
        # Pass through; function API accepts nested dicts or dotted keys
        return _settings_mod.update_and_save(updates)

    @staticmethod
    def reset_to_defaults():
        return _settings_mod.reset_to_defaults()

# Optional: export in __all__ for clarity (not required for direct import)
try:
    __all__.append("Settings")  # type: ignore[name-defined]
except Exception:
    pass
