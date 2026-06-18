# c3dBox/Step4_check/emg.py
"""
EMG utilities for Cycle Checker UI.

Functions
---------
is_emg_key(k) -> bool
    True if a dict key represents an EMG channel (case-insensitive substring match).

process_emg(signal, fs_analog) -> np.ndarray
    Amplitude-normalize the raw signal using max(signal).
    Returns a 1D array with the same shape as input.

process_emg_dict(analog_dict, fs_analog) -> dict[str, np.ndarray]
    Finds all EMG* channels in `analog_dict`, computes amplitude-normalized signals,
    caches results to avoid redundant re-processing, and
    adds each processed array back into `analog_dict` under "<key>_proc".
    Returns {original_key: processed_array}.
"""

from __future__ import annotations

from typing import Dict, MutableMapping
import hashlib

import numpy as np
# Name of the per-dict cache bucket
_CACHE_KEY = "_emg_cache"


def is_emg_key(k: str) -> bool:
    """Return True if key `k` looks like an EMG channel (case-insensitive 'emg' substring)."""
    return isinstance(k, str) and ("emg" in k.lower())


def _array_signature(x: np.ndarray) -> str:
    """
    Create a lightweight signature for caching.
    Uses shape, dtype, and a SHA1 digest of the raw bytes.
    """
    x = np.asarray(x)
    h = hashlib.sha1()
    # For speed, handle small and medium arrays directly;
    # for very large arrays, hashing is still fast enough at typical EMG sizes.
    h.update(x.tobytes(order="C"))
    return f"{x.shape}|{x.dtype.str}|{h.hexdigest()}"


def process_emg(signal: np.ndarray, fs_analog: float) -> np.ndarray:
    """
    Amplitude-normalize a raw EMG channel.

    Parameters
    ----------
    signal : np.ndarray
        1D signal.
    fs_analog : float
        Sampling frequency [Hz] of the analog data. (Unused here, kept for API compatibility.)

    Returns
    -------
    np.ndarray
        Amplitude-normalized EMG, same length as `signal`.
    """
    if fs_analog is None or fs_analog <= 0:
        raise ValueError("fs_analog must be a positive float.")

    x = np.asarray(signal, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("process_emg expects a 1D array per channel.")

    # Replace non-finite values to keep normalization stable
    if not np.isfinite(x).all():
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            return np.zeros_like(x)
        # Simple in-place interpolation of NaNs
        idx = np.arange(x.size)
        x[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], x[finite_mask])

    # Amplitude normalize using max(signal), per instructions.
    max_val = float(np.max(x)) if x.size else 0.0
    if max_val == 0.0:
        return np.zeros_like(x)
    return (x / max_val) * 100.0


def process_emg_dict(analog_dict: MutableMapping, fs_analog: float) -> Dict[str, np.ndarray]:
    """
    Process and cache all EMG channels found in `analog_dict`.

    - Detects EMG keys via `is_emg_key`.
    - Computes amplitude-normalized signals with `process_emg`.
    - Caches results in `analog_dict[_emg_cache]` using a content hash and fs_analog.
    - Writes processed arrays back into `analog_dict` under "<key>_proc".

    Parameters
    ----------
    analog_dict : MutableMapping
        Dictionary that contains analog signals (e.g., from the .mat structure).
        It will be updated in-place with processed keys.
    fs_analog : float
        Analog sampling rate in Hz.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of original EMG keys -> processed arrays.
    """
    if fs_analog is None or fs_analog <= 0:
        raise ValueError("fs_analog must be a positive float.")

    # Ensure cache bucket exists
    cache = analog_dict.setdefault(_CACHE_KEY, {})  # type: ignore[assignment]

    out: Dict[str, np.ndarray] = {}

    # Iterate stable over keys snapshot (since we modify the dict by adding *_proc)
    for key in list(analog_dict.keys()):
        if not is_emg_key(key):
            continue

        x = analog_dict.get(key, None)
        if x is None:
            continue

        arr = np.asarray(x)
        if arr.ndim != 1:
            # Only process 1D analog channels
            continue

        sig = _array_signature(arr)
        cache_entry = cache.get(key)

        if (
            isinstance(cache_entry, dict)
            and cache_entry.get("fs") == float(fs_analog)
            and cache_entry.get("sig") == sig
            and isinstance(cache_entry.get("data"), np.ndarray)
            and cache_entry["data"].shape == arr.shape
        ):
            env = cache_entry["data"]
        else:
            env = process_emg(arr, fs_analog)
            cache[key] = {"fs": float(fs_analog), "sig": sig, "data": env}

        # Store alongside original with a clear suffix
        proc_key = f"{key}_proc"
        analog_dict[proc_key] = env
        out[key] = env

    return out


if __name__ == "__main__":
    # Tiny self-check (not exhaustive):
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    # Fake EMG: narrow-band noise around 100 Hz plus DC
    rng = np.random.default_rng(0)
    raw = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.05 * rng.standard_normal(t.size) + 0.1
    env = process_emg(raw, fs)

    # Quick smoke usage for dict
    ad = {"EMG_01_R_M_tibialis_anterior": raw.copy(), "time": t}
    out_map = process_emg_dict(ad, fs)
    assert "EMG_01_R_M_tibialis_anterior" in out_map
    assert "EMG_01_R_M_tibialis_anterior_proc" in ad
