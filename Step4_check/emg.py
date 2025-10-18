# c3dBox/Step4_check/emg.py
"""
EMG utilities for Cycle Checker UI.

Functions
---------
is_emg_key(k) -> bool
    True if a dict key represents an EMG channel (case-insensitive substring match).

process_emg(signal, fs_analog) -> np.ndarray
    Mean-remove -> 4th-order Butterworth band-pass (30–300 Hz, zero-phase)
    -> full-wave rectify -> 4th-order Butterworth low-pass (6 Hz, zero-phase).
    Returns the linear envelope with the same shape as input (1D).

process_emg_dict(analog_dict, fs_analog) -> dict[str, np.ndarray]
    Finds all EMG* channels in `analog_dict`, computes their envelopes,
    caches results to avoid redundant re-processing, and
    adds each processed array back into `analog_dict` under "<key>_proc".
    Returns {original_key: processed_array}.
"""

from __future__ import annotations

from typing import Dict, MutableMapping
import hashlib

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Default filter specs
_BP_LOW = 30.0     # Hz
_BP_HIGH = 300.0   # Hz
_ENV_LP = 6.0      # Hz
_ORDER = 4         # Butterworth order

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


def _design_bandpass(low_hz: float, high_hz: float, fs: float):
    nyq = fs * 0.5
    low = max(0.0, float(low_hz))
    high = max(0.0, float(high_hz))

    # Clamp to Nyquist with a small safety margin
    high = min(high, nyq * 0.99)
    if high <= 0:
        raise ValueError("High cutoff must be > 0 after clamping to Nyquist.")

    # Ensure low < high; if fs is small, degrade gracefully
    if low >= high:
        # If we cannot form a proper band, nudge low just below high
        low = max(1e-6, 0.5 * high)

    wn = [low / nyq, high / nyq]
    return butter(_ORDER, wn, btype="bandpass", output="sos")


def _design_lowpass(cut_hz: float, fs: float):
    nyq = fs * 0.5
    cut = max(1e-6, min(float(cut_hz), nyq * 0.99))
    wn = cut / nyq
    return butter(_ORDER, wn, btype="lowpass", output="sos")


def _safe_sosfiltfilt(sos, x: np.ndarray) -> np.ndarray:
    """
    sosfiltfilt with a conservative padlen to avoid errors on short signals.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("process_emg expects a 1D array per channel.")

    # Default padlen for filtfilt is 3 * (max(len(a), len(b)) - 1).
    # For SOS, use a small multiple of the number of sections.
    sections = sos.shape[0]
    default_pad = 3 * (2 * sections)  # rough, stable for most cases
    padlen = min(default_pad, max(0, x.size - 1))
    if padlen <= 0:
        # Too short to filter meaningfully; return zeros-like envelope
        return np.zeros_like(x)
    padlen = 0 # reset by HP
    return sosfiltfilt(sos, x, padlen=padlen)


def process_emg(signal: np.ndarray, fs_analog: float) -> np.ndarray:
    """
    Process a raw EMG channel to a linear envelope.

    Steps:
      1) Remove mean
      2) 4th-order Butterworth band-pass 30–300 Hz (zero-phase)
      3) Full-wave rectify
      4) 4th-order Butterworth low-pass 6 Hz (zero-phase)

    Parameters
    ----------
    signal : np.ndarray
        1D signal.
    fs_analog : float
        Sampling frequency [Hz] of the analog data.

    Returns
    -------
    np.ndarray
        Processed EMG (linear envelope), same length as `signal`.
    """
    if fs_analog is None or fs_analog <= 0:
        raise ValueError("fs_analog must be a positive float.")

    x = np.asarray(signal, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("process_emg expects a 1D array per channel.")

    # Replace non-finite values to keep filters happy
    if not np.isfinite(x).all():
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            return np.zeros_like(x)
        # Simple in-place interpolation of NaNs
        idx = np.arange(x.size)
        x[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], x[finite_mask])

    # 1) Mean remove (use nanmean equivalently now that we removed NaNs)
    x = x - float(x.mean())

    # 2) Band-pass
    sos_bp = _design_bandpass(_BP_LOW, _BP_HIGH, fs_analog)
    x_bp = _safe_sosfiltfilt(sos_bp, x)

    # 3) Rectify
    x_rect = np.abs(x_bp)

    # 4) Low-pass to get the envelope
    sos_lp = _design_lowpass(_ENV_LP, fs_analog)
    env = _safe_sosfiltfilt(sos_lp, x_rect)
    
    # 5) Normalize per cycle
    env = env / np.max(env) * 100

    return env


def process_emg_dict(analog_dict: MutableMapping, fs_analog: float) -> Dict[str, np.ndarray]:
    """
    Process and cache all EMG channels found in `analog_dict`.

    - Detects EMG keys via `is_emg_key`.
    - Computes envelopes with `process_emg`.
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
