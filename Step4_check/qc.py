"""Quality control helpers for IK and SO results.

This module evaluates per-cycle OpenSim outputs for inverse kinematics (IK)
marker errors and static optimization (SO) residuals. It adds boolean flags
onto each cycle dict, treating missing data as *pass* (True).

Flags added when the respective data exist (and True if data missing):
- "IK_RMS_ok":     max_t(marker_error_RMS) <= settings.ik.marker_error_RMS_max
- "IK_MAX_ok":     max_t(marker_error_max) <= settings.ik.marker_error_max_max
- "SO_F_RMS_ok":   RMS(FX,FY,FZ in SLS window) <= settings.so.force_rms_max
- "SO_F_MAX_ok":   MAX(|FX|,|FY|,|FZ| in SLS window) <= settings.so.force_max_max
- "SO_M_RMS_ok":   RMS(MX,MY,MZ in SLS window) <= settings.so.moment_rms_max
- "SO_M_MAX_ok":   MAX(|MX|,|MY|,|MZ| in SLS window) <= settings.so.moment_max_max

SLS window = contralateral Foot_Off → contralateral Foot_Strike mapped to
SO_forces["time"]. We skip n_skip frames at both window ends where
n_skip = int(points_fs / so_frames_tol), with points_fs = meta['header']['points']['frame_rate'].

If any required arrays are missing or empty, the corresponding flag is set to True
("pass") rather than failing the cycle.

Usage:
    from .qc import qc_cycle, qc_all

"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import math

import numpy as np

try:
    # settings.get(key_path, default=None)
    from .settings import get as settings_get
except Exception:  # pragma: no cover
    settings_get = None  # Fallback later


# ------------------------------
# Internal helpers
# ------------------------------

def _get_setting(path: str, default: Any) -> Any:
    """Fetch a setting value from settings.py if available, otherwise fallback.

    Parameters
    ----------
    path : str
        Dotted path key, e.g., 'ik.marker_error_RMS_max'.
    default : Any
        Default to return if settings are unavailable or the key is missing.
    """
    if settings_get is None:
        return default
    try:
        val = settings_get(path)
    except Exception:
        return default
    return default if val is None else val


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely walk nested dicts: _safe_get(d, 'a', 'b', default=...) -> d['a']['b'] or default."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_array_like(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _rms(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr, dtype=float))))


def _max_abs(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _contralateral_events(side: str) -> Tuple[str, str]:
    """Return (Foot_Off_key, Foot_Strike_key) to use as contralateral events.

    Parameters
    ----------
    side : {'left_stride','right_stride'}
        The *indexing* side of the cycle dict.
    """
    if side == 'left_stride':
        return 'Right_Foot_Off', 'Right_Foot_Strike'
    else:
        return 'Left_Foot_Off', 'Left_Foot_Strike'


def _window_indices_from_times(
    t_series: np.ndarray,
    t0: Optional[float],
    t1: Optional[float],
    n_skip: int,
) -> Optional[Tuple[int, int]]:
    """Map [t0, t1] to (i0, i1) inclusive indices on t_series and crop by n_skip.

    Returns None if mapping fails or window is invalid after cropping.
    """
    if t_series is None or len(t_series) == 0 or t0 is None or t1 is None:
        return None
    if not (np.isfinite(t0) and np.isfinite(t1)):
        return None
    if t1 <= t0:
        return None

    # Find first idx >= t0 and last idx <= t1
    # Assumption: t_series is non-decreasing
    i0 = int(np.searchsorted(t_series, t0, side='left'))
    i1 = int(np.searchsorted(t_series, t1, side='right')) - 1

    # Bound and crop
    i0 = max(0, i0 + max(0, int(n_skip)))
    i1 = min(len(t_series) - 1, i1 - max(0, int(n_skip)))

    if i1 < i0:
        return None
    return i0, i1


# ------------------------------
# Public evaluation functions
# ------------------------------

def eval_ik_flags(cycle: Dict[str, Any]) -> Dict[str, bool]:
    """Evaluate IK marker error flags for a single cycle.

    Parameters
    ----------
    cycle : dict
        Cycle dict which may contain 'IK_markerErr' sub-dict with time-discrete
        arrays 'marker_error_RMS' and 'marker_error_max'.

    Returns
    -------
    dict
        {'IK_RMS_ok': bool, 'IK_MAX_ok': bool}
        Missing data -> both True.
    """
    rms_thr = float(_get_setting('ik.marker_error_RMS_max', 0.002))
    max_thr = float(_get_setting('ik.marker_error_max_max', 0.004))

    ikm = _safe_get(cycle, 'IK_markerErr', default=None)
    if not isinstance(ikm, dict):
        return {'IK_RMS_ok': True, 'IK_MAX_ok': True}

    rms = _safe_get(ikm, 'marker_error_RMS', default=None)
    mxx = _safe_get(ikm, 'marker_error_max', default=None)

    if not _is_array_like(rms) or not _is_array_like(mxx):
        return {'IK_RMS_ok': True, 'IK_MAX_ok': True}

    try:
        rms_max_over_time = float(np.max(np.asarray(rms, dtype=float)))
        max_max_over_time = float(np.max(np.asarray(mxx, dtype=float)))
    except Exception:
        return {'IK_RMS_ok': True, 'IK_MAX_ok': True}

    return {
        'IK_RMS_ok': rms_max_over_time <= rms_thr,
        'IK_MAX_ok': max_max_over_time <= max_thr,
    }


def eval_so_flags(
    cycle: Dict[str, Any],
    side: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Evaluate SO residual force/moment flags for a single cycle.

    The evaluation is restricted to single-limb stance (contralateral Foot_Off
    → contralateral Foot_Strike) mapped into SO_forces["time"]. We skip a fixed
    number of frames at both window ends: n_skip = int(points_fs / so_frames_tol).

    Parameters
    ----------
    cycle : dict
        Cycle dict with sub-dicts: '<Contra>_Foot_Off', '<Contra>_Foot_Strike',
        and 'SO_forces' containing 'time', 'FX','FY','FZ','MX','MY','MZ'.
    side : {'left_stride','right_stride'}
        Which stride this cycle belongs to (determines contralateral events).
    meta : dict, optional
        Root 'meta' dict to obtain points frame rate at ['header']['points']['frame_rate'].

    Returns
    -------
    dict
        {'SO_F_RMS_ok','SO_F_MAX_ok','SO_M_RMS_ok','SO_M_MAX_ok'} booleans.
        Missing/invalid data -> all True.
    """
    # Thresholds
    f_rms_thr = float(_get_setting('so.force_rms_max', 10.0))
    f_max_thr = float(_get_setting('so.force_max_max', 25.0))
    m_rms_thr = float(_get_setting('so.moment_rms_max', 50.0))
    m_max_thr = float(_get_setting('so.moment_max_max', 75.0))
    so_frames_tol = float(_get_setting('so.so_frames_tol', 10.0))

    # Sampling: points fs for skip calculation
    points_fs = float(_safe_get(meta or {}, 'header', 'points', 'frame_rate', default=0.0) or 0.0)
    n_skip = int(points_fs / so_frames_tol) if so_frames_tol > 0 else 0

    # Access contralateral events
    off_key, str_key = _contralateral_events(side)
    t_off = _safe_get(cycle, off_key, 'time', default=None)
    t_str = _safe_get(cycle, str_key, 'time', default=None)

    so = _safe_get(cycle, 'SO_forces', default=None)
    if not isinstance(so, dict):
        return {
            'SO_F_RMS_ok': True,
            'SO_F_MAX_ok': True,
            'SO_M_RMS_ok': True,
            'SO_M_MAX_ok': True,
        }

    t_series = _safe_get(so, 'time', default=None)
    FX = _safe_get(so, 'FX', default=None)
    FY = _safe_get(so, 'FY', default=None)
    FZ = _safe_get(so, 'FZ', default=None)
    MX = _safe_get(so, 'MX', default=None)
    MY = _safe_get(so, 'MY', default=None)
    MZ = _safe_get(so, 'MZ', default=None)

    # Validate availability
    arrays_ok = all(_is_array_like(x) for x in (t_series, FX, FY, FZ, MX, MY, MZ))
    if not arrays_ok:
        return {
            'SO_F_RMS_ok': True,
            'SO_F_MAX_ok': True,
            'SO_M_RMS_ok': True,
            'SO_M_MAX_ok': True,
        }

    t_series = np.asarray(t_series, dtype=float)
    FX = np.asarray(FX, dtype=float)
    FY = np.asarray(FY, dtype=float)
    FZ = np.asarray(FZ, dtype=float)
    MX = np.asarray(MX, dtype=float)
    MY = np.asarray(MY, dtype=float)
    MZ = np.asarray(MZ, dtype=float)

    if t_series.size == 0:
        return {
            'SO_F_RMS_ok': True,
            'SO_F_MAX_ok': True,
            'SO_M_RMS_ok': True,
            'SO_M_MAX_ok': True,
        }

    # Map window and crop by n_skip
    win = _window_indices_from_times(t_series, t_off, t_str, n_skip)
    if win is None:
        return {
            'SO_F_RMS_ok': True,
            'SO_F_MAX_ok': True,
            'SO_M_RMS_ok': True,
            'SO_M_MAX_ok': True,
        }

    i0, i1 = win
    sl = slice(i0, i1 + 1)

    # Compute per-component RMS and MAX within window
    f_rms = max(_rms(FX[sl]), _rms(FY[sl]), _rms(FZ[sl]))
    f_max = max(_max_abs(FX[sl]), _max_abs(FY[sl]), _max_abs(FZ[sl]))

    m_rms = max(_rms(MX[sl]), _rms(MY[sl]), _rms(MZ[sl]))
    m_max = max(_max_abs(MX[sl]), _max_abs(MY[sl]), _max_abs(MZ[sl]))

    return {
        'SO_F_RMS_ok': f_rms <= f_rms_thr,
        'SO_F_MAX_ok': f_max <= f_max_thr,
        'SO_M_RMS_ok': m_rms <= m_rms_thr,
        'SO_M_MAX_ok': m_max <= m_max_thr,
    }


def qc_cycle(
    cycle: Dict[str, Any],
    side: str,
    meta: Optional[Dict[str, Any]] = None,
    set_missing_flags_true: bool = True,
) -> Dict[str, Any]:
    """Run QC on a single cycle and set flags in-place.

    Parameters
    ----------
    cycle : dict
        A cycle dictionary to annotate.
    side : {'left_stride','right_stride'}
        Which stride the cycle belongs to.
    meta : dict, optional
        Root meta dict (for points frame rate).
    set_missing_flags_true : bool
        If True, explicitly set flags to True when data are missing; if False,
        leave those flags absent.

    Returns
    -------
    dict
        The same cycle dict (mutated) for convenience.
    """
    # IK
    ik_flags = eval_ik_flags(cycle)
    # SO
    so_flags = eval_so_flags(cycle, side=side, meta=meta)

    # When set_missing_flags_true=False, only write flags if we actually evaluated
    # (i.e., not all True by default due to missing data). We detect this by
    # checking presence of the relevant sub-dicts.
    ik_present = isinstance(_safe_get(cycle, 'IK_markerErr', default=None), dict)
    so_present = isinstance(_safe_get(cycle, 'SO_forces', default=None), dict)

    if set_missing_flags_true or ik_present:
        cycle['IK_RMS_ok'] = bool(ik_flags['IK_RMS_ok'])
        cycle['IK_MAX_ok'] = bool(ik_flags['IK_MAX_ok'])
    if set_missing_flags_true or so_present:
        cycle['SO_F_RMS_ok'] = bool(so_flags['SO_F_RMS_ok'])
        cycle['SO_F_MAX_ok'] = bool(so_flags['SO_F_MAX_ok'])
        cycle['SO_M_RMS_ok'] = bool(so_flags['SO_M_RMS_ok'])
        cycle['SO_M_MAX_ok'] = bool(so_flags['SO_M_MAX_ok'])

    return cycle


def qc_all(root: Dict[str, Any]) -> Dict[str, Any]:
    """Run QC over all cycles in root dict (mutates in place).

    The function looks for 'left_stride' and 'right_stride' keys, then for
    children 'cycle*' (any key starting with 'cycle'). Missing branches are
    simply skipped.

    Parameters
    ----------
    root : dict
        Root dictionary containing side branches and 'meta'.

    Returns
    -------
    dict
        The same root dict (mutated) for convenience.
    """
    if not isinstance(root, dict):
        return root

    meta = _safe_get(root, 'meta', default={})

    for side in ('left_stride', 'right_stride'):
        side_dict = _safe_get(root, side, default=None)
        if not isinstance(side_dict, dict):
            continue
        for cyc_key, cyc_val in list(side_dict.items()):
            if not (isinstance(cyc_key, str) and cyc_key.lower().startswith('cycle')):
                continue
            if not isinstance(cyc_val, dict):
                continue
            qc_cycle(cyc_val, side=side, meta=meta)

    return root


# ------------------------------
# __main__ smoke test (optional)
# ------------------------------
if __name__ == '__main__':  # pragma: no cover
    # Minimal synthetic arrays to sanity-check execution without real files
    t = np.linspace(0, 1, 200)
    cyc = {
        'Left_Foot_Off': {'time': 0.2},
        'Left_Foot_Strike': {'time': 0.8},
        'SO_forces': {
            'time': t,
            'FX': np.zeros_like(t),
            'FY': np.zeros_like(t),
            'FZ': np.zeros_like(t),
            'MX': np.zeros_like(t),
            'MY': np.zeros_like(t),
            'MZ': np.zeros_like(t),
        },
        'IK_markerErr': {
            'marker_error_RMS': 0.001 * np.ones_like(t),
            'marker_error_max': 0.003 * np.ones_like(t),
        },
    }
    meta = {'header': {'points': {'frame_rate': 200.0}}}

    print(eval_ik_flags(cyc))
    print(eval_so_flags(cyc, side='right_stride', meta=meta))
    print(qc_cycle(cyc, side='right_stride', meta=meta))
