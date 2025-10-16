"""Quality control helpers for IK and SO results (GRF/BW/ABS configurable; FY-only decision).

This module evaluates per-cycle OpenSim outputs for inverse kinematics (IK)
marker errors and static optimization (SO) residuals.

SO force thresholds (choose EXACTLY ONE per metric in settings.json → section "so"):
  RMS metric (applies to each component FX/FY/FZ; mapping SO↔FP below):
    - absolute N:      "force_rms_max": <float N>
    - % of GRF RMS:    "force_rms_max_pcGRF": <float %>   (uses per-component FP RMS in SLS)
    - % of BW in N:    "force_rms_max_pcBW": <float %>    (BW = meta.PROCESSING.Bodymass * 9.81 N)
  MAX metric:
    - absolute N:      "force_max_max": <float N>
    - % of GRF MAX:    "force_max_max_pcGRF": <float %>   (uses per-component FP MAX in SLS)
    - % of BW in N:    "force_max_max_pcBW": <float %>

Mapping for reference components (absolute values used throughout):
  SO.FX ↔ FP.Y   (Force_Fy#),   SO.FY ↔ FP.Z (Force_Fz#),   SO.FZ ↔ FP.X (Force_Fx#)

SLS window = contralateral Foot_Off → contralateral Foot_Strike mapped to SO_forces['time'].
Trim both ends by "so.so_ms_tol" milliseconds (required) on the SO timeline.
Forceplate streams are windowed by mapping the trimmed SO index window to analog indices
via linear fraction of the cycle (works with/without analog time vector).

IMPORTANT (temporary): Although thresholds are computed for FX/FY/FZ, the final decision
for forces is intentionally evaluated **only for FY** (vs FP.Z) for now, so it’s easy to
switch back to all components later.

This module is strict: it raises clear errors if required settings/data are missing.
Use the literal DEBUG_REPORT_ALWAYS toggle inside eval_so_flags() to print the report at
every call; otherwise it prints only for the first evaluated cycle.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import re
import json
from pathlib import Path
import numpy as np

try:
    from .settings import get as settings_get  # type: ignore
except Exception:  # pragma: no cover
    settings_get = None


# ------------------------------
# Small helpers
# ------------------------------

def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_array_like(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _rms(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _max_abs(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _contralateral_events(side: str) -> Tuple[str, str]:
    if side == 'left_stride':
        return 'Right_Foot_Off', 'Right_Foot_Strike'
    else:
        return 'Left_Foot_Off', 'Left_Foot_Strike'


def _as_scalar_time(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        try:
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return None
    if arr.shape == ():
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def _window_indices_from_times(t_series: np.ndarray, t0: Optional[float], t1: Optional[float], n_skip: int) -> Tuple[int, int]:
    if t_series is None or len(t_series) == 0:
        raise ValueError("SO_forces.time is missing or empty")
    if t0 is None or t1 is None:
        raise ValueError("Contralateral Foot_Off/Foot_Strike time is missing")
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise ValueError("Contralateral event time(s) are not finite")
    if t1 <= t0:
        raise ValueError(f"Invalid SLS window: Foot_Strike ({t1}) <= Foot_Off ({t0})")

    ts = np.asarray(t_series, dtype=float)
    i0 = int(np.searchsorted(ts, t0, side='left'))
    i1 = int(np.searchsorted(ts, t1, side='right')) - 1

    i0 = max(0, i0 + max(0, int(n_skip)))
    i1 = min(len(ts) - 1, i1 - max(0, int(n_skip)))
    if i1 < i0:
        raise ValueError(f"SLS window after trimming is empty: indices [{i0}:{i1}]")
    return i0, i1


# ------------------------------
# Settings helpers
# ------------------------------

def _load_local_settings() -> Dict[str, Any]:
    path = Path(__file__).with_name("settings.json").resolve()
    return json.loads(path.read_text(encoding="utf-8"))  # let JSON errors raise


def _deep_get(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing settings key: {dotted}")
        cur = cur[part]
    return cur


# ------------------------------
# Forceplate utilities
# ------------------------------

def _collect_fp_streams(analog: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
    """Collect Force_Fx#, Force_Fy#, Force_Fz# arrays (abs) for any number of plates."""
    if not isinstance(analog, dict):
        raise ValueError("cycle['analog'] is missing or not a dict")
    out: Dict[str, List[np.ndarray]] = {'Fx': [], 'Fy': [], 'Fz': []}
    pat = {
        'Fx': re.compile(r'^Force_Fx(\d+)$', re.IGNORECASE),
        'Fy': re.compile(r'^Force_Fy(\d+)$', re.IGNORECASE),
        'Fz': re.compile(r'^Force_Fz(\d+)$', re.IGNORECASE),
    }
    for key, val in analog.items():
        if not _is_array_like(val):
            continue
        for comp, rx in pat.items():
            if rx.match(key):
                arr = np.asarray(val, dtype=float)
                if arr.size == 0:
                    continue
                out[comp].append(np.abs(arr))
                break
    if not any(out.values()):
        raise ValueError("No Force_Fx#/Force_Fy#/Force_Fz# channels found in cycle['analog']")
    return out


def _fp_window_stat(arr_list: List[np.ndarray], frac0: float, frac1: float, mode: str) -> float:
    """Grand statistic across plates in window; mode in {'rms','max'}."""
    vals: List[float] = []
    for a in arr_list:
        L = len(a)
        if L == 0:
            continue
        j0 = int(round(frac0 * (L - 1)))
        j1 = int(round(frac1 * (L - 1)))
        j0 = max(0, min(L - 1, j0))
        j1 = max(0, min(L - 1, j1))
        if j1 < j0:
            continue
        seg = a[j0:j1 + 1]
        if seg.size == 0:
            continue
        if mode == 'rms':
            vals.append(_rms(seg))
        elif mode == 'max':
            vals.append(_max_abs(seg))
        else:
            raise ValueError(f"Unknown mode '{mode}' for _fp_window_stat")
    if not vals:
        raise ValueError("Unable to compute forceplate window statistic — window outside analog data for all plates")
    return max(vals)  # grand max across plates


# ------------------------------
# Public evaluation functions
# ------------------------------

def eval_ik_flags(cycle: Dict[str, Any]) -> Dict[str, bool]:
    """IK remains tolerant; uses settings if available (raises only on numeric errors)."""
    rms_thr = float(settings_get('ik.marker_error_RMS_max', 0.02) if settings_get else 0.02)
    max_thr = float(settings_get('ik.marker_error_max_max', 0.04) if settings_get else 0.04)

    ikm = _safe_get(cycle, 'IK_markerErr', default=None)
    if not isinstance(ikm, dict):
        return {'IK_RMS_ok': True, 'IK_MAX_ok': True}

    rms = _safe_get(ikm, 'marker_error_RMS', default=None)
    mxx = _safe_get(ikm, 'marker_error_max', default=None)

    if not _is_array_like(rms) or not _is_array_like(mxx):
        return {'IK_RMS_ok': True, 'IK_MAX_ok': True}

    rms_max_over_time = float(np.max(np.asarray(rms, dtype=float)))
    max_max_over_time = float(np.max(np.asarray(mxx, dtype=float)))

    return {'IK_RMS_ok': rms_max_over_time <= rms_thr, 'IK_MAX_ok': max_max_over_time <= max_thr}


def eval_so_flags(cycle: Dict[str, Any], side: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Evaluate SO residual force/moment flags.

    Strict behavior: raises clear errors if settings/data are missing.
    Computes thresholds for all components, but only applies the final decision to FY.
    """
    # Toggle detailed report at every call by changing this literal:
    if False:
        DEBUG_REPORT_ALWAYS = True
    else:
        DEBUG_REPORT_ALWAYS = False

    # --- Settings (must exist) ---
    conf = _load_local_settings()
    so_conf = _deep_get(conf, 'so')
    ms_tol = float(_deep_get(so_conf, 'so_ms_tol'))  # milliseconds (required)

    # Decide RMS spec: exactly one of the three
    rms_keys = ['force_rms_max', 'force_rms_max_pcGRF', 'force_rms_max_pcBW']
    rms_present = [k for k in rms_keys if k in so_conf]
    if len(rms_present) != 1:
        raise ValueError(f"Ambiguous or missing RMS spec in settings.so — found {rms_present}, expected exactly one of {rms_keys}")
    rms_key = rms_present[0]
    rms_val = float(so_conf[rms_key])

    # Decide MAX spec: exactly one of the three
    max_keys = ['force_max_max', 'force_max_max_pcGRF', 'force_max_max_pcBW']
    max_present = [k for k in max_keys if k in so_conf]
    if len(max_present) != 1:
        raise ValueError(f"Ambiguous or missing MAX spec in settings.so — found {max_present}, expected exactly one of {max_keys}")
    max_key = max_present[0]
    max_val = float(so_conf[max_key])

    # --- Sampling ---
    points_fs = float(_safe_get(meta or {}, 'header', 'points', 'frame_rate'))
    if not np.isfinite(points_fs) or points_fs <= 0:
        raise ValueError("meta['header']['points']['frame_rate'] is missing or invalid")
    n_skip = int(points_fs * max(0.0, ms_tol) / 1000.0)

    # --- SLS events ---
    off_key, str_key = _contralateral_events(side)
    t_off = _as_scalar_time(_safe_get(cycle, off_key, 'time'))
    t_str = _as_scalar_time(_safe_get(cycle, str_key, 'time'))

    so = _safe_get(cycle, 'SO_forces')
    if not isinstance(so, dict):
        raise ValueError("cycle['SO_forces'] is missing or not a dict")

    t_series = np.asarray(_safe_get(so, 'time'), dtype=float)
    FX = np.asarray(_safe_get(so, 'FX'), dtype=float)
    FY = np.asarray(_safe_get(so, 'FY'), dtype=float)
    FZ = np.asarray(_safe_get(so, 'FZ'), dtype=float)
    MX = np.asarray(_safe_get(so, 'MX'), dtype=float)
    MY = np.asarray(_safe_get(so, 'MY'), dtype=float)
    MZ = np.asarray(_safe_get(so, 'MZ'), dtype=float)

    if any(a.size == 0 for a in (t_series, FX, FY, FZ, MX, MY, MZ)):
        raise ValueError("Empty arrays in SO_forces")

    # SO window (trimmed by ms on points timeline)
    i0, i1 = _window_indices_from_times(t_series, t_off, t_str, n_skip)
    sl = slice(i0, i1 + 1)

    # SO values in SLS (absolute)
    so_rms = {'FX': _rms(np.abs(FX[sl])), 'FY': _rms(np.abs(FY[sl])), 'FZ': _rms(np.abs(FZ[sl]))}
    so_max = {'FX': _max_abs(FX[sl]),     'FY': _max_abs(FY[sl]),     'FZ': _max_abs(FZ[sl])}

    # --- Forceplate streams (must exist) ---
    analog = _safe_get(cycle, 'analog')
    grf = _collect_fp_streams(analog)

    # Map trimmed SO index window to analog via *fractions of cycle*
    N_so = len(t_series)
    if N_so < 2:
        raise ValueError("SO_forces.time has insufficient length")
    frac0 = i0 / (N_so - 1)
    frac1 = i1 / (N_so - 1)

    # Per-component grand stats across plates (in SLS)
    fp_rms = {
        'Fx': _fp_window_stat(grf['Fx'], frac0, frac1, mode='rms'),
        'Fy': _fp_window_stat(grf['Fy'], frac0, frac1, mode='rms'),
        'Fz': _fp_window_stat(grf['Fz'], frac0, frac1, mode='rms'),
    }
    fp_max = {
        'Fx': _fp_window_stat(grf['Fx'], frac0, frac1, mode='max'),
        'Fy': _fp_window_stat(grf['Fy'], frac0, frac1, mode='max'),
        'Fz': _fp_window_stat(grf['Fz'], frac0, frac1, mode='max'),
    }

    # Bodyweight if needed
    BW_N = None
    if rms_key.endswith('_pcBW') or max_key.endswith('_pcBW'):
        bodymass_kg = _safe_get(meta or {}, 'PROCESSING', 'Bodymass')
        if bodymass_kg is None:
            raise ValueError("Bodymass missing at meta['PROCESSING']['Bodymass'] required for %BW thresholds")
        BW_N = float(bodymass_kg) * 9.81

    # --- Build per-component thresholds (compute for all components) ---
    # RMS
    if rms_key == 'force_rms_max':
        thr_rms = {'FX': rms_val, 'FY': rms_val, 'FZ': rms_val}
    elif rms_key == 'force_rms_max_pcGRF':
        pct = rms_val / 100.0
        thr_rms = {
            'FX': pct * fp_rms['Fy'],
            'FY': pct * fp_rms['Fz'],
            'FZ': pct * fp_rms['Fx'],
        }
    elif rms_key == 'force_rms_max_pcBW':
        thr_val = (rms_val / 100.0) * float(BW_N)  # BW_N validated above
        thr_rms = {'FX': thr_val, 'FY': thr_val, 'FZ': thr_val}
    else:
        raise RuntimeError("Unreachable RMS config case")

    # MAX
    if max_key == 'force_max_max':
        thr_max = {'FX': max_val, 'FY': max_val, 'FZ': max_val}
    elif max_key == 'force_max_max_pcGRF':
        pct = max_val / 100.0
        thr_max = {
            'FX': pct * fp_max['Fy'],
            'FY': pct * fp_max['Fz'],
            'FZ': pct * fp_max['Fx'],
        }
    elif max_key == 'force_max_max_pcBW':
        thr_val = (max_val / 100.0) * float(BW_N)  # BW_N validated above
        thr_max = {'FX': thr_val, 'FY': thr_val, 'FZ': thr_val}
    else:
        raise RuntimeError("Unreachable MAX config case")

    # --- Final decisions (TEMP: FY-only) ---
    so_f_rms_ok = all(so_rms[k] < thr_rms[k] for k in ('FY',)) # add FX or FZ here to apply their check to the decission
    so_f_max_ok = all(so_max[k] < thr_max[k] for k in ('FY',)) # add FX or FZ here to apply their check to the decission

    # Moments (absolute thresholds; required in settings)
    m_rms_thr = float(_deep_get(so_conf, 'moment_rms_max'))
    m_max_thr = float(_deep_get(so_conf, 'moment_max_max'))
    m_rms = max(_rms(np.abs(MX[sl])), _rms(np.abs(MY[sl])), _rms(np.abs(MZ[sl])))
    m_max = max(_max_abs(MX[sl]), _max_abs(MY[sl]), _max_abs(MZ[sl]))

    # ------------- Detailed console report -------------
    if DEBUG_REPORT_ALWAYS or not getattr(eval_so_flags, "_printed", False):
        denom = max(1, (len(t_series) - 1))
        p0 = (i0 / denom) * 100.0
        p1 = (i1 / denom) * 100.0
        def _f(v: float) -> str: return f"{v:.3f}"

        def _mode_str(which: str) -> str:
            if which == 'rms':
                if rms_key == 'force_rms_max': return f"{rms_val:.3f} N (absolute)"
                if rms_key == 'force_rms_max_pcGRF': return f"{rms_val:.1f}% of GRF component RMS"
                if rms_key == 'force_rms_max_pcBW': return f"{rms_val:.1f}% of BW"
            else:
                if max_key == 'force_max_max': return f"{max_val:.3f} N (absolute)"
                if max_key == 'force_max_max_pcGRF': return f"{max_val:.1f}% of GRF component MAX"
                if max_key == 'force_max_max_pcBW': return f"{max_val:.1f}% of BW"
            return "?"

        print("=== QC thresholds & decisions (FY-only decision active) ===")
        print(f"SLS: indices [{i0}:{i1}] | %GC [{p0:.1f}%–{p1:.1f}%] | so_ms_tol={ms_tol:.1f} ms")
        print(f"FP grand RMS (SLS): Fx={_f(fp_rms['Fx'])} N, Fy={_f(fp_rms['Fy'])} N, Fz={_f(fp_rms['Fz'])} N")
        print(f"FP grand MAX (SLS): Fx={_f(fp_max['Fx'])} N, Fy={_f(fp_max['Fy'])} N, Fz={_f(fp_max['Fz'])} N")

        print(f"SO forces RMS mode: {_mode_str('rms')}")
        for comp, ref in (('FX','Fy'), ('FY','Fz'), ('FZ','Fx')):
            print(f"  {comp}: value={_f(so_rms[comp])} N  < thr={_f(thr_rms[comp])} N  (ref=FP.{ref})  → {'OK' if so_rms[comp] < thr_rms[comp] else 'FAIL'}")
        print(f"  → SO_F_RMS_ok (FY-only) = {'OK' if so_f_rms_ok else 'FAIL'}")

        print(f"SO forces MAX mode: {_mode_str('max')}")
        for comp, ref in (('FX','Fy'), ('FY','Fz'), ('FZ','Fx')):
            print(f"  {comp}: value={_f(so_max[comp])} N  < thr={_f(thr_max[comp])} N  (ref=FP.{ref})  → {'OK' if so_max[comp] < thr_max[comp] else 'FAIL'}")
        print(f"  → SO_F_MAX_ok (FY-only) = {'OK' if so_f_max_ok else 'FAIL'}")

        print("SO moments (absolute thresholds):")
        print(f"  RMS: value={_f(m_rms)} Nm  <= thr={_f(m_rms_thr)} Nm  → {'OK' if m_rms <= m_rms_thr else 'FAIL'}")
        print(f"  MAX: value={_f(m_max)} Nm  <= thr={_f(m_max_thr)} Nm  → {'OK' if m_max <= m_max_thr else 'FAIL'}")

        if not DEBUG_REPORT_ALWAYS:
            eval_so_flags._printed = True

    return {
        'SO_F_RMS_ok': so_f_rms_ok,
        'SO_F_MAX_ok': so_f_max_ok,
        'SO_M_RMS_ok': m_rms <= m_rms_thr,
        'SO_M_MAX_ok': m_max <= m_max_thr,
    }


def qc_cycle(cycle: Dict[str, Any], side: str, meta: Optional[Dict[str, Any]] = None, set_missing_flags_true: bool = True) -> Dict[str, Any]:
    ik_flags = eval_ik_flags(cycle)
    so_flags = eval_so_flags(cycle, side=side, meta=meta)

    cycle['IK_RMS_ok'] = bool(ik_flags['IK_RMS_ok'])
    cycle['IK_MAX_ok'] = bool(ik_flags['IK_MAX_ok'])
    cycle['SO_F_RMS_ok'] = bool(so_flags['SO_F_RMS_ok'])
    cycle['SO_F_MAX_ok'] = bool(so_flags['SO_F_MAX_ok'])
    cycle['SO_M_RMS_ok'] = bool(so_flags['SO_M_RMS_ok'])
    cycle['SO_M_MAX_ok'] = bool(so_flags['SO_M_MAX_ok'])
    return cycle


def qc_all(root: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(root, dict):
        raise ValueError("qc_all expects the root dict")
    meta = _safe_get(root, 'meta')
    for side in ('left_stride', 'right_stride'):
        side_dict = _safe_get(root, side)
        if not isinstance(side_dict, dict):
            continue
        for cyc_key, cyc_val in list(side_dict.items()):
            if not (isinstance(cyc_key, str) and cyc_key.lower().startswith('cycle')):
                continue
            if not isinstance(cyc_val, dict):
                continue
            qc_cycle(cyc_val, side=side, meta=meta)
    return root
