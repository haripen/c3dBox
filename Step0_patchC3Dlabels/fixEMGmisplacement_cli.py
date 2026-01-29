#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixEMGmisplacement_cli.py
CLI tool to detect and remap misplaced EMG labels in C3D files.
Outputs plots + JSON mapping; optionally applies mapping to COPIED C3Ds.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.signal as signal

try:
    import ezc3d
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ezc3d is required for this tool.") from exc

_PLOT = None


def get_plotting():
    global _PLOT
    if _PLOT is not None:
        return _PLOT
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting JPEG outputs.") from exc
    _PLOT = plt
    return _PLOT


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LABEL_MAP_PATH = REPO_ROOT / "Step0_patchC3Dlabels" / "label_mapping.json"

EMG_RIGHT_NUMS = set(range(1, 14))
EMG_LEFT_NUMS = set(list(range(17, 28)) + [31, 32])

def normalize_muscle_key(k):
    k = (k or "").lower()
    if k.startswith("m"):
        k = k[1:]
    return k


def canonical_muscle(k):
    k = normalize_muscle_key(k)
    aliases = [
        ("tibialis_ant", ["tibialisanterior", "tibant", "tibialisant"]),
        ("peroneus_long", ["peroneuslong", "peroneuslongus", "perlong"]),
        ("gastro_lat", ["gastroclat", "gastrolat", "gastroclateral", "gastro. lat", "gastro.lat"]),
        ("gastro_med", ["gastrocmed", "gastromed", "gastrocmedial", "gastro. med", "gastro.med"]),
        ("soleus", ["soleus"]),
        ("vastus_lat", ["vastuslat", "vastuslateral"]),
        ("vastus_med", ["vastusmed", "vastusmedial"]),
        ("rectus_fem", ["rectusfem", "rectusfemoris"]),
        ("semitendinosus", ["semitendinosus", "semitend"]),
        ("biceps_femoris", ["bicepsfemoris", "bicepsfem"]),
        ("gluteus_medius", ["gluteusmedius", "glutmed"]),
        ("gluteus_maximus", ["gluteusmaximus", "glutmax"]),
        ("erector_spinae", ["erectorspine", "erectorspinae", "erectorsp"]),
    ]
    for canon, parts in aliases:
        for p in parts:
            if p in k:
                return canon
    return k or "unknown"


def close_muscle_keys(canon_key):
    groups = [
        {"tibialis_ant", "peroneus_long", "gastro_lat", "gastro_med", "soleus"},
        {"vastus_lat", "vastus_med", "rectus_fem"},
        {"semitendinosus", "biceps_femoris"},
        {"gluteus_medius", "gluteus_maximus"},
    ]
    for g in groups:
        if canon_key in g:
            return g
    return set()

try:
    from Step4_check.emg import process_emg as step4_process_emg
    _HAS_STEP4_EMG = True
except Exception:
    step4_process_emg = None
    _HAS_STEP4_EMG = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect and remap misplaced EMG channel labels in C3D files."
    )
    p.add_argument(
        "--data-root",
        default=r"D:\Data_local\Pros_5er_hybrid_plain",
        help="Root folder to scan for .c3d files.",
    )
    p.add_argument(
        "--task",
        required=True,
        help="Substring to match DESCRIPTION= in .enf (case-insensitive).",
    )
    p.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "Step0_patchC3Dlabels" / "outputs"),
        help="Output base folder for plots, logs, JSON.",
    )
    p.add_argument(
        "--agg",
        choices=["mean", "median", "pca"],
        default="pca",
        help="Aggregation method for ID-specific and global traces.",
    )
    p.add_argument("--min-cycles", type=int, default=3, help="Min cycles per ID/channel.")
    p.add_argument(
        "--min-cycle-samples",
        type=int,
        default=50,
        help="Min samples per cycle before normalization.",
    )
    p.add_argument(
        "--bandpass",
        nargs=2,
        type=float,
        default=[30.0, 350.0],
        metavar=("LOW", "HIGH"),
        help="Bandpass cutoff Hz for raw EMG.",
    )
    p.add_argument(
        "--lowpass",
        type=float,
        default=6.0,
        help="Lowpass cutoff Hz for rectified EMG envelope.",
    )
    p.add_argument(
        "--emg-method",
        choices=["internal", "step4"],
        default="step4" if _HAS_STEP4_EMG else "internal",
        help="EMG envelope method. 'step4' uses Step4_check.emg.process_emg.",
    )
    p.add_argument(
        "--within-lag-max",
        type=int,
        default=5,
        help="Max allowed lag (samples) for within-ID cycle vs trace alignment.",
    )
    p.add_argument(
        "--lag-lr-range",
        nargs=2,
        type=int,
        default=[55, 65],
        metavar=("MIN", "MAX"),
        help="Expected lag range (samples) for L/R mix-up validation.",
    )
    p.add_argument(
        "--corr-gain",
        type=float,
        default=0.10,
        help="Required correlation gain for mapping suggestions.",
    )
    p.add_argument(
        "--mad-gain",
        type=float,
        default=0.20,
        help="Required relative MAD improvement for mapping suggestions.",
    )
    p.add_argument(
        "--use-ci",
        action="store_true",
        help="Require candidate metrics to fall within CI for mapping suggestions.",
    )
    p.add_argument(
        "--threshold-pass",
        type=int,
        default=2,
        help="Number of passes for empirical threshold estimation (1 or 2).",
    )
    p.add_argument(
        "--remap-scope",
        choices=["flagged", "all"],
        default="flagged",
        help="Candidate targets during remapping.",
    )
    p.add_argument(
        "--interactive-unsure",
        action="store_true",
        help="At end, ask for user decisions on borderline unsure cases.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Alias for --interactive-unsure.",
    )
    p.add_argument(
        "--interactive-close-frac",
        type=float,
        default=0.80,
        help="Borderline fraction of corr/mad gains to include in interactive review.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of TASK-MATCHED C3Ds for testing (0 = no cap).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply mapping (requires --apply-mode).",
    )
    p.add_argument(
        "--apply-mode",
        choices=["C3Doverwrite", "C3Dcopy"],
        default="",
        help="Required when --apply: C3Dcopy (default) or C3Doverwrite.",
    )
    p.add_argument(
        "--apply-root",
        default="",
        help="Root folder to place copied C3Ds when --apply is set.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (not recommended).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging.",
    )
    return p.parse_args()


def log(msg, log_lines, also_print=True):
    if also_print:
        print(msg)
    log_lines.append(msg)


def read_label_mapping(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_c3ds(root: Path, max_files=0):
    c3ds = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".c3d"):
                c3ds.append(Path(dp) / f)
    return c3ds


def parse_enf_description(c3d_path: Path):
    enf = c3d_path.with_suffix(".enf")
    if not enf.exists():
        return None
    desc = None
    with open(enf, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.upper().startswith("DESCRIPTION="):
                desc = line.split("=", 1)[1].strip()
                break
    return desc


def match_task(desc, task_substring):
    if not desc:
        return False
    return task_substring.lower() in desc.lower()


def get_subject_id(c3d):
    names = c3d["parameters"].get("SUBJECTS", {}).get("NAMES", {}).get("value", None)
    if names is None:
        return "UNKNOWN"
    if isinstance(names, (list, tuple)):
        return str(names[0]).strip()
    return str(names).strip()


def extract_event_times(c3d):
    ev = c3d["parameters"].get("EVENT", {})
    labels = list(ev.get("LABELS", {}).get("value", []))
    contexts = list(ev.get("CONTEXTS", {}).get("value", []))
    times = np.array(ev.get("TIMES", {}).get("value", []), dtype=float)
    if times.ndim == 2 and times.shape[0] >= 2:
        # Choose the row with non-zero sum (often row 1)
        row0 = times[0]
        row1 = times[1]
        if np.sum(np.abs(row1)) >= np.sum(np.abs(row0)):
            times = row1
        else:
            times = row0
    elif times.ndim == 1:
        pass
    else:
        times = times.reshape(-1)
    return times, labels, contexts


def get_foot_strike_times(times, labels, contexts, side):
    out = []
    for t, lab, ctx in zip(times, labels, contexts):
        if str(lab).strip().lower() == "foot strike" and str(ctx).strip().lower() == side.lower():
            out.append(float(t))
    out = sorted(set(out))
    return out


def analog_matrix(c3d):
    analogs = np.array(c3d["data"]["analogs"], dtype=float)
    if analogs.ndim != 3:
        raise RuntimeError(f"Unexpected analogs shape: {analogs.shape}")
    subframes, n_channels, n_frames = analogs.shape
    # Flatten subframes to full-rate samples
    data = np.transpose(analogs, (2, 0, 1)).reshape(n_frames * subframes, n_channels)
    rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    return data, rate


def time_c3d0(c3d):
    try:
        point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
        actual_start = float(c3d["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0])
        if point_rate <= 0:
            return 0.0
        return (actual_start / point_rate) - (1.0 / point_rate)
    except Exception:
        return 0.0


def analog_time_vector(c3d, n_samples):
    rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    if rate <= 0:
        return np.arange(n_samples, dtype=float)
    t0 = time_c3d0(c3d)
    return (np.arange(n_samples, dtype=float) / rate) + t0


def emg_channel_info(label):
    s = str(label).strip()
    if "emg" not in s.lower():
        return None
    # Ignore NFU
    if "nfu" in s.lower():
        return {"label": s, "is_nfu": True}
    m = re.search(r"emg[_\.\s-]*([0-9]{1,2})", s, re.IGNORECASE)
    if not m:
        return None
    emg_num = int(m.group(1))
    side = None
    m2 = re.search(r"emg[_\.\s-]*[0-9]{1,2}[_\.\s-]*([LR])\b", s, re.IGNORECASE)
    if m2:
        side = m2.group(1).upper()
    # Enforce expected 13 left/13 right channels
    if side == "R" and emg_num not in EMG_RIGHT_NUMS:
        return None
    if side == "L" and emg_num not in EMG_LEFT_NUMS:
        return None
    muscle = ""
    m3 = re.search(r"emg[_\.\s-]*[0-9]{1,2}[_\.\s-]*[LR]\s*(.*)$", s, re.IGNORECASE)
    if m3:
        muscle = m3.group(1).strip()
    muscle_key = re.sub(r"[^a-z0-9]", "", muscle.lower())
    return {
        "label": s,
        "emg_num": emg_num,
        "side": side,
        "muscle": muscle,
        "muscle_key": muscle_key,
        "is_nfu": False,
    }


def build_emg_label_list(labels):
    infos = []
    for idx, lab in enumerate(labels):
        info = emg_channel_info(lab)
        if info is None:
            continue
        info["index"] = idx
        infos.append(info)
    return infos


def bandpass_filter(sig, fs, low, high):
    nyq = fs * 0.5
    low = max(1.0, low)
    high = min(high, nyq * 0.95)
    if high <= low:
        raise ValueError("Invalid bandpass bounds after adjustment.")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig)


def lowpass_filter(sig, fs, cutoff):
    nyq = fs * 0.5
    cutoff = min(cutoff, nyq * 0.95)
    sos = signal.butter(2, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig)


def process_emg_segment(raw, fs, bp_low, bp_high, lp_cut, method="internal"):
    if raw.size < 5:
        return None
    if method == "step4" and step4_process_emg is not None:
        try:
            env = step4_process_emg(raw, fs)
        except Exception:
            env = None
        if env is None:
            return None
        env = np.asarray(env, dtype=float)
        env[~np.isfinite(env)] = 0.0
        env[env < 0] = 0.0
        return env
    x = raw - np.mean(raw)
    x = bandpass_filter(x, fs, bp_low, bp_high)
    x = np.abs(x)
    x = lowpass_filter(x, fs, lp_cut)
    x[x < 0] = 0.0
    return x


def time_normalize(sig, n=101):
    if sig.size < 2:
        return None
    x_old = np.linspace(0, 1, sig.size)
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, sig)


def cycle_snr(envelope):
    if envelope.size < 5:
        return np.nan
    thresh = np.percentile(envelope, 20)
    low = envelope[envelope <= thresh]
    noise = np.std(low) if low.size else np.std(envelope)
    signal_level = np.mean(envelope)
    return float(signal_level / (noise + 1e-8))


def aggregate_cycles(cycles, method):
    X = np.asarray(cycles)
    if X.ndim != 2:
        return None
    if method == "mean":
        return np.mean(X, axis=0)
    if method == "median":
        return np.median(X, axis=0)
    # PCA-like: first component without centering, sign-aligned to mean
    # SVD on raw data to capture dominant shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    mean = np.mean(X, axis=0)
    if np.std(v) > 1e-12 and np.std(mean) > 1e-12:
        if np.corrcoef(v, mean)[0, 1] < 0:
            v = -v
    # scale to mean RMS for interpretability
    scale = (np.linalg.norm(mean) / (np.linalg.norm(v) + 1e-8))
    return v * scale


def cross_corr(a, b, max_lag=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("Signals must be same length for cross-correlation.")
    n = a.size
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    best_corr = -np.inf
    best_lag = 0
    if max_lag is None:
        lag_range = range(-(n - 1), n)
    else:
        max_lag = int(max_lag)
        lag_range = range(-max_lag, max_lag + 1)
    for lag in lag_range:
        if lag < 0:
            aa = a0[-lag:]
            bb = b0[: n + lag]
        else:
            aa = a0[: n - lag]
            bb = b0[lag:]
        if aa.size < 3:
            continue
        denom = (np.std(aa) * np.std(bb) + 1e-8)
        corr = float(np.mean((aa - np.mean(aa)) * (bb - np.mean(bb))) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag


def circ_shift(a, lag):
    a = np.asarray(a, dtype=float)
    if lag == 0:
        return a
    return np.roll(a, int(lag))


def corr_for_lag_circular(a, b, lag):
    aa = circ_shift(a, lag)
    bb = np.asarray(b, dtype=float)
    if aa.size < 3:
        return np.nan
    denom = (np.std(aa) * np.std(bb) + 1e-8)
    return float(np.mean((aa - np.mean(aa)) * (bb - np.mean(bb))) / denom)


def best_corr_circular(a, b, lags):
    best = (-np.inf, 0)
    for lag in lags:
        corr = corr_for_lag_circular(a, b, lag)
        if np.isfinite(corr) and corr > best[0]:
            best = (corr, int(lag))
    return best[0], best[1]


def aligned_for_lag(a, b, lag):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if lag < 0:
        aa = a[-lag:]
        bb = b[: n + lag]
    elif lag > 0:
        aa = a[: n - lag]
        bb = b[lag:]
    else:
        aa = a
        bb = b
    return aa, bb


def mad_for_lag(a, b, lag=0):
    aa = circ_shift(a, lag)
    bb = np.asarray(b, dtype=float)
    if aa.size < 3:
        return np.nan
    return float(np.mean(np.abs(aa - bb)))


def best_mad_for_lags(a, b, lags):
    best = (np.inf, 0)
    for lag in lags:
        m = mad_for_lag(a, b, lag)
        if np.isfinite(m) and m < best[0]:
            best = (m, lag)
    return best[0], best[1]


def shift_trace_for_plot(a, lag):
    a = np.asarray(a, dtype=float)
    if lag == 0:
        return a
    if lag > 0:
        return np.concatenate([np.full(lag, np.nan), a[:-lag]])
    return np.concatenate([a[-lag:], np.full(-lag, np.nan)])


def safe_slug(s):
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_")


def compute_thresholds(values, mode="lower", quantile=0.025):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan
    if mode == "lower":
        return float(np.quantile(vals, quantile))
    # upper for absolute lag
    return float(np.quantile(vals, 1.0 - quantile))


def compute_ci(values, level=0.80):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size < 2:
        return (np.nan, np.nan, vals.size)
    alpha = (1.0 - level) / 2.0
    low = float(np.quantile(vals, alpha))
    high = float(np.quantile(vals, 1.0 - alpha))
    return (low, high, vals.size)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class ProgressBar:
    def __init__(self, label, total, every_pct=10):
        self.label = label
        self.total = max(1, int(total))
        self.every_pct = max(1, int(every_pct))
        self.last_pct = -1

    def update(self, current):
        pct = int((current / self.total) * 100)
        if pct >= 100 or (pct % self.every_pct == 0 and pct != self.last_pct):
            bar_len = 20
            filled = int(round(bar_len * pct / 100))
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"[progress] {self.label}: [{bar}] {pct}%")
            self.last_pct = pct

def plot_within_id(
    out_path,
    cycles,
    outlier_mask,
    id_trace,
    xcorr_med,
    lag_med,
    status,
    agg_label,
    snr,
    n_cycles,
    n_outliers,
    within_corr_med,
    within_lag_med,
    within_lag_med_lim,
    within_lag_abs_med,
    early_late_corr,
    early_late_lag,
    lag_limit,
):
    plt = get_plotting()
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(3, 1, 1)
    for c, is_out in zip(cycles, outlier_mask):
        if is_out:
            ax1.plot(c, color="0.6", linewidth=0.8, linestyle=":")
        else:
            ax1.plot(c, color="0.7", linewidth=0.8)
    ax1.plot(id_trace, color="C1", linewidth=2.0)
    ax1.set_title(f"ID cycles (n={n_cycles}, outliers={n_outliers}) + ID trace (thick)")
    ax1.set_xlim(0, len(id_trace) - 1)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(id_trace, color="C1", linewidth=2.0)
    ax2.set_title("ID trace (processing)")
    ax2.set_xlim(0, len(id_trace) - 1)

    ax3 = plt.subplot(3, 1, 3)
    if xcorr_med.size == (2 * len(id_trace) - 1):
        lags = np.arange(-(len(id_trace) - 1), len(id_trace))
    else:
        lags = np.arange(-lag_limit, lag_limit + 1)
    ax3.plot(lags, xcorr_med, color="0.2", linewidth=1.2)
    ax3.axvline(lag_med, color="C3", linestyle="--")
    ax3.axvline(lag_limit, color="C1", linestyle=":")
    ax3.axvline(-lag_limit, color="C1", linestyle=":")
    ax3.set_title("Cycle vs ID trace cross-correlation (median)")
    ax3.set_xlabel("Lag (samples)")

    line1 = f"{status} | agg={agg_label} | snr={snr:.2f}"
    line2 = (
        f"within: corr_med={within_corr_med:.3f}, lag_med_full={within_lag_med:.1f}, "
        f"lag_abs_med={within_lag_abs_med:.1f}, lag_med_lim={within_lag_med_lim:.1f}"
    )
    line3 = f"early/late: corr={early_late_corr:.3f}, lag={early_late_lag:.1f}"
    fig.text(0.5, 0.985, line1, ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.958, line2, ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.935, line3, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_id_channel(
    out_path,
    cycles,
    outlier_mask,
    id_trace,
    global_trace,
    xcorr_within_med,
    within_lag_med,
    xcorr_between,
    between_lag,
    status,
    agg_label,
    snr,
    n_cycles,
    n_outliers,
    n_ids_used,
    n_ids_total,
    within_corr_med,
    within_lag_med_full,
    within_lag_abs_med,
    within_lag_med_lim,
    early_late_corr,
    early_late_lag,
    between_corr,
    lag_limit,
    mad_info=None,
    shifted_trace=None,
    shifted_label=None,
    mapping_text=None,
):
    plt = get_plotting()
    fig = plt.figure(figsize=(13, 9))

    ax1 = plt.subplot(3, 1, 1)
    for c, is_out in zip(cycles, outlier_mask):
        if is_out:
            ax1.plot(c, color="0.6", linewidth=0.8, linestyle=":")
        else:
            ax1.plot(c, color="0.7", linewidth=0.8)
    ax1.plot(id_trace, color="C1", linewidth=2.0)
    ax1.set_title(f"ID cycles (n={n_cycles}, outliers={n_outliers}) + ID trace (thick)")
    ax1.set_xlim(0, len(id_trace) - 1)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(id_trace, color="C1", linewidth=2.0, label="ID trace")
    ax2.plot(global_trace, color="C0", linewidth=2.0, label="Global trace (PCA dim1)")
    if shifted_trace is not None:
        lab = shifted_label or "ID trace (shifted)"
        ax2.plot(shifted_trace, color="C2", linewidth=1.4, linestyle="--", label=lab)
    ax2.set_title(f"ID vs Global (IDs used/total={n_ids_used}/{n_ids_total})")
    ax2.set_xlim(0, len(id_trace) - 1)
    ax2.legend(loc="upper right", frameon=False)

    ax3 = plt.subplot(3, 1, 3)
    if xcorr_within_med.size == (2 * len(id_trace) - 1):
        lags_within = np.arange(-(len(id_trace) - 1), len(id_trace))
    else:
        lags_within = np.arange(-lag_limit, lag_limit + 1)
    ax3.plot(lags_within, xcorr_within_med, color="0.2", linewidth=1.2, label="within (median)")
    ax3.axvline(within_lag_med, color="C3", linestyle="--")
    ax3.axvline(lag_limit, color="C1", linestyle=":")
    ax3.axvline(-lag_limit, color="C1", linestyle=":")
    lags_between = np.arange(-(len(id_trace) - 1), len(id_trace))
    ax3.plot(lags_between, xcorr_between, color="C0", linewidth=1.0, alpha=0.7, label="between")
    ax3.axvline(between_lag, color="C0", linestyle="--")
    ax3.set_title("Cross-correlation (within + between)")
    ax3.set_xlabel("Lag (samples)")
    ax3.legend(loc="upper right", frameon=False)

    def mapping_color(text):
        if not text:
            return "black"
        tl = text.lower()
        if "success" in tl and "-> ok" in tl:
            return "green"
        if "success" in tl and ("l/r switch" in tl or "muscle switch" in tl):
            return "blue"
        if "fail" in tl and ("l/r switch" in tl or "muscle switch" in tl):
            return "red"
        if "fail" in tl and "-> remap" in tl:
            return "orange"
        return "black"

    title_top = mapping_text or ""
    line2 = (
        f"{status} | agg={agg_label} | snr={snr:.2f} | "
        f"within: corr_med={within_corr_med:.3f}, lag_med_full={within_lag_med_full:.1f}, "
        f"lag_abs_med={within_lag_abs_med:.1f}, lag_med_lim={within_lag_med_lim:.1f}"
    )
    line3 = (
        f"early/late: corr={early_late_corr:.3f}, lag={early_late_lag:.1f} | "
        f"between: corr={between_corr:.3f}, lag={between_lag}"
    )
    if mad_info:
        line3 = line3 + f" | {mad_info}"
    if title_top:
        fig.text(0.5, 0.985, title_top, ha="center", va="top", fontsize=9, color=mapping_color(title_top))
    fig.text(0.5, 0.958, line2, ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.935, line3, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_global_channel(out_path, traces, global_trace, label, n_ids_used, n_ids_total):
    plt = get_plotting()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    for t in traces:
        ax.plot(t, color="0.8", linewidth=0.8)
    ax.plot(global_trace, color="C0", linewidth=2.0)
    ax.set_title(f"{label} | Global trace (PCA dim1) | IDs used/total={n_ids_used}/{n_ids_total}")
    ax.set_xlim(0, len(global_trace) - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def interactive_review(items, globals_map, out_path):
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    decisions = []

    for idx, item in enumerate(items, start=1):
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(3, 1, 1)
        for c, is_out in zip(item["cycles"], item["outliers"]):
            ax1.plot(c, color="0.6" if is_out else "0.8", linewidth=0.8, linestyle=":" if is_out else "-")
        ax1.plot(item["trace"], color="C1", linewidth=2.0)
        ax1.set_title("ID cycles + ID trace")
        ax1.set_xlim(0, len(item["trace"]) - 1)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(item["trace"], color="C1", linewidth=2.0, label="ID trace")
        ax2.plot(globals_map[item["label"]], color="C0", linewidth=1.8, label="Global (original)")
        ax2.plot(globals_map[item["candidate"]], color="C2", linewidth=1.8, label="Global (candidate)")
        ax2.set_title("ID vs Global (original + candidate)")
        ax2.set_xlim(0, len(item["trace"]) - 1)
        ax2.legend(loc="upper right", frameon=False)

        ax3 = plt.subplot(3, 1, 3)
        corr1, _ = cross_corr(item["trace"], globals_map[item["label"]], max_lag=5)
        corr2, _ = cross_corr(item["trace"], globals_map[item["candidate"]], max_lag=5)
        ax3.bar([0, 1], [corr1, corr2], color=["C0", "C2"])
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["orig", "cand"])
        ax3.set_ylim(0, 1)
        ax3.set_title(f"corr@±5 (orig={corr1:.3f}, cand={corr2:.3f})")

        fig.text(0.5, 0.99, f"{idx} of {len(items)}", ha="center", va="top", fontsize=10)
        fig.text(0.5, 0.965, f"{item['sid']} | {item['label']} -> {item['candidate']} ({item['reason']})",
                 ha="center", va="top", fontsize=10)

        resp = {"value": None}

        def set_resp(v):
            resp["value"] = v
            plt.close(fig)

        ax_swap = plt.axes([0.15, 0.01, 0.2, 0.05])
        ax_unsure = plt.axes([0.40, 0.01, 0.2, 0.05])
        ax_ok = plt.axes([0.65, 0.01, 0.2, 0.05])
        Button(ax_swap, "to be swapped (1)").on_clicked(lambda _evt: set_resp("swap"))
        Button(ax_unsure, "unsure (2)").on_clicked(lambda _evt: set_resp("unsure"))
        Button(ax_ok, "looks good (3)").on_clicked(lambda _evt: set_resp("ok"))

        def on_key(event):
            if event.key == "1":
                set_resp("swap")
            elif event.key == "2":
                set_resp("unsure")
            elif event.key == "3":
                set_resp("ok")

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        if resp["value"] is None:
            break
        decisions.append({**item, "response": resp["value"]})

    if decisions:
        with open(out_path, "a", encoding="utf-8") as f:
            for rec in decisions:
                safe = {
                    "sid": rec.get("sid"),
                    "label": rec.get("label"),
                    "candidate": rec.get("candidate"),
                    "reason": rec.get("reason"),
                    "response": rec.get("response"),
                }
                f.write(json.dumps(safe) + "\n")
    return decisions


def main():
    args = parse_args()
    if args.apply and not args.apply_mode:
        raise ValueError("--apply requires --apply-mode C3Doverwrite or C3Dcopy.")
    if args.interactive:
        args.interactive_unsure = True
    log_lines = []

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.output_root) / f"fixEMG_{safe_slug(args.task)}_{timestamp}"
    out_processing = out_base / "processing"
    out_mapping = out_base / "mapping"
    out_global = out_base / "global"
    for p in [out_processing, out_mapping, out_global]:
        ensure_dir(p)

    log(f"[start] task='{args.task}', data_root='{data_root}'", log_lines)
    log(f"[start] output_root='{out_base}'", log_lines)
    log(f"[start] emg_method='{args.emg_method}'", log_lines)
    log(f"[start] id_agg='{args.agg}'", log_lines)
    log(f"[start] within_lag_max={args.within_lag_max}", log_lines)
    log(f"[start] lag_lr_range={args.lag_lr_range[0]}-{args.lag_lr_range[1]}", log_lines)
    log(f"[start] corr_gain={args.corr_gain:.2f}", log_lines)
    log(f"[start] mad_gain={args.mad_gain:.2f}", log_lines)
    log(f"[start] use_ci={args.use_ci}", log_lines)

    label_map = read_label_mapping(LABEL_MAP_PATH)
    emg_map = label_map.get("analog", {}).get("group_to_mostfrequent_raw", {})

    c3ds = list_c3ds(data_root)
    if not c3ds:
        raise FileNotFoundError("No C3D files found under data root.")

    log(f"[scan] found {len(c3ds)} C3D files", log_lines)

    # Data containers
    id_channels = defaultdict(lambda: defaultdict(list))  # id -> label -> list of cycles
    id_cycle_snr = defaultdict(dict)  # id -> label -> snr
    id_label_sets = defaultdict(lambda: defaultdict(set))  # id -> emg_num -> set(labels)
    id_channel_info = defaultdict(dict)  # id -> label -> info
    id_files = defaultdict(set)
    mismatch_warned = set()

    skipped_no_enf = 0
    skipped_no_task = 0
    processed_files = 0

    matched_count = 0
    for c3d_path in c3ds:
        desc = parse_enf_description(c3d_path)
        if desc is None:
            skipped_no_enf += 1
            continue
        if not match_task(desc, args.task):
            skipped_no_task += 1
            continue
        if args.max_files and matched_count >= args.max_files:
            break
        try:
            c3d = ezc3d.c3d(str(c3d_path))
        except Exception as exc:
            log(f"[warn] failed to read {c3d_path}: {exc}", log_lines)
            continue

        subject_id = get_subject_id(c3d)
        times, labels, contexts = extract_event_times(c3d)
        if len(times) == 0:
            log(f"[warn] no events in {c3d_path}", log_lines)
            continue

        data, rate = analog_matrix(c3d)
        analog_time = analog_time_vector(c3d, data.shape[0])
        analog_labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
        emg_infos = build_emg_label_list(analog_labels)

        # Collect strike times per side
        left_strikes = get_foot_strike_times(times, labels, contexts, "Left")
        right_strikes = get_foot_strike_times(times, labels, contexts, "Right")
        if len(left_strikes) < 2 and len(right_strikes) < 2:
            log(f"[warn] insufficient foot strikes in {c3d_path}", log_lines)
            continue

        processed_files += 1
        matched_count += 1
        id_files[subject_id].add(str(c3d_path))

        for info in emg_infos:
            if info.get("is_nfu"):
                continue
            if not info.get("side"):
                continue
            emg_num = info.get("emg_num")
            label = info.get("label")
            emg_key = f"emg#{emg_num}"
            if emg_key in emg_map:
                expected = emg_map.get(emg_key)
                if expected != label:
                    key = (emg_key, label)
                    if key not in mismatch_warned:
                        log(f"[warn] label mismatch {emg_key}: '{label}' != '{expected}'", log_lines)
                        mismatch_warned.add(key)
            id_label_sets[subject_id][emg_num].add(label)
            id_channel_info[subject_id][label] = info

            strikes = left_strikes if info["side"] == "L" else right_strikes
            if len(strikes) < 2:
                continue
            idx = info["index"]
            cycles = []
            snrs = []
            for s0, s1 in zip(strikes[:-1], strikes[1:]):
                if s1 <= s0:
                    continue
                i0 = int(np.searchsorted(analog_time, s0, side="left"))
                i1 = int(np.searchsorted(analog_time, s1, side="right"))
                i0 = max(0, i0)
                i1 = min(data.shape[0], i1)
                if i1 <= i0:
                    continue
                if i1 - i0 < args.min_cycle_samples:
                    continue
                seg = data[i0:i1, idx]
                try:
                    env = process_emg_segment(
                        seg,
                        rate,
                        args.bandpass[0],
                        args.bandpass[1],
                        args.lowpass,
                        method=args.emg_method,
                    )
                except Exception:
                    continue
                if env is None or env.size < 5:
                    continue
                env_norm = time_normalize(env, n=101)
                if env_norm is None:
                    continue
                cycles.append(env_norm)
                snrs.append(cycle_snr(env_norm))

            if cycles:
                id_channels[subject_id][label].extend(cycles)
                if snrs:
                    # Keep median SNR across cycles per file; later combine by median across files
                    snr_med = float(np.nanmedian(snrs))
                    prev = id_cycle_snr[subject_id].get(label, [])
                    prev = prev if isinstance(prev, list) else [prev]
                    prev.append(snr_med)
                    id_cycle_snr[subject_id][label] = prev

    log(f"[scan] processed files: {processed_files}", log_lines)
    log(f"[scan] skipped (no .enf): {skipped_no_enf}", log_lines)
    log(f"[scan] skipped (task mismatch): {skipped_no_task}", log_lines)

    # Progress bar 1: trace computation and threshold prep
    n_items = sum(
        1
        for sid, ch_dict in id_channels.items()
        for label, cycles in ch_dict.items()
        if len(cycles) >= args.min_cycles
    )
    pb1 = ProgressBar("compute traces", total=max(1, n_items * 2))
    pb1_step = 0

    # Check label consistency per ID/emg#
    inconsistent = []
    for sid, emg_map_sets in id_label_sets.items():
        for emg_num, labels_set in emg_map_sets.items():
            if len(labels_set) > 1:
                inconsistent.append((sid, emg_num, list(labels_set)))
    if inconsistent:
        log("[warn] inconsistent EMG labels within ID (per emg#):", log_lines)
        for sid, emg_num, labs in inconsistent:
            log(f"  {sid} emg#{emg_num}: {labs}", log_lines)

    # Map SNR to single value per ID/channel
    id_snr = defaultdict(dict)
    for sid, ch_dict in id_cycle_snr.items():
        for label, vals in ch_dict.items():
            vals = np.array(vals, dtype=float).reshape(-1)
            id_snr[sid][label] = float(np.nanmedian(vals))

    # Compute amplitude normalization, ID traces
    id_traces = defaultdict(dict)
    id_cycles_norm = defaultdict(dict)
    id_cycles_all = defaultdict(dict)
    within_stats = {}
    cycle_corrs_all = []
    cycle_corr_map = defaultdict(dict)
    cycle_lag_map = defaultdict(dict)
    # Pass 1: initial ID traces and within-cycle correlations
    for sid, ch_dict in id_channels.items():
        for label, cycles in ch_dict.items():
            if len(cycles) < args.min_cycles:
                continue
            X = np.asarray(cycles)
            mu = float(np.mean(X))
            sd = float(np.std(X))
            norm = mu + 3.0 * sd
            if norm <= 0:
                norm = 1.0
            Xn = X / norm
            id_trace = aggregate_cycles(Xn, args.agg)
            if id_trace is None:
                continue
            corr_list = []
            lag_list = []
            for i, cyc in enumerate(Xn):
                c_lim, l_lim = cross_corr(cyc, id_trace, max_lag=args.within_lag_max)
                c_full, l_full = cross_corr(cyc, id_trace, max_lag=None)
                corr_list.append(c_lim)
                lag_list.append(l_lim)
                cycle_corrs_all.append(c_lim)
                cycle_corr_map[(sid, label)][i] = c_lim
                cycle_lag_map[(sid, label)][i] = l_lim
            id_cycles_norm[sid][label] = Xn
            id_cycles_all[sid][label] = Xn
            id_traces[sid][label] = id_trace
            within_stats[(sid, label)] = {
                "n_cycles": int(Xn.shape[0]),
                "corr_med": float(np.nanmedian(corr_list)) if corr_list else np.nan,
                "lag_med": float(np.nanmedian(lag_list)) if lag_list else np.nan,
                "corr_mean": float(np.nanmean(corr_list)) if corr_list else np.nan,
                "lag_mean": float(np.nanmean(lag_list)) if lag_list else np.nan,
            }
            pb1_step += 1
            pb1.update(pb1_step)

    # Homogeneity test: flag outlier cycles (low corr to ID trace)
    within_corr_low = compute_thresholds(cycle_corrs_all, mode="lower", quantile=0.025)
    outlier_cycles = defaultdict(dict)
    for sid, ch_dict in id_cycles_norm.items():
        for label, Xn in ch_dict.items():
            n = Xn.shape[0]
            mask = np.zeros(n, dtype=bool)
            for i in range(n):
                c = cycle_corr_map.get((sid, label), {}).get(i, np.nan)
                if np.isfinite(within_corr_low) and np.isfinite(c) and c < within_corr_low:
                    mask[i] = True
            outlier_cycles[(sid, label)] = mask

    # Pass 2: remove outliers and recompute ID traces + within stats
    for sid, ch_dict in list(id_cycles_norm.items()):
        for label, Xn in list(ch_dict.items()):
            mask = outlier_cycles.get((sid, label), np.zeros(Xn.shape[0], dtype=bool))
            Xf = Xn[~mask] if mask.size else Xn
            if Xf.shape[0] < args.min_cycles:
                continue
            id_trace = aggregate_cycles(Xf, args.agg)
            id_cycles_norm[sid][label] = Xf
            id_traces[sid][label] = id_trace
            corr_list = []
            lag_list = []
            corr_full_list = []
            lag_full_list = []
            lag_lim_list = []
            for cyc in Xf:
                c_lim, l_lim = cross_corr(cyc, id_trace, max_lag=args.within_lag_max)
                c_full, l_full = cross_corr(cyc, id_trace, max_lag=None)
                corr_list.append(c_lim)
                lag_list.append(l_lim)
                corr_full_list.append(c_full)
                lag_full_list.append(l_full)
                lag_lim_list.append(l_lim)
            # Early vs late homogeneity
            early_late_corr = np.nan
            early_late_lag = np.nan
            if Xf.shape[0] >= 4:
                mid = Xf.shape[0] // 2
                early = aggregate_cycles(Xf[:mid], args.agg)
                late = aggregate_cycles(Xf[mid:], args.agg)
                if early is not None and late is not None:
                    early_late_corr, early_late_lag = cross_corr(early, late, max_lag=args.within_lag_max)
            homogeneity_flag = False
            if np.isfinite(within_corr_low) and np.isfinite(early_late_corr):
                homogeneity_flag = early_late_corr < within_corr_low
            within_stats[(sid, label)] = {
                "n_cycles": int(Xf.shape[0]),
                "n_outliers": int(mask.sum()),
                "corr_med": float(np.nanmedian(corr_list)) if corr_list else np.nan,
                "lag_med": float(np.nanmedian(lag_full_list)) if lag_full_list else np.nan,
                "lag_abs_med": float(np.nanmedian(np.abs(lag_full_list))) if lag_full_list else np.nan,
                "lag_abs_med_lim": float(np.nanmedian(np.abs(lag_lim_list))) if lag_lim_list else np.nan,
                "corr_mean": float(np.nanmean(corr_list)) if corr_list else np.nan,
                "lag_mean": float(np.nanmean(lag_full_list)) if lag_full_list else np.nan,
                "corr_med_full": float(np.nanmedian(corr_full_list)) if corr_full_list else np.nan,
                "lag_med_lim": float(np.nanmedian(lag_list)) if lag_list else np.nan,
                "early_late_corr": float(early_late_corr) if np.isfinite(early_late_corr) else np.nan,
                "early_late_lag": float(early_late_lag) if np.isfinite(early_late_lag) else np.nan,
                "homogeneity_flag": bool(homogeneity_flag),
            }
            pb1_step += 1
            pb1.update(pb1_step)

    # Build global traces by label
    label_id_traces = defaultdict(list)
    label_id_counts = defaultdict(list)
    label_id_raw_ids = defaultdict(set)
    for sid, ch_dict in id_traces.items():
        for label, trace in ch_dict.items():
            if trace is not None:
                label_id_traces[label].append(trace)
                ncy = within_stats.get((sid, label), {}).get("n_cycles", 0)
                label_id_counts[label].append(int(ncy))
    # Count IDs that have any raw cycles for each label (before filtering)
    for sid, ch_dict in id_channels.items():
        for label, cycles in ch_dict.items():
            if cycles:
                label_id_raw_ids[label].add(sid)

    global_traces = {}
    for label, traces in label_id_traces.items():
        if traces:
            global_traces[label] = aggregate_cycles(traces, "pca")
    total_ids = len(id_files)
    global_label_summary = {}
    for label, counts in label_id_counts.items():
        if counts:
            counts_arr = np.array(counts, dtype=float)
            global_label_summary[label] = {
                "n_ids_used": int(len(counts)),
                "n_ids_total": int(total_ids),
                "n_ids_raw": int(len(label_id_raw_ids.get(label, set()))),
                "cycles_min": float(np.min(counts_arr)),
                "cycles_median": float(np.median(counts_arr)),
                "cycles_max": float(np.max(counts_arr)),
            }

    # Build label->info map for side/muscle lookups
    label_info_global = {}
    for sid, ch_info in id_channel_info.items():
        for label, info in ch_info.items():
            if "canon" not in info:
                info = dict(info)
                info["canon"] = canonical_muscle(info.get("muscle_key", ""))
            label_info_global[label] = info

    # Compute correlation and lag per ID/channel
    corr_stats = {}
    for sid, ch_dict in id_traces.items():
        for label, trace in ch_dict.items():
            g = global_traces.get(label)
            if g is None or trace is None:
                continue
            corr, lag = cross_corr(trace, g)
            corr_stats[(sid, label)] = (corr, lag)

    # MAD stats (ID trace vs global trace, full-length circular shift at lag=0)
    mad_stats = {}
    for sid, ch_dict in id_traces.items():
        for label, trace in ch_dict.items():
            g = global_traces.get(label)
            if g is None or trace is None:
                continue
            mad0 = mad_for_lag(trace, g, 0)
            mad_stats[(sid, label)] = {"mad0": mad0}

    # Empirical thresholds
    snr_vals = [v for sid in id_snr for v in id_snr[sid].values() if np.isfinite(v)]
    snr_low = compute_thresholds(snr_vals, mode="lower", quantile=0.025)

    def compute_corr_lag_thresholds(masked_keys):
        corrs = []
        lags = []
        for key in masked_keys:
            corr, lag = corr_stats.get(key, (np.nan, np.nan))
            if np.isfinite(corr):
                corrs.append(corr)
            if np.isfinite(lag):
                lags.append(abs(lag))
        corr_thr = compute_thresholds(corrs, mode="lower", quantile=0.025)
        lag_thr = compute_thresholds(lags, mode="upper", quantile=0.025)
        return corr_thr, lag_thr

    # pass 1
    keys_all = list(corr_stats.keys())
    keys_good_snr = [k for k in keys_all if id_snr.get(k[0], {}).get(k[1], np.inf) >= snr_low]
    corr_thr_1, lag_thr_1 = compute_corr_lag_thresholds(keys_good_snr)

    corr_thr, lag_thr = corr_thr_1, lag_thr_1
    if args.threshold_pass >= 2:
        # exclude obvious failures before recompute
        keys_pass = []
        for k in keys_good_snr:
            corr, lag = corr_stats.get(k, (np.nan, np.nan))
            if corr >= corr_thr_1 and abs(lag) <= lag_thr_1:
                keys_pass.append(k)
        if keys_pass:
            corr_thr, lag_thr = compute_corr_lag_thresholds(keys_pass)

    log(f"[threshold] SNR low (2.5%): {snr_low:.3f}", log_lines)
    log(f"[threshold] corr low (2.5%): {corr_thr:.3f}", log_lines)
    log(f"[threshold] |lag| high (97.5%): {lag_thr:.1f}", log_lines)
    log(f"[threshold] within corr low (2.5%): {within_corr_low:.3f}", log_lines)

    # Validation step
    status = {}
    for (sid, label), (corr, lag) in corr_stats.items():
        snr_val = id_snr.get(sid, {}).get(label, np.nan)
        if np.isfinite(snr_low) and snr_val < snr_low:
            status[(sid, label)] = "low_snr"
        elif (not np.isfinite(corr_thr)) or (not np.isfinite(lag_thr)):
            status[(sid, label)] = "unknown"
        elif corr < corr_thr or abs(lag) > lag_thr:
            status[(sid, label)] = "flagged"
        else:
            status[(sid, label)] = "validated"

    # Enforce within-ID constraints (lag limit + homogeneity)
    validation_reasons = {}
    for (sid, label), st in list(status.items()):
        if st in ("low_snr", "unknown"):
            continue
        ws = within_stats.get((sid, label))
        if not ws:
            status[(sid, label)] = "flagged_within_missing"
            validation_reasons[(sid, label)] = ["within_missing"]
            continue
        reasons = []
        if np.isfinite(ws.get("lag_abs_med_lim", np.nan)) and ws.get("lag_abs_med_lim", 0.0) > args.within_lag_max:
            reasons.append("within_lag")
        if ws.get("homogeneity_flag", False):
            reasons.append("homogeneity")
        if np.isfinite(within_corr_low) and np.isfinite(ws.get("corr_med", np.nan)) and ws.get("corr_med", 0.0) < within_corr_low:
            reasons.append("within_corr")
        if reasons:
            status[(sid, label)] = "flagged_within"
            validation_reasons[(sid, label)] = reasons

    log("[summary] validation status (per ID/label):", log_lines, also_print=False)
    for (sid, label), st in status.items():
        reasons = validation_reasons.get((sid, label), [])
        reason_txt = f" reasons={','.join(reasons)}" if reasons else ""
        log(f"  {sid}::{label}: {st}{reason_txt}", log_lines, also_print=False)

    # Build CI stats from validated members (per label) using limited lag
    validated_stats = defaultdict(lambda: {"corr": [], "lag": [], "mad": []})
    for (sid, label), st in status.items():
        if st != "validated":
            continue
        trace = id_traces.get(sid, {}).get(label)
        g = global_traces.get(label)
        if trace is None or g is None:
            continue
        lags_lim = range(-args.within_lag_max, args.within_lag_max + 1)
        corr_v, lag_v = best_corr_circular(trace, g, lags_lim)
        mad_v = mad_for_lag(trace, g, lag_v)
        validated_stats[label]["corr"].append(corr_v)
        validated_stats[label]["lag"].append(lag_v)
        validated_stats[label]["mad"].append(mad_v)

    validated_ci = {}
    for label, vals in validated_stats.items():
        corr_ci = compute_ci(vals["corr"], level=0.80)
        lag_ci = compute_ci(vals["lag"], level=0.80)
        mad_ci = compute_ci(vals["mad"], level=0.80)
        validated_ci[label] = {"corr": corr_ci, "lag": lag_ci, "mad": mad_ci}

    log("[summary] validated_ci (per label, 80%):", log_lines, also_print=False)
    for label, ci in validated_ci.items():
        c_low, c_high, c_n = ci["corr"]
        l_low, l_high, l_n = ci["lag"]
        m_low, m_high, m_n = ci["mad"]
        log(
            f"  {label}: corr_ci=[{c_low:.3f},{c_high:.3f}] n={c_n} "
            f"lag_ci=[{l_low:.1f},{l_high:.1f}] n={l_n} "
            f"mad_ci=[{m_low:.4f},{m_high:.4f}] n={m_n}",
            log_lines,
            also_print=False,
        )

    # (LR CI removed; LR mapping uses normal validated CI after phase-shift alignment)

    # Re-check status against label CI (more strict)
    for (sid, label), st in list(status.items()):
        if st in ("low_snr", "unknown"):
            continue
        g = global_traces.get(label)
        trace = id_traces.get(sid, {}).get(label)
        if g is None or trace is None:
            continue
        lags_lim = range(-args.within_lag_max, args.within_lag_max + 1)
        corr_c, lag_c = best_corr_circular(trace, g, lags_lim)
        mad_c = mad_for_lag(trace, g, lag_c)
        ci = validated_ci.get(label)
        if not ci:
            continue
        if not (ci["corr"][0] <= corr_c <= ci["corr"][1] and ci["lag"][0] <= lag_c <= ci["lag"][1] and ci["mad"][0] <= mad_c <= ci["mad"][1]):
            status[(sid, label)] = "flagged_ci"

    log("[summary] validation status after ci (per ID/label):", log_lines, also_print=False)
    for (sid, label), st in status.items():
        log(f"  {sid}::{label}: {st}", log_lines, also_print=False)

    # Progress bar 2: validation, plots, remap, mapping write
    remap_candidates = [k for k, st in status.items() if st.startswith("flagged")]
    plot_count = (len(label_id_traces) + len(status) + len(remap_candidates)) if not args.no_plots else 0
    total_pb2 = plot_count + len(remap_candidates)
    pb2 = ProgressBar("remap+plots", total=max(1, total_pb2))
    pb2_step = 0

    # Log summaries
    log(f"[summary] total IDs processed: {total_ids}", log_lines, also_print=False)
    log("[summary] global_label_summary (per label):", log_lines, also_print=False)
    for label, summ in global_label_summary.items():
        log(
            f"  {label}: n_ids_used={summ['n_ids_used']} n_ids_raw={summ['n_ids_raw']} n_ids_total={summ['n_ids_total']} "
            f"cycles_min={summ['cycles_min']:.0f} cycles_med={summ['cycles_median']:.0f} cycles_max={summ['cycles_max']:.0f}",
            log_lines,
            also_print=False,
        )
    log("[summary] within_id_stats (per ID/label):", log_lines, also_print=False)
    for (sid, label), ws in within_stats.items():
        log(
            f"  {sid}::{label}: n_cycles={ws.get('n_cycles', 0)} n_outliers={ws.get('n_outliers', 0)} "
            f"corr_med={ws.get('corr_med', np.nan):.3f} lag_med_full={ws.get('lag_med', np.nan):.1f} "
            f"lag_abs_med={ws.get('lag_abs_med', np.nan):.1f} lag_abs_med_lim={ws.get('lag_abs_med_lim', np.nan):.1f} "
            f"lag_med_lim={ws.get('lag_med_lim', np.nan):.1f} "
            f"early_late_corr={ws.get('early_late_corr', np.nan):.3f} early_late_lag={ws.get('early_late_lag', np.nan):.1f} "
            f"homogeneity_flag={ws.get('homogeneity_flag', False)}",
            log_lines,
            also_print=False,
        )
    log("[summary] between_id_stats (per ID/label):", log_lines, also_print=False)
    for (sid, label), (corr, lag) in corr_stats.items():
        log(
            f"  {sid}::{label}: corr={corr:.3f} lag={lag} n_ids_used={len(label_id_traces.get(label, []))} n_ids_total={total_ids}",
            log_lines,
            also_print=False,
        )

    # Plots for validation (per ID/channel + global per muscle)
    if not args.no_plots:
        # Global plots per muscle/label
        for label, traces in label_id_traces.items():
            g = global_traces.get(label)
            if g is None:
                continue
            fname = safe_slug(f"{label}__global.jpg")
            plot_global_channel(
                out_global / fname,
                traces,
                g,
                label,
                len(traces),
                total_ids,
            )
            pb2_step += 1
            pb2.update(pb2_step)

        # Per ID/channel plots
        for (sid, label), st in status.items():
            if st == "unknown":
                continue
            cycles_used = id_cycles_norm.get(sid, {}).get(label)
            cycles_all = id_cycles_all.get(sid, {}).get(label)
            trace = id_traces.get(sid, {}).get(label)
            g = global_traces.get(label)
            if cycles_used is None or cycles_all is None or trace is None or g is None:
                continue
            within = within_stats.get((sid, label), {})
            out_dir = out_mapping
            # within-ID xcorr median (cycles vs ID trace)
            xcorr_stack = []
            for cyc in cycles_used:
                xcorr_stack.append(build_xcorr_series(cyc, trace, max_lag=None))
            xcorr_med = (
                np.median(np.asarray(xcorr_stack), axis=0)
                if xcorr_stack
                else build_xcorr_series(trace, trace, max_lag=None)
            )
            # between-ID xcorr (ID trace vs global)
            xcorr_between = build_xcorr_series(trace, g, max_lag=None)
            within_fname = safe_slug(f"{sid}__{label}__idplot__{st}.jpg")
            plot_id_channel(
                out_dir / within_fname,
                cycles_all,
                outlier_cycles.get((sid, label), np.zeros(cycles_all.shape[0], dtype=bool)),
                trace,
                g,
                xcorr_med,
                within.get("lag_med", np.nan),
                xcorr_between,
                corr_stats.get((sid, label), (np.nan, 0))[1],
                st,
                args.agg,
                id_snr.get(sid, {}).get(label, np.nan),
                within.get("n_cycles", 0),
                within.get("n_outliers", 0),
                len(label_id_traces.get(label, [])),
                total_ids,
                within.get("corr_med", np.nan),
                within.get("lag_med", np.nan),
                within.get("lag_abs_med", np.nan),
                within.get("lag_med_lim", np.nan),
                within.get("early_late_corr", np.nan),
                within.get("early_late_lag", np.nan),
                corr_stats.get((sid, label), (np.nan, 0))[0],
                args.within_lag_max,
                mapping_text=(
                    f"mapped from {sid}::{label} to {label}: "
                    f"{'success' if st == 'validated' else 'fail'} -> "
                    f"{'ok' if st == 'validated' else 'remap'}"
                ),
            )
            pb2_step += 1
            pb2.update(pb2_step)

    # Remapping step (MAD-based suggestions)
    mapping = defaultdict(dict)
    remap_results = {}
    used_targets = set()

    def candidate_labels_for_id(sid):
        return list(global_traces.keys())

    def within_ci(val, ci):
        low, high, n = ci
        if not np.isfinite(val) or not np.isfinite(low) or not np.isfinite(high):
            return False
        return low <= val <= high

    def ci_ok(label, corr, lag, mad):
        ci = validated_ci.get(label)
        if not ci:
            return False
        return (
            within_ci(corr, ci["corr"])
            and within_ci(lag, ci["lag"])
            and within_ci(mad, ci["mad"])
        )


    flagged_keys = [k for k, st in status.items() if st.startswith("flagged")]
    for sid, label in flagged_keys:
        trace = id_traces.get(sid, {}).get(label)
        if trace is None:
            continue
        snr_val = id_snr.get(sid, {}).get(label, np.nan)
        low_snr_flag = bool(np.isfinite(snr_low) and snr_val < snr_low)
        info = label_info_global.get(label, {})
        side = info.get("side")
        muscle_key = info.get("muscle_key")
        canon = info.get("canon")

        g_same = global_traces.get(label)
        lags_lim = range(-args.within_lag_max, args.within_lag_max + 1)
        if g_same is not None:
            base_corr, base_lag = best_corr_circular(trace, g_same, lags_lim)
            base_mad = mad_for_lag(trace, g_same, base_lag)
        else:
            base_corr, base_lag = np.nan, 0
            base_mad = mad_stats.get((sid, label), {}).get("mad0", np.nan)

        def improvement_ok(corr_c, mad_c):
            if not np.isfinite(base_mad) or base_mad <= 0:
                return False
            corr_up = corr_c - base_corr
            mad_down = (base_mad - mad_c) / max(base_mad, 1e-8)
            return (corr_up >= args.corr_gain) and (mad_down >= args.mad_gain)

        suggestion = None
        best_candidate = None  # (label, mad, lag, corr, tag)
        # Phase-shift (L/R) test
        if side and muscle_key:
            opp_side = "L" if side == "R" else "R"
            candidates = [
                lab for lab, inf in label_info_global.items()
                if inf.get("side") == opp_side and inf.get("muscle_key") == muscle_key
            ]
            if candidates:
                cand = candidates[0]
                g = global_traces.get(cand)
                if g is not None:
                    lag_min, lag_max = args.lag_lr_range
                    lags = list(range(-lag_max, -lag_min + 1)) + list(range(lag_min, lag_max + 1))
                    corr_c, best_lag = best_corr_circular(trace, g, lags)
                    shifted = circ_shift(trace, best_lag)
                    corr_align, lag_align = best_corr_circular(shifted, g, lags_lim)
                    mad_align = mad_for_lag(shifted, g, lag_align)
                    best_candidate = (cand, mad_align, best_lag, corr_align, "lr")
                    if improvement_ok(corr_align, mad_align) and (not args.use_ci or ci_ok(cand, corr_align, lag_align, mad_align)):
                        suggestion = ("suggest_lr", cand, mad_align, best_lag, corr_align)

        # Close muscle test (same side, no shift)
        if suggestion is None and side and canon:
            close_set = close_muscle_keys(canon)
            cand_labels = []
            for cand in candidate_labels_for_id(sid):
                if cand == label:
                    continue
                info_c = label_info_global.get(cand, {})
                if info_c.get("side") != side:
                    continue
                if info_c.get("canon") in close_set:
                    cand_labels.append(cand)
            best = (None, np.inf, 0, -np.inf)
            for cand in cand_labels:
                g = global_traces.get(cand)
                if g is None:
                    continue
                corr_c, lag_c = best_corr_circular(trace, g, lags_lim)
                cand_mad = mad_for_lag(trace, g, lag_c)
                if np.isfinite(cand_mad):
                    score = (corr_c - base_corr) + ((base_mad - cand_mad) / max(base_mad, 1e-8))
                    if score > best[3]:
                        best = (cand, cand_mad, lag_c, corr_c)
            if best[0] is not None:
                if best_candidate is None or (np.isfinite(best[1]) and best[1] < best_candidate[1]):
                    best_candidate = (best[0], best[1], best[2], best[3], "close")
                if improvement_ok(best[3], best[1]) and (not args.use_ci or ci_ok(best[0], best[3], best[2], best[1])):
                    suggestion = ("suggest_close", best[0], best[1], best[2], best[3])

        # General muscle test (same side, no shift)
        if suggestion is None and side:
            cand_labels = []
            for cand in candidate_labels_for_id(sid):
                if cand == label:
                    continue
                info_c = label_info_global.get(cand, {})
                if info_c.get("side") != side:
                    continue
                if muscle_key and info_c.get("muscle_key") == muscle_key:
                    continue
                cand_labels.append(cand)
            best = (None, np.inf, 0, -np.inf)
            for cand in cand_labels:
                g = global_traces.get(cand)
                if g is None:
                    continue
                corr_c, lag_c = best_corr_circular(trace, g, lags_lim)
                cand_mad = mad_for_lag(trace, g, lag_c)
                if np.isfinite(cand_mad):
                    score = (corr_c - base_corr) + ((base_mad - cand_mad) / max(base_mad, 1e-8))
                    if score > best[3]:
                        best = (cand, cand_mad, lag_c, corr_c)
            if best[0] is not None:
                if best_candidate is None or (np.isfinite(best[1]) and best[1] < best_candidate[1]):
                    best_candidate = (best[0], best[1], best[2], best[3], "general")
                if improvement_ok(best[3], best[1]) and (not args.use_ci or ci_ok(best[0], best[3], best[2], best[1])):
                    suggestion = ("suggest_general", best[0], best[1], best[2], best[3])

        if suggestion and not low_snr_flag:
            st, target, cand_mad, best_lag, corr_c = suggestion
            if st == "suggest_lr":
                tag = "lr"
            elif st == "suggest_close":
                tag = "close"
            else:
                tag = "general"
            if target not in used_targets:
                mapping[sid][label] = target
                remap_results[(sid, label)] = (st, target, (cand_mad, best_lag, base_mad, corr_c, tag))
                used_targets.add(target)
            else:
                remap_results[(sid, label)] = ("unsure_target_used", None, None)
        else:
            if best_candidate is not None:
                bc_lab, bc_mad, bc_lag, bc_corr, bc_tag = best_candidate
                reason = "low_snr" if low_snr_flag else "ci"
                remap_results[(sid, label)] = (f"unsure_{reason}", bc_lab, (bc_mad, bc_lag, base_mad, bc_corr, bc_tag))
            else:
                reason = "low_snr" if low_snr_flag else "ci"
                remap_results[(sid, label)] = (f"unsure_{reason}", None, (base_mad,))
        pb2_step += 1
        pb2.update(pb2_step)

    log("[summary] remap_suggestions (per ID/label):", log_lines, also_print=False)
    for (sid, label), (st, target, stats) in remap_results.items():
        if target:
            cand_mad = stats[0] if stats else np.nan
            best_lag = stats[1] if stats else np.nan
            base_mad = stats[2] if stats and len(stats) > 2 else np.nan
            corr_c = stats[3] if stats and len(stats) > 3 else np.nan
            tag = stats[4] if stats and len(stats) > 4 else ""
            log(
                f"  {sid}::{label}: {st} -> {target} mad0={base_mad:.4f} mad_cand={cand_mad:.4f} "
                f"corr_cand={corr_c:.3f} lag={best_lag} tag={tag}",
                log_lines,
                also_print=False,
            )
        else:
            base_mad = stats[0] if stats else np.nan
            log(
                f"  {sid}::{label}: {st} mad0={base_mad:.4f}",
                log_lines,
                also_print=False,
            )

    # LR flip diagnostics (data-driven)
    lr_lags = []
    lr_by_id = defaultdict(int)
    lr_total_by_id = defaultdict(int)
    for (sid, label), (st, target, stats) in remap_results.items():
        if st == "suggest_lr" and stats:
            lr_lags.append(float(stats[1]))
            lr_by_id[sid] += 1
        lr_total_by_id[sid] += 1
    if lr_lags:
        larr = np.array(lr_lags, dtype=float)
        p10, p50, p90 = np.nanpercentile(np.abs(larr), [10, 50, 90])
        log(f"[summary] lr_suggest_lag_abs_p10_p50_p90={p10:.1f},{p50:.1f},{p90:.1f}", log_lines, also_print=False)
    log("[summary] lr_suggest_ratio (per ID):", log_lines, also_print=False)
    for sid, tot in lr_total_by_id.items():
        if tot <= 0:
            continue
        ratio = lr_by_id.get(sid, 0) / float(tot)
        log(f"  {sid}: {lr_by_id.get(sid, 0)}/{tot} ratio={ratio:.2f}", log_lines, also_print=False)

    # Plots for remapping results
    if not args.no_plots:
        for (sid, label), (st, target, stats) in remap_results.items():
            trace = id_traces.get(sid, {}).get(label)
            cycles_used = id_cycles_norm.get(sid, {}).get(label)
            cycles_all = id_cycles_all.get(sid, {}).get(label)
            if trace is None or cycles_used is None or cycles_all is None:
                continue
            within = within_stats.get((sid, label), {})
            mad_info = None
            shifted_trace = None
            shifted_label = None
            tag = ""
            if stats and len(stats) >= 5:
                tag = stats[4]
            if st.startswith("suggest_"):
                outcome = "success"
                if tag == "lr":
                    reason = "l/r switch"
                elif tag == "close":
                    reason = "muscle switch/close"
                elif tag == "general":
                    reason = "muscle switch/general"
                else:
                    reason = "muscle switch/ok"
            else:
                outcome = "fail"
                if tag == "lr":
                    reason = f"l/r switch/{st}"
                elif tag == "close":
                    reason = f"muscle switch/close/{st}"
                elif tag == "general":
                    reason = f"muscle switch/general/{st}"
                else:
                    reason = f"muscle switch/{st}"
            if target:
                g = global_traces.get(target)
                out_dir = out_mapping
                used_label = target
                if stats and len(stats) >= 2:
                    cand_mad = stats[0]
                    best_lag = stats[1]
                    base_mad = stats[2] if len(stats) >= 3 else np.nan
                    corr_c = stats[3] if len(stats) >= 4 else np.nan
                    mad_info = f"mad0={base_mad:.4f}, mad_cand={cand_mad:.4f}, corr_cand={corr_c:.3f}, lag={best_lag}"
                    if tag:
                        mad_info = mad_info + f", tag={tag}"
                    if st == "suggest_lr" and np.isfinite(best_lag):
                        shifted_trace = shift_trace_for_plot(trace, int(best_lag))
                        shifted_label = f"ID trace shifted (lag={int(best_lag)})"
            else:
                g = global_traces.get(label)
                out_dir = out_mapping
                used_label = label
                if stats and len(stats) >= 1:
                    base_mad = stats[0]
                    mad_info = f"mad0={base_mad:.4f}"
            if g is None:
                continue
            corr, lag = cross_corr(trace, g)
            xcorr_between = build_xcorr_series(trace, g, max_lag=None)
            xcorr_stack = [build_xcorr_series(cyc, trace, max_lag=None) for cyc in cycles_used]
            xcorr_med = np.median(np.asarray(xcorr_stack), axis=0) if xcorr_stack else build_xcorr_series(trace, trace, max_lag=None)
            fname = safe_slug(f"{sid}__{label}__remap_idplot__{st}.jpg")
            plot_id_channel(
                out_dir / fname,
                cycles_all,
                outlier_cycles.get((sid, label), np.zeros(cycles_all.shape[0], dtype=bool)),
                trace,
                g,
                xcorr_med,
                within.get("lag_med", np.nan),
                xcorr_between,
                lag,
                st,
                args.agg,
                id_snr.get(sid, {}).get(label, np.nan),
                within.get("n_cycles", 0),
                within.get("n_outliers", 0),
                len(label_id_traces.get(used_label, [])),
                total_ids,
                within.get("corr_med", np.nan),
                within.get("lag_med", np.nan),
                within.get("lag_abs_med", np.nan),
                within.get("lag_med_lim", np.nan),
                within.get("early_late_corr", np.nan),
                within.get("early_late_lag", np.nan),
                corr,
                args.within_lag_max,
                mad_info,
                shifted_trace,
                shifted_label,
                mapping_text=(
                    f"mapped from {sid}::{label} to {used_label}: "
                    f"{outcome} -> {reason}"
                ),
            )

    # Interactive review for borderline unsure cases
    if args.interactive_unsure:
        review_items = []
        seen_review = set()
        # Include suggested swaps for review
        for (sid, label), (st, target, stats) in remap_results.items():
            if not st.startswith("suggest_"):
                continue
            if not target:
                continue
            trace = id_traces.get(sid, {}).get(label)
            cycles_all = id_cycles_all.get(sid, {}).get(label)
            if trace is None or cycles_all is None:
                continue
            out_mask = outlier_cycles.get((sid, label), np.zeros(cycles_all.shape[0], dtype=bool))
            if (sid, label) in seen_review:
                continue
            review_items.append(
                {
                    "sid": sid,
                    "label": label,
                    "candidate": target,
                    "reason": st,
                    "trace": trace,
                    "cycles": cycles_all,
                    "outliers": out_mask,
                }
            )
            seen_review.add((sid, label))
        close_frac = max(0.0, min(1.0, args.interactive_close_frac))
        for (sid, label), (st, target, stats) in remap_results.items():
            if not st.startswith("unsure_"):
                continue
            if not target or not stats or len(stats) < 5:
                continue
            trace = id_traces.get(sid, {}).get(label)
            cycles_all = id_cycles_all.get(sid, {}).get(label)
            if trace is None or cycles_all is None:
                continue
            out_mask = outlier_cycles.get((sid, label), np.zeros(cycles_all.shape[0], dtype=bool))
            cand_mad = stats[0]
            base_mad = stats[2] if len(stats) >= 3 else np.nan
            corr_c = stats[3] if len(stats) >= 4 else np.nan
            g_same = global_traces.get(label)
            if g_same is None:
                continue
            lags_lim = range(-args.within_lag_max, args.within_lag_max + 1)
            base_corr, base_lag = best_corr_circular(trace, g_same, lags_lim)
            if not np.isfinite(base_mad) or base_mad <= 0:
                continue
            corr_up = corr_c - base_corr
            mad_down = (base_mad - cand_mad) / max(base_mad, 1e-8)
            if corr_up >= args.corr_gain * close_frac and mad_down >= args.mad_gain * close_frac:
                if (sid, label) not in seen_review:
                    review_items.append(
                        {
                            "sid": sid,
                            "label": label,
                            "candidate": target,
                            "reason": st,
                            "trace": trace,
                            "cycles": cycles_all,
                            "outliers": out_mask,
                        }
                    )
                    seen_review.add((sid, label))
        if review_items:
            feedback_path = out_processing / "interactive_feedback.jsonl"
            decisions = interactive_review(review_items, global_traces, feedback_path)
            for rec in decisions:
                if rec.get("response") == "swap":
                    mapping[rec["sid"]][rec["label"]] = rec["candidate"]
                    remap_results[(rec["sid"], rec["label"])] = ("user_swap", rec["candidate"], None)
                elif rec.get("response") == "ok":
                    if rec["sid"] in mapping and rec["label"] in mapping[rec["sid"]]:
                        del mapping[rec["sid"]][rec["label"]]
                    remap_results[(rec["sid"], rec["label"])] = ("user_keep", None, None)
            pb2_step += 1
            pb2.update(pb2_step)

    # Normalize reciprocal swaps to single entries
    mapping = normalize_mapping_pairs(mapping)
    # Write mapping JSON
    mapping_out = {sid: lab_map for sid, lab_map in mapping.items()}
    mapping_path = out_base / "mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_out, f, indent=2, ensure_ascii=False)
    log(f"[write] mapping: {mapping_path}", log_lines)

    # Write log
    log_path = out_base / "processing" / "log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Apply mapping to COPIED C3Ds if requested
    if args.apply:
        apply_root = Path(args.apply_root) if args.apply_root else (out_processing / "applied_c3d")
        ensure_dir(apply_root)
        if args.apply_mode == "C3Dcopy":
            apply_mapping_to_copies(
                data_root, c3ds, mapping, apply_root, args.task, log_lines
            )
        elif args.apply_mode == "C3Doverwrite":
            apply_mapping_in_place(
                data_root, c3ds, mapping, args.task, log_lines
            )
        else:
            raise ValueError("--apply-mode must be C3Dcopy or C3Doverwrite.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    log("[done]", log_lines)


def build_xcorr_series(a, b, max_lag=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    out = []
    if max_lag is None:
        lag_range = range(-(n - 1), n)
    else:
        max_lag = int(max_lag)
        lag_range = range(-max_lag, max_lag + 1)
    for lag in lag_range:
        if lag < 0:
            aa = a[-lag:]
            bb = b[: n + lag]
        else:
            aa = a[: n - lag]
            bb = b[lag:]
        if aa.size < 3:
            out.append(0.0)
            continue
        denom = (np.std(aa) * np.std(bb) + 1e-8)
        corr = float(np.mean((aa - np.mean(aa)) * (bb - np.mean(bb))) / denom)
        out.append(corr)
    return np.array(out, dtype=float)


def normalize_mapping_pairs(mapping):
    out = defaultdict(dict)
    for sid, lab_map in mapping.items():
        used = set()
        for src, dst in lab_map.items():
            if src == dst:
                continue
            if src in used:
                continue
            if dst in lab_map and lab_map.get(dst) == src:
                key = src if src < dst else dst
                val = dst if src < dst else src
                out[sid][key] = val
                used.add(src)
                used.add(dst)
            else:
                out[sid][src] = dst
                used.add(src)
    return out


def apply_label_mapping(labels, label_map):
    idx = {str(lab): i for i, lab in enumerate(labels)}
    processed = set()
    changed = 0
    for src, dst in label_map.items():
        src = str(src)
        dst = str(dst)
        if src not in idx:
            continue
        i = idx[src]
        if i in processed:
            continue
        if dst in idx:
            j = idx[dst]
            if j in processed:
                continue
            labels[i], labels[j] = labels[j], labels[i]
            processed.add(i)
            processed.add(j)
            idx[labels[i]] = i
            idx[labels[j]] = j
            changed += 2
        else:
            labels[i] = dst
            processed.add(i)
            idx[dst] = i
            changed += 1
    return labels, changed


def apply_mapping_to_copies(data_root, c3ds, mapping, apply_root, task_substring, log_lines):
    # Copy only task-matched C3Ds to apply_root, preserving relative structure
    for c3d_path in c3ds:
        desc = parse_enf_description(c3d_path)
        if not match_task(desc, task_substring):
            continue
        rel = c3d_path.relative_to(data_root)
        out_path = apply_root / rel
        ensure_dir(out_path.parent)
        shutil.copy2(c3d_path, out_path)

        try:
            c3d = ezc3d.c3d(str(out_path))
        except Exception:
            log(f"[warn] apply: failed to read {out_path}", log_lines)
            continue
        sid = get_subject_id(c3d)
        if sid not in mapping:
            continue
        label_map = mapping[sid]
        if not label_map:
            continue
        asec = c3d["parameters"]["ANALOG"]
        labels = list(asec["LABELS"]["value"])
        changed = 0
        labels, changed = apply_label_mapping(labels, label_map)
        asec["LABELS"]["value"] = labels
        if changed:
            c3d.write(str(out_path))
            log(f"[apply] patched {out_path} (changed {changed})", log_lines)


def apply_mapping_in_place(data_root, c3ds, mapping, task_substring, log_lines):
    for c3d_path in c3ds:
        desc = parse_enf_description(c3d_path)
        if not match_task(desc, task_substring):
            continue
        try:
            c3d = ezc3d.c3d(str(c3d_path))
        except Exception:
            log(f"[warn] apply: failed to read {c3d_path}", log_lines)
            continue
        sid = get_subject_id(c3d)
        if sid not in mapping:
            continue
        label_map = mapping[sid]
        if not label_map:
            continue
        asec = c3d["parameters"]["ANALOG"]
        labels = list(asec["LABELS"]["value"])
        changed = 0
        labels, changed = apply_label_mapping(labels, label_map)
        asec["LABELS"]["value"] = labels
        if changed:
            c3d.write(str(c3d_path))
            log(f"[apply] patched {c3d_path} (changed {changed})", log_lines)


if __name__ == "__main__":
    main()
