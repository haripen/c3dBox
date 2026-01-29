#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for sEMG processing heterogeneity.

Compares:
  1) MAT split cycles (PRO_checked_clustered_mat) processed with:
       - Step4_check.emg.process_emg
       - sim2kin_synergies.nnmf_muscle_synergy.preprocess_emg_trial
  2) C3D raw cycles processed with Step4_check.emg.process_emg

Outputs CSV summaries + a few example plots to help pinpoint mismatches.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
SIM2KIN_ROOT = Path(r"D:\Git\sim2kin_synergies")
DATA_ROOT_MAT = Path(r"D:\Data_local\PRO_checked_clustered_mat")
DATA_ROOT_C3D = Path(r"D:\Data_local\Pros_5er_hybrid_plain")

for p in [REPO_ROOT, SIM2KIN_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ezc3d  # noqa: E402
from utils_py.mat2dict import loadmat_to_dict  # noqa: E402
from utils_py.cluster_spm_helpers import iter_cycles  # noqa: E402

from Step4_check.emg import process_emg as step4_process_emg  # noqa: E402
from semg_compare_utils import participant_id_from_path  # noqa: E402
from nnmf_muscle_synergy import (  # noqa: E402
    preprocess_emg_trial,
    compute_fs,
    resample_to_points,
    build_emg_key_map,
)


EMG_RIGHT_NUMS = set(range(1, 14))
EMG_LEFT_NUMS = set(list(range(17, 28)) + [31, 32])


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose sEMG processing heterogeneity.")
    p.add_argument("--mat-root", default=str(DATA_ROOT_MAT))
    p.add_argument("--c3d-root", default=str(DATA_ROOT_C3D))
    p.add_argument("--task-substr", default="walk", help="Substring in filename for filtering.")
    p.add_argument("--max-mat", type=int, default=0, help="Limit number of MAT files (0 = all).")
    p.add_argument("--max-c3d", type=int, default=0, help="Limit number of C3D files for index (0 = all).")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    p.add_argument("--plot-max", type=int, default=20, help="Max number of example plots.")
    return p.parse_args()


def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def emg_key_side_num_mat(key: str):
    m = re.match(r"EMG_(\d+)_([LR])_M_(.+)", key)
    if not m:
        return None
    num = int(m.group(1))
    side = m.group(2)
    if side == "R" and num not in EMG_RIGHT_NUMS:
        return None
    if side == "L" and num not in EMG_LEFT_NUMS:
        return None
    return num, side, m.group(3)


def emg_key_from_c3d_label(label: str):
    s = str(label).strip()
    m = re.search(r"EMG[._\s-]*([0-9]{1,2})[._\s-]*([LR])\s*(.*)$", s, re.IGNORECASE)
    if not m:
        return None
    num = int(m.group(1))
    side = m.group(2).upper()
    if side == "R" and num not in EMG_RIGHT_NUMS:
        return None
    if side == "L" and num not in EMG_LEFT_NUMS:
        return None
    muscle = m.group(3).strip()
    muscle = re.sub(r"^m[\s\.]+", "", muscle, flags=re.IGNORECASE)
    muscle_key = re.sub(r"[^a-z0-9]+", "_", muscle.lower()).strip("_")
    return f"EMG_{num:02d}_{side}_M_{muscle_key}"


def list_files(root: Path, ext: str, max_files: int = 0, task_substr: str = ""):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.lower().endswith(ext.lower()):
                continue
            if task_substr and task_substr.lower() not in f.lower():
                continue
            out.append(Path(dp) / f)
            if max_files and len(out) >= max_files:
                return out
    return out


def build_c3d_index(root: Path, max_files=0, task_substr=""):
    idx = {}
    for path in list_files(root, ".c3d", max_files=max_files, task_substr=task_substr):
        idx[path.stem] = path
    return idx


def time_c3d0(c3d):
    try:
        point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
        actual_start = float(c3d["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0])
        if point_rate <= 0:
            return 0.0
        return (actual_start / point_rate) - (1.0 / point_rate)
    except Exception:
        return 0.0


def extract_c3d_cycles(c3d_path: Path):
    c3d = ezc3d.c3d(str(c3d_path))
    labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    analogs = np.array(c3d["data"]["analogs"], dtype=float)
    subframes, n_channels, n_frames = analogs.shape
    data = np.transpose(analogs, (2, 0, 1)).reshape(n_frames * subframes, n_channels)
    t0 = time_c3d0(c3d)
    time_full = (np.arange(data.shape[0], dtype=float) / rate) + t0

    # Event times
    ev = c3d["parameters"].get("EVENT", {})
    labels_ev = list(ev.get("LABELS", {}).get("value", []))
    contexts_ev = list(ev.get("CONTEXTS", {}).get("value", []))
    times = np.array(ev.get("TIMES", {}).get("value", []), dtype=float)
    if times.ndim == 2 and times.shape[0] >= 2:
        times = times[1] if np.sum(np.abs(times[1])) >= np.sum(np.abs(times[0])) else times[0]
    elif times.ndim != 1:
        times = times.reshape(-1)

    def strikes(side):
        out = []
        for t, lab, ctx in zip(times, labels_ev, contexts_ev):
            if str(lab).lower() == "foot strike" and str(ctx).lower() == side:
                out.append(float(t))
        return sorted(set(out))

    left_strikes = strikes("left")
    right_strikes = strikes("right")
    all_events = left_strikes + right_strikes
    duration = data.shape[0] / rate if rate > 0 else 0.0
    t_min = min(all_events) if all_events else np.nan
    t_max = max(all_events) if all_events else np.nan
    id_cycles = defaultdict(list)  # processed env
    id_cycles_raw = defaultdict(list)

    for ch_idx, lab in enumerate(labels):
        key = emg_key_from_c3d_label(lab)
        if not key:
            continue
        _, side, _ = emg_key_side_num_mat(key)
        strikes_use = left_strikes if side == "L" else right_strikes
        if len(strikes_use) < 2:
            continue
        for s0, s1 in zip(strikes_use[:-1], strikes_use[1:]):
            i0 = int(np.searchsorted(time_full, s0, side="left"))
            i1 = int(np.searchsorted(time_full, s1, side="right"))
            i0 = max(0, i0)
            i1 = min(data.shape[0], i1)
            if i1 - i0 < 5:
                continue
            raw = data[i0:i1, ch_idx]
            t_seg = time_full[i0:i1]
            r0 = resample_to_points(t_seg, raw, 101)
            if r0 is not None:
                id_cycles_raw[key].append(r0)
            env = step4_process_emg(raw, rate)
            resamp = resample_to_points(t_seg, env, 101)
            if resamp is None:
                continue
            id_cycles[key].append(resamp)
    meta = {
        "duration": duration,
        "t_min": t_min,
        "t_max": t_max,
        "t_range": (t_max - t_min) if all_events else np.nan,
        "time_c3d0": t0,
    }
    return id_cycles, id_cycles_raw, meta


def corr_and_lag(a: np.ndarray, b: np.ndarray):
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    if denom <= 0:
        return np.nan, 0
    corr = np.correlate(a0, b0, mode="full") / denom
    lag = int(np.argmax(corr)) - (len(a0) - 1)
    return float(np.max(corr)), lag


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = list_files(Path(args.mat_root), ".mat", max_files=args.max_mat, task_substr=args.task_substr)
    if not mat_files:
        raise FileNotFoundError("No MAT files found.")

    c3d_index = build_c3d_index(Path(args.c3d_root), max_files=args.max_c3d, task_substr=args.task_substr)

    # Containers
    mat_cycles_step4 = defaultdict(lambda: defaultdict(list))  # key -> id -> cycles
    mat_cycles_sim2 = defaultdict(lambda: defaultdict(list))
    mat_cycles_raw = defaultdict(lambda: defaultdict(list))
    c3d_cycles = defaultdict(lambda: defaultdict(list))
    c3d_cycles_raw = defaultdict(lambda: defaultdict(list))
    counts_rows = []
    compare_rows = []
    hom_rows = []
    event_rows = []
    plot_count = 0

    for mat_path in mat_files:
        data = loadmat_to_dict(str(mat_path))
        pid = participant_id_from_path(str(mat_path))

        # MAT cycles
        emg_keys = None
        for side, cycle_id, cycle in iter_cycles(data):
            analog = cycle.get("analog", {})
            if not isinstance(analog, dict):
                continue
            if emg_keys is None:
                emg_keys = [k for k in analog.keys() if k.lower().startswith("emg_")]
                if not emg_keys:
                    break
                key_map = build_emg_key_map(emg_keys)
            time_vec = analog.get("time")
            if time_vec is None:
                continue
            fs = compute_fs(time_vec)
            if fs is None:
                continue
            for k in emg_keys:
                parsed = emg_key_side_num_mat(k)
                if not parsed:
                    continue
                _, k_side, _ = parsed
                if (side == "left" and k_side != "L") or (side == "right" and k_side != "R"):
                    continue
                raw = np.asarray(analog.get(k, []), dtype=float).reshape(-1)
                if raw.size < 5:
                    continue
                env1 = step4_process_emg(raw, fs)
                env2 = preprocess_emg_trial(raw.reshape(-1, 1), fs).reshape(-1)
                r0 = resample_to_points(time_vec, raw, 101)
                r1 = resample_to_points(time_vec, env1, 101)
                r2 = resample_to_points(time_vec, env2, 101)
                if r0 is not None:
                    mat_cycles_raw[k][pid].append(r0)
                if r1 is not None:
                    mat_cycles_step4[k][pid].append(r1)
                if r2 is not None:
                    mat_cycles_sim2[k][pid].append(r2)

        # C3D cycles
        stem = mat_path.stem
        stem = stem.replace("_splitCycles_osim_check_c", "")
        c3d_path = c3d_index.get(stem)
        if c3d_path:
            c3d_id_cycles, c3d_id_cycles_raw, meta = extract_c3d_cycles(c3d_path)
            meta_row = {
                "c3d_path": str(c3d_path),
                "participant_id": pid,
                "duration": meta.get("duration"),
                "t_min": meta.get("t_min"),
                "t_max": meta.get("t_max"),
                "t_range": meta.get("t_range"),
                "time_c3d0": meta.get("time_c3d0"),
            }
            event_rows.append(meta_row)
            for key, waves in c3d_id_cycles.items():
                c3d_cycles[key][pid].extend(waves)
            for key, waves in c3d_id_cycles_raw.items():
                c3d_cycles_raw[key][pid].extend(waves)

    # Summaries
    for key in sorted(set(mat_cycles_step4.keys()) | set(c3d_cycles.keys()) | set(mat_cycles_raw.keys()) | set(c3d_cycles_raw.keys())):
        ids = sorted(set(mat_cycles_step4[key].keys()) | set(c3d_cycles[key].keys()) | set(mat_cycles_raw[key].keys()) | set(c3d_cycles_raw[key].keys()))
        for pid in ids:
            n_mat = len(mat_cycles_step4[key].get(pid, []))
            n_sim2 = len(mat_cycles_sim2[key].get(pid, []))
            n_c3d = len(c3d_cycles[key].get(pid, []))
            n_mat_raw = len(mat_cycles_raw[key].get(pid, []))
            n_c3d_raw = len(c3d_cycles_raw[key].get(pid, []))
            counts_rows.append([key, pid, n_mat, n_sim2, n_c3d, n_mat_raw, n_c3d_raw])

            # Homogeneity stats (mat vs c3d)
            for label, waves in [
                ("mat_step4", mat_cycles_step4[key].get(pid, [])),
                ("mat_sim2", mat_cycles_sim2[key].get(pid, [])),
                ("mat_raw", mat_cycles_raw[key].get(pid, [])),
                ("c3d_step4", c3d_cycles[key].get(pid, [])),
                ("c3d_raw", c3d_cycles_raw[key].get(pid, [])),
            ]:
                if len(waves) < 2:
                    continue
                mean = np.mean(waves, axis=0)
                corrs = []
                lags = []
                for w in waves:
                    c, l = corr_and_lag(w, mean)
                    corrs.append(c)
                    lags.append(l)
                hom_rows.append([
                    key, pid, label,
                    len(waves),
                    float(np.nanmedian(corrs)), float(np.nanmedian(lags)),
                    float(np.nanmean(corrs)), float(np.nanmean(lags)),
                ])

            # Compare mat vs c3d mean
            if mat_cycles_step4[key].get(pid) and c3d_cycles[key].get(pid):
                mean_mat = np.mean(mat_cycles_step4[key][pid], axis=0)
                mean_c3d = np.mean(c3d_cycles[key][pid], axis=0)
                corr, lag = corr_and_lag(mean_mat, mean_c3d)
                compare_rows.append([key, pid, "step4", corr, lag, len(mat_cycles_step4[key][pid]), len(c3d_cycles[key][pid])])
            if mat_cycles_raw[key].get(pid) and c3d_cycles_raw[key].get(pid):
                mean_mat = np.mean(mat_cycles_raw[key][pid], axis=0)
                mean_c3d = np.mean(c3d_cycles_raw[key][pid], axis=0)
                corr, lag = corr_and_lag(mean_mat, mean_c3d)
                compare_rows.append([key, pid, "raw", corr, lag, len(mat_cycles_raw[key][pid]), len(c3d_cycles_raw[key][pid])])

    # Write CSVs
    def write_csv(path: Path, header, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join([str(x) for x in r]) + "\n")

    write_csv(out_dir / "counts_per_id.csv",
              ["emg_key", "participant_id", "n_mat_step4", "n_mat_sim2", "n_c3d_step4", "n_mat_raw", "n_c3d_raw"],
              counts_rows)

    write_csv(out_dir / "homogeneity_stats.csv",
              ["emg_key", "participant_id", "source", "n_cycles", "corr_med", "lag_med", "corr_mean", "lag_mean"],
              hom_rows)

    write_csv(out_dir / "compare_mat_vs_c3d.csv",
              ["emg_key", "participant_id", "source", "corr", "lag", "n_mat", "n_c3d"],
              compare_rows)

    if event_rows:
        with open(out_dir / "event_time_offsets.csv", "w", encoding="utf-8") as f:
            f.write("c3d_path,participant_id,duration,t_min,t_max,t_range,time_c3d0\n")
            for r in event_rows:
                f.write(
                    f"{r['c3d_path']},{r['participant_id']},{r['duration']},{r['t_min']},"
                    f"{r['t_max']},{r['t_range']},{r['time_c3d0']}\n"
                )

    print(f"[done] wrote diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
