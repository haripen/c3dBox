#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export global EMG traces and per-ID processed data (cycles + ID traces).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Step0_patchC3Dlabels import fixEMGmisplacement_cli as fx


def parse_args():
    p = argparse.ArgumentParser(description="Export global and per-ID EMG traces.")
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
        "--run-dir",
        default="",
        help="fixEMG run dir name (e.g., fixEMG_walk_YYYYMMDD_HHMMSS).",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Override output directory (defaults to run_dir/processing/exports).",
    )
    p.add_argument(
        "--ids",
        default="Rd0079,Rd0012",
        help="Comma-separated subject IDs to export (e.g., Rd0079,Rd0012).",
    )
    p.add_argument(
        "--random-ids",
        type=int,
        default=0,
        help="If >0, ignore --ids and export this many random IDs found in the scan.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for --random-ids sampling.",
    )
    p.add_argument(
        "--agg",
        choices=["mean", "median", "pca"],
        default="pca",
        help="Aggregation method for ID and global traces.",
    )
    p.add_argument(
        "--emg-method",
        choices=["internal", "step4"],
        default="step4" if fx._HAS_STEP4_EMG else "internal",
        help="EMG envelope method.",
    )
    p.add_argument("--min-cycles", type=int, default=3)
    p.add_argument("--min-cycle-samples", type=int, default=50)
    p.add_argument("--bandpass", nargs=2, type=float, default=[30.0, 350.0])
    p.add_argument("--lowpass", type=float, default=6.0)
    p.add_argument("--max-files", type=int, default=0)
    return p.parse_args()


def safe_key(label: str) -> str:
    return fx.safe_slug(label).replace("-", "_")


def latest_run(output_root: Path) -> Path:
    runs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("fixEMG_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    out_root = Path(fx.REPO_ROOT) / "Step0_patchC3Dlabels" / "outputs"
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(out_root)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError("Could not locate a fixEMG run directory.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = run_dir / "processing" / "exports"
    fx.ensure_dir(out_dir)

    target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    target_ids_set = {s.lower() for s in target_ids}

    label_map = fx.read_label_mapping(fx.LABEL_MAP_PATH)
    emg_map = label_map.get("analog", {}).get("group_to_mostfrequent_raw", {})

    c3ds = fx.list_c3ds(data_root)
    if not c3ds:
        raise FileNotFoundError("No C3D files found under data root.")

    id_channels = fx.defaultdict(lambda: fx.defaultdict(list))
    id_cycle_snr = fx.defaultdict(dict)
    id_label_sets = fx.defaultdict(lambda: fx.defaultdict(set))

    matched_count = 0
    for c3d_path in c3ds:
        desc = fx.parse_enf_description(c3d_path)
        if desc is None:
            continue
        if not fx.match_task(desc, args.task):
            continue
        if args.max_files and matched_count >= args.max_files:
            break
        try:
            c3d = fx.ezc3d.c3d(str(c3d_path))
        except Exception:
            continue

        subject_id = fx.get_subject_id(c3d)
        times, labels, contexts = fx.extract_event_times(c3d)
        if len(times) == 0:
            continue

        data, rate = fx.analog_matrix(c3d)
        analog_time = fx.analog_time_vector(c3d, data.shape[0])
        analog_labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
        emg_infos = fx.build_emg_label_list(analog_labels)

        left_strikes = fx.get_foot_strike_times(times, labels, contexts, "Left")
        right_strikes = fx.get_foot_strike_times(times, labels, contexts, "Right")
        if len(left_strikes) < 2 and len(right_strikes) < 2:
            continue

        matched_count += 1

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
                    pass
            id_label_sets[subject_id][emg_num].add(label)

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
                    env = fx.process_emg_segment(
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
                env_norm = fx.time_normalize(env, n=101)
                if env_norm is None:
                    continue
                cycles.append(env_norm)
                snrs.append(fx.cycle_snr(env_norm))

            if cycles:
                id_channels[subject_id][label].extend(cycles)
                if snrs:
                    snr_med = float(np.nanmedian(snrs))
                    prev = id_cycle_snr[subject_id].get(label, [])
                    prev = prev if isinstance(prev, list) else [prev]
                    prev.append(snr_med)
                    id_cycle_snr[subject_id][label] = prev

    id_traces = fx.defaultdict(dict)
    id_cycles_norm = fx.defaultdict(dict)
    outlier_cycles = fx.defaultdict(dict)
    cycle_corrs_all = []
    cycle_corr_map = fx.defaultdict(dict)

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
            id_trace = fx.aggregate_cycles(Xn, args.agg)
            if id_trace is None:
                continue
            for i, cyc in enumerate(Xn):
                c_lim, _ = fx.cross_corr(cyc, id_trace, max_lag=5)
                cycle_corrs_all.append(c_lim)
                cycle_corr_map[(sid, label)][i] = c_lim
            id_cycles_norm[sid][label] = Xn
            id_traces[sid][label] = id_trace

    within_corr_low = fx.compute_thresholds(cycle_corrs_all, mode="lower", quantile=0.025)
    for sid, ch_dict in id_cycles_norm.items():
        for label, Xn in ch_dict.items():
            n = Xn.shape[0]
            mask = np.zeros(n, dtype=bool)
            for i in range(n):
                c = cycle_corr_map.get((sid, label), {}).get(i, np.nan)
                if np.isfinite(within_corr_low) and np.isfinite(c) and c < within_corr_low:
                    mask[i] = True
            outlier_cycles[(sid, label)] = mask

    for sid, ch_dict in list(id_cycles_norm.items()):
        for label, Xn in list(ch_dict.items()):
            mask = outlier_cycles.get((sid, label), np.zeros(Xn.shape[0], dtype=bool))
            Xf = Xn[~mask] if mask.size else Xn
            if Xf.shape[0] < args.min_cycles:
                continue
            id_trace = fx.aggregate_cycles(Xf, args.agg)
            id_cycles_norm[sid][label] = Xf
            id_traces[sid][label] = id_trace

    label_id_traces = fx.defaultdict(list)
    for sid, ch_dict in id_traces.items():
        for label, trace in ch_dict.items():
            if trace is not None:
                label_id_traces[label].append(trace)

    global_traces = {}
    for label, traces in label_id_traces.items():
        if traces:
            global_traces[label] = fx.aggregate_cycles(traces, "pca")

    # Export globals
    g_npz = {}
    g_index = {}
    for label, trace in global_traces.items():
        key = safe_key(label)
        g_npz[key] = np.asarray(trace)
        g_index[key] = label
    np.savez_compressed(out_dir / "global_traces.npz", **g_npz)
    with open(out_dir / "global_traces_index.json", "w", encoding="utf-8") as f:
        json.dump(g_index, f, indent=2)

    # Determine IDs to export
    all_ids = list(id_cycles_norm.keys())
    export_ids = []
    if args.random_ids > 0:
        import random
        random.seed(args.seed)
        if len(all_ids) < args.random_ids:
            export_ids = all_ids
        else:
            export_ids = random.sample(all_ids, args.random_ids)
    else:
        for sid in all_ids:
            sid_l = sid.lower()
            if sid_l in target_ids_set:
                export_ids.append(sid)
            else:
                for token in target_ids_set:
                    if token and token in sid_l:
                        export_ids.append(sid)
                        break

    # Export per-ID processed data (match by exact or substring)
    for sid in export_ids:
        sid_l = sid.lower()
        export_id = sid
        for token in target_ids_set:
            if token and token in sid_l:
                export_id = token
                break
        npz = {}
        index = {"id": sid, "export_id": export_id, "labels": []}
        for label, cycles in id_cycles_norm[sid].items():
            key = safe_key(label)
            npz[f"cycles__{key}"] = np.asarray(cycles)
            npz[f"trace__{key}"] = np.asarray(id_traces[sid].get(label))
            mask = outlier_cycles.get((sid, label), np.zeros(cycles.shape[0], dtype=bool))
            npz[f"outlier__{key}"] = np.asarray(mask)
            index["labels"].append({"label": label, "key": key, "n_cycles": int(cycles.shape[0])})
        file_id = export_id.replace(" ", "_")
        np.savez_compressed(out_dir / f"processed_{file_id}.npz", **npz)
        with open(out_dir / f"processed_{file_id}_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    print(f"[done] exports in: {out_dir}")


if __name__ == "__main__":
    main()
