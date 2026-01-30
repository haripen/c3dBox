#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check boundary continuity (sample 0 vs 100) across processing steps.
Compares filtering after split vs before split and reports where discontinuities arise.
"""

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Step0_patchC3Dlabels import fixEMGmisplacement_cli as fx


def parse_args():
    p = argparse.ArgumentParser(description="Check cycle boundary continuity across processing steps.")
    p.add_argument("--data-root", default=r"D:\Data_local\Pros_5er_hybrid_plain")
    p.add_argument("--task", required=True)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--run-dir", default="", help="fixEMG run dir for mapping.json (optional).")
    p.add_argument("--emg-method", choices=["internal", "step4"], default="step4" if fx._HAS_STEP4_EMG else "internal")
    p.add_argument("--bandpass", nargs=2, type=float, default=[30.0, 350.0])
    p.add_argument("--lowpass", type=float, default=6.0)
    p.add_argument("--min-cycle-samples", type=int, default=50)
    p.add_argument("--min-cycles", type=int, default=3)
    p.add_argument("--within-lag-max", type=int, default=5)
    p.add_argument("--out-top-n", type=int, default=5)
    return p.parse_args()


def latest_run(output_root: Path) -> Path:
    runs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("fixEMG_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def boundary_metrics(sig):
    sig = np.asarray(sig, dtype=float)
    if sig.size < 3:
        return None
    diff = float(abs(sig[-1] - sig[0]))
    d0 = sig[1] - sig[0]
    d1 = sig[-1] - sig[-2]
    ddiff = float(abs(d1 - d0))
    sd = float(np.std(sig) + 1e-8)
    diff_n = diff / sd
    ddiff_n = ddiff / sd
    # 50% shift check
    half = sig.size // 2
    shift = np.concatenate([sig[half:], sig[:half]])
    diff_shift = float(abs(shift[-1] - shift[0]))
    return {
        "diff": diff,
        "diff_n": diff_n,
        "ddiff": ddiff,
        "ddiff_n": ddiff_n,
        "diff_shift50": diff_shift,
    }


def summarize_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list if k in m and np.isfinite(m[k])], dtype=float)
        if vals.size == 0:
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
        }
    return out


def process_file(args):
    (c3d_path, params) = args
    result = {
        "cycles_after": {},
        "cycles_before": {},
    }
    desc = fx.parse_enf_description(c3d_path)
    if desc is None or not fx.match_task(desc, params["task"]):
        return result
    try:
        c3d = fx.ezc3d.c3d(str(c3d_path))
    except Exception:
        return result

    subject_id = fx.get_subject_id(c3d)
    times, labels, contexts = fx.extract_event_times(c3d)
    if len(times) == 0:
        return result
    data, rate = fx.analog_matrix(c3d)
    analog_time = fx.analog_time_vector(c3d, data.shape[0])
    analog_labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    emg_infos = fx.build_emg_label_list(analog_labels)
    left_strikes = fx.get_foot_strike_times(times, labels, contexts, "Left")
    right_strikes = fx.get_foot_strike_times(times, labels, contexts, "Right")
    if len(left_strikes) < 2 and len(right_strikes) < 2:
        return result

    for info in emg_infos:
        if info.get("is_nfu"):
            continue
        if not info.get("side"):
            continue
        label = info.get("label")
        strikes = left_strikes if info["side"] == "L" else right_strikes
        if len(strikes) < 2:
            continue
        idx = info["index"]
        full_raw = data[:, idx]
        try:
            full_env = fx.process_emg_segment(
                full_raw,
                rate,
                params["bandpass"][0],
                params["bandpass"][1],
                params["lowpass"],
                method=params["emg_method"],
            )
        except Exception:
            full_env = None

        for s0, s1 in zip(strikes[:-1], strikes[1:]):
            if s1 <= s0:
                continue
            i0 = int(np.searchsorted(analog_time, s0, side="left"))
            i1 = int(np.searchsorted(analog_time, s1, side="right"))
            i0 = max(0, i0)
            i1 = min(data.shape[0], i1)
            if i1 <= i0:
                continue
            if i1 - i0 < params["min_cycle_samples"]:
                continue
            seg = full_raw[i0:i1]
            try:
                env_after = fx.process_emg_segment(
                    seg,
                    rate,
                    params["bandpass"][0],
                    params["bandpass"][1],
                    params["lowpass"],
                    method=params["emg_method"],
                )
            except Exception:
                continue
            if env_after is None or env_after.size < 5:
                continue
            env_after_n = fx.time_normalize(env_after, n=101)
            if env_after_n is None:
                continue
            result["cycles_after"].setdefault(subject_id, {}).setdefault(label, []).append(env_after_n)

            if full_env is not None:
                seg_b = full_env[i0:i1]
                if seg_b.size >= 5:
                    env_before_n = fx.time_normalize(seg_b, n=101)
                    if env_before_n is not None:
                        result["cycles_before"].setdefault(subject_id, {}).setdefault(label, []).append(env_before_n)

    return result


def muscle_worker(args_m):
    muscle, traces, meta, out_n = args_m
    if not traces:
        return muscle, {}
    pooled = fx.aggregate_cycles(traces, "pca")
    pooled_metrics = boundary_metrics(pooled)
    # Outlier ranking by boundary diff
    diffs = []
    for item in meta:
        m = boundary_metrics(item["trace"])
        if m:
            diffs.append((m["diff_n"], item["sid"], item["label"]))
    diffs.sort(reverse=True)
    top = [{"sid": s, "label": l, "diff_n": d} for d, s, l in diffs[:out_n]]
    return muscle, {"pooled_pca": pooled_metrics, "top_boundary_outliers": top}


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    out_root = REPO_ROOT / "Step0_patchC3Dlabels" / "outputs"
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(out_root)
    mapping = {}
    if run_dir and run_dir.exists():
        mapping_path = run_dir / "mapping.json"
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping = fx.normalize_mapping_pairs(mapping)

    c3ds = fx.list_c3ds(data_root)
    if not c3ds:
        raise FileNotFoundError("No C3D files found under data root.")
    if args.max_files:
        c3ds = c3ds[: args.max_files]

    params = {
        "task": args.task,
        "emg_method": args.emg_method,
        "bandpass": args.bandpass,
        "lowpass": args.lowpass,
        "min_cycle_samples": args.min_cycle_samples,
    }

    n_proc = min(max(1, args.processes), cpu_count())
    jobs = [(p, params) for p in c3ds]
    if n_proc > 1:
        with Pool(processes=n_proc) as pool:
            results = pool.map(process_file, jobs)
    else:
        results = [process_file(j) for j in jobs]

    cycles_after = defaultdict(lambda: defaultdict(list))
    cycles_before = defaultdict(lambda: defaultdict(list))
    for res in results:
        for sid, ch_dict in res["cycles_after"].items():
            for label, cycs in ch_dict.items():
                cycles_after[sid][label].extend(cycs)
        for sid, ch_dict in res["cycles_before"].items():
            for label, cycs in ch_dict.items():
                cycles_before[sid][label].extend(cycs)

    # Build ID traces (mean + PCA) and metrics
    per_id_label = {}
    id_traces_after = defaultdict(dict)
    id_traces_before = defaultdict(dict)
    for sid, ch_dict in cycles_after.items():
        for label, cycs in ch_dict.items():
            if len(cycs) < args.min_cycles:
                continue
            X = np.asarray(cycs)
            mean_trace = np.mean(X, axis=0)
            pca_trace = fx.aggregate_cycles(X, "pca")
            id_traces_after[sid][label] = pca_trace
            metrics_cycles = [boundary_metrics(c) for c in cycs]
            per_id_label[(sid, label)] = {
                "n_cycles": int(len(cycs)),
                "cycles_after": summarize_metrics([m for m in metrics_cycles if m]),
                "id_mean_after": boundary_metrics(mean_trace),
                "id_pca_after": boundary_metrics(pca_trace),
            }
    for sid, ch_dict in cycles_before.items():
        for label, cycs in ch_dict.items():
            if len(cycs) < args.min_cycles:
                continue
            X = np.asarray(cycs)
            pca_trace = fx.aggregate_cycles(X, "pca")
            id_traces_before[sid][label] = pca_trace
            metrics_cycles = [boundary_metrics(c) for c in cycs]
            entry = per_id_label.get((sid, label), {})
            entry["cycles_before"] = summarize_metrics([m for m in metrics_cycles if m])
            entry["id_pca_before"] = boundary_metrics(pca_trace)
            per_id_label[(sid, label)] = entry

    # Apply mapping and pool by muscle (after-split)
    mapped = fx.normalize_mapping_pairs(mapping)
    muscle_traces = defaultdict(list)
    muscle_meta = defaultdict(list)
    for sid, ch_dict in id_traces_after.items():
        for label, trace in ch_dict.items():
            if mapped.get(sid) and label in mapped[sid]:
                label = mapped[sid][label]
            info = fx.emg_channel_info(label) or {}
            canon = fx.canonical_muscle(info.get("muscle_key", ""))
            muscle_traces[canon].append(trace)
            muscle_meta[canon].append({"sid": sid, "label": label, "trace": trace})

    work = [(m, muscle_traces[m], muscle_meta[m], args.out_top_n) for m in muscle_traces.keys()]
    if n_proc > 1:
        with Pool(processes=n_proc) as pool:
            muscle_results = pool.map(muscle_worker, work)
    else:
        muscle_results = [muscle_worker(w) for w in work]

    muscle_stats = {m: st for m, st in muscle_results if st}

    out = {
        "metadata": {
            "task": args.task,
            "data_root": str(data_root),
            "emg_method": args.emg_method,
            "bandpass": args.bandpass,
            "lowpass": args.lowpass,
            "time_normalize_n": 101,
            "note": "cycles_after: filter after split; cycles_before: filter before split",
        },
        "per_id_label": {f"{sid}::{label}": stats for (sid, label), stats in per_id_label.items()},
        "per_muscle_after": muscle_stats,
    }

    out_dir = (run_dir / "processing" / "diagnostics") if run_dir else (out_root / "diagnostics")
    fx.ensure_dir(out_dir)
    out_path = out_dir / f"boundary_continuity_{fx.safe_slug(args.task)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
