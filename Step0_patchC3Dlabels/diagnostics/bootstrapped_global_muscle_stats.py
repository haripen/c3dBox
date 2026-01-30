#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute swap-updated, left/right pooled global muscle stats with bootstrap PCA.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Step0_patchC3Dlabels import fixEMGmisplacement_cli as fx


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap global per-muscle stats (L/R pooled).")
    p.add_argument("--data-root", default=r"D:\Data_local\Pros_5er_hybrid_plain")
    p.add_argument("--task", required=True)
    p.add_argument(
        "--run-dir",
        default="",
        help="fixEMG run dir name (e.g., fixEMG_walk_YYYYMMDD_HHMMSS).",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--processes", type=int, default=24)
    p.add_argument("--max-muscles", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--emg-method", choices=["internal", "step4"], default="step4" if fx._HAS_STEP4_EMG else "internal")
    p.add_argument("--bandpass", nargs=2, type=float, default=[30.0, 350.0])
    p.add_argument("--lowpass", type=float, default=6.0)
    p.add_argument("--min-cycles", type=int, default=3)
    p.add_argument("--min-cycle-samples", type=int, default=50)
    p.add_argument("--within-lag-max", type=int, default=5)
    p.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.05,
        help="Quantile for corr low / mad high outlier detection (e.g., 0.05).",
    )
    p.add_argument(
        "--outlier-top-frac",
        type=float,
        default=0.05,
        help="Top fraction by dissimilarity to mark as outliers.",
    )
    return p.parse_args()


def latest_run(output_root: Path) -> Path:
    runs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("fixEMG_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def apply_mapping_to_traces(id_traces, mapping):
    out = defaultdict(dict)
    for sid, ch_dict in id_traces.items():
        lab_map = mapping.get(sid, {})
        used = set()
        for label, trace in ch_dict.items():
            if label in used:
                continue
            if label in lab_map:
                target = lab_map[label]
                if target in ch_dict:
                    # swap
                    out[sid][target] = trace
                    out[sid][label] = ch_dict[target]
                    used.add(label)
                    used.add(target)
                else:
                    out[sid][target] = trace
                    used.add(label)
            else:
                out[sid][label] = trace
                used.add(label)
    return out


def bootstrap_stats(args):
    muscle, traces, n_boot, seed = args
    rng = np.random.default_rng(seed)
    n = len(traces)
    if n == 0:
        return muscle, None
    boot_traces = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [traces[i] for i in idx]
        boot = fx.aggregate_cycles(sample, "pca")
        boot_traces.append(boot)
    B = np.asarray(boot_traces)
    stats = {
        "mean": np.mean(B, axis=0).tolist(),
        "std": np.std(B, axis=0).tolist(),
        "p025": np.percentile(B, 2.5, axis=0).tolist(),
        "p50": np.percentile(B, 50.0, axis=0).tolist(),
        "p975": np.percentile(B, 97.5, axis=0).tolist(),
        "n_boot": int(n_boot),
        "n_traces": int(n),
    }
    return muscle, stats


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    out_root = Path(fx.REPO_ROOT) / "Step0_patchC3Dlabels" / "outputs"
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(out_root)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError("Could not locate a fixEMG run directory.")

    mapping_path = run_dir / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = run_dir / "processing" / "bootstrap_stats"
    fx.ensure_dir(out_dir)

    label_map = fx.read_label_mapping(fx.LABEL_MAP_PATH)
    emg_map = label_map.get("analog", {}).get("group_to_mostfrequent_raw", {})

    c3ds = fx.list_c3ds(data_root)
    if not c3ds:
        raise FileNotFoundError("No C3D files found under data root.")

    id_channels = defaultdict(lambda: defaultdict(list))
    id_label_sets = defaultdict(lambda: defaultdict(set))
    id_channel_info = defaultdict(dict)
    matched_count = 0

    for c3d_path in c3ds:
        desc = fx.parse_enf_description(c3d_path)
        if desc is None or not fx.match_task(desc, args.task):
            continue
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
            info = dict(info)
            info["canon"] = fx.canonical_muscle(info.get("muscle_key", ""))
            id_channel_info[subject_id][label] = info

            strikes = left_strikes if info["side"] == "L" else right_strikes
            if len(strikes) < 2:
                continue
            idx = info["index"]
            try:
                full_env = fx.process_emg_segment(
                    data[:, idx],
                    rate,
                    args.bandpass[0],
                    args.bandpass[1],
                    args.lowpass,
                    method=args.emg_method,
                )
            except Exception:
                full_env = None
            if full_env is None or full_env.size < 5:
                continue
            cycles = []
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
                env = full_env[i0:i1]
                if env is None or env.size < 5:
                    continue
                env_norm = fx.time_normalize(env, n=101)
                if env_norm is None:
                    continue
                cycles.append(env_norm)
            if cycles:
                id_channels[subject_id][label].extend(cycles)

    # Compute ID traces with outlier removal
    id_traces = defaultdict(dict)
    id_cycles_norm = defaultdict(dict)
    cycle_corrs_all = []
    cycle_corr_map = defaultdict(dict)
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
            id_trace = fx.aggregate_cycles(Xn, "pca")
            if id_trace is None:
                continue
            for i, cyc in enumerate(Xn):
                c_lim, _ = fx.cross_corr(cyc, id_trace, max_lag=args.within_lag_max)
                cycle_corrs_all.append(c_lim)
                cycle_corr_map[(sid, label)][i] = c_lim
            id_cycles_norm[sid][label] = Xn
            id_traces[sid][label] = id_trace

    within_corr_low = fx.compute_thresholds(cycle_corrs_all, mode="lower", quantile=0.025)
    for sid, ch_dict in list(id_cycles_norm.items()):
        for label, Xn in list(ch_dict.items()):
            mask = np.array(
                [cycle_corr_map.get((sid, label), {}).get(i, np.nan) < within_corr_low for i in range(Xn.shape[0])],
                dtype=bool,
            )
            Xf = Xn[~mask] if mask.size else Xn
            if Xf.shape[0] < args.min_cycles:
                continue
            id_traces[sid][label] = fx.aggregate_cycles(Xf, "pca")

    # Apply mapping to traces (swap-aware) before pooling
    mapped_traces = apply_mapping_to_traces(id_traces, mapping)

    # Pool L/R per muscle (mapped labels)
    muscle_traces = defaultdict(list)
    muscle_traces_side = defaultdict(lambda: defaultdict(list))
    muscle_meta = defaultdict(list)  # list of dicts with sid/label/side/trace
    muscle_ids = defaultdict(set)
    muscle_ids_side = defaultdict(lambda: defaultdict(set))
    for sid, ch_dict in mapped_traces.items():
        for label, trace in ch_dict.items():
            info = fx.emg_channel_info(label) or {}
            canon = fx.canonical_muscle(info.get("muscle_key", ""))
            side = info.get("side", "")
            muscle_traces[canon].append(trace)
            muscle_meta[canon].append({"sid": sid, "label": label, "side": side, "trace": trace})
            muscle_ids[canon].add(sid)
            if side in ("L", "R"):
                muscle_traces_side[canon][side].append(trace)
                muscle_ids_side[canon][side].add(sid)

    muscles = list(muscle_traces.keys())
    if args.max_muscles > 0:
        muscles = muscles[: args.max_muscles]

    n_boot = int(args.bootstrap)
    n_proc = min(max(1, args.processes), cpu_count())
    seed = int(args.seed)

    work = []
    for i, m in enumerate(muscles):
        work.append((m, muscle_traces[m], n_boot, seed + i))

    t0 = time.time()
    if n_proc > 1:
        with Pool(processes=n_proc) as pool:
            results = pool.map(bootstrap_stats, work)
    else:
        results = [bootstrap_stats(w) for w in work]
    elapsed = time.time() - t0

    out = {
        "metadata": {
            "task": args.task,
            "data_root": str(data_root),
            "run_dir": str(run_dir),
            "mapping_path": str(mapping_path),
            "mapping_applied_to_traces": True,
            "emg_method": args.emg_method,
            "bandpass": args.bandpass,
            "lowpass": args.lowpass,
            "time_normalize_n": 101,
            "amplitude_norm": "per-ID channel: mean+3*SD",
            "outlier_removal": "within-cycle corr < lower 2.5% across all cycles",
            "agg_method": "PCA dim1 (SVD, sign aligned to mean)",
            "id_trace_method": "PCA dim1 (SVD) per ID/channel, then remapped by mapping.json",
            "bootstrap": n_boot,
            "processes": n_proc,
            "elapsed_sec": round(elapsed, 2),
        },
        "muscles": {},
    }
    for muscle, stats in results:
        if stats is None:
            continue
        stats["n_ids"] = int(len(muscle_ids[muscle]))
        stats["n_ids_left"] = int(len(muscle_ids_side[muscle].get("L", set())))
        stats["n_ids_right"] = int(len(muscle_ids_side[muscle].get("R", set())))
        stats["n_traces_left"] = int(len(muscle_traces_side[muscle].get("L", [])))
        stats["n_traces_right"] = int(len(muscle_traces_side[muscle].get("R", [])))
        out["muscles"][muscle] = stats

    # Plots: pooled PCA dim1 + SD, side PCA means
    plot_dir = out_dir / "plots"
    fx.ensure_dir(plot_dir)
    for muscle in muscles:
        traces = muscle_traces.get(muscle, [])
        if not traces:
            continue
        X = np.asarray(traces)
        pooled_pca = fx.aggregate_cycles(traces, "pca")
        sd = np.std(X, axis=0)
        mean_l = None
        mean_r = None
        if muscle_traces_side[muscle].get("L"):
            mean_l = fx.aggregate_cycles(muscle_traces_side[muscle]["L"], "pca")
        if muscle_traces_side[muscle].get("R"):
            mean_r = fx.aggregate_cycles(muscle_traces_side[muscle]["R"], "pca")

        # Outlier detection vs leave-one-out pooled PCA
        corrs = []
        mads = []
        dissim = []
        for i, item in enumerate(muscle_meta[muscle]):
            t = item["trace"]
            others = [m["trace"] for j, m in enumerate(muscle_meta[muscle]) if j != i]
            if not others:
                continue
            loo_pca = fx.aggregate_cycles(others, "pca")
            c0, _ = fx.cross_corr(t, loo_pca, max_lag=0)
            mad0 = fx.mad_for_lag(t, loo_pca, 0)
            corrs.append(c0)
            mads.append(mad0)
            item["corr_to_pooled"] = float(c0)
            item["mad_to_pooled"] = float(mad0)
        q = max(0.0, min(0.49, float(args.outlier_quantile)))
        corr_thr = float(np.percentile(corrs, 100.0 * q)) if corrs else np.nan
        mad_thr = float(np.percentile(mads, 100.0 * (1.0 - q))) if mads else np.nan
        mad_med = float(np.median(mads)) if mads else np.nan
        for item in muscle_meta[muscle]:
            if "corr_to_pooled" not in item:
                continue
            mad0 = item["mad_to_pooled"]
            score = (1.0 - item["corr_to_pooled"]) + (mad0 / (mad_med + 1e-8))
            item["dissim"] = float(score)
            dissim.append(score)
        top_frac = max(0.0, min(1.0, float(args.outlier_top_frac)))
        top_k = max(1, int(round(top_frac * max(1, len(dissim))))) if top_frac > 0 else 0
        top_idx = set()
        if dissim and top_k > 0:
            order = np.argsort(-np.asarray(dissim))
            top_idx = set(order[:top_k].tolist())
        outliers = [
            {
                "sid": item["sid"],
                "label": item["label"],
                "side": item["side"],
                "corr_to_pooled": item["corr_to_pooled"],
                "mad_to_pooled": item["mad_to_pooled"],
                "dissim": item.get("dissim", np.nan),
            }
            for i, item in enumerate(muscle_meta[muscle])
            if (np.isfinite(corr_thr) and item.get("corr_to_pooled", np.inf) < corr_thr)
            or (np.isfinite(mad_thr) and item.get("mad_to_pooled", -np.inf) > mad_thr)
            or (i in top_idx)
        ]
        out["muscles"][muscle]["outliers"] = outliers
        out["muscles"][muscle]["outlier_method"] = (
            f"LOO PCA vs trace, corr<{q:.2f} or mad>{1.0-q:.2f} or top {top_frac:.2f} dissimilarity"
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.linspace(0, 100, pooled_pca.size)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 1, 1)
        outlier_idx = set()
        for i, item in enumerate(muscle_meta[muscle]):
            if "dissim" in item:
                if item.get("corr_to_pooled", 1.0) < corr_thr or item.get("mad_to_pooled", 0.0) > mad_thr:
                    outlier_idx.add(i)
        for i, t in enumerate(X):
            if i in outlier_idx:
                ax.plot(x, t, color="0.35", alpha=0.30, linewidth=0.8, linestyle=":")
            else:
                ax.plot(x, t, color="0.6", alpha=0.20, linewidth=0.6)
        ax.fill_between(x, pooled_pca - sd, pooled_pca + sd, color="C0", alpha=0.2, label="Pooled traces ±1 SD")
        ax.plot(x, pooled_pca, color="C0", linewidth=2.0, label="Pooled PCA dim1")
        if mean_l is not None:
            ax.plot(x, mean_l, color="C3", linestyle="--", linewidth=1.5, label="Left PCA dim1")
        if mean_r is not None:
            ax.plot(x, mean_r, color="C2", linestyle="--", linewidth=1.5, label="Right PCA dim1")
        ax.set_title(f"{muscle} (n_traces={len(traces)}, outliers={len(outliers)})")
        ax.set_xlabel("% stride")
        ax.set_ylabel("normalized EMG (a.u.)")
        ax.set_xlim(0, 100)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{muscle}.jpg", dpi=150)
        fig.savefig(plot_dir / f"{muscle}.pdf")
        plt.close(fig)
    out_path = out_dir / f"global_muscle_stats_{fx.safe_slug(args.task)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[done] wrote {out_path} (plots: {plot_dir})")


if __name__ == "__main__":
    main()
