#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize remap thresholds using exported traces with LOO and iterative globals.
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
    p = argparse.ArgumentParser(description="Optimize misplacement detection thresholds.")
    p.add_argument(
        "--export-dir",
        default="",
        help="Exports directory (defaults to latest fixEMG run exports).",
    )
    p.add_argument(
        "--ids",
        default="rd0079,rd0012",
        help="Comma-separated export IDs (lowercase in filenames).",
    )
    p.add_argument(
        "--expected",
        default="rd0079:lr,rd0012:ok",
        help="Expected mode per id (lr or ok).",
    )
    p.add_argument("--lag-lr-range", nargs=2, type=int, default=[55, 65])
    p.add_argument("--within-lag-max", type=int, default=5)
    p.add_argument("--iterations", type=int, default=2, help="Remap->rebuild global iterations.")
    p.add_argument("--feedback", default="user_feedback.jsonl", help="Feedback jsonl filename in export dir.")
    p.add_argument("--feedback-weight", type=float, default=1.0, help="Weight for feedback score.")
    p.add_argument("--expected-weight", type=float, default=1.0, help="Weight for expected-mode score.")
    p.add_argument(
        "--metric",
        choices=["score", "f1_feedback", "f1_expected", "f1_combined"],
        default="score",
        help="Optimization target metric.",
    )
    p.add_argument("--corr-gain-grid", default="0.00,0.02,0.05,0.08,0.10")
    p.add_argument("--mad-gain-grid", default="0.00,0.10,0.20,0.30")
    p.add_argument("--out", default="optimize_results.json")
    return p.parse_args()


def latest_exports_dir() -> Path:
    out_root = REPO_ROOT / "Step0_patchC3Dlabels" / "outputs"
    runs = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("fixEMG_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] / "processing" / "exports"


def load_globals(export_dir: Path):
    idx = json.loads((export_dir / "global_traces_index.json").read_text(encoding="utf-8"))
    npz = np.load(export_dir / "global_traces.npz")
    globals_map = {}
    for key, label in idx.items():
        globals_map[label] = np.asarray(npz[key], dtype=float)
    return globals_map


def load_id(export_dir: Path, export_id: str):
    idx_path = export_dir / f"processed_{export_id}_index.json"
    npz_path = export_dir / f"processed_{export_id}.npz"
    if not idx_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Missing processed files for {export_id}")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)
    traces = {}
    for item in idx["labels"]:
        key = item["key"]
        traces[item["label"]] = np.asarray(npz[f"trace__{key}"], dtype=float)
    return traces


def resolve_export_id(export_dir: Path, export_id: str):
    base = export_id.replace(" ", "_")
    if base.lower().endswith("_index"):
        base = base[:-6]
    idx_path = export_dir / f"processed_{base}_index.json"
    npz_path = export_dir / f"processed_{base}.npz"
    if idx_path.exists() and npz_path.exists():
        return base
    target = f"processed_{base}_index.json".lower()
    for p in export_dir.glob("processed_*_index.json"):
        if p.name.lower() == target:
            return p.stem.replace("processed_", "").removesuffix("_index")
    return None


def load_feedback(export_dir: Path, filename: str):
    path = export_dir / filename
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            resp = rec.get("response", "").lower()
            if resp not in ("swap", "ok"):
                continue
            items.append(rec)
    return items


def f1_from_counts(tp, fp, fn):
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return (2 * tp) / denom


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


DEFAULT_CLOSE_GROUPS = [
    # shank muscles: all pairwise combos allowed
    {"tibialis_ant", "peroneus_long", "gastro_lat", "gastro_med", "soleus"},
    # thigh groups
    {"vastus_lat", "vastus_med", "rectus_fem"},
    {"semitendinosus", "biceps_femoris"},
    {"gluteus_medius", "gluteus_maximus"},
]


def match_group(canon_key, group):
    return canon_key in group


def close_muscle_keys(canon_key):
    keys = set()
    for group in DEFAULT_CLOSE_GROUPS:
        if match_group(canon_key, group):
            keys.update(group)
    return keys


def build_label_index(labels):
    idx = {}
    for lab in labels:
        info = fx.emg_channel_info(lab)
        if info is None:
            continue
        info["canon"] = canonical_muscle(info.get("muscle_key", ""))
        idx[lab] = info
    return idx


def opposite_side(side):
    return "L" if side == "R" else "R"


def build_globals_from_mapping(id_traces, id_mapping, base_globals, exclude_sid=None):
    buckets = {lab: [g] for lab, g in base_globals.items()}
    for sid, labs in id_traces.items():
        if exclude_sid is not None and sid == exclude_sid:
            continue
        for label, trace in labs.items():
            mapped_label = id_mapping.get(sid, {}).get(label, label)
            buckets.setdefault(mapped_label, []).append(trace)
    global_traces = {}
    for label, traces in buckets.items():
        if traces:
            global_traces[label] = fx.aggregate_cycles(traces, "pca")
    return global_traces


def pick_best_candidate(t, baseline, candidates, global_traces, lags):
    best = None
    corr_same, lag_same = fx.best_corr_circular(t, baseline, lags)
    mad_same = fx.mad_for_lag(t, baseline, lag_same)
    for cand in candidates:
        g = global_traces.get(cand)
        if g is None:
            continue
        corr_c, lag_c = fx.best_corr_circular(t, g, lags)
        mad_c = fx.mad_for_lag(t, g, lag_c)
        corr_up = corr_c - corr_same
        mad_down = (mad_same - mad_c) / max(mad_same, 1e-8)
        score = corr_up + mad_down
        if best is None or score > best["score"]:
            best = {
                "label": cand,
                "corr_up": corr_up,
                "mad_down": mad_down,
                "score": score,
            }
    return best


def suggest_mapping_for_label(
    lab,
    t,
    info,
    global_traces,
    label_index,
    corr_gain,
    mad_gain,
    lag_lr_range,
    lag_lim,
):
    g_same = global_traces.get(lab)
    if g_same is None:
        return {"pred": lab, "reason": "ok"}

    # 1) LR swap with phase shift
    if info is not None:
        lr_lab = None
        for cand, cand_info in label_index.items():
            if cand_info["muscle_key"] == info["muscle_key"] and cand_info["side"] == opposite_side(info["side"]):
                lr_lab = cand
                break
    if lr_lab is not None:
        corr_same, lag_same = fx.best_corr_circular(t, g_same, range(-lag_lim, lag_lim + 1))
        mad_same = fx.mad_for_lag(t, g_same, lag_same)
        g_lr = global_traces.get(lr_lab)
        corr_lr, lag_lr = fx.best_corr_circular(t, g_lr, range(lag_lr_range[0], lag_lr_range[1] + 1))
        mad_lr = fx.mad_for_lag(t, g_lr, lag_lr)
        corr_up = corr_lr - corr_same
        mad_down = (mad_same - mad_lr) / max(mad_same, 1e-8)
        if corr_up >= corr_gain and mad_down >= mad_gain:
            return {"pred": lr_lab, "reason": "lr"}

    # 2) Close muscle swap (same side), no phase shift
    if info is not None:
        norm_key = normalize_muscle_key(info["muscle_key"])
        close_keys = close_muscle_keys(canonical_muscle(info["muscle_key"]))
        close_labels = []
        for cand, cand_info in label_index.items():
            if cand_info["side"] != info["side"]:
                continue
            cand_canon = cand_info.get("canon", canonical_muscle(cand_info["muscle_key"]))
            if cand != lab and cand_canon in close_keys:
                close_labels.append(cand)
        if close_labels:
            best = pick_best_candidate(t, g_same, close_labels, global_traces, range(-lag_lim, lag_lim + 1))
            if best and best["corr_up"] >= corr_gain and best["mad_down"] >= mad_gain:
                return {"pred": best["label"], "reason": "close"}

    # 3) General swap (same side), no phase shift
    if info is not None:
        general = []
        for cand, cand_info in label_index.items():
            if cand_info["side"] != info["side"]:
                continue
            if cand != lab:
                general.append(cand)
        if general:
            best = pick_best_candidate(t, g_same, general, global_traces, range(-lag_lim, lag_lim + 1))
            if best and best["corr_up"] >= corr_gain and best["mad_down"] >= mad_gain:
                return {"pred": best["label"], "reason": "general"}

    return {"pred": lab, "reason": "ok"}


def eval_mapping(id_traces, globals_full, expected_mode, corr_gain, mad_gain, lr_range, lag_lim):
    results = []
    labels = sorted(id_traces.keys())
    label_index = build_label_index(globals_full.keys())
    for lab in labels:
        info = fx.emg_channel_info(lab)
        g_loo = globals_full.get(lab)
        pred = lab
        reason = "ok"
        if g_loo is not None:
            sug = suggest_mapping_for_label(
                lab,
                id_traces[lab],
                info,
                globals_full,
                label_index,
                corr_gain,
                mad_gain,
                lr_range,
                lag_lim,
            )
            pred = sug["pred"]
            reason = sug["reason"]
        results.append({"label": lab, "pred": pred, "reason": reason})

    if expected_mode == "lr":
        correct = 0
        total = 0
        for r in results:
            info = fx.emg_channel_info(r["label"])
            if info is None:
                continue
            total += 1
            if r["reason"] == "lr":
                info_pred = fx.emg_channel_info(r["pred"])
                if info_pred and info_pred["muscle_key"] == info["muscle_key"] and info_pred["side"] != info["side"]:
                    correct += 1
        acc = correct / total if total else 0.0
    else:
        correct = sum(1 for r in results if r["pred"] == r["label"])
        total = len(results)
        acc = correct / total if total else 0.0
    return acc, results


def main():
    args = parse_args()
    export_dir = Path(args.export_dir) if args.export_dir else latest_exports_dir()
    if export_dir is None or not export_dir.exists():
        raise FileNotFoundError("Could not locate exports directory.")

    ids = [s.strip().lower() for s in args.ids.split(",") if s.strip()]
    expected = {}
    for pair in args.expected.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        expected[k.strip().lower()] = v.strip().lower()

    id_traces = {sid: load_id(export_dir, sid) for sid in ids}
    globals_full = load_globals(export_dir)
    feedback = load_feedback(export_dir, args.feedback)

    corr_gain_grid = [float(x) for x in args.corr_gain_grid.split(",") if x.strip() != ""]
    mad_gain_grid = [float(x) for x in args.mad_gain_grid.split(",") if x.strip() != ""]

    summaries = []
    best = {"score": -1}

    for cg in corr_gain_grid:
        for mg in mad_gain_grid:
            score = 0.0
            per_id = {}
            id_mapping = {sid: {} for sid in ids}

            for _ in range(max(1, args.iterations)):
                for sid in ids:
                    globals_loo = build_globals_from_mapping(id_traces, id_mapping, globals_full, exclude_sid=sid)
                    mode = expected.get(sid, "ok")
                    acc, res = eval_mapping(
                        id_traces[sid],
                        globals_loo,
                        mode,
                        cg,
                        mg,
                        args.lag_lr_range,
                        args.within_lag_max,
                    )
                    per_id[sid] = acc
                    # update mapping for next iteration
                    id_mapping[sid] = {r["label"]: r["pred"] for r in res}

            score = sum(per_id.values())
            # Feedback score + F1
            fb_total = 0
            fb_ok = 0
            fb_tp = fb_fp = fb_fn = 0
            if feedback:
                for rec in feedback:
                    rid = resolve_export_id(export_dir, str(rec.get("id", "")))
                    if rid is None:
                        continue
                    traces = id_traces.get(rid)
                    if traces is None:
                        traces = load_id(export_dir, rid)
                        id_traces[rid] = traces
                    lab = rec.get("label")
                    cand = rec.get("candidate")
                    if lab not in traces:
                        continue
                    info = fx.emg_channel_info(lab)
                    sug = suggest_mapping_for_label(
                        lab,
                        traces[lab],
                        info,
                        globals_full,
                        build_label_index(globals_full.keys()),
                        cg,
                        mg,
                        args.lag_lr_range,
                        args.within_lag_max,
                    )
                    pred = sug["pred"]
                    resp = rec.get("response", "").lower()
                    fb_total += 1
                    if resp == "swap":
                        if pred == cand and pred != lab:
                            fb_ok += 1
                            fb_tp += 1
                        elif pred == lab:
                            fb_fn += 1
                        else:
                            fb_fp += 1
                            fb_fn += 1
                    elif resp == "ok":
                        if pred == lab:
                            fb_ok += 1
                        else:
                            fb_fp += 1
            fb_acc = fb_ok / fb_total if fb_total else 0.0
            fb_f1 = f1_from_counts(fb_tp, fb_fp, fb_fn)

            # Expected-mode F1 (LR as positives, OK as negatives)
            exp_tp = exp_fp = exp_fn = 0
            for sid in ids:
                mode = expected.get(sid, "ok")
                traces = id_traces.get(sid)
                if traces is None:
                    traces = load_id(export_dir, sid)
                    id_traces[sid] = traces
                globals_loo = build_globals_from_mapping(id_traces, id_mapping, globals_full, exclude_sid=sid)
                label_index = build_label_index(globals_loo.keys())
                for lab, t in traces.items():
                    info = fx.emg_channel_info(lab)
                    sug = suggest_mapping_for_label(
                        lab,
                        t,
                        info,
                        globals_loo,
                        label_index,
                        cg,
                        mg,
                        args.lag_lr_range,
                        args.within_lag_max,
                    )
                    pred = sug["pred"]
                    if mode == "lr":
                        info_pred = fx.emg_channel_info(pred) if pred != lab else None
                        is_lr = bool(info_pred and info and info_pred["muscle_key"] == info["muscle_key"] and info_pred["side"] != info["side"])
                        if is_lr:
                            exp_tp += 1
                        else:
                            exp_fn += 1
                            if pred != lab:
                                exp_fp += 1
                    else:
                        if pred != lab:
                            exp_fp += 1
            exp_f1 = f1_from_counts(exp_tp, exp_fp, exp_fn)

            if args.metric == "f1_feedback":
                metric = fb_f1
            elif args.metric == "f1_expected":
                metric = exp_f1
            elif args.metric == "f1_combined":
                metric = 0.5 * (fb_f1 + exp_f1)
            else:
                metric = args.expected_weight * score + args.feedback_weight * fb_acc

            combined = metric
            summaries.append(
                {
                    "corr_gain": cg,
                    "mad_gain": mg,
                    "score": combined,
                    "expected_score": score,
                    "feedback_acc": fb_acc,
                    "feedback_f1": fb_f1,
                    "expected_f1": exp_f1,
                    "per_id": per_id,
                    "feedback_n": fb_total,
                }
            )
            if combined > best.get("score", -1):
                best = summaries[-1]

    out_path = export_dir / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "summaries": summaries}, f, indent=2)

    print(f"[done] wrote {out_path}")
    print(f"[best] corr_gain={best['corr_gain']} mad_gain={best['mad_gain']} score={best['score']:.3f}")
    print(
        f"[best] expected_score={best.get('expected_score', 0):.3f} "
        f"feedback_acc={best.get('feedback_acc', 0):.3f} n={best.get('feedback_n', 0)} "
        f"feedback_f1={best.get('feedback_f1', 0):.3f} expected_f1={best.get('expected_f1', 0):.3f}"
    )
    for sid, acc in best.get("per_id", {}).items():
        print(f"  {sid}: acc={acc:.3f}")


if __name__ == "__main__":
    main()
