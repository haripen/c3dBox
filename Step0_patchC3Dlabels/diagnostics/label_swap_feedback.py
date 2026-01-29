#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive labeling tool for swap candidates.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Step0_patchC3Dlabels import fixEMGmisplacement_cli as fx
from Step0_patchC3Dlabels.diagnostics.optimize_detection import (
    canonical_muscle,
    close_muscle_keys,
    build_label_index,
    suggest_mapping_for_label,
)


def parse_args():
    p = argparse.ArgumentParser(description="Collect user feedback on swap candidates.")
    p.add_argument(
        "--export-dir",
        default="",
        help="Exports directory (defaults to latest fixEMG run exports).",
    )
    p.add_argument(
        "--ids",
        default="rd0079",
        help="Comma-separated export IDs (lowercase in filenames).",
    )
    p.add_argument(
        "--random-ids",
        type=int,
        default=0,
        help="If >0, ignore --ids and sample this many IDs from exports.",
    )
    p.add_argument(
        "--expected",
        default="rd0079:lr,rd0012:ok",
        help="Expected mode per id (lr or ok) to store in feedback.",
    )
    p.add_argument("--corr-gain", type=float, default=0.0)
    p.add_argument("--mad-gain", type=float, default=0.3)
    p.add_argument("--lag-lr-range", nargs=2, type=int, default=[55, 65])
    p.add_argument("--within-lag-max", type=int, default=5)
    p.add_argument("--n-swap", type=int, default=20)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--per-id", type=int, default=0, help="If >0, cap items per ID.")
    p.add_argument("--out", default="user_feedback.jsonl")
    p.add_argument("--seed", type=int, default=123)
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


def resolve_export_id(export_dir: Path, export_id: str):
    base = export_id.replace(" ", "_")
    if base.lower().endswith("_index"):
        base = base[:-6]
    idx_path = export_dir / f"processed_{base}_index.json"
    npz_path = export_dir / f"processed_{base}.npz"
    if idx_path.exists() and npz_path.exists():
        return base
    # case-insensitive search
    target = f"processed_{base}_index.json".lower()
    for p in export_dir.glob("processed_*_index.json"):
        if p.name.lower() == target:
            return p.stem.replace("processed_", "")
    return None


def load_id(export_dir: Path, export_id: str):
    resolved = resolve_export_id(export_dir, export_id)
    if resolved is None:
        raise FileNotFoundError(f"Missing processed files for {export_id}")
    idx_path = export_dir / f"processed_{resolved}_index.json"
    npz_path = export_dir / f"processed_{resolved}.npz"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)
    traces = {}
    cycles = {}
    outliers = {}
    for item in idx["labels"]:
        key = item["key"]
        traces[item["label"]] = np.asarray(npz[f"trace__{key}"], dtype=float)
        cycles[item["label"]] = np.asarray(npz[f"cycles__{key}"], dtype=float)
        outliers[item["label"]] = np.asarray(npz[f"outlier__{key}"], dtype=bool)
    return traces, cycles, outliers


def load_existing(out_path: Path):
    seen = set()
    if not out_path.exists():
        return seen
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                key = (item.get("id"), item.get("label"), item.get("candidate"))
                seen.add(key)
            except Exception:
                continue
    return seen


def plot_item(item, globals_map):
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(3, 1, 1)
    cycles = item["cycles"]
    outliers = item["outliers"]
    for c, is_out in zip(cycles, outliers):
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
    corr1, lag1 = fx.cross_corr(item["trace"], globals_map[item["label"]], max_lag=5)
    corr2, lag2 = fx.cross_corr(item["trace"], globals_map[item["candidate"]], max_lag=5)
    ax3.bar([0, 1], [corr1, corr2], color=["C0", "C2"])
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["orig", "cand"])
    ax3.set_ylim(0, 1)
    ax3.set_title(f"corr@±5 (orig={corr1:.3f}, cand={corr2:.3f})")

    fig.text(0.5, 0.98, f"{item['id']} | {item['label']} -> {item['candidate']} ({item['reason']})",
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
    return resp["value"]


def main():
    args = parse_args()
    random.seed(args.seed)
    export_dir = Path(args.export_dir) if args.export_dir else latest_exports_dir()
    if export_dir is None or not export_dir.exists():
        raise FileNotFoundError("Could not locate exports directory.")

    globals_map = load_globals(export_dir)
    label_index = build_label_index(globals_map.keys())

    expected = {}
    for pair in args.expected.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        expected[k.strip().lower()] = v.strip().lower()

    if args.random_ids > 0:
        pool = sorted({p.stem.replace("processed_", "").removesuffix("_index") for p in export_dir.glob("processed_*_index.json")})
        if len(pool) < args.random_ids:
            print(f"[warn] only {len(pool)} IDs in exports, using all.")
            ids = pool
        else:
            ids = random.sample(pool, args.random_ids)
    else:
        ids = [s.strip().lower() for s in args.ids.split(",") if s.strip()]

    out_path = export_dir / args.out
    seen = load_existing(out_path)

    swap_candidates = []
    ok_candidates = []

    for sid in ids:
        traces, cycles_map, outliers_map = load_id(export_dir, sid)
        for lab, t in traces.items():
            info = label_index.get(lab)
            sug = suggest_mapping_for_label(
                lab,
                t,
                info,
                globals_map,
                label_index,
                args.corr_gain,
                args.mad_gain,
                args.lag_lr_range,
                args.within_lag_max,
            )
            cand = sug["pred"]
            reason = sug["reason"]
            item = {
                "id": sid,
                "label": lab,
                "candidate": cand,
                "reason": reason,
                "trace": t,
                "cycles": cycles_map[lab],
                "outliers": outliers_map[lab],
            }
            if reason != "ok" and cand in globals_map:
                swap_candidates.append(item)
            else:
                ok_candidates.append(item)

    random.shuffle(swap_candidates)
    random.shuffle(ok_candidates)
    chosen = swap_candidates[: args.n_swap] + ok_candidates[: args.n_random]
    if args.per_id > 0:
        per_id = {}
        filtered = []
        for item in chosen:
            count = per_id.get(item["id"], 0)
            if count >= args.per_id:
                continue
            per_id[item["id"]] = count + 1
            filtered.append(item)
        chosen = filtered
    random.shuffle(chosen)

    with open(out_path, "a", encoding="utf-8") as f:
        for item in chosen:
            key = (item["id"], item["label"], item["candidate"])
            if key in seen:
                continue
            resp = plot_item(item, globals_map)
            if resp is None:
                break
            rec = {
                "id": item["id"],
                "label": item["label"],
                "candidate": item["candidate"],
                "reason": item["reason"],
                "response": resp,
                "expected": expected.get(str(item["id"]).lower(), ""),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            seen.add(key)

    print(f"[done] feedback saved to {out_path}")


if __name__ == "__main__":
    main()
