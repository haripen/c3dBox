#!/usr/bin/env python3
"""
Patch split-cycle MAT files with continuously preprocessed sEMG.

For each target MAT, this script:
  1. Finds the matching full-trial MAT by removing the "_splitCycles..." suffix.
  2. Filters the full-trial analog EMG* channels before any cycle slicing.
  3. Replaces only nested cycle analog EMG* arrays in-place.

The script is dry-run by default. Pass --apply to overwrite target files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_py.emg_processing import process_semg_signal


DEFAULT_TARGET_ROOT = Path(r"D:\Data_local\PRO_checked_clustered_mat")
DEFAULT_SOURCE_ROOT = Path(r"D:\Data_local\PROs_Walk_stitched_wM_2025-02-05")


@dataclass
class PatchStats:
    file: Path
    source: Optional[Path] = None
    status: str = "pending"
    cycles_seen: int = 0
    cycles_touched: int = 0
    arrays_replaced: int = 0
    missing_source_keys: int = 0
    max_time_error: float = 0.0
    message: str = ""


def _load_mat(path: Path) -> Dict[str, Any]:
    return scipy.io.loadmat(str(path), simplify_cells=True)


def _save_mat(path: Path, data: Mapping[str, Any]) -> None:
    out = {k: v for k, v in data.items() if not (k.startswith("__") and k.endswith("__"))}
    scipy.io.savemat(
        str(path),
        out,
        do_compression=True,
        long_field_names=True,
        format="5",
        oned_as="column",
    )


def _target_trial_stem(target: Path) -> str:
    return re.sub(r"_splitCycles.*$", "", target.stem)


def _is_target_mat(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".mat") and "_splitcycles" in name


def _is_source_candidate(path: Path, needed_stems: set[str]) -> bool:
    if path.suffix.lower() != ".mat":
        return False
    if "_splitcycles" in path.stem.lower():
        return False
    return path.stem in needed_stems


def _build_source_index(source_roots: Sequence[Path], needed_stems: set[str]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {stem: [] for stem in needed_stems}
    for root in source_roots:
        if root.is_file() and _is_source_candidate(root, needed_stems):
            index[root.stem].append(root)
            continue
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*.mat"), key=lambda p: str(p).lower()):
            if _is_source_candidate(path, needed_stems):
                index[path.stem].append(path)
    return index


def _analog_rate(analog: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> Optional[float]:
    t = analog.get("time")
    if t is not None:
        arr = np.asarray(t, dtype=float).ravel()
        if arr.size >= 2:
            dt = float(np.nanmedian(np.diff(arr)))
            if np.isfinite(dt) and dt > 0:
                return 1.0 / dt
    if isinstance(meta, Mapping) and "analog_rate" in meta:
        try:
            rate = float(meta["analog_rate"])
            return rate if np.isfinite(rate) and rate > 0 else None
        except Exception:
            return None
    return None


def _is_emg_key(key: str) -> bool:
    return str(key).upper().startswith("EMG")


def _preprocess_source_emg(source_data: Mapping[str, Any], settings: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    analog = source_data.get("analog")
    if not isinstance(analog, Mapping):
        raise ValueError("source has no analog struct")
    if "time" not in analog:
        raise ValueError("source analog struct has no time field")

    source_time = np.asarray(analog["time"], dtype=float).ravel()
    if source_time.size < 2:
        raise ValueError("source analog time is too short")

    fs = _analog_rate(analog, source_data.get("meta") if isinstance(source_data.get("meta"), Mapping) else None)
    if fs is None:
        raise ValueError("could not determine source analog sampling rate")

    processed: Dict[str, np.ndarray] = {}
    for key, value in analog.items():
        if not _is_emg_key(key):
            continue
        proc = process_semg_signal(
            np.asarray(value, dtype=float).ravel(),
            fs,
            bp_low=float(settings["bp_low"]),
            bp_high=float(settings["bp_high"]),
            lp_cut=float(settings["lp_cut"]),
            demean=bool(settings["demean"]),
            rectify=bool(settings["rectify"]),
            non_negative=bool(settings["non_negative"]),
        )
        if proc is not None:
            processed[key] = np.asarray(proc, dtype=float).ravel()

    if not processed:
        raise ValueError("source has no processable EMG* analog fields")
    return source_time, processed


def _nearest_time_indices(source_time: np.ndarray, target_time: np.ndarray, tolerance: float) -> Tuple[np.ndarray, float]:
    target = np.asarray(target_time, dtype=float).ravel()
    if target.size == 0:
        return np.array([], dtype=int), 0.0
    pos = np.searchsorted(source_time, target)
    pos = np.clip(pos, 0, source_time.size - 1)
    prev = np.clip(pos - 1, 0, source_time.size - 1)
    use_prev = np.abs(source_time[prev] - target) < np.abs(source_time[pos] - target)
    idx = np.where(use_prev, prev, pos).astype(int)
    max_error = float(np.nanmax(np.abs(source_time[idx] - target))) if target.size else 0.0
    if max_error > tolerance:
        raise ValueError(f"target cycle time not aligned to source analog time; max error {max_error:.9g}s")
    return idx, max_error


def _iter_cycle_analogs(data: Mapping[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for stride_name in ("left_stride", "right_stride", "exercise"):
        stride = data.get(stride_name)
        if not isinstance(stride, Mapping):
            continue
        for cycle_name, cycle in stride.items():
            if not str(cycle_name).startswith("cycle") or not isinstance(cycle, Mapping):
                continue
            analog = cycle.get("analog")
            if isinstance(analog, dict):
                yield stride_name, str(cycle_name), analog


def _select_source_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        try:
            data = _load_mat(path)
            analog = data.get("analog")
            if isinstance(analog, Mapping) and "time" in analog and any(_is_emg_key(k) for k in analog):
                return path
        except Exception:
            continue
    return None


def patch_one(
    target: Path,
    source_index: Mapping[str, Sequence[Path]],
    settings: Mapping[str, Any],
    apply: bool,
) -> PatchStats:
    stats = PatchStats(file=target)
    trial_stem = _target_trial_stem(target)
    source_path = _select_source_path(source_index.get(trial_stem, []))
    if source_path is None:
        stats.status = "missing_source"
        stats.message = f"no usable full-trial source MAT found for stem {trial_stem}"
        return stats

    stats.source = source_path
    try:
        target_data = _load_mat(target)
        source_data = _load_mat(source_path)
        source_time, source_emg = _preprocess_source_emg(source_data, settings)

        source_dt = float(np.nanmedian(np.diff(source_time)))
        tolerance = max(abs(source_dt) * 0.51, float(settings["time_tolerance"]))

        for _stride_name, _cycle_name, analog in _iter_cycle_analogs(target_data):
            stats.cycles_seen += 1
            if "time" not in analog:
                continue
            cycle_time = np.asarray(analog["time"], dtype=float).ravel()
            idx, max_error = _nearest_time_indices(source_time, cycle_time, tolerance)
            stats.max_time_error = max(stats.max_time_error, max_error)

            touched_cycle = False
            for key in list(analog.keys()):
                if not _is_emg_key(key):
                    continue
                replacement = source_emg.get(key)
                if replacement is None:
                    stats.missing_source_keys += 1
                    continue
                old_arr = np.asarray(analog[key])
                new_arr = replacement[idx]
                if old_arr.size != new_arr.size:
                    raise ValueError(
                        f"{key}: target size {old_arr.size} != source slice size {new_arr.size}"
                    )
                analog[key] = new_arr.reshape(old_arr.shape) if old_arr.shape else float(new_arr[0])
                stats.arrays_replaced += 1
                touched_cycle = True
            if touched_cycle:
                stats.cycles_touched += 1

        if stats.arrays_replaced == 0:
            stats.status = "no_emg_replaced"
            stats.message = "target had no replaceable nested analog EMG* fields"
            return stats

        if apply:
            _save_mat(target, target_data)
            stats.status = "patched"
        else:
            stats.status = "dry_run"
        return stats
    except Exception as exc:
        stats.status = "error"
        stats.message = str(exc)
        return stats


def _write_report(path: Path, rows: Sequence[PatchStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "source",
                "status",
                "cycles_seen",
                "cycles_touched",
                "arrays_replaced",
                "missing_source_keys",
                "max_time_error",
                "message",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": str(row.file),
                    "source": "" if row.source is None else str(row.source),
                    "status": row.status,
                    "cycles_seen": row.cycles_seen,
                    "cycles_touched": row.cycles_touched,
                    "arrays_replaced": row.arrays_replaced,
                    "missing_source_keys": row.missing_source_keys,
                    "max_time_error": f"{row.max_time_error:.9g}",
                    "message": row.message,
                }
            )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=None,
        help="Root containing full-trial Step1 MAT files. May be passed multiple times.",
    )
    parser.add_argument("--apply", action="store_true", help="Overwrite target MAT files in-place.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of target MAT files for testing.")
    parser.add_argument("--report", type=Path, default=Path("patch_split_cycle_semg_report.csv"))
    parser.add_argument("--bp-low", type=float, default=10.0)
    parser.add_argument("--bp-high", type=float, default=490.0)
    parser.add_argument("--lp-cut", type=float, default=15.0)
    parser.add_argument("--time-tolerance", type=float, default=1e-7)
    parser.add_argument("--keep-signed", action="store_true", help="Do not clamp processed sEMG below zero.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    source_roots = args.source_root if args.source_root else [DEFAULT_SOURCE_ROOT]

    targets = sorted(p for p in args.target_root.rglob("*.mat") if _is_target_mat(p))
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets:
        print(f"No split-cycle target MAT files found under {args.target_root}")
        return 2

    needed_stems = {_target_trial_stem(p) for p in targets}
    print(f"Targets: {len(targets)}")
    print(f"Unique trial stems: {len(needed_stems)}")
    print("Source roots:")
    for root in source_roots:
        print(f"  {root}")
    source_index = _build_source_index(source_roots, needed_stems)
    indexed = sum(1 for paths in source_index.values() if paths)
    print(f"Matched source stems before validation: {indexed}/{len(needed_stems)}")
    print("Mode:", "APPLY in-place overwrite" if args.apply else "DRY RUN")

    settings = {
        "bp_low": args.bp_low,
        "bp_high": args.bp_high,
        "lp_cut": args.lp_cut,
        "demean": True,
        "rectify": True,
        "non_negative": not args.keep_signed,
        "time_tolerance": args.time_tolerance,
    }

    rows: List[PatchStats] = []
    for i, target in enumerate(targets, 1):
        row = patch_one(target, source_index, settings, apply=args.apply)
        rows.append(row)
        print(
            f"[{i}/{len(targets)}] {target.name}: {row.status}, "
            f"cycles={row.cycles_touched}/{row.cycles_seen}, arrays={row.arrays_replaced}"
        )
        if row.message:
            print(f"  {row.message}")

    _write_report(args.report, rows)
    print(f"Report: {args.report}")

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if any(row.status == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
