#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
updatePoints.py

Recursively scans a root folder for files named "*_splitCycles_osim_check.mat".
For each one, finds the matching base "*_splitCycles_osim.mat" in the same folder
(by replacing "_osim_check" with "_osim"), and overwrites selected per-cycle fields
from the base into the OSIM-CHECK file for both "left_stride" and "right_stride".

What gets copied per cycle (if present in the base):
  - "point"
  - every key that starts with "SO_" (e.g., "SO_activation", "SO_forces")
    **including ALL nested subfields**, not just "time".

The updated OSIM-CHECK file is written to a FLAT output directory (ignoring
the input subfolder structure) using the SAME FILENAME as the input OSIM-CHECK.

Assumes your loader returns a dict-of-dicts like:
    d["left_stride"]["cycle1"]["SO_activation"]["time"], ...

USAGE
-----
python -m utils_py.updatePoints "D:\\Data\\root" "D:\\Data\\output"
# or
python updatePoints.py "D:\\Data\\root" "D:\\Data\\output"
"""

import argparse
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy.io import savemat

# Import the user's loader
try:
    from .mat2dict import loadmat_to_dict  # when used as "python -m utils_py.updatePoints"
except Exception:
    try:
        from mat2dict import loadmat_to_dict  # when run directly in same folder
    except Exception as e:
        raise ImportError("Could not import loadmat_to_dict from mat2dict. "
                          "Place updatePoints.py next to mat2dict.py or use a package import.") from e


CYCLE_PAT = re.compile(r"^\s*cycle\s*(\d+)\s*$", re.IGNORECASE)


def nat_key(s: str):
    """Natural sort so 'cycle2' < 'cycle10'."""
    parts = re.split(r'(\d+)', str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def cycle_map_from_side(side: Any) -> Dict[str, Dict[str, Any]]:
    """
    Given side dict like d['left_stride'], return a map:
        {'cycle1': <cycle1_dict>, 'cycle2': <cycle2_dict>, ...}
    Only includes keys that look like 'cycleN' (case-insensitive).
    """
    if not isinstance(side, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in side.items():
        ks = str(k)
        m = CYCLE_PAT.match(ks)
        if m and isinstance(v, dict):
            out[f"cycle{int(m.group(1))}"] = v
    return out


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> int:
    """
    Recursively merge all key/values from src into dst (in place).
    Returns the number of keys written (leaf keys set/overwritten).
    """
    written = 0
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            written += deep_merge(dst[k], v)
        else:
            dst[k] = deepcopy(v)
            written += 1
    return written


def ensure_dict(obj: Any) -> Dict[str, Any]:
    """If obj is a dict, return it; otherwise return a new empty dict."""
    return obj if isinstance(obj, dict) else {}


def fields_to_copy_from_cycle(base_cycle: Dict[str, Any]) -> List[str]:
    """
    Decide which top-level fields to copy from a base cycle.
    Always includes 'point' if present, plus *all* keys that start with 'SO_'.
    """
    if not isinstance(base_cycle, dict):
        return []
    keys: List[str] = []
    if "point" in base_cycle:
        keys.append("point")
    keys.extend([k for k in base_cycle.keys() if isinstance(k, str) and k.startswith("SO_")])
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for k in keys:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def copy_cycle_fields(base_cycle: Dict[str, Any], dst_cycle: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """
    Copy eligible fields from base_cycle into dst_cycle.
    For 'point': replace entirely.
    For 'SO_*': deep-merge all subfields so *every* nested key transfers.
    Returns: (total_fields_written, per_block_keycount)
    """
    per_block_counts: Dict[str, int] = {}
    total = 0
    for field in fields_to_copy_from_cycle(base_cycle):
        src_block = base_cycle.get(field)
        if not isinstance(src_block, dict):
            # Non-dict payload (unexpected for these blocks), just overwrite
            dst_cycle[field] = deepcopy(src_block)
            per_block_counts[field] = per_block_counts.get(field, 0) + 1
            total += 1
            continue

        if field == "point":
            # Replace whole point block
            dst_cycle[field] = deepcopy(src_block)
            per_block_counts[field] = len(src_block)
            total += len(src_block)
        else:
            # SO_* : ensure destination exists and deep-merge all nested keys
            dst_block = ensure_dict(dst_cycle.get(field))
            wrote = deep_merge(dst_block, src_block)
            dst_cycle[field] = dst_block
            per_block_counts[field] = wrote
            total += wrote
    return total, per_block_counts


def update_side_fields(side_name: str, base: Dict[str, Any], osim: Dict[str, Any], file_label: str) -> None:
    """
    Copy desired fields for all matching cycles under side_name.
    - 'point' is fully replaced
    - 'SO_*' blocks are deep-merged (all subfields copied)
    """
    if side_name not in base or side_name not in osim:
        print(f"[INFO] {side_name} not present in both base and osim in {file_label}, skipping this side.")
        return

    base_side = base[side_name]
    osim_side = osim[side_name]

    if not isinstance(base_side, dict) or not isinstance(osim_side, dict):
        print(f"[WARN] {side_name} is not a dict in one of the files in {file_label}; skipping this side.")
        return

    base_cycles = cycle_map_from_side(base_side)
    osim_cycles = cycle_map_from_side(osim_side)

    if not base_cycles:
        print(f"[WARN] No 'cycle*' keys found in base {side_name} for {file_label}.")
    if not osim_cycles:
        print(f"[WARN] No 'cycle*' keys found in osim {side_name} for {file_label}.")

    common = sorted(set(base_cycles.keys()) & set(osim_cycles.keys()), key=nat_key)

    for k in sorted(set(base_cycles.keys()) - set(osim_cycles.keys()), key=nat_key):
        print(f"[WARN] {side_name}: {k} exists in base but not in osim in {file_label}.")
    for k in sorted(set(osim_cycles.keys()) - set(base_cycles.keys()), key=nat_key):
        print(f"[WARN] {side_name}: {k} exists in osim but not in base in {file_label}.")

    grand_counts: Dict[str, int] = {}

    for cyc in common:
        b_cycle = base_cycles[cyc]
        o_cycle = osim_cycles[cyc]

        if not isinstance(o_cycle, dict):
            print(f"[WARN] Destination {side_name}[{cyc}] is not a dict in {file_label}, skipping this cycle.")
            continue

        wrote, block_counts = copy_cycle_fields(b_cycle, o_cycle)
        for blk, cnt in block_counts.items():
            grand_counts[blk] = grand_counts.get(blk, 0) + cnt

    # Summary
    if grand_counts:
        summary = ", ".join([f"{k}:{v}" for k, v in sorted(grand_counts.items())])
        print(f"[OK] Updated {side_name} in {file_label} :: {summary}")
    else:
        print(f"[WARN] No eligible fields found to copy for {side_name} in {file_label}.")


def process_pair(base_path: Path, osim_check_path: Path, out_dir: Path) -> None:
    file_label = osim_check_path.name
    print(f"\n[PROCESS] {file_label}")
    print(f"         base(osim): {base_path}")
    print(f"         osim_check: {osim_check_path}")

    if not base_path.exists():
        print(f"[ERROR] Base OSIM file not found for {file_label}: {base_path}")
        return

    base_dict = loadmat_to_dict(str(base_path))
    osim_dict  = loadmat_to_dict(str(osim_check_path))

    update_side_fields("left_stride",  base_dict, osim_dict, file_label)
    update_side_fields("right_stride", base_dict, osim_dict, file_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / osim_check_path.name

    print(f"[SAVE] Writing updated OSIM-CHECK file to: {out_path}")
    savemat(
        str(out_path),
        osim_dict,
        oned_as="column",
        long_field_names=True,
        do_compression=True,
    )


def walk_and_process(root_dir: Path, out_dir: Path) -> None:
    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)
        for fname in filenames:
            # ONLY process *_splitCycles_osim_check.mat
            if not fname.endswith("_splitCycles_osim_check.mat"):
                continue

            osim_check_path = dirpath / fname
            base_name = fname.replace("_osim_check", "_osim")  # base is the *_osim.mat file
            base_path = dirpath / base_name

            process_pair(base_path, osim_check_path, out_dir)


def main():
    p = argparse.ArgumentParser(description=(
        "Copy 'point' and all 'SO_*' blocks (deeply) for every cycle ('left_stride'/'right_stride') "
        "from sibling *_splitCycles_osim.mat into matching *_splitCycles_osim_check.mat files, "
        "saving updated OSIM-CHECK files into a flat output folder."
    ))
    p.add_argument("root_dir", type=str, help="Root directory to scan recursively.")
    p.add_argument("out_dir", type=str, help="Output directory (flat).")
    args = p.parse_args()

    root_dir = Path(args.root_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()

    print(f"[START] Scanning root: {root_dir}")
    print(f"[OUTPUT] Updated files will be written to: {out_dir}")
    walk_and_process(root_dir, out_dir)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
