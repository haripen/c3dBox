#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a mapping.json to C3D files in place (swap-aware).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Step0_patchC3Dlabels import fixEMGmisplacement_cli as fx


def parse_args():
    p = argparse.ArgumentParser(description="Apply mapping.json to C3D files in place.")
    p.add_argument("--data-root", default=r"D:\Data_local\Pros_5er_hybrid_plain")
    p.add_argument("--task", required=True, help="Substring to match DESCRIPTION= in .enf (case-insensitive).")
    p.add_argument(
        "--mapping",
        required=True,
        help="Path to mapping.json (from fixEMG output).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = fx.normalize_mapping_pairs(mapping)

    c3ds = fx.list_c3ds(data_root)
    if not c3ds:
        raise FileNotFoundError("No C3D files found under data root.")

    changed_files = 0
    for c3d_path in c3ds:
        desc = fx.parse_enf_description(c3d_path)
        if not fx.match_task(desc, args.task):
            continue
        try:
            c3d = fx.ezc3d.c3d(str(c3d_path))
        except Exception:
            continue
        sid = fx.get_subject_id(c3d)
        if sid not in mapping:
            continue
        label_map = mapping[sid]
        if not label_map:
            continue
        asec = c3d["parameters"]["ANALOG"]
        labels = list(asec["LABELS"]["value"])
        labels, changed = fx.apply_label_mapping(labels, label_map)
        asec["LABELS"]["value"] = labels
        if changed:
            c3d.write(str(c3d_path))
            changed_files += 1
            print(f"[apply] patched {c3d_path} (changed {changed})")

    print(f"[done] changed files: {changed_files}")


if __name__ == "__main__":
    main()
