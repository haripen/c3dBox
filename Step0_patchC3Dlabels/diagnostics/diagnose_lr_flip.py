#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse fixEMGmisplacement_cli log.txt and summarize L/R flip diagnostics.
"""

import argparse
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Summarize L/R flip diagnostics from fixEMG logs.")
    p.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    p.add_argument("--run-dir", default="", help="Specific run folder name (e.g., fixEMG_walk_YYYYMMDD_HHMMSS).")
    return p.parse_args()


def latest_run(output_root: Path) -> Path:
    runs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("fixEMG_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = out_root / run_dir
    else:
        run_dir = latest_run(out_root)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError("No fixEMG output folder found.")

    log_path = run_dir / "processing" / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log.txt in {run_dir}")

    lr_line = None
    ratios = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("[summary] lr_suggest_lag_abs_p10_p50_p90="):
                lr_line = line.strip()
            if re.match(r"^\s+Rd\d{4}:", line):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].startswith("ratio="):
                    ratios.append((parts[0].replace(":", ""), float(parts[1].split("=")[1])))

    print(f"[run] {run_dir}")
    if lr_line:
        print(lr_line)
    if ratios:
        ratios.sort(key=lambda x: x[1], reverse=True)
        print("[top lr_suggest_ratio]")
        for sid, r in ratios[:10]:
            print(f"  {sid}: {r:.2f}")
    else:
        print("[warn] no lr_suggest_ratio lines found; re-run fixEMG with current code.")


if __name__ == "__main__":
    main()
