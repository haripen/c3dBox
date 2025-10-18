# c3dBox/Step4_check/audit_log.py
"""
Audit logging utilities for Step4_check.

This module writes selection-change audit entries per (pid, trial_type) into
CSV files stored under:
    <root>/Step4_check_logs/{PID}__{TrialType}.csv

Features
--------
- list_usernames(root): scan all CSVs in Step4_check_logs and return unique
  usernames (case-sensitive), sorted.
- log_selection_change(...): append a row with a fixed column schema using an
  atomic write (temp file + os.replace). Creates the directory if missing and
  writes a header if the CSV does not yet exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv
import datetime as _dt
import os
import tempfile

LOG_DIR_NAME = "Step4_check_logs"

LOG_COLUMNS: List[str] = [
    "iso_datetime",
    "username",
    "pid",
    "trial_type",
    # change deltas (counts that changed since last save)
    "left_kinetic_changed",
    "left_kinematic_changed",
    "right_kinetic_changed",
    "right_kinematic_changed",
    # final counts snapshot after save
    "left_kinetic_sel",
    "left_kinetic_unsel",
    "left_kinematic_sel",
    "left_kinematic_unsel",
    "right_kinetic_sel",
    "right_kinetic_unsel",
    "right_kinematic_sel",
    "right_kinematic_unsel",
]


def _log_dir(root: str | os.PathLike[str]) -> Path:
    return Path(root) / LOG_DIR_NAME


def list_usernames(root: str | os.PathLike[str]) -> List[str]:
    """
    Scan all CSVs under <root>/Step4_check_logs and return unique usernames.

    Returns
    -------
    list[str]
        Unique usernames sorted in ascending order. If no logs are present,
        returns an empty list.
    """
    log_dir = _log_dir(root)
    if not log_dir.exists():
        return []

    usernames: set[str] = set()
    for csv_path in log_dir.glob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    u = (row.get("username") or "").strip()
                    if u:
                        usernames.add(u)
        except Exception:
            # Ignore unreadable files; keep scanning
            continue
    return sorted(usernames)


def _safe_get(dct: Dict, *keys, default=0):
    cur = dct
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default


def log_selection_change(
    root: str | os.PathLike[str],
    pid: str,
    trial_type: str,
    username: str,
    changes_breakdown: Dict[str, Dict[str, int]],
    final_counts: Dict[str, Dict[str, Dict[str, int]]],
) -> None:
    """
    Append a row to <root>/Step4_check_logs/{PID}__{TrialType}.csv.

    Parameters
    ----------
    changes_breakdown
        Structure like:
        {
          "left": {"kinetic": <int>, "kinematic": <int>},
          "right": {"kinetic": <int>, "kinematic": <int>}
        }

    final_counts
        Structure like:
        {
          "left": {"kinetic": {"selected": X, "unselected": Y},
                   "kinematic": {"selected": X, "unselected": Y}},
          "right": {...}
        }
    """
    log_dir = _log_dir(root)
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"{pid}__{trial_type}.csv"

    now_iso = _dt.datetime.now().isoformat(timespec="seconds")

    payload = {
        "iso_datetime": now_iso,
        "username": username,
        "pid": pid,
        "trial_type": trial_type,
        "left_kinetic_changed": _safe_get(changes_breakdown, "left", "kinetic", default=0),
        "left_kinematic_changed": _safe_get(changes_breakdown, "left", "kinematic", default=0),
        "right_kinetic_changed": _safe_get(changes_breakdown, "right", "kinetic", default=0),
        "right_kinematic_changed": _safe_get(changes_breakdown, "right", "kinematic", default=0),
        "left_kinetic_sel": _safe_get(final_counts, "left", "kinetic", "selected", default=0),
        "left_kinetic_unsel": _safe_get(final_counts, "left", "kinetic", "unselected", default=0),
        "left_kinematic_sel": _safe_get(final_counts, "left", "kinematic", "selected", default=0),
        "left_kinematic_unsel": _safe_get(final_counts, "left", "kinematic", "unselected", default=0),
        "right_kinetic_sel": _safe_get(final_counts, "right", "kinetic", "selected", default=0),
        "right_kinetic_unsel": _safe_get(final_counts, "right", "kinetic", "unselected", default=0),
        "right_kinematic_sel": _safe_get(final_counts, "right", "kinematic", "selected", default=0),
        "right_kinematic_unsel": _safe_get(final_counts, "right", "kinematic", "unselected", default=0),
    }

    # Read existing file, if any, so we can append atomically.
    existing_text: Optional[str] = None
    if target.exists():
        try:
            existing_text = target.read_text(encoding="utf-8")
        except Exception:
            existing_text = None

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=str(log_dir)
    ) as tf:
        tmp_path = Path(tf.name)
        writer = csv.DictWriter(tf, fieldnames=LOG_COLUMNS)
        if existing_text:
            # Write back the old content verbatim, then a new row.
            # Ensure there is a trailing newline to keep csv aligned.
            if not existing_text.endswith(("\n", "\r")):
                existing_text = existing_text + "\n"
            tf.write(existing_text)
            writer.writerow({k: payload.get(k, "") for k in LOG_COLUMNS})
        else:
            # New file: header + first row
            writer.writeheader()
            writer.writerow({k: payload.get(k, "") for k in LOG_COLUMNS})

    os.replace(str(tmp_path), str(target))
