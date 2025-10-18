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

Schema
------
Columns (in order):
    iso_datetime,
    username,
    pid,
    trial_type,
    left_kinetic_changed,
    left_kinematic_changed,
    right_kinetic_changed,
    right_kinematic_changed,
    left_kinetic_sel,
    left_kinetic_unsel,
    left_kinematic_sel,
    left_kinematic_unsel,
    right_kinetic_sel,
    right_kinematic_sel,
    right_kinetic_unsel,
    right_kinematic_unsel

Expected arguments
------------------
- changes_breakdown: nested dict with counts of cycles whose *final selection
  state changed* per side×mode, e.g.:
    {
      "left": {"kinetic": 3, "kinematic": 0},
      "right": {"kinetic": 1, "kinematic": 2}
    }
  (Missing keys are treated as 0.)

- final_counts: nested dict from status.compute_counts_by_side_and_mode(cycles),
  shape:
    { side: { mode: {"selected": int, "unselected": int} } }
  where side ∈ {"left","right"}, mode ∈ {"kinetic","kinematic"}.
  (Missing keys are treated as 0.)

All writes are UTF-8 and newline-normalized for CSV compatibility.
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Column order is fixed by specification.
LOG_COLUMNS: Tuple[str, ...] = (
    "iso_datetime",
    "username",
    "pid",
    "trial_type",
    "left_kinetic_changed",
    "left_kinematic_changed",
    "right_kinetic_changed",
    "right_kinematic_changed",
    "left_kinetic_sel",
    "left_kinetic_unsel",
    "left_kinematic_sel",
    "left_kinematic_unsel",
    "right_kinetic_sel",
    "right_kinematic_sel",
    "right_kinetic_unsel",
    "right_kinematic_unsel",
)


def _log_dir(root: os.PathLike | str) -> Path:
    """Return the Path to the Step4_check_logs directory under the given root."""
    return Path(root) / "Step4_check_logs"


def _log_path(root: os.PathLike | str, pid: str, trial_type: str) -> Path:
    """Return the CSV path for a given (pid, trial_type)."""
    # Filenames are specified as PID__TrialType.csv (case preserved).
    fname = f"{pid}__{trial_type}.csv"
    return _log_dir(root) / fname


def _ensure_dir(p: Path) -> None:
    """Create parent directory for path p if it doesn't exist."""
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_get(d: Mapping, *keys: str, default: int = 0) -> int:
    """
    Safely traverse nested mappings using keys; return default if any level is missing.
    Example: _safe_get(final_counts, "left", "kinetic", "selected", default=0)
    """
    cur: Mapping = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]  # type: ignore[index]
    try:
        return int(cur)  # type: ignore[return-value]
    except Exception:
        return default


def list_usernames(root: os.PathLike | str) -> List[str]:
    """
    Scan all CSV log files under <root>/Step4_check_logs/ and return a sorted
    list of unique usernames (case-sensitive).

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
                # Only consider files with the expected header having "username"
                if reader.fieldnames and "username" in reader.fieldnames:
                    for row in reader:
                        val = row.get("username")
                        if val:
                            usernames.add(val)
        except Exception:
            # Corrupt or unreadable file: skip silently per robustness principle.
            continue

    return sorted(usernames)


def log_selection_change(
    root: os.PathLike | str,
    pid: str,
    trial_type: str,
    username: str,
    changes_breakdown: Mapping[str, Mapping[str, int]],
    final_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> None:
    """
    Append an audit log row for the given selection change using an atomic write.

    Parameters
    ----------
    root : path-like
        The chosen project root.
    pid : str
        Participant ID.
    trial_type : str
        Trial type (e.g., "WalkA01").
    username : str
        Name chosen by the user (case-sensitive).
    changes_breakdown : Mapping
        Counts of cycles whose *final selection state* changed per side×mode:
            {"left": {"kinetic": int, "kinematic": int},
             "right": {"kinetic": int, "kinematic": int}}
        Missing keys are treated as zero.
    final_counts : Mapping
        Output of status.compute_counts_by_side_and_mode(cycles):
            { side: { mode: {"selected": int, "unselected": int} } }
        Missing keys are treated as zero.

    Notes
    -----
    - Creates <root>/Step4_check_logs if it does not exist.
    - If the target CSV does not exist, writes a header first.
    - Uses a temp file + os.replace(...) to ensure atomicity of the append.
    """
    target = _log_path(root, pid, trial_type)
    _ensure_dir(target)

    # Build the row payload according to the fixed schema.
    payload: Dict[str, str | int] = {
        "iso_datetime": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "pid": pid,
        "trial_type": trial_type,
        # Changes (per side × mode)
        "left_kinetic_changed": _safe_get(changes_breakdown, "left", "kinetic", default=0),
        "left_kinematic_changed": _safe_get(changes_breakdown, "left", "kinematic", default=0),
        "right_kinetic_changed": _safe_get(changes_breakdown, "right", "kinetic", default=0),
        "right_kinematic_changed": _safe_get(changes_breakdown, "right", "kinematic", default=0),
        # Final counts (per side × mode)
        "left_kinetic_sel": _safe_get(final_counts, "left", "kinetic", "selected", default=0),
        "left_kinetic_unsel": _safe_get(final_counts, "left", "kinetic", "unselected", default=0),
        "left_kinematic_sel": _safe_get(final_counts, "left", "kinematic", "selected", default=0),
        "left_kinematic_unsel": _safe_get(final_counts, "left", "kinematic", "unselected", default=0),
        "right_kinetic_sel": _safe_get(final_counts, "right", "kinetic", "selected", default=0),
        "right_kinematic_sel": _safe_get(final_counts, "right", "kinematic", "selected", default=0),
        "right_kinetic_unsel": _safe_get(final_counts, "right", "kinetic", "unselected", default=0),
        "right_kinematic_unsel": _safe_get(final_counts, "right", "kinematic", "unselected", default=0),
    }

    # Read the existing file (if any) so we can write a complete new copy atomically.
    existing_text: Optional[str] = None
    if target.exists():
        try:
            existing_text = target.read_text(encoding="utf-8")
        except Exception:
            # If unreadable, fall back to treating as new file with header.
            existing_text = None

    # Prepare temp file in the same directory for atomic replace.
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=str(target.parent),
        suffix=".tmp",
        encoding="utf-8",
        newline=""
    ) as tf:
        tmp_path = Path(tf.name)

        # If we had an existing file and could read it, start by copying it verbatim.
        if existing_text is not None:
            # Ensure it ends with a newline before we append a new CSV row.
            if existing_text and not existing_text.endswith(("\n", "\r")):
                existing_text = existing_text + "\n"
            tf.write(existing_text)
            writer = csv.DictWriter(tf, fieldnames=LOG_COLUMNS)
            # Do NOT write header again; existing_text already contains it.
            writer.writerow({k: payload.get(k, "") for k in LOG_COLUMNS})
        else:
            # New file: write header + first row.
            writer = csv.DictWriter(tf, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerow({k: payload.get(k, "") for k in LOG_COLUMNS})

    # Atomic replacement of the target.
    os.replace(str(tmp_path), str(target))
