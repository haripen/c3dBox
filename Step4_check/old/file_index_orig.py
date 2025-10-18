#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c3dBox/Step4_check/file_index.py

Utilities to scan result folders and index processed files for the PySide6 app.

Pattern
-------
We match files using FILE_REGEX:

    ^(?P<pid>[^_]+(?:_[^_]+)?)_
      (?P<datetime>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_
      (?P<trial_orig>[^_]+)_
      (?P<trial_type>[^_]+)_
      (?P<proc>fp0_clean_cropped_splitCycles(?:_osim)?)
      (?P<check>_check)?
      \.mat$

Captures:
- pid:         participant id (allows exactly one underscore inside, e.g. "P01" or "Doe_John")
- datetime:    "YYYY-MM-DD_HH-MM-SS"
- trial_orig:  original trial label (free of underscores)
- trial_type:  standardized trial type (free of underscores)
- proc:        "fp0_clean_cropped_splitCycles" with optional "_osim"
- check:       optional literal "_check"
- extension:   ".mat"

Priority
--------
Priority is used to deduplicate variants of the same (pid, trial_type, datetime, trial_orig):

    _osim_check  >  _osim  >  base _splitCycles

Concretely:
- If proc contains "_osim" and check == "_check"   -> priority = 3
- If proc contains "_osim" and no check            -> priority = 2
- Otherwise (base, with or without "_check")       -> priority = 1

API
---
scan_root(root) -> list[FileMeta]
    Recursively scan 'root' and return FileMeta records.

build_indices(files) -> (participants, trial_types_by_pid, files_by_pid_type)
    Deduplicate by highest priority per (pid, trial_type, datetime, trial_orig) and
    build indices:
      - participants: sorted list of unique pids
      - trial_types_by_pid: dict[pid] -> sorted list[str]
      - files_by_pid_type: dict[pid] -> dict[trial_type] -> list[FileMeta] (sorted by datetime)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


# Keep this pattern EXACTLY as specified.
FILE_REGEX: str = (
    r'^(?P<pid>[^_]+(?:_[^_]+)?)_'
    r'(?P<datetime>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_'
    r'(?P<trial_orig>[^_]+)_'
    r'(?P<trial_type>[^_]+)_'
    r'(?P<proc>fp0_clean_cropped_splitCycles(?:_osim)?)'
    r'(?P<check>_check)?\.mat$'
)
_file_rx = re.compile(FILE_REGEX)


@dataclass(frozen=True)
class FileMeta:
    """Lightweight record describing a matched file."""
    pid: str
    trial_type: str
    datetime: str
    trial_orig: str
    path: str
    priority: int

    @property
    def key(self) -> Tuple[str, str, str, str]:
        """Grouping key for deduplication."""
        return (self.pid, self.trial_type, self.datetime, self.trial_orig)


def _compute_priority(proc: str, check: str | None) -> int:
    """
    Map (proc, check) to priority according to:
        _osim_check (3) > _osim (2) > base (1)
    Note: base with '_check' (if present) does NOT increase priority beyond base.
    """
    is_osim = "_osim" in proc
    has_check = (check == "_check")
    if is_osim and has_check:
        return 3
    if is_osim:
        return 2
    return 1


def _iter_mat_files(root: str) -> Iterable[str]:
    """Yield absolute file paths for all .mat files under 'root' (recursive)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mat"):
                yield os.path.join(dirpath, fn)


def scan_root(root: str) -> List[FileMeta]:
    """
    Recursively scan 'root' for files matching FILE_REGEX.

    Returns:
        list[FileMeta]
    """
    results: List[FileMeta] = []
    for fpath in _iter_mat_files(root):
        fname = os.path.basename(fpath)
        m = _file_rx.match(fname)
        if not m:
            continue

        pid = m.group("pid")
        dt = m.group("datetime")
        trial_orig = m.group("trial_orig")
        trial_type = m.group("trial_type")
        proc = m.group("proc")
        check = m.group("check")
        prio = _compute_priority(proc, check)

        results.append(
            FileMeta(
                pid=pid,
                trial_type=trial_type,
                datetime=dt,
                trial_orig=trial_orig,
                path=os.path.abspath(fpath),
                priority=prio,
            )
        )
    return results


def build_indices(files: Iterable[FileMeta]):
    """
    Deduplicate and build indices for quick access in the UI layer.

    Deduping key: (pid, trial_type, datetime, trial_orig)
    Keep the record with the highest 'priority' (see module docstring).

    Returns:
        participants: list[str]
            Sorted unique participant IDs.

        trial_types_by_pid: dict[str, list[str]]
            For each participant, a sorted list of trial types.

        files_by_pid_type: dict[str, dict[str, list[FileMeta]]]
            For each (pid, trial_type), a list of FileMeta sorted by datetime (ascending).
            Note: 'datetime' is lexicographically sortable (YYYY-MM-DD_HH-MM-SS).
    """
    # 1) Deduplicate by highest priority.
    best: Dict[Tuple[str, str, str, str], FileMeta] = {}
    for fm in files:
        key = fm.key
        cur = best.get(key)
        if (cur is None) or (fm.priority > cur.priority):
            best[key] = fm
        # If equal priority, keep the first encountered deterministically.

    deduped = list(best.values())

    # 2) Build indices
    participants = sorted({fm.pid for fm in deduped})

    trial_types_by_pid: Dict[str, List[str]] = {}
    files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}

    for pid in participants:
        trial_types = sorted({fm.trial_type for fm in deduped if fm.pid == pid})
        trial_types_by_pid[pid] = trial_types
        files_by_pid_type[pid] = {}

    for fm in deduped:
        files_by_pid_type[fm.pid].setdefault(fm.trial_type, []).append(fm)

    # 3) Sort each list of files by datetime (ascending). The timestamp format is sortable as a string.
    for pid, by_type in files_by_pid_type.items():
        for ttype, items in by_type.items():
            by_type[ttype] = sorted(items, key=lambda r: r.datetime)

    return participants, trial_types_by_pid, files_by_pid_type


# ----------------------- tiny smoke test -----------------------

if __name__ == '__main__':
    """
    Minimal smoke test:
    Run this module directly to scan a local "./testing" folder (if present).
    Prints a small summary to stdout. Does not raise on empty / missing folder.
    """
    test_root = os.path.join(os.path.dirname(__file__) or ".", "testing")
    if not os.path.isdir(test_root):
        print(f"[file_index] No './testing' folder found next to this file. Nothing to scan.")
    else:
        print(f"[file_index] Scanning: {os.path.abspath(test_root)}")
        found = scan_root(test_root)
        print(f"[file_index] Matched files: {len(found)}")

        participants, trial_types_by_pid, files_by_pid_type = build_indices(found)
        print(f"[file_index] Participants: {participants}")
        for pid in participants:
            ttypes = trial_types_by_pid.get(pid, [])
            print(f"  - {pid}: trial types={ttypes}")
            for ttype in ttypes:
                items = files_by_pid_type[pid].get(ttype, [])
                print(f"      * {ttype}: {len(items)} file(s)")
                for fm in items[:3]:  # show up to 3 per type
                    print(f"          - {fm.datetime}  {fm.trial_orig}  (prio={fm.priority})")
