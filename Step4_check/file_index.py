# c3dBox/Step4_check/file_index.py

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from pathlib import Path

def _variant_priority(path: str) -> tuple[int, str]:
    n = Path(path).name.lower()
    # Highest → lowest
    if n.endswith("_splitcycles_osim_check.mat"):
        return (4, "osim_check")
    if n.endswith("_splitcycles_check.mat"):
        return (3, "check")
    if n.endswith("_splitcycles_osim.mat"):
        return (2, "osim")
    if n.endswith("_splitcycles.mat"):
        return (1, "base")
    return (0, "other")

@dataclass(frozen=True)
class FileMeta:
    path: str
    pid: str
    trial_type: str          # base (e.g., WalkA)
    original_trial_type: str # as in filename (e.g., WalkA03)
    session: str             # e.g., Dynamic03
    groups: Tuple[str, ...] = tuple()
    datetime: str = ""       # e.g., 2022-10-25_13-52-18
    variant: str = "base"
    priority: int = 1


_FILENAME_RE = re.compile(
    r"""
    ^
    (?P<pid>[^_]+)        _    # Rd0001
    (?P<pid2>[^_]+)       _    # Rd0001 (repeated)
    (?P<date>[^_]+)       _    # 2022-10-25
    (?P<time>[^_]+)       _    # 13-52-18
    (?P<session>[^_]+)    _    # Dynamic03
    (?P<trial>[^_]+)      _    # WalkA03
    (?P<rest>.*)               # fp0_clean_cropped_splitCycles_osim.mat
    $
    """,
    re.X,
)


def _strip_trailing_digits(s: str) -> str:
    # WalkA01 -> WalkA; WalkA -> WalkA
    return re.sub(r"\d+$", "", s)


def _candidate_groups() -> Tuple[str, ...]:
    # Hints for cycles.collect_cycles filtering; safe to keep broad
    return ("point", "analog", "JRL", "IK", "IK_markerErr", "SO_forces", "residual", "grf", "jrf")


def scan_root(root: str | Path) -> List[FileMeta]:
    root = Path(root)
    out: List[FileMeta] = []
    for p in sorted(root.rglob("*_splitCycles*.mat")):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        pid = m.group("pid")
        trial_full = m.group("trial")
        session = m.group("session")
        base = _strip_trailing_digits(trial_full)
        dt_str = f"{m.group('date')}_{m.group('time')}"
        prio, var = _variant_priority(str(p))
        out.append(
            FileMeta(
                path=str(p),
                pid=pid,
                trial_type=base,
                original_trial_type=trial_full,
                session=session,
                groups=_candidate_groups(),
                datetime=dt_str,
                variant=var,
                priority=prio,
            )
        )
    return out


def build_indices(files: Iterable[FileMeta]) -> tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[FileMeta]]],
]:
    # key = (pid, base_trial_type, datetime, original_trial_type)
    best: Dict[tuple, FileMeta] = {}
    for f in files:
        k = (f.pid, _strip_trailing_digits(f.trial_type), f.datetime, f.original_trial_type)
        cur = best.get(k)
        if cur is None or f.priority > cur.priority:
            best[k] = f

    dedup = list(best.values())

    participants = sorted({f.pid for f in dedup})
    trial_types_by_pid: Dict[str, List[str]] = {}
    files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}
    for f in dedup:
        t = _strip_trailing_digits(f.trial_type)
        trial_types_by_pid.setdefault(f.pid, [])
        if t not in trial_types_by_pid[f.pid]:
            trial_types_by_pid[f.pid].append(t)
        files_by_pid_type.setdefault(f.pid, {}).setdefault(t, []).append(f)

    for pid in trial_types_by_pid:
        trial_types_by_pid[pid].sort()
    for pid, inner in files_by_pid_type.items():
        for t in inner:
            inner[t].sort(key=lambda fm: (fm.datetime, fm.original_trial_type))

    return participants, trial_types_by_pid, files_by_pid_type
