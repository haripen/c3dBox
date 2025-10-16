# c3dBox/Step4_check/save_flow.py
"""
Save & audit flow for the Cycle Selection Tool.

Key points in this revision:
- Decides the target path **before** writing (no late override).
- Overwrites the same file when you opened a `_check.mat`, with a one-time confirm.
- Uses `derive_check_path(...)` idempotently (no `_check_check.mat`).
- Adds `save_as(...)` to write to a user-chosen path (enforces `_check.mat` suffix).
- Keeps cycle flags in sync, recomputes QC if the module is available, and logs changes.

This module is UI-light: it optionally uses Qt (guarded imports) but all heavy lifting
is pure Python so tests can run headless.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Iterable

import copy

# Optional Qt bits
try:
    from PySide6 import QtWidgets, QtGui
except Exception:  # pragma: no cover
    QtWidgets = None  # type: ignore
    QtGui = None  # type: ignore

# Writing MAT files directly (for Save As)
try:
    import scipy.io as _sio  # type: ignore
except Exception:  # pragma: no cover
    _sio = None  # type: ignore

from . import audit_log
from .io_mat import derive_check_path
from . import io_mat as _iomod

# Optional QC and counts helpers
try:
    from . import qc as _qc  # type: ignore
except Exception:
    _qc = None  # type: ignore

try:
    from .status import compute_counts_by_side_and_mode as _counts_side_mode  # type: ignore
except Exception:
    _counts_side_mode = None  # type: ignore


FIELDS_TO_SAVE = {
    "manually_selected",
    "reconstruction_ok",
    # IK/SO typical flags
    "IK_RMS_ok", "IK_MAX_ok",
    "SO_F_RMS_ok", "SO_F_MAX_ok",
    "SO_M_RMS_ok", "SO_M_MAX_ok",
    # generic shortcuts if present
    "IK_ok", "SO_ok",
    # bookkeeping
    "kinetic",
}


def _normalize_side(s: str) -> str:
    s = (s or "").lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    if s in ("l", "r"):
        return "left" if s == "l" else "right"
    return s or "unknown"


def _cycle_id(side: str, name: str) -> str:
    return f"{_normalize_side(side)}::{name}"


def _internal_counts_by_side_and_mode(cycles: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    counts = {
        "left": {"kinetic": {"selected": 0, "unselected": 0},
                 "kinematic": {"selected": 0, "unselected": 0}},
        "right": {"kinetic": {"selected": 0, "unselected": 0},
                  "kinematic": {"selected": 0, "unselected": 0}},
    }
    for c in cycles:
        side = _normalize_side(c.get("side") or c.get("stride_side") or "")
        if side not in ("left", "right"):
            continue
        flags = c.get("flags") or c
        kinetic = 1 if (flags.get("kinetic", 1) in (1, True)) else 0
        mode = "kinetic" if kinetic else "kinematic"
        sel = 1 if (flags.get("manually_selected", 1) in (1, True)) else 0
        key = "selected" if sel else "unselected"
        counts[side][mode][key] += 1
    return counts


def _write_mat_to_path(path: Path, data: Dict[str, Any]) -> Path:
    """
    Write `data` to an explicit `path` using the same options as io_mat.save_dict_check.
    """
    if _sio is None:
        # fallback: let io_mat save to a derived path near `path`
        _iomod.save_dict_check(str(path), data)  # type: ignore[attr-defined]
        return path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _sio.savemat(
        str(path),
        data,
        do_compression=True,
        long_field_names=True,
        format="5",
    )
    return path


@dataclass
class SaveController:
    root: str
    pid: str
    trial_type: str
    original_path: str                # file that was opened
    data: Dict[str, Any]
    cycles: List[Dict[str, Any]]
    username: Optional[str] = None

    # Internals
    _baseline_sel: Dict[str, int] = field(default_factory=dict)  # cycle_id -> 0/1
    _have_saved_once: bool = False
    save_path: Optional[Path] = None                             # where subsequent saves go

    def __post_init__(self) -> None:
        # Default save path: if opened a _check → same file, else derive sibling _check
        op = Path(self.original_path)
        self.save_path = op if op.name.lower().endswith("_check.mat") else derive_check_path(op)
        self.update_baseline()
        # If we opened a _check that already exists, consider it "already saved once"
        self._have_saved_once = self.save_path.exists()

    # ---------- Baseline & dirty ----------

    def update_baseline(self) -> None:
        self._baseline_sel.clear()
        for c in self.cycles:
            side = c.get("side") or ""
            name = c.get("name") or ""
            cid = _cycle_id(side, name)
            flags = c.get("flags", {})
            cur = 1 if (flags.get("manually_selected", 1) in (1, True)) else 0
            self._baseline_sel[cid] = cur

    def is_dirty(self) -> bool:
        if not self._baseline_sel:
            return False
        for c in self.cycles:
            side = c.get("side") or ""
            name = c.get("name") or ""
            cid = _cycle_id(side, name)
            flags = c.get("flags", {})
            cur = 1 if (flags.get("manually_selected", 1) in (1, True)) else 0
            if self._baseline_sel.get(cid, cur) != cur:
                return True
        return False

    # ---------- QC & merge flags ----------

    def _ensure_qc_flags_before_save(self) -> None:
        data = self.data or {}
        meta = data.get("meta", {})
        for side in ("left_stride", "right_stride"):
            block = data.get(side, {})
            if not isinstance(block, dict):
                continue
            for name, cyc in block.items():
                if not isinstance(cyc, dict):
                    continue
                cyc.setdefault("reconstruction_ok", True)
                if _qc is not None:
                    try:
                        cyc.update(_qc.eval_ik_flags(cyc))
                    except Exception:
                        pass
                    try:
                        cyc.update(_qc.eval_so_flags(cyc, side, meta))
                    except Exception:
                        pass
                cyc.setdefault("IK_RMS_ok", True)
                cyc.setdefault("IK_MAX_ok", True)
                for k in ("SO_F_RMS_ok","SO_F_MAX_ok","SO_M_RMS_ok","SO_M_MAX_ok"):
                    cyc.setdefault(k, True)

    def _merge_cycle_flags_back(self) -> None:
        for c in self.cycles:
            side = c.get("side") or ""
            name = c.get("name") or ""
            if not side or not name:
                continue
            owner_side: Dict[str, Any] = self.data.setdefault(side, {})
            target: Dict[str, Any] = owner_side.setdefault(name, {})
            target.setdefault("manually_selected", 1)
            target.setdefault("reconstruction_ok", True)
            flags = (c.get("flags") or {}).copy()
            for k, v in list(flags.items()):
                if (k in FIELDS_TO_SAVE) or (k.endswith("_ok") and isinstance(v, (bool, int))):
                    target[k] = int(v) if isinstance(v, bool) else v

    # ---------- Counts & logging ----------

    def _changes_breakdown(self) -> Dict[str, Dict[str, int]]:
        changes = {"left": {"kinetic": 0, "kinematic": 0},
                   "right": {"kinetic": 0, "kinematic": 0}}
        for c in self.cycles:
            side = _normalize_side(c.get("side") or "")
            if side not in ("left", "right"):
                continue
            name = c.get("name") or ""
            cid = _cycle_id(side, name)
            flags = c.get("flags", {})
            now_sel = 1 if (flags.get("manually_selected", 1) in (1, True)) else 0
            before_sel = self._baseline_sel.get(cid, now_sel)
            if now_sel != before_sel:
                kinetic = 1 if (flags.get("kinetic", 1) in (1, True)) else 0
                mode = "kinetic" if kinetic else "kinematic"
                changes[side][mode] += 1
        return changes

    def _final_counts(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        if _counts_side_mode is not None:
            try:
                return _counts_side_mode(self.cycles)  # type: ignore[misc]
            except Exception:
                pass
        return _internal_counts_by_side_and_mode(self.cycles)

    # ---------- Save paths ----------

    def _decide_target_path(self) -> Path:
        assert self.save_path is not None
        return Path(self.save_path)

    # ---------- Public save APIs ----------

    def save_now(self, parent: Any = None) -> str:
        """
        Write to the current target path (overwrite if it's the opened `_check.mat`).

        Returns:
            The path to the saved file (str). Empty string if user aborted.
        """
        # 0) Determine target before writing
        target = self._decide_target_path()

        # 1) Optional overwrite prompt only the *first* time when saving over an opened _check
        if (str(target).lower().endswith("_check.mat")
            and target.exists()
            and not self._have_saved_once
            and QtWidgets is not None
        ):
            r = QtWidgets.QMessageBox.question(
                parent, "Overwrite existing",
                f"Overwrite existing file?\n{target.name}",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if r != QtWidgets.QMessageBox.Yes:
                return ""  # aborted

        # 2) Make sure QC present then merge flags back
        self._ensure_qc_flags_before_save()
        self._merge_cycle_flags_back()

        # 3) Compute changes & counts *before* writing
        changes_breakdown = self._changes_breakdown()
        final_counts = self._final_counts()

        # 4) Write file
        if target == Path(self.original_path) or str(target).lower().endswith("_check.mat"):
            # Write exactly to `target`
            _write_mat_to_path(target, self.data)
        else:
            # Fallback: use io_mat (derives a _check near original)
            target = _iomod.save_dict_check(self.original_path, self.data)  # type: ignore[attr-defined]
            target = Path(target)

        # 5) Audit log only if anything changed
        total_changes = sum(changes_breakdown.get("left", {}).values()) + \
                        sum(changes_breakdown.get("right", {}).values())
        if total_changes > 0:
            try:
                audit_log.log_selection_change(  # type: ignore[attr-defined]
                    self.root, self.pid, self.trial_type, self.username or "unknown",
                    changes_breakdown, final_counts
                )
            except Exception:
                pass  # never block saving on logging failures

        # 6) Update baseline and mark as saved
        self.update_baseline()
        self._have_saved_once = True
        self.save_path = target

        return str(target)

    def save_as(self, parent: Any = None) -> str:
        """
        Prompt for a new target and save there. Enforces `_check.mat` suffix.
        """
        if QtWidgets is None:
            # Headless fallback: append `_check.mat` if needed and write
            tgt = derive_check_path(Path(self.original_path))
            self.save_path = tgt
            return self.save_now(parent)

        start_dir = str(Path(self.save_path or self.original_path).parent)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent, "Save As (_check.mat)", start_dir, "MAT files (*.mat)"
        )
        if not fn:
            return ""
        tgt = Path(fn)
        if not tgt.name.lower().endswith("_check.mat"):
            tgt = derive_check_path(tgt)
        self.save_path = tgt
        # When choosing a new target, reset the one-time overwrite confirm
        self._have_saved_once = tgt.exists()
        return self.save_now(parent)
