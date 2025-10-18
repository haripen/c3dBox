# c3dBox/Step4_check/save_flow.py
"""
Save & audit flow for the Cycle Selection Tool.

Responsibilities
----------------
- Merge per-cycle fields (manually_selected, reconstruction_ok, IK/SO flags) back to the
  owning data dict.
- Write sibling *_check.mat via io_mat.save_dict_check().
- Maintain a baseline snapshot of manually_selected to detect "dirty" changes since last save.
- Compute per-save changes by side×mode and final counts.
- Call audit_log.log_selection_change(...) when there are any changes.
- Provide a confirm-to-save prompt suitable for navigation or app close.
- Install a Ctrl+S shortcut to save.

This module is UI-light: it optionally uses a QMessageBox prompt and QShortcut,
but all heavy lifting is pure Python and import-guarded so tests can run headless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
import copy

# Optional Qt bits (safe imports)
try:
    from PySide6 import QtWidgets, QtGui
    from PySide6.QtGui import QKeySequence
except Exception:  # pragma: no cover - optional
    QtWidgets = None  # type: ignore
    QtGui = None  # type: ignore
    QKeySequence = None  # type: ignore

# Project-local imports
try:
    from . import io_mat  # type: ignore
except Exception:
    import io_mat  # type: ignore

# status is optional; we fall back to an internal implementation if missing
try:
    from .status import compute_counts_by_side_and_mode as _counts_side_mode  # type: ignore
except Exception:
    _counts_side_mode = None  # type: ignore

try:
    from . import audit_log  # type: ignore
except Exception:
    import audit_log  # type: ignore

FIELDS_TO_SAVE = {
    "manually_selected",
    "reconstruction_ok",
    # Allow any *_ok style boolean flags (IK/SO etc.)
    "IK_RMS_ok", "IK_MAX_ok",
    "SO_F_RMS_ok", "SO_F_MAX_ok",
    "SO_M_RMS_ok", "SO_M_MAX_ok",
    # Some datasets use generic flags too
    "IK_ok", "SO_ok",
    "kinetic",  # keep this for mode bookkeeping if present
}

def _normalize_side(s: str) -> str:
    s = (s or "").lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    # allow 'l'/'r'
    if s in ("l", "r"):
        return "left" if s == "l" else "right"
    return s or "unknown"


def _cycle_id(side: str, name: str) -> str:
    return f"{_normalize_side(side)}::{name}"


def _ensure_cycle_fields(cyc: Dict[str, Any]) -> None:
    cyc.setdefault("manually_selected", 1)
    cyc.setdefault("reconstruction_ok", True)


def _internal_counts_by_side_and_mode(cycles: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Fallback count computation if status.compute_counts_by_side_and_mode is unavailable.
    """
    counts = {
        "left": {"kinetic": {"selected": 0, "unselected": 0},
                 "kinematic": {"selected": 0, "unselected": 0}},
        "right": {"kinetic": {"selected": 0, "unselected": 0},
                  "kinematic": {"selected": 0, "unselected": 0}},
    }
    for c in cycles:
        side = _normalize_side(c.get("side") or c.get("stride_side") or c.get("stride") or "")
        if side not in ("left", "right"):
            continue
        flags = c.get("flags") or c
        kinetic = 1 if (flags.get("kinetic", 1) in (1, True)) else 0
        mode = "kinetic" if kinetic else "kinematic"
        sel = 1 if (flags.get("manually_selected", 1) in (1, True)) else 0
        key = "selected" if sel else "unselected"
        counts[side][mode][key] += 1
    return counts


@dataclass
class SaveController:
    root: str
    pid: str
    trial_type: str
    original_path: str
    data: Dict[str, Any]
    cycles: List[Dict[str, Any]]
    username: Optional[str] = None
    # function to show a username picker; receives (parent, known_usernames) and returns a string or None
    get_username_cb: Optional[Callable[[Any, List[str]], Optional[str]]] = None

    # internals
    _baseline_sel: Dict[str, int] = field(default_factory=dict)  # cycle_id -> manually_selected(0/1)
    _have_saved_once: bool = False

    # ---------------- API ----------------
    def install_ctrl_s_shortcut(self, widget: Any) -> None:
        """Install Ctrl+S (or Cmd+S on mac) on the given widget to trigger save_now."""
        if QtWidgets is None or QKeySequence is None:
            return
        sc = QtGui.QShortcut(QKeySequence.Save, widget)  # type: ignore[arg-type]
        sc.setContext(QtGui.Qt.ApplicationShortcut)  # type: ignore[attr-defined]
        sc.activated.connect(lambda: self.save_now(parent=widget))  # type: ignore[attr-defined]

    def is_dirty(self) -> bool:
        """Return True if any cycle's manually_selected differs from baseline."""
        if not self._baseline_sel:
            return False
        for c in self.cycles:
            side = c.get("side") or ""
            name = c.get("name") or ""
            cid = _cycle_id(side, name)
            current = 1 if (c.get("flags", {}).get("manually_selected", 1) in (1, True)) else 0
            if self._baseline_sel.get(cid, current) != current:
                return True
        return False

    def update_baseline(self) -> None:
        """Refresh the baseline snapshot from current cycles."""
        self._baseline_sel.clear()
        for c in self.cycles:
            side = c.get("side") or ""
            name = c.get("name") or ""
            cid = _cycle_id(side, name)
            flags = c.get("flags", {})
            cur = 1 if (flags.get("manually_selected", 1) in (1, True)) else 0
            self._baseline_sel[cid] = cur

    def _merge_cycle_flags_back(self) -> None:
        """Write whitelisted cycle flags back into the main data dict."""
        for c in self.cycles:
            side = c.get("side") or ""
            name = c.get("name") or ""
            if not side or not name:
                continue
            side_key = side  # already in correct keys like 'left_stride'/'right_stride'
            owner_side: Dict[str, Any] = self.data.setdefault(side_key, {})
            target: Dict[str, Any] = owner_side.setdefault(name, {})
            _ensure_cycle_fields(target)

            flags = (c.get("flags") or {}).copy()
            # Allow saving any "*_ok" flags too
            for k, v in list(flags.items()):
                if (k in FIELDS_TO_SAVE) or (k.endswith("_ok") and isinstance(v, (bool, int))):
                    target[k] = int(v) if isinstance(v, bool) else v

    def _changes_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Count how many cycles changed selection since baseline by side×mode.
        """
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

    def _choose_username(self, parent: Any) -> str:
        # If a username is already set, keep it.
        if self.username and self.username.strip():
            return self.username.strip()

        # Try to suggest from logs
        known_users: List[str] = []
        try:
            known_users = audit_log.list_usernames(self.root)  # type: ignore[attr-defined]
        except Exception:
            known_users = []

        if self.get_username_cb is not None:
            try:
                name = self.get_username_cb(parent, known_users)
                if name:
                    self.username = name.strip()
                    return self.username
            except Exception:
                pass

        # As a minimal fallback, use Qt input dialog if available
        if QtWidgets is not None:
            try:
                text, ok = QtWidgets.QInputDialog.getText(
                    parent, "Save – Username",
                    "Who is saving this selection?\n(used in the audit log)",
                )
                if ok and text.strip():
                    self.username = text.strip()
                    return self.username
            except Exception:
                pass

        self.username = "unknown"
        return self.username

    # -------- UI integrated helpers --------
    def maybe_prompt_to_save_on_navigation(self, parent: Any) -> Optional[bool]:
        """
        Call before switching PID or trial_type or before closing the app.

        Returns:
            True  -> proceed (user saved or discarded)
            False -> abort navigation
            None  -> no action required (no changes)
        """
        if not self.is_dirty() and self._have_saved_once:
            return None  # nothing to do

        if not self.is_dirty():
            return True

        if QtWidgets is None:
            # Headless default: auto-save
            self.save_now(parent=parent)
            return True

        mbox = QtWidgets.QMessageBox(parent)
        mbox.setWindowTitle("Unsaved changes")
        mbox.setText("You have unsaved cycle selections. Save before switching?")
        mbox.setIcon(QtWidgets.QMessageBox.Warning)
        save_btn = mbox.addButton("Save", QtWidgets.QMessageBox.AcceptRole)
        dont_btn = mbox.addButton("Don't Save", QtWidgets.QMessageBox.DestructiveRole)
        cancel_btn = mbox.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        mbox.setDefaultButton(save_btn)
        mbox.exec()

        clicked = mbox.clickedButton()
        if clicked is save_btn:
            self.save_now(parent=parent)
            return True
        elif clicked is dont_btn:
            return True
        else:
            return False

    # -------- core save --------
    def save_now(self, parent: Any = None) -> str:
        """
        Write <original>_check.mat and append an audit log line if anything changed.

        Returns:
            The path to the saved *_check.mat.
        """
        # username – this may bring up a prompt
        self._choose_username(parent)

        # 1) merge flags back
        self._merge_cycle_flags_back()

        # 2) compute change breakdown and final counts
        changes_breakdown = self._changes_breakdown()
        final_counts = self._final_counts()

        # 3) write out the check file
        out_path = io_mat.save_dict_check(self.original_path, self.data)  # type: ignore[attr-defined]

        # 4) audit log if there are changes
        total_changes = sum(changes_breakdown.get("left", {}).values()) + \
                        sum(changes_breakdown.get("right", {}).values())
        if total_changes > 0:
            try:
                audit_log.log_selection_change(  # type: ignore[attr-defined]
                    self.root, self.pid, self.trial_type, self.username or "unknown",
                    changes_breakdown, final_counts
                )
            except Exception:
                # logging should not block saving
                pass

        # 5) refresh baseline & flags
        self.update_baseline()
        self._have_saved_once = True
        return str(out_path)
