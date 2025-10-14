# c3dBox/Step4_check/save_flow.py
"""
Save & audit flow for Cycle Selection Tool.

Responsibilities
- Merge per-cycle fields (manually_selected, reconstruction_ok, IK/SO flags) back to the
  owning data dict.
- Write sibling *_check.mat via io_mat.save_dict_check().
- Maintain a baseline snapshot of manually_selected to detect "dirty" changes since last save.
- Compute per-save changes by side×mode and final counts using status.compute_counts_by_side_and_mode().
- Call audit_log.log_selection_change(...) when there are any changes.
- Provide a confirm-to-save prompt suitable for navigation or app close.

This module is UI-light: it uses a QMessageBox prompt but is otherwise logic-focused.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .io_mat import save_dict_check, derive_check_path
from .status import compute_counts_by_side_and_mode

try:
    # Optional logging backend; keep this import soft to avoid hard dependency while developing.
    from . import audit_log  # noqa: F401
except Exception:  # pragma: no cover
    audit_log = None  # type: ignore

try:
    from PySide6.QtWidgets import QMessageBox, QWidget
except Exception:  # pragma: no cover
    # Fallback shims so this file can be imported in headless contexts (tests, etc.)
    QMessageBox = None  # type: ignore
    QWidget = object  # type: ignore


# Keys we always persist back into the source dict if present on a cycle's flags
FIELDS_TO_SAVE = {
    "manually_selected",
    "reconstruction_ok",
    "IK_RMS_ok",
    "IK_MAX_ok",
    "SO_F_RMS_ok",
    "SO_F_MAX_ok",
    "SO_M_RMS_ok",
    "SO_M_MAX_ok",
}


CycleLike = Union[Dict[str, Any], Any]  # object with attrs or dict-like


def _get(cycle: CycleLike, key: str, default: Any = None) -> Any:
    """Attribute-or-dict getter."""
    if isinstance(cycle, dict):
        return cycle.get(key, default)
    return getattr(cycle, key, default)


def _normalize_side(side_key: str) -> str:
    s = side_key.lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return side_key


def _nested_get(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _ensure_cycle_fields(target_cycle: Dict[str, Any]) -> None:
    """Create required fields on the owner dict cycle if missing."""
    target_cycle.setdefault("manually_selected", 1)
    target_cycle.setdefault("reconstruction_ok", True)


def _mode_of_cycle(data: Dict[str, Any], side_key: str, cycle_name: str) -> str:
    """Return 'kinetic' if cycle has kinetic==1 else 'kinematic'."""
    kin = _nested_get(data, [side_key, cycle_name, "kinetic"], default=0)
    try:
        kin_val = int(kin)
    except Exception:
        kin_val = 0
    return "kinetic" if kin_val == 1 else "kinematic"


def _init_breakdown() -> Dict[str, Dict[str, int]]:
    return {
        "left": {"kinetic": 0, "kinematic": 0},
        "right": {"kinetic": 0, "kinematic": 0},
    }


class SaveManager:
    """
    Orchestrates merging flags, saving *_check.mat, auditing, and dirty detection.

    Parameters
    ----------
    root : str
        Project root directory (where log files live).
    pid : str
        Participant ID (from filename).
    trial_type : str
        Trial type (from filename).
    original_path : str
        Path to the source *_splitCycles*.mat that was loaded.
    data : dict
        The full data dict currently loaded (will be persisted).
    cycles : list
        List of cycle references (from cycles.collect_cycles). Each cycle must provide:
        - side: 'left_stride'|'right_stride' (or similar; will be normalized to 'left'/'right')
        - name: e.g., 'cycle1'
        - flags: dict containing at least 'manually_selected', 'reconstruction_ok', and IK/SO flags.
    username : Optional[str]
        Current username (if already known via UI/menu).
    get_username_cb : Optional[Callable[[Optional[QWidget], List[str]], str]]
        Callback to obtain username if not set. Should return a non-empty string or raise/cancel.
        Signature: fn(parent_widget, known_usernames) -> username
    """

    def __init__(
        self,
        root: str,
        pid: str,
        trial_type: str,
        original_path: str,
        data: Dict[str, Any],
        cycles: List[CycleLike],
        username: Optional[str] = None,
        get_username_cb: Optional[Callable[[Optional[QWidget], List[str]], str]] = None,
    ) -> None:
        self.root = root
        self.pid = pid
        self.trial_type = trial_type
        self.original_path = original_path
        self.data = data
        self.cycles = cycles
        self.username = username or ""
        self.get_username_cb = get_username_cb

        # Baseline of manually_selected per (side, name)
        self._baseline: Dict[Tuple[str, str], int] = self._snapshot_manually_selected()
        self._saved_once: bool = False
        self._last_saved_check_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def is_dirty(self) -> bool:
        """Return True if any cycle's manually_selected differs from the last saved baseline."""
        cur = self._snapshot_manually_selected()
        return any(cur[k] != self._baseline.get(k, cur[k]) for k in cur.keys())

    def save(self, parent: Optional[QWidget] = None) -> Optional[str]:
        """
        Save current state to sibling *_check.mat.

        - Merge flags back into data dict
        - Compute change breakdown vs baseline
        - Persist .mat
        - Audit (if any change)
        - Update baseline

        Returns
        -------
        path : Optional[str]
            Path to the saved *_check.mat (if derivable), or None.
        """
        # 1) Ensure username for auditing if we have changes
        changes_breakdown = self._compute_changes_breakdown()
        any_changes = any(changes_breakdown[s][m] > 0 for s in changes_breakdown for m in changes_breakdown[s])

        if any_changes and not self.username:
            self.username = self._obtain_username(parent)

        # 2) Merge all cycle flags into the data dict
        self._merge_cycles_into_data()

        # 3) Persist *_check.mat
        save_dict_check(self.original_path, self.data)
        check_path = derive_check_path(self.original_path)
        self._last_saved_check_path = check_path

        # 4) Final counts "after save"
        final_counts = compute_counts_by_side_and_mode(self.cycles)

        # 5) Audit logging
        if any_changes and audit_log is not None:
            try:
                # If audit_log exposes helper to list usernames for dropdowns, that's used elsewhere.
                audit_log.log_selection_change(
                    root=self.root,
                    pid=self.pid,
                    trial_type=self.trial_type,
                    username=self.username or "unknown",
                    changes_breakdown=changes_breakdown,
                    final_counts=final_counts,
                )
            except Exception:
                # Stay robust: saving must not fail due to logging errors
                pass

        # 6) Update baseline and flags
        self._baseline = self._snapshot_manually_selected()
        self._saved_once = True
        return check_path

    def handle_navigation_or_close(self, parent: Optional[QWidget] = None) -> bool:
        """
        Call this before navigating away (switch PID/trial) or on app close.

        If dirty since last baseline, prompts to save. Returns:
        - True  to proceed with navigation/close
        - False to cancel (user pressed "Cancel" or dialog unavailable)

        Behavior matches: "if dirty and no save yet, prompt to save".
        """
        if not self.is_dirty():
            return True
        if self._saved_once:
            # You asked specifically: prompt if dirty AND no save yet.
            # If you also want prompting after subsequent edits, remove this guard.
            return True

        # Prompt only if we can (GUI context). In headless, default to proceed without saving.
        if QMessageBox is None:
            return True

        msg = QMessageBox(parent)
        msg.setWindowTitle("Unsaved changes")
        msg.setText("You have unsaved manual selections.\nDo you want to save them before continuing?")
        save_btn = msg.addButton("Save", QMessageBox.AcceptRole)
        discard_btn = msg.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.setIcon(QMessageBox.Warning)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is save_btn:
            self.save(parent=parent)
            return True
        if clicked is discard_btn:
            # Do NOT update baseline; caller is navigating away / reloading anyway.
            return True
        # Cancel
        return False

    def set_username(self, username: str) -> None:
        self.username = (username or "").strip()

    def last_saved_path(self) -> Optional[str]:
        return self._last_saved_check_path

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _snapshot_manually_selected(self) -> Dict[Tuple[str, str], int]:
        snap: Dict[Tuple[str, str], int] = {}
        for c in self.cycles:
            side_key = _get(c, "side", "")
            name = _get(c, "name", "")
            flags = _get(c, "flags", {}) or {}
            val = flags.get("manually_selected", 1)
            try:
                val_i = int(val)
            except Exception:
                val_i = 1
            snap[(side_key, name)] = val_i
        return snap

    def _compute_changes_breakdown(self) -> Dict[str, Dict[str, int]]:
        breakdown = _init_breakdown()
        current = self._snapshot_manually_selected()

        for (side_key, name), cur_val in current.items():
            base_val = self._baseline.get((side_key, name), cur_val)
            if cur_val == base_val:
                continue
            side = _normalize_side(side_key)
            mode = _mode_of_cycle(self.data, side_key, name)
            if side not in breakdown:
                breakdown[side] = {"kinetic": 0, "kinematic": 0}
            breakdown[side][mode] = breakdown[side].get(mode, 0) + 1
        return breakdown

    def _merge_cycles_into_data(self) -> None:
        """
        Push flags from self.cycles back into the owning dict at:
            data[side_key][cycle_name][field] = value
        For safety, we only persist boolean/int-like flags and known FIELDS_TO_SAVE.
        """
        for c in self.cycles:
            side_key = _get(c, "side")
            name = _get(c, "name")
            flags: Dict[str, Any] = _get(c, "flags", {}) or {}

            if not side_key or not name:
                continue

            # Obtain the target cycle dict
            owner_side: Dict[str, Any] = self.data.setdefault(side_key, {})
            target_cycle: Dict[str, Any] = owner_side.setdefault(name, {})
            _ensure_cycle_fields(target_cycle)

            # Write back whitelisted flags if present
            for k, v in flags.items():
                if (k in FIELDS_TO_SAVE) or (isinstance(v, (bool, int)) and k.endswith("_ok")):
                    try:
                        # Normalize ints for MATLAB convenience
                        if isinstance(v, bool):
                            vv = int(v)
                        elif isinstance(v, (int,)):
                            vv = int(v)
                        else:
                            vv = v
                        target_cycle[k] = vv
                    except Exception:
                        # Skip non-serializable values
                        continue

    def _obtain_username(self, parent: Optional[QWidget]) -> str:
        """
        Ask UI (via callback) for username. If unavailable, return 'unknown'.

        If audit_log provides a list of known usernames, offer them via the callback's dropdown.
        """
        known_users: List[str] = []
        if audit_log is not None and hasattr(audit_log, "list_known_usernames"):
            try:
                known_users = list(audit_log.list_known_usernames(self.root))  # type: ignore[attr-defined]
            except Exception:
                known_users = []

        if callable(self.get_username_cb):
            try:
                name = self.get_username_cb(parent, known_users)  # UI-owned dialog
                return (name or "").strip() or "unknown"
            except Exception:
                return "unknown"
        return "unknown"
