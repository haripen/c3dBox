# c3dBox/Step4_check/status.py
"""
Status helpers and widget for Cycle Selection Tool.

Exports:
- compute_counts(cycles) -> dict
- compute_counts_by_side_and_mode(cycles) -> dict
- StatusWidget(QStatusBar)

Assumptions about each entry in `cycles`:
- It is either a dict-like mapping representing a single cycle or an object
  with one of the dict attributes: 'cycle', 'cycle_dict', 'data', or 'dict'.
- The cycle mapping contains:
    - 'manually_selected' (int 0/1), default 1
    - 'kinetic' (int 0/1); 1 => 'kinetic', else => 'kinematic'
- Side information is available on the wrapper object or mapping via one of:
    - attribute: .side (e.g., 'left_stride'/'right_stride' or 'left'/'right')
    - key: 'side'/'stride'/'side_name'/'stride_side'
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    # GUI only needed when using StatusWidget
    from PySide6.QtWidgets import QStatusBar, QLabel
    from PySide6.QtCore import Slot
except Exception:  # pragma: no cover - allows logic use without PySide6 installed
    QStatusBar = object  # type: ignore
    QLabel = object  # type: ignore
    def Slot(*_a, **_k):  # type: ignore
        def _wrap(fn): return fn
        return _wrap


def _as_cycle_mapping(item: Any) -> Mapping[str, Any]:
    """Extract the dict holding per-cycle keys from various wrappers."""
    if isinstance(item, Mapping):
        return item
    for attr in ("cycle", "cycle_dict", "data", "dict"):
        if hasattr(item, attr):
            maybe = getattr(item, attr)
            if isinstance(maybe, Mapping):
                return maybe
    # last resort: treat the object itself as mapping (empty)
    return {}


def _side_of(item: Any) -> Optional[str]:
    """Return 'left' or 'right' if discoverable, else None."""
    val = None
    # attributes first
    for attr in ("side", "side_name", "stride", "stride_side"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            break
    # mapping keys next
    if val is None and isinstance(item, Mapping):
        for k in ("side", "side_name", "stride", "stride_side"):
            if k in item:
                val = item[k]
                break
    if not isinstance(val, str):
        return None
    v = val.lower()
    if "left" in v:
        return "left"
    if "right" in v:
        return "right"
    if v in ("l", "r"):
        return "left" if v == "l" else "right"
    return None


def _mode_of(cycle_map: Mapping[str, Any]) -> str:
    """Return 'kinetic' if kinetic==1, else 'kinematic'."""
    try:
        kin = int(cycle_map.get("kinetic", 0))
    except Exception:
        kin = 0
    return "kinetic" if kin == 1 else "kinematic"


def _is_selected(cycle_map: Mapping[str, Any]) -> bool:
    """Return True if manually_selected is truthy (default True)."""
    try:
        sel = int(cycle_map.get("manually_selected", 1))
    except Exception:
        sel = 1
    return sel == 1


def compute_counts(cycles: Iterable[Any]) -> Dict[str, Dict[str, int]]:
    """
    Compute total counts across all cycles.

    Returns:
        {
          "kinetic":   {"selected": int, "unselected": int},
          "kinematic": {"selected": int, "unselected": int},
        }
    """
    out = {
        "kinetic": {"selected": 0, "unselected": 0},
        "kinematic": {"selected": 0, "unselected": 0},
    }
    for item in cycles or []:
        c = _as_cycle_mapping(item)
        mode = _mode_of(c)
        if _is_selected(c):
            out[mode]["selected"] += 1
        else:
            out[mode]["unselected"] += 1
    return out


def compute_counts_by_side_and_mode(
    cycles: Iterable[Any],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Compute nested counts per side and mode.

    Returns:
        {
          "left": {
            "kinetic":   {"selected": int, "unselected": int},
            "kinematic": {"selected": int, "unselected": int},
          },
          "right": {...}
        }
    """
    template = {"selected": 0, "unselected": 0}
    out = {
        "left": {"kinetic": dict(template), "kinematic": dict(template)},
        "right": {"kinetic": dict(template), "kinematic": dict(template)},
    }
    for item in cycles or []:
        c = _as_cycle_mapping(item)
        side = _side_of(item)
        if side not in ("left", "right"):
            # Skip if side cannot be resolved
            continue
        mode = _mode_of(c)
        bucket = out[side][mode]
        if _is_selected(c):
            bucket["selected"] += 1
        else:
            bucket["unselected"] += 1
    return out


class StatusWidget(QStatusBar):
    """
    Compact status bar showing:
    Sel KIN | Unsel KIN | Sel KINEM | Unsel KINEM

    Methods:
      - update_from_cycles(cycles)
      - update_from_counts(totals)
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._lbl_sel_kin = QLabel("Sel KIN: 0")
        self._lbl_uns_kin = QLabel("Unsel KIN: 0")
        self._lbl_sel_kim = QLabel("Sel KINEM: 0")
        self._lbl_uns_kim = QLabel("Unsel KINEM: 0")

        for i, w in enumerate(
            (self._lbl_sel_kin, self._lbl_uns_kin, self._lbl_sel_kim, self._lbl_uns_kim)
        ):
            self.addPermanentWidget(w)
            if i < 3:
                sep = QLabel(" | ")
                sep.setStyleSheet("color: palette(mid);")
                self.addPermanentWidget(sep)

        # Avoid size jitter
        self.setSizeGripEnabled(False)

    @Slot(object)
    def update_from_cycles(self, cycles: Iterable[Any]) -> None:
        """Convenience: recompute totals from a cycles collection and update labels."""
        totals = compute_counts(cycles)
        self.update_from_counts(totals)

    @Slot(dict)
    def update_from_counts(self, totals: Dict[str, Dict[str, int]]) -> None:
        """Update labels from a compute_counts(totals) result."""
        kin = totals.get("kinetic", {})
        kim = totals.get("kinematic", {})
        self._lbl_sel_kin.setText(f"Sel KIN: {int(kin.get('selected', 0))}")
        self._lbl_uns_kin.setText(f"Unsel KIN: {int(kin.get('unselected', 0))}")
        self._lbl_sel_kim.setText(f"Sel KINEM: {int(kim.get('selected', 0))}")
        self._lbl_uns_kim.setText(f"Unsel KINEM: {int(kim.get('unselected', 0))}")


__all__ = [
    "compute_counts",
    "compute_counts_by_side_and_mode",
    "StatusWidget",
]


if __name__ == "__main__":  # Tiny visual smoke test (optional)
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow
        import sys

        app = QApplication(sys.argv)
        win = QMainWindow()
        bar = StatusWidget()
        win.setStatusBar(bar)
        win.resize(800, 200)
        win.show()

        # Fake data: 6 cycles
        fake_cycles = [
            {"manually_selected": 1, "kinetic": 1, "side": "left_stride"},
            {"manually_selected": 0, "kinetic": 1, "side": "left_stride"},
            {"manually_selected": 1, "kinetic": 0, "side": "right_stride"},
            {"manually_selected": 0, "kinetic": 0, "side": "right_stride"},
            {"manually_selected": 1, "kinetic": 1, "side": "left"},
            {"manually_selected": 0, "kinetic": 0, "side": "right"},
        ]
        bar.update_from_cycles(fake_cycles)

        sys.exit(app.exec())
    except Exception as e:
        # Allow module import/run in headless or without PySide6
        print("StatusWidget demo skipped:", e)
