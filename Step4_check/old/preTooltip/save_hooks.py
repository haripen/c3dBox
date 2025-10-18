# c3dBox/Step4_check/save_hooks.py
"""
Qt wiring helpers for the save flow.

- add_ctrl_s(widget, controller): installs Ctrl+S to trigger save
- guard_combo_change(combo, controller, on_change): prompts to save if dirty before allowing change
"""
from __future__ import annotations

from typing import Callable, Optional

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    from PySide6.QtGui import QKeySequence
    from PySide6.QtCore import Qt
except Exception:  # pragma: no cover
    QtWidgets = None  # type: ignore
    QtGui = None  # type: ignore
    QtCore = None  # type: ignore
    QKeySequence = None  # type: ignore

def add_ctrl_s(widget, controller) -> None:
    if QtWidgets is None:
        return
    sc = QtGui.QShortcut(QKeySequence.Save, widget)  # type: ignore[arg-type]
    sc.setContext(QtCore.Qt.ApplicationShortcut)
    sc.activated.connect(lambda: controller.save_now(parent=widget))  # type: ignore[attr-defined]

def guard_combo_change(combo, controller, on_change: Callable[[int], None]) -> None:
    """
    Ensure that when the combobox changes, we prompt to save if there are unsaved edits.

    Parameters
    ----------
    combo : QComboBox
    controller : SaveController
    on_change : callable(index) -> None
        Called only if the navigation is allowed.
    """
    if QtWidgets is None:
        # headless fallback: just connect
        combo.currentIndexChanged.connect(on_change)  # type: ignore[attr-defined]
        return

    # Track previous index to revert on cancel
    combo._prev_index = combo.currentIndex()

    def _before_change(index: int):
        # Ignore spurious -1 or same index
        if index == getattr(combo, "_prev_index", index):
            return
        decision = controller.maybe_prompt_to_save_on_navigation(combo)
        if decision is False:
            # cancel -> revert
            combo.blockSignals(True)
            combo.setCurrentIndex(getattr(combo, "_prev_index", 0))
            combo.blockSignals(False)
            return
        # proceed
        combo._prev_index = index
        on_change(index)

    combo.currentIndexChanged.connect(_before_change)  # type: ignore[attr-defined]
