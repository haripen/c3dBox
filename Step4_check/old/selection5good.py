
# c3dBox/Step4_check/selection.py
"""
Rectangle-selection helper (Matplotlib RectangleSelector).
Public API:
- set_mode("select"/"deselect")
- disconnect()
- set_active(True/False)
- on_result callback -> (items, mode, bbox)

Each SelectionItem.cycle_ref can be any identifier you choose (e.g., (file_idx, stride_side, cycle_name)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.widgets import RectangleSelector


@dataclass(frozen=True)
class SelectionItem:
    cycle_ref: Any   # e.g., (file_idx, stride_side, cycle_name)
    param: str       # parameter key for this line


def _line_intersects_rect(line: Line2D, bbox: Bbox) -> bool:
    if not line.get_visible():
        return False
    x = np.asarray(line.get_xdata(orig=False))
    y = np.asarray(line.get_ydata(orig=False))
    if x.size == 0 or y.size == 0:
        return False
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return False
    x = x[mask]; y = y[mask]
    return bool(np.any((x >= bbox.xmin) & (x <= bbox.xmax) & (y >= bbox.ymin) & (y <= bbox.ymax)))


class SelectionManager:
    def __init__(
        self,
        ax: Axes,
        line_to_item: Dict[Line2D, SelectionItem],
        mode: str = "select",
        on_result: Optional[Callable[[List[SelectionItem], str, Bbox], None]] = None,
        minspanx: float = 0.0,
        minspany: float = 0.0,
    ) -> None:
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.mode = mode
        self.on_result = on_result
        self._line_to_item = dict(line_to_item)

        self._rs = RectangleSelector(
            self.ax,
            onselect=self._on_select,
            useblit=True,
            button=[1],  # left click & drag
            minspanx=minspanx,
            minspany=minspany,
            spancoords="data",
            interactive=False,
            drag_from_anywhere=True,
        )

    def set_mode(self, mode: str) -> None:
        if mode not in ("select", "deselect"):
            raise ValueError("mode must be 'select' or 'deselect'")
        self.mode = mode

    def set_active(self, active: bool) -> None:
        try:
            self._rs.set_active(bool(active))
        except Exception:
            self._rs.active = bool(active)  # type: ignore[attr-defined]

    def disconnect(self) -> None:
        try:
            self._rs.set_active(False)
        except Exception:
            pass
        try:
            self._rs.disconnect_events()
        except Exception:
            pass
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return
        from matplotlib.transforms import Bbox
        bbox = Bbox.from_extents(min(eclick.xdata, erelease.xdata),
                                 min(eclick.ydata, erelease.ydata),
                                 max(eclick.xdata, erelease.xdata),
                                 max(erelease.ydata, eclick.ydata))
        hits: List[SelectionItem] = []
        for line, item in self._line_to_item.items():
            if _line_intersects_rect(line, bbox):
                hits.append(item)
        if self.on_result is not None:
            try:
                self.on_result(hits, self.mode, bbox)
            except Exception:
                pass
