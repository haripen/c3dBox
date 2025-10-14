# c3dBox/Step4_check/selection.py
"""
Rectangle-selection helper for the Step4_check UI.

- Pure Matplotlib object API (no pyplot).
- Uses matplotlib.widgets.RectangleSelector.
- Exposes a small manager that:
    * Tracks a 'mode' in {'select', 'deselect'} (toggle externally).
    * Hit-tests Line2D artists intersecting a drawn rectangle.
    * Returns a list of (cycle_ref, param) for lines hit.

The module also contains a minimal PySide6 + QtAgg demo in __main__ that
generates synthetic left/right "cycles" and shows how to wire the selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import RectangleSelector
from matplotlib.transforms import Bbox

# QtAgg canvas import without pyplot
try:  # Matplotlib 3.5+ recommended with Qt6
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # Fallback for environments that still ship qt5agg
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


@dataclass(frozen=True)
class SelectionItem:
    """Represents a target to modify after rectangle selection."""
    cycle_ref: Any  # Whatever your app uses to identify a cycle (e.g., CycleRef dataclass)
    param: str      # Parameter key for the subplot (e.g., "LHipAngles" or "Force_Fx1")


def _line_intersects_rect(line: Line2D, bbox: Bbox) -> bool:
    """
    Fast, dense-sampled hit test: returns True if any (x, y) vertex lies inside bbox.
    With 101 time-normalized samples per cycle, point-in-rect works well in practice.
    """
    if not line.get_visible():
        return False

    xdata = np.asarray(line.get_xdata(orig=False))
    ydata = np.asarray(line.get_ydata(orig=False))
    if xdata.size == 0 or ydata.size == 0:
        return False

    # Guard against NaNs
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    if not np.any(mask):
        return False
    x = xdata[mask]
    y = ydata[mask]

    xmin, xmax = bbox.xmin, bbox.xmax
    ymin, ymax = bbox.ymin, bbox.ymax

    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return bool(np.any(inside))


class SelectionManager:
    """
    Rectangle selection manager for a single Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to attach the RectangleSelector to.
    line_to_item : dict[Line2D, SelectionItem]
        Mapping from drawn lines to (cycle_ref, param) targets.
    mode : {'select', 'deselect'}
        Initial selection mode. Can be changed later via set_mode().
    on_result : callable | None
        Callback called as on_result(items: list[SelectionItem], mode: str, bbox: Bbox)
        whenever a rectangle is completed.
    minspanx, minspany : float
        Minimum drag spans in data units to consider a rectangle.
    """

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
        self._line_to_item = line_to_item

        # Interactive rectangle selector
        self._rs = RectangleSelector(
            self.ax,
            onselect=self._on_select,
            useblit=True,
            button=[1],          # left mouse button
            minspanx=minspanx,
            minspany=minspany,
            spancoords="data",
            interactive=False,
            drag_from_anywhere=True,
        )

        # Public: last selection outcome
        self.last_bbox: Optional[Bbox] = None
        self.last_items: List[SelectionItem] = []

    # ---------- Public API ----------

    def set_mode(self, mode: str) -> None:
        """Set selection mode: 'select' or 'deselect'."""
        if mode not in ("select", "deselect"):
            raise ValueError("mode must be 'select' or 'deselect'")
        self.mode = mode

    def register_lines(self, line_to_item: Dict[Line2D, SelectionItem]) -> None:
        """Merge additional lines into the mapping (useful after redraws)."""
        self._line_to_item.update(line_to_item)

    def clear_lines(self) -> None:
        """Clear the mapping; does not remove artists from the Axes."""
        self._line_to_item.clear()

    def set_active(self, active: bool) -> None:
        """Enable/disable the rectangle selector."""
        self._rs.set_active(active)

    def is_active(self) -> bool:
        return self._rs.active

    # ---------- Internals ----------

    def _on_select(self, eclick, erelease) -> None:
        # Ignore if selection started/ended outside data coords
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)

        hits: List[SelectionItem] = []
        # Iterate over mapped lines only (fast) and honor visibility
        for line, item in self._line_to_item.items():
            if _line_intersects_rect(line, bbox):
                hits.append(item)

        self.last_bbox = bbox
        self.last_items = hits

        if self.on_result is not None:
            self.on_result(hits, self.mode, bbox)

        # Optional visual feedback: draw a brief outline (not required)
        self.canvas.draw_idle()


# ------------------------- Demo (no pyplot) -------------------------

if __name__ == "__main__":
    """
    Minimal, synthetic demo:
    - PySide6 window with a single Axes
    - Ten "cycles" per side (left=red, right=blue)
    - Press 'S' to set mode='select', 'D' for mode='deselect'
    - Click-drag a rectangle to see which items are hit
    - Selected lines brighten; deselected dim to alpha=0.2 (demo behavior)
    """
    import sys
    from PySide6 import QtWidgets, QtCore

    class DemoWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Rectangle Selection Demo (Step4_check)")
            self.resize(900, 600)

            fig = Figure(constrained_layout=True)
            self.canvas = FigureCanvas(fig)
            self.setCentralWidget(self.canvas)
            self.ax: Axes = fig.add_subplot(1, 1, 1)

            # Generate synthetic cycles
            # Left cycles (red), Right cycles (blue)
            rng = np.random.default_rng(42)
            t = np.linspace(0, 100, 101)  # percent cycle on x-axis
            line_to_item: Dict[Line2D, SelectionItem] = {}

            def make_cycle(side: str, i: int) -> Tuple[Line2D, SelectionItem]:
                base = np.sin(2 * np.pi * (t / 100.0)) * (10 if side == "left_stride" else 12)
                noise = rng.normal(0, 0.8, size=t.shape)
                y = base + noise + (i * 0.5 if side == "left_stride" else -i * 0.5)
                color = "red" if side == "left_stride" else "blue"
                line = Line2D(t, y, color=color, alpha=1.0, linewidth=1.25)
                self.ax.add_line(line)
                # Attach mapping info; in the real app, cycle_ref would be your CycleRef object
                cycle_ref = {"side": side, "cycle": f"cycle{i+1}"}
                param = "demo_param"
                return line, SelectionItem(cycle_ref=cycle_ref, param=param)

            for i in range(10):
                ln, it = make_cycle("left_stride", i)
                line_to_item[ln] = it
                ln2, it2 = make_cycle("right_stride", i)
                line_to_item[ln2] = it2

            self.ax.set_xlabel("% cycle")
            self.ax.set_ylabel("amplitude (a.u.)")
            self.ax.set_title("Drag to select • Press S=select, D=deselect • Esc=quit")
            self.ax.grid(True, alpha=0.3)
            self.ax.relim()
            self.ax.autoscale_view()

            # Manager with callback that updates line alpha for quick visual feedback
            def on_result(items: List[SelectionItem], mode: str, bbox: Bbox) -> None:
                # Reverse-map items to lines (demo-only search)
                # In your app, you likely keep both directions for O(1) updates.
                hits_lines: List[Line2D] = []
                for line, item in line_to_item.items():
                    if item in items:
                        hits_lines.append(line)

                # Apply demo styling: select → alpha=1.0; deselect → alpha=0.2
                new_alpha = 1.0 if mode == "select" else 0.2
                for ln in hits_lines:
                    ln.set_alpha(new_alpha)

                # Print what would be sent upstream for state change
                print(f"[{mode.upper()}] {len(items)} lines in bbox"
                      f" x[{bbox.xmin:.2f},{bbox.xmax:.2f}] y[{bbox.ymin:.2f},{bbox.ymax:.2f}]")
                for it in items[:5]:
                    print("   ->", it.cycle_ref, "|", it.param)
                if len(items) > 5:
                    print(f"   ... and {len(items) - 5} more")

                self.canvas.draw_idle()

            self.manager = SelectionManager(
                ax=self.ax,
                line_to_item=line_to_item,
                mode="select",
                on_result=on_result,
            )
            self.manager.set_active(True)

            # Key handling (figure-level) to toggle modes
            self.cid_key = self.canvas.mpl_connect("key_press_event", self._on_key_press)

        def _on_key_press(self, event) -> None:
            if event.key is None:
                return
            k = event.key.lower()
            if k == "s":
                self.manager.set_mode("select")
                self.statusBar().showMessage("Mode: SELECT", 1500)
            elif k == "d":
                self.manager.set_mode("deselect")
                self.statusBar().showMessage("Mode: DESELECT", 1500)
            elif k == "escape":
                QtWidgets.QApplication.instance().quit()

    app = QtWidgets.QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec())
