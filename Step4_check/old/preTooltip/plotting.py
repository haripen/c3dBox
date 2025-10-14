# c3dBox/Step4_check/plotting.py
"""
PySide6/Matplotlib plotting helpers for Cycle Selection UI (no pyplot).

Key features implemented:
- CycleAxes wrapper around a Matplotlib Axes (not pyplot) that draws many lines efficiently.
- Colors: left=red, right=blue. De-selected lines alpha=0.2.
- Dashed style if ANY QC flag fails (e.g., reconstruction_ok, IK_*, SO_*).
- Supports 1-D arrays and (N,3) arrays (draws one line per coordinate).
- Efficient redraw via Line2D.set_data(...) and draw_idle().
- Hover tooltips on motion_notify_event using Qt QToolTip.
- autoscale_y_from_selected(...) utility to set y-limits from currently visible,
  manually_selected==1 lines (aggregating across all coords) and applying a margin.

This module intentionally does not import matplotlib.pyplot.

Assumptions:
- The outer UI manages visibility filters (kinetic/kinematic, left/right, etc.)
  and sets per-line metadata (e.g., 'manually_selected', 'kinetic', QC flags).
- Each drawn line receives a small metadata dict stored on the Line2D object as
  attribute `_meta` (Python allows dynamic attributes on artists).

Author: Cycle Checker UI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# Qt tooltip (ensure your app uses a QtAgg canvas, e.g., FigureCanvasQTAgg)
try:
    from PySide6.QtWidgets import QToolTip
    from PySide6.QtGui import QCursor
except Exception:  # pragma: no cover - allows non-Qt environments to import this module
    QToolTip = None
    QCursor = None

Color = Tuple[float, float, float]


# --- Styling constants --------------------------------------------------------

LEFT_COLOR: str = "#d62728"   # mpl default red
RIGHT_COLOR: str = "#1f77b4"  # mpl default blue

ALPHA_SELECTED: float = 1.0
ALPHA_DESELECTED: float = 0.2

# dash pattern roughly like '--'
DASH_FAILED: Tuple[int, Tuple[int, int]] = (0, (5, 3))


# --- Helpers ------------------------------------------------------------------

def _is_finite_array(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.size > 0 and np.isfinite(a).any()


def _ensure_xy(x: Optional[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure x and y are 1D and aligned. If x is None, build 0..100 linspace."""
    if y.ndim == 1:
        yy = y
    elif y.ndim == 2 and y.shape[1] == 1:
        yy = y[:, 0]
    else:
        # For (N,3) we handle in caller by iterating columns; here keep as 1D.
        yy = y  # caller must pass 1D here
    n = yy.shape[0]
    xx = np.linspace(0.0, 100.0, n) if x is None else np.asarray(x).reshape(-1)
    if xx.shape[0] != n:
        raise ValueError(f"x and y must have same length (got {xx.shape[0]} vs {n})")
    return xx, yy


def _qc_failed(meta: Dict[str, Any]) -> bool:
    """Return True if ANY available QC flag is False."""
    flags = (
        "reconstruction_ok",
        "IK_RMS_ok",
        "IK_MAX_ok",
        "SO_F_RMS_ok",
        "SO_F_MAX_ok",
        "SO_M_RMS_ok",
        "SO_M_MAX_ok",
    )
    for k in flags:
        if k in meta and meta[k] is not None and meta[k] is False:
            return True
    return False


def _apply_style(line: Line2D) -> None:
    """Style a line based on its _meta fields."""
    meta = getattr(line, "_meta", {}) or {}
    side = (meta.get("side") or "").lower()
    color = LEFT_COLOR if side.startswith("left") else RIGHT_COLOR
    line.set_color(color)

    # Selection alpha
    sel = meta.get("manually_selected", 1)
    line.set_alpha(ALPHA_SELECTED if int(sel) == 1 else ALPHA_DESELECTED)

    # QC dashed if any QC flag fails
    if _qc_failed(meta):
        line.set_linestyle("--")
        line.set_dashes(DASH_FAILED[1])
    else:
        line.set_linestyle("-")
        # reset to solid (clear explicit dash pattern)
        line.set_dashes(())  # empty tuple = solid


def _format_tooltip(meta: Dict[str, Any]) -> str:
    """Build tooltip text from metadata."""
    # Expected keys from upstream: filename, trial_type, side, cycle_no
    fn = meta.get("filename", "file: ?")
    trial = meta.get("trial_type", "?")
    side = meta.get("side", "?")
    cyc = meta.get("cycle_no", "?")
    param = meta.get("param", "")
    coord = meta.get("coord", "")
    bits = [f"{fn}", f"{trial}", f"{side}", f"cycle {cyc}"]
    if param:
        bits.append(str(param))
    if coord:
        bits.append(str(coord))
    return " • ".join(bits)


# --- Public API ---------------------------------------------------------------

@dataclass
class CycleLine:
    """Lightweight container that keeps a reference to a Line2D and its meta."""
    line: Line2D
    meta: Dict[str, Any]

    def update_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.line.set_data(x, y)

    def refresh_style(self) -> None:
        _apply_style(self.line)

    def set_visible(self, visible: bool) -> None:
        self.line.set_visible(visible)


class CycleAxes:
    """
    Manage many cycle lines on a single Matplotlib Axes (no pyplot).

    Typical usage:
        cx = CycleAxes(ax)
        cx.add_cycle_series(y=array_or_(N,3), x=None, side='left_stride',
                            param='LHipAngles', cycle_no=1,
                            meta_extra={'filename': 'Rd0001...', 'trial_type': 'WalkA01'})
        cx.draw_idle()

    Notes:
    - For (N,3) arrays, three lines are added (coords x,y,z), all sharing the same side/param/cycle.
    - Hover tooltips are enabled when PySide6 is available and the canvas is QtAgg.
    """
    def __init__(self, ax: Axes, hover_pick_tol: int = 6) -> None:
        self.ax: Axes = ax
        self._lines: List[CycleLine] = []
        self._hover_pick_tol = int(hover_pick_tol)
        self._last_hovered: Optional[Line2D] = None

        canvas = getattr(self.ax, "figure", None) and self.ax.figure.canvas
        if canvas is not None:
            # connect motion for hover
            self._cid_motion = canvas.mpl_connect("motion_notify_event", self._on_motion)
        else:
            self._cid_motion = None

    # ------------- adding & updating -----------------------------------------

    def add_cycle_series(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        *,
        side: str,
        param: str,
        cycle_no: Union[int, str],
        meta_extra: Optional[Dict[str, Any]] = None,
        manually_selected: int = 1,
        kinetic: Optional[int] = None,
        qc_flags: Optional[Dict[str, Any]] = None,
        linewidth: float = 1.2,
    ) -> List[CycleLine]:
        """
        Add one cycle's data to this axes.

        If y is (N,), draws one line.
        If y is (N,3), draws three lines (coords x,y,z) in the same color.

        Returns references to the created CycleLine objects.
        """
        y = np.asarray(y)
        meta_base: Dict[str, Any] = dict(
            side=side,
            param=param,
            cycle_no=cycle_no,
            manually_selected=int(manually_selected),
        )
        if meta_extra:
            meta_base.update(meta_extra)
        if qc_flags:
            meta_base.update(qc_flags)
        if kinetic is not None:
            meta_base["kinetic"] = int(kinetic)

        created: List[CycleLine] = []

        if y.ndim == 1:
            xx, yy = _ensure_xy(x, y)
            ln = self._plot_single(xx, yy, linewidth=linewidth)
            ln._meta = dict(meta_base)  # attach metadata
            ln._meta["coord"] = ""
            ln.set_pickradius(self._hover_pick_tol)
            _apply_style(ln)
            created.append(CycleLine(ln, ln._meta))
        elif y.ndim == 2 and y.shape[1] == 3:
            # three coordinates: x, y, z
            coord_labels = ("x", "y", "z")
            for k in range(3):
                xx, yy = _ensure_xy(x, y[:, k])
                ln = self._plot_single(xx, yy, linewidth=linewidth)
                ln._meta = dict(meta_base)
                ln._meta["coord"] = coord_labels[k]
                ln.set_pickradius(self._hover_pick_tol)
                _apply_style(ln)
                created.append(CycleLine(ln, ln._meta))
        else:
            raise ValueError("y must be 1-D or (N,3) array")

        self._lines.extend(created)
        return created

    def _plot_single(self, x: np.ndarray, y: np.ndarray, *, linewidth: float) -> Line2D:
        ln = Line2D(x, y, linewidth=linewidth)
        ln.set_animated(False)
        ln.set_visible(True)
        self.ax.add_line(ln)
        return ln

    def update_cycle_series(
        self,
        *,
        side: Optional[str] = None,
        param: Optional[str] = None,
        cycle_no: Optional[Union[int, str]] = None,
        new_x: Optional[np.ndarray] = None,
        new_y: Optional[np.ndarray] = None,
        refresh_style: bool = True,
    ) -> int:
        """
        Update data for lines that match a (side, param, cycle_no) triple.
        new_y may be 1-D or (N,3) matching the number of lines previously drawn.

        Returns number of updated Line2D artists.
        """
        matchers = []
        if side is not None:
            matchers.append(("side", side))
        if param is not None:
            matchers.append(("param", param))
        if cycle_no is not None:
            matchers.append(("cycle_no", cycle_no))

        # collect target lines in consistent coord order
        targets: List[Tuple[int, CycleLine]] = []
        for idx, cl in enumerate(self._lines):
            ok = all(cl.meta.get(k) == v for k, v in matchers)
            if ok:
                targets.append((idx, cl))

        if not targets:
            return 0

        if new_y is None:
            # only style refresh
            if refresh_style:
                for _, cl in targets:
                    _apply_style(cl.line)
            self.draw_idle()
            return 0

        new_y = np.asarray(new_y)
        if new_y.ndim == 1:
            # expect single line
            assert len(targets) == 1, "Updating 1-D data but multiple lines matched"
            xx, yy = _ensure_xy(new_x, new_y)
            targets[0][1].update_data(xx, yy)
            if refresh_style:
                targets[0][1].refresh_style()
            self.draw_idle()
            return 1

        elif new_y.ndim == 2 and new_y.shape[1] == 3:
            assert len(targets) == 3, "Updating (N,3) data but !=3 lines matched"
            for k, (_, cl) in enumerate(sorted(targets, key=lambda t: t[1].meta.get("coord", ""))):
                xx, yy = _ensure_xy(new_x, new_y[:, k])
                cl.update_data(xx, yy)
                if refresh_style:
                    cl.refresh_style()
            self.draw_idle()
            return 3

        else:
            raise ValueError("new_y must be 1-D or (N,3) array")

    # ------------- bulk operations & queries ----------------------------------

    def iter_lines(self) -> Iterable[CycleLine]:
        return iter(self._lines)

    def set_manual_selection(self, lines: Iterable[Line2D], selected: int) -> None:
        """Set 'manually_selected' flag on given Line2D and restyle."""
        for ln in lines:
            meta = getattr(ln, "_meta", None)
            if meta is None:
                continue
            meta["manually_selected"] = int(selected)
            _apply_style(ln)
        self.draw_idle()

    def draw_idle(self) -> None:
        canvas = self.ax.figure.canvas if self.ax and self.ax.figure else None
        if canvas is not None:
            canvas.draw_idle()

    # ------------- hover tooltips ---------------------------------------------

    def _on_motion(self, event) -> None:  # Matplotlib event
        if event.inaxes is not self.ax:
            self._hide_tooltip()
            return
        if QToolTip is None or QCursor is None:
            return  # Qt not available -> silently do nothing

        # Try fast containment tests; stop at the first hit for stability
        hit_line: Optional[Line2D] = None
        for cl in self._lines:
            ln = cl.line
            if not ln.get_visible():
                continue
            contains, _ = ln.contains(event)
            if contains:
                hit_line = ln
                break

        if hit_line is None:
            self._hide_tooltip()
            return

        if self._last_hovered is hit_line:
            # already showing this tooltip
            return

        self._last_hovered = hit_line
        meta = getattr(hit_line, "_meta", {}) or {}
        tip = _format_tooltip(meta)
        # Global cursor position (Qt)
        pos = QCursor.pos()
        QToolTip.showText(pos, tip)

    def _hide_tooltip(self) -> None:
        if self._last_hovered is not None:
            self._last_hovered = None
            if QToolTip is not None:
                QToolTip.hideText()


# --- Y autoscale utility ------------------------------------------------------

def autoscale_y_from_selected(
    lines: Sequence[Line2D],
    margin_pct: float = 0.05,
) -> bool:
    """
    Compute y-limits across currently visible & manually_selected==1 lines
    (aggregating all coords) and apply them to that subplot's Axes with a margin.

    Returns True if limits were updated, False if no finite data was found.

    NOTE: All given lines must belong to the same Axes.
    """
    if not lines:
        return False

    # Filter: visible + selected
    vals: List[np.ndarray] = []
    ax: Optional[Axes] = None

    for ln in lines:
        if not isinstance(ln, Line2D):
            continue
        if not ln.get_visible():
            continue
        meta = getattr(ln, "_meta", {}) or {}
        if int(meta.get("manually_selected", 1)) != 1:
            continue
        y = np.asarray(ln.get_ydata())
        if y.size == 0 or not np.isfinite(y).any():
            continue
        vals.append(y[np.isfinite(y)])
        if ax is None:
            ax = ln.axes

    if not vals or ax is None:
        return False

    y_all = np.concatenate(vals) if len(vals) > 1 else vals[0]
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    if not np.isfinite([y_min, y_max]).all() or y_min == y_max:
        # If identical or non-finite, avoid changing limits
        return False

    # Apply margin
    span = y_max - y_min
    m = float(margin_pct) if margin_pct is not None else 0.0
    m = max(0.0, m)
    pad = span * m
    ax.set_ylim(y_min - pad, y_max + pad)

    canvas = ax.figure.canvas if ax and ax.figure else None
    if canvas is not None:
        canvas.draw_idle()
    return True
