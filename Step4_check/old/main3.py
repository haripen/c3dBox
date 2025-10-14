
# c3dBox/Step4_check/main.py  (overhaul)
# - Immediate re-scan + initialize when root is set
# - Trial-type shows base names (e.g., "WalkA") and aggregates WalkA01..09
# - Loads & plots all sessions for the selected participant+trial-type
# - Robust data discovery; per-subplot dropdown lists all available keys
#   (handles 3D arrays (N,3) and 1D keys with _fx/_fy/_fz, etc.)
# - Counts kinetic/kinematic cycles and updates status
# - Rectangle selection is active immediately (no need to toggle checkboxes)
# - Rich debug console (Dock) showing key-access attempts and data discovery
#
# Run:  python -m c3dBox.Step4_check.main

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QMessageBox

from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Local modules
from .settings import load as settings_load, get as settings_get, update_and_save as settings_update, reset_to_defaults as settings_reset  # noqa
from .file_index import FileMeta, scan_root, build_indices
from .io_mat import load_dict
from .cycles import collect_cycles, CycleRef
from .selection import SelectionManager, SelectionItem
from .history import History, SelectionEdit
from .status import StatusWidget, compute_counts
from .plotting import CycleAxes
from . import qc as qc_mod
from . import emg as emg_mod


# ----------------------------- small helpers ---------------------------------

def _side_from_key(side_key: str) -> str:
    s = (side_key or "").lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return side_key or ""


def _is_emg_key(k: str) -> bool:
    return emg_mod.is_emg_key(k)


def _interpolate_101(arr: np.ndarray) -> Optional[np.ndarray]:
    """Simple 1D interpolation to 101 points. Returns None if not a 1D array >=2."""
    try:
        a = np.asarray(arr).reshape(-1)
    except Exception:
        return None
    if a.size < 2:
        return None
    n = a.shape[0]
    x = np.linspace(0.0, 1.0, n)
    xx = np.linspace(0.0, 1.0, 101)
    return np.interp(xx, x, a)


def _timenorm_array(arr: np.ndarray, is_3d: bool) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.size == 0:
        return None
    if a.ndim == 1:
        return _interpolate_101(a)
    if is_3d and a.ndim == 2 and a.shape[1] == 3:
        out = np.zeros((101, 3), dtype=float)
        for k in range(3):
            col = _interpolate_101(a[:, k])
            if col is None:
                return None
            out[:, k] = col
        return out
    return None


# ----------------------------- Debug console ---------------------------------

class DebugConsole(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Debug", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)  # avoid unbounded growth
        self.setWidget(self.text)
        self._buffer: List[str] = []

    def log(self, msg: str) -> None:
        self._buffer.append(msg)
        self.text.appendPlainText(msg)

    def clear(self) -> None:
        self.text.clear()
        self._buffer.clear()


# --------------------------- Data discovery ----------------------------------

@dataclass
class LoadedFile:
    meta: FileMeta
    data: Dict[str, Any]
    cycles: List[CycleRef]


def _safe_arr(x: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None
        return a
    except Exception:
        return None


def _list_point_3d_keys(cycle_dict: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    pt = cycle_dict.get("point", {})
    if isinstance(pt, dict):
        for k, v in pt.items():
            a = _safe_arr(v)
            if a is not None and a.ndim == 2 and a.shape[1] == 3:
                out.append(k)
    return sorted(set(out))


def _list_group_1d_keys(group_dict: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if isinstance(group_dict, dict):
        for k, v in group_dict.items():
            a = _safe_arr(v)
            if a is not None:
                a = np.asarray(a)
                if a.ndim == 1 or (a.ndim == 2 and min(a.shape) == 1):
                    out.append(k)
    return sorted(set(out))


def discover_all_param_keys(files: List[LoadedFile], dbg: DebugConsole) -> List[str]:
    """Build a flat list of available parameter keys across all cycles of all files.

    - point (Nx3): returns key name (e.g., 'LHipAngles')
    - IK_markerErr: 'IK_markerErr.marker_error_RMS', etc.
    - SO_forces: 'SO_forces.FX', ...
    - analog: channel names (e.g., 'Force_Fx1', 'EMG_...')
    - JRL/GRF/JRF/residual: direct 1D keys (e.g., 'hip_r_on_femur_r_in_femur_r_fx')
    """
    keys: set[str] = set()

    for lf in files:
        for cref in lf.cycles:
            # Each CycleRef points into lf.data via its side and cycle name
            side_block = lf.data.get(cref.side, {})
            cdict = side_block.get(cref.cycle, {}) if isinstance(side_block, dict) else {}

            # point Nx3
            for k in _list_point_3d_keys(cdict):
                keys.add(k)

            # IK marker errors
            ikm = cdict.get("IK_markerErr", {})
            if isinstance(ikm, dict):
                for nm in ("total_squared_error", "marker_error_RMS", "marker_error_max"):
                    if nm in ikm:
                        keys.add(f"IK_markerErr.{nm}")

            # SO residuals
            so = cdict.get("SO_forces", {})
            if isinstance(so, dict):
                for nm in ("FX", "FY", "FZ", "MX", "MY", "MZ"):
                    if nm in so:
                        keys.add(f"SO_forces.{nm}")

            # analog channels
            analog = cdict.get("analog", {})
            for k in _list_group_1d_keys(analog):
                keys.add(k)

            # JRL/GRF/JRF/residual groups (flat 1D keys)
            for grp in ("JRL", "grf", "jrf", "residual"):
                g = cdict.get(grp, {})
                for k in _list_group_1d_keys(g):
                    keys.add(k)

    lst = sorted(keys)
    dbg.log(f"[discover] total keys discovered: {len(lst)}")
    return lst


def _find_param_array(cycle_dict: Dict[str, Any], key: str, dbg: Optional[DebugConsole]) -> tuple[Optional[Any], str]:
    """
    Returns (array_or_list, kind) where kind in {'1d','3d','analog_many'}.
    Looks into common groups: point, IK_markerErr, SO_forces, analog, grf, jrf, residual, JRL.
    """
    if dbg:
        dbg.log(f"[access] try key='{key}' on cycle groups: {list(cycle_dict.keys())}")

    if not isinstance(cycle_dict, dict) or not key:
        return None, ""

    # point (angles/moments etc.), (N,3) commonly
    pt = cycle_dict.get("point", {})
    if isinstance(pt, dict) and key in pt:
        arr = pt.get(key)
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 3:
            if dbg: dbg.log(f"  -> found in 'point' as 3D: {a.shape}")
            return a, "3d"
        if dbg: dbg.log(f"  -> found in 'point' as 1D: {a.shape}")
        return a.reshape(-1), "1d"

    # IK marker error components
    if key.startswith("IK_markerErr."):
        _, _, name = key.partition("IK_markerErr.")
        ikm = cycle_dict.get("IK_markerErr", {})
        if isinstance(ikm, dict) and name in ikm:
            a = np.asarray(ikm[name]).reshape(-1)
            if dbg: dbg.log(f"  -> found in 'IK_markerErr': {name}, shape={a.shape}")
            return a, "1d"

    # SO residuals (forces/moments)
    if key.startswith("SO_forces."):
        _, _, name = key.partition("SO_forces.")
        so = cycle_dict.get("SO_forces", {})
        if isinstance(so, dict) and name in so:
            a = np.asarray(so[name]).reshape(-1)
            if dbg: dbg.log(f"  -> found in 'SO_forces': {name}, shape={a.shape}")
            return a, "1d"

    # Generic analog by exact name
    analog = cycle_dict.get("analog", {})
    if isinstance(analog, dict) and key in analog:
        a = np.asarray(analog[key]).reshape(-1)
        if _is_emg_key(key):
            # EMG envelope on-the-fly
            fs_guess = 1000.0  # fallback
            # Try to derive from time vector if present
            t = analog.get("time")
            if isinstance(t, (list, np.ndarray)) and len(t) > 1:
                t = np.asarray(t, dtype=float).reshape(-1)
                dt = np.nanmedian(np.diff(t))
                if np.isfinite(dt) and dt > 0:
                    fs_guess = 1.0 / dt
            a = emg_mod.process_emg(a, float(fs_guess))
            if dbg: dbg.log(f"  -> found in 'analog' EMG: processed to envelope, fs~{fs_guess:.1f} Hz")
        else:
            if dbg: dbg.log(f"  -> found in 'analog': shape={a.shape}")
        return a, "1d"

    # GRF/JRF/residuals/JRL by exact name
    for group in ("grf", "jrf", "residual", "JRL"):
        g = cycle_dict.get(group, {})
        if isinstance(g, dict) and key in g:
            a = np.asarray(g[key]).reshape(-1)
            if dbg: dbg.log(f"  -> found in '{group}': shape={a.shape}")
            return a, "1d"

    if dbg:
        dbg.log("  !! not found in any known group")
    return None, ""


# -------------------------- Plot page/widgets ---------------------------------

class PlotCell(QtWidgets.QWidget):
    selectionMade = QtCore.Signal(list, str)  # (items, mode) -> bubble up
    selectionChanged = QtCore.Signal()   # emitted on combo change

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title = title
        self.current_key = ""

        self.fig = Figure(figsize=(3, 2), layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)  # allow filtering
        self.combo.currentTextChanged.connect(self._on_param_changed)

        self.btn_ylim = QtWidgets.QToolButton()
        self.btn_ylim.setText("↕︎")
        self.btn_ylim.setToolTip("Autoscale y from selected lines")

        top = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(self.title)
        lab.setMinimumWidth(120)
        top.addWidget(lab)
        top.addStretch(1)
        top.addWidget(self.combo)
        top.addWidget(self.btn_ylim)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

        self.lines: List[Any] = []
        self.cycle_axes = CycleAxes(self.ax)
        self.selector: Optional[SelectionManager] = None

        self.btn_ylim.clicked.connect(self.autoscale)

    def set_options(self, keys: Sequence[str], default_key: Optional[str] = None) -> None:
        # preserve current selection if possible
        prev = self.combo.currentText().strip()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(keys))
        if default_key and default_key in keys:
            self.combo.setCurrentText(default_key)
        elif prev in keys:
            self.combo.setCurrentText(prev)
        elif keys:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        self.current_key = self.combo.currentText().strip()

    def _on_param_changed(self, key: str):
        self.current_key = key.strip()
        self.selectionChanged.emit()

    def autoscale(self):
        # Simple autoscale: compute y-range of visible lines and set with small margin
        ymin, ymax = None, None
        for ln in self.ax.lines:
            if not ln.get_visible():
                continue
            y = np.asarray(ln.get_ydata(orig=False))
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            lo, hi = float(np.min(y)), float(np.max(y))
            ymin = lo if ymin is None else min(ymin, lo)
            ymax = hi if ymax is None else max(ymax, hi)
        if ymin is None or ymax is None or not np.isfinite([ymin, ymax]).all():
            return
        if ymax <= ymin:
            pad = 1.0
            self.ax.set_ylim(ymin - pad, ymax + pad)
        else:
            span = ymax - ymin
            pad = 0.05 * span
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.canvas.draw_idle()

    def clear(self):
        self.ax.cla()
        self.ax.set_xlabel("% cycle")
        self.ax.set_title(self.title)
        self.lines.clear()
        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None
        self.canvas.draw_idle()

    def plot_cycles(self,
                    loaded_files: List[LoadedFile],
                    key: str,
                    show_left: bool,
                    show_right: bool,
                    dbg: DebugConsole):
        self.clear()
        self.current_key = key

        line_to_item: Dict[Any, SelectionItem] = {}

        for lf in loaded_files:
            for cref in lf.cycles:
                side = _side_from_key(cref.side)
                if side == "left" and not show_left:
                    continue
                if side == "right" and not show_right:
                    continue

                # Dive into the owning cycle dict
                side_block = lf.data.get(cref.side, {})
                cdict = side_block.get(cref.cycle, {}) if isinstance(side_block, dict) else {}
                arr, kind = _find_param_array(cdict, key, dbg)

                if arr is None:
                    continue

                # Timenorm + draw
                if kind == "3d":
                    arr101 = _timenorm_array(arr, is_3d=True)
                    if arr101 is None:
                        continue
                    for j, coord in enumerate(("x", "y", "z")):
                        x = np.linspace(0.0, 100.0, 101)
                        y = arr101[:, j].reshape(-1)
                        ln, = self.ax.plot(x, y, lw=1.25, alpha=0.9)
                        ln._meta = {
                            "side": side,
                            "param": key,
                            "coord": coord,
                            "filename": os.path.basename(lf.meta.path),
                            "trial_type": lf.meta.original_trial_type,
                            "cycle_no": cref.cycle,
                            "manually_selected": int(cdict.get("manually_selected", 1)),
                        }
                        line_to_item[ln] = SelectionItem(cycle_ref=cref, param=key)
                elif kind == "1d":
                    arr101 = _timenorm_array(np.asarray(arr).reshape(-1), is_3d=False)
                    if arr101 is None:
                        continue
                    x = np.linspace(0.0, 100.0, 101)
                    y = arr101.reshape(-1)
                    ln, = self.ax.plot(x, y, lw=1.25, alpha=0.9)
                    ln._meta = {
                        "side": side,
                        "param": key,
                        "coord": "",
                        "filename": os.path.basename(lf.meta.path),
                        "trial_type": lf.meta.original_trial_type,
                        "cycle_no": cref.cycle,
                        "manually_selected": int(cdict.get("manually_selected", 1)),
                    }
                    line_to_item[ln] = SelectionItem(cycle_ref=cref, param=key)

        self.ax.set_xlabel("% cycle")
        self.ax.set_title(self.title)
        self.canvas.draw_idle()

        # Activate rectangle selection
        self.selector = SelectionManager(
            self.ax,
            line_to_item=line_to_item,
            mode="select",
            on_result=self._on_rect_selection,
        )
        self.selector.set_active(True)

    # Rectangle selection result
    def _on_rect_selection(self, items: List[SelectionItem], mode: str, bbox):
        # Bubble up to page/window
        self.selectionMade.emit(items, mode)


class PlotPage(QtWidgets.QWidget):

    @QtCore.Slot(list, str)
    def _on_cell_selection(self, items, mode):
        # Convert to (CycleRef, new_val) list and emit
        changes = []
        new_val = 1 if mode == "select" else 0
        for it in items:
            try:
                cref = it.cycle_ref
                changes.append((cref, new_val))
            except Exception:
                continue
        if changes:
            self.selectionApplied.emit(changes)

    selectionApplied = QtCore.Signal(list)  # list of (CycleRef, new_val)
    def __init__(self, parent, title: str, rows: int, cols: int):
        super().__init__(parent)
        self.title = title
        self.rows = rows
        self.cols = cols
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)

        self.cells: List[PlotCell] = []
        for r in range(rows):
            for c in range(cols):
                cell = PlotCell(self, title=f"{title} [{r+1},{c+1}]")
                self.grid.addWidget(cell, r, c)
                cell.selectionMade.connect(self._on_cell_selection)
                self.cells.append(cell)

    def set_options_all(self, keys: Sequence[str]) -> None:
        # Assign options; keep previous selection if possible
        for i, cell in enumerate(self.cells):
            default = keys[i] if i < len(keys) else (keys[0] if keys else None)
            cell.set_options(keys, default_key=default)

    def redraw_all(self,
                   loaded_files: List[LoadedFile],
                   show_left: bool,
                   show_right: bool,
                   dbg: DebugConsole) -> None:
        for cell in self.cells:
            key = cell.current_key or (cell.combo.currentText().strip() if cell.combo.count() else "")
            if not key and cell.combo.count() > 0:
                key = cell.combo.itemText(0)
            if key:
                cell.plot_cycles(loaded_files, key, show_left, show_right, dbg)


# ------------------------------- MainWindow -----------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cycle Selection Tool — Step 4 (check)")
        self.resize(1400, 900)

        # Debug console
        self.debug = DebugConsole(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.debug)

        # Central splitter: controls on top, tabs below
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        self.root_edit = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("Set Root…")
        self.btn_browse.clicked.connect(self.on_browse_root)

        self.cmb_pid = QtWidgets.QComboBox()
        self.cmb_pid.currentTextChanged.connect(self.on_pid_changed)

        self.cmb_trial = QtWidgets.QComboBox()
        self.cmb_trial.currentTextChanged.connect(self.on_trial_changed)

        self.chk_left = QtWidgets.QCheckBox("Left")
        self.chk_left.setChecked(True)
        self.chk_right = QtWidgets.QCheckBox("Right")
        self.chk_right.setChecked(True)
        for chk in (self.chk_left, self.chk_right):
            chk.toggled.connect(self.on_filters_changed)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Root:"))
        ctrl.addWidget(self.root_edit, 1)
        ctrl.addWidget(self.btn_browse)
        ctrl.addSpacing(16)
        ctrl.addWidget(QtWidgets.QLabel("Participant:"))
        ctrl.addWidget(self.cmb_pid)
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Trial type:"))
        ctrl.addWidget(self.cmb_trial)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.chk_left)
        ctrl.addWidget(self.chk_right)
        ctrl.addStretch(1)

        # Tabs with two pages (3x3 each by default)
        self.tabs = QtWidgets.QTabWidget()
        self.page1 = PlotPage(self, "Page 1", rows=3, cols=3)
        self.page2 = PlotPage(self, "Page 2", rows=3, cols=3)
        self.tabs.addTab(self.page1, "Page 1")
        self.tabs.addTab(self.page2, "Page 2")

        self.page1.selectionApplied.connect(self.on_selection_applied)
        self.page2.selectionApplied.connect(self.on_selection_applied)

        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(ctrl)
        lay.addWidget(self.tabs, 1)

        # Status bar with counts
        self.status_widget = StatusWidget(self)
        self.setStatusBar(self.status_widget)

        # Data state
        self.root: str = ""
        self.participants: List[str] = []
        self.trial_types_by_pid: Dict[str, List[str]] = {}
        self.files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}

        self.loaded: List[LoadedFile] = []  # current selection
        self.all_keys: List[str] = []

        # Initialize settings (and use ui.start_fullscreen if set)
        try:
            s = settings_load()
            if bool(s.get("ui", {}).get("start_fullscreen", False)):
                self.showMaximized()
        except Exception as e:
            self.debug.log(f"[settings] load failed: {e!r}")

        # If an environment variable provides a root, auto-init
        env_root = os.getenv("CYCLE_TOOL_ROOT")
        if env_root and os.path.isdir(env_root):
            self.root_edit.setText(env_root)
            self.set_root(env_root)

    # ------------ Root / Indexing ------------

    def on_browse_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project root")
        if not path:
            return
        self.root_edit.setText(path)
        self.set_root(path)

    def set_root(self, root: str):
        """Scan root and initialize participant + trial types immediately."""
        self.root = root
        self.debug.clear()
        self.debug.log(f"[root] set to: {root}")

        files = scan_root(root)
        self.debug.log(f"[scan] found {len(files)} *_splitCycles*.mat files")
        self.participants, self.trial_types_by_pid, self.files_by_pid_type = build_indices(files)

        # Populate participant combo
        self.cmb_pid.blockSignals(True)
        self.cmb_pid.clear()
        self.cmb_pid.addItems(self.participants)
        self.cmb_pid.blockSignals(False)

        # Auto-select first participant if available
        if self.participants:
            self.cmb_pid.setCurrentIndex(0)
            self.on_pid_changed(self.participants[0])
        else:
            # Clear everything
            self.cmb_trial.clear()
            self.loaded = []
            self.all_keys = []
            self.page1.set_options_all([])
            self.page2.set_options_all([])
            self.redraw_all()

    def on_pid_changed(self, pid: str):
        if not pid:
            return
        types = self.trial_types_by_pid.get(pid, [])
        self.debug.log(f"[pid] {pid} -> trial types: {types}")

        self.cmb_trial.blockSignals(True)
        self.cmb_trial.clear()
        self.cmb_trial.addItems(types)
        self.cmb_trial.blockSignals(False)

        if types:
            self.cmb_trial.setCurrentIndex(0)
            self.on_trial_changed(types[0])
        else:
            self.loaded = []
            self.all_keys = []
            self.page1.set_options_all([])
            self.page2.set_options_all([])
            self.redraw_all()

    def on_trial_changed(self, trial_type: str):
        """Load all sessions for the selected pid+trial_type and refresh UI."""
        pid = self.cmb_pid.currentText().strip()
        if not pid or not trial_type:
            return
        metas = self.files_by_pid_type.get(pid, {}).get(trial_type, [])
        self.debug.log(f"[trial] load {pid} / {trial_type}: {len(metas)} file(s)")

        self.loaded = []
        for fm in metas:
            try:
                d = load_dict(fm.path)
                cycles = collect_cycles(d, {"groups": fm.groups})
                self.loaded.append(LoadedFile(meta=fm, data=d, cycles=cycles))
                self.debug.log(f"  - {os.path.basename(fm.path)}: cycles={len(cycles)}")
            except Exception as e:
                self.debug.log(f"  !! failed to load {fm.path}: {e!r}")

        # Recompute counts and status
        all_cycle_objs: List[Any] = []
        for lf in self.loaded:
            # Each CycleRef points to cycle dict with flags already ensured
            all_cycle_objs.extend(lf.cycles)
        totals = compute_counts([c.flags for c in all_cycle_objs])
        self.status_widget.update_from_counts(totals)

        # Discover keys -> update dropdowns
        self.all_keys = discover_all_param_keys(self.loaded, self.debug)
        self.page1.set_options_all(self.all_keys)
        self.page2.set_options_all(self.all_keys)

        # Trigger redraw
        self.redraw_all()

    def on_filters_changed(self, _state: bool):
        self.redraw_all()

    def redraw_all(self):
        show_left = self.chk_left.isChecked()
        show_right = self.chk_right.isChecked()
        self.page1.redraw_all(self.loaded, show_left, show_right, self.debug)
        self.page2.redraw_all(self.loaded, show_left, show_right, self.debug)



    @QtCore.Slot(list)
    def on_selection_applied(self, changes: List[tuple]):
        # changes: List[(CycleRef, new_val)]
        applied = 0
        for cref, new_val in changes:
            # navigate to owner cycle dict
            for lf in self.loaded:
                if cref in lf.cycles:
                    side_block = lf.data.get(cref.side, {})
                    cdict = side_block.get(cref.cycle, {}) if isinstance(side_block, dict) else {}
                    try:
                        old = int(cdict.get("manually_selected", 1))
                    except Exception:
                        old = 1
                    if old != int(new_val):
                        cdict["manually_selected"] = int(new_val)
                        applied += 1
        if applied:
            self.debug.log(f"[select] applied changes to {applied} cycle(s)")
            # Update counts and redraw
            all_cycle_objs: List[Any] = []
            for lf in self.loaded:
                all_cycle_objs.extend(lf.cycles)
            totals = compute_counts([c.flags for c in all_cycle_objs])
            self.status_widget.update_from_counts(totals)
            self.redraw_all()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
