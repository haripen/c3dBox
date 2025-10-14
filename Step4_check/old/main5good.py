
# c3dBox/Step4_check/main.py
# Implements stride-aware plotting, filters & page layouts per spec.
# - Print debug to console (no docks)
# - Fixed 3x6 layouts (no JSON)
# - Checkboxes: left_stride (✓), right_stride (✓), kinetic (✓), kinematic (✗)
# - Dropdown width ~83 chars on both pages
# - 's' toggles SELECT, 'd' toggles DESELECT; title + toolbar show mode
# - Click-drag opens rectangle; selection toggles 'manually_selected' and refreshes counts
# - Plot ipsilateral data only: for left_stride plot left-side params only; for right_stride right-side only
# - Colors: left_stride red (#d62728), right_stride blue (#1f77b4); de-selected alpha=0.2
# - QC fail (any *_ok flag false) → dashed linestyle
# - Y autoscale per subplot (↕︎) and global (Ctrl+U)

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot

from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .file_index import scan_root, build_indices, FileMeta
from .io_mat import load_dict
from .selection import SelectionManager, SelectionItem
from .status import compute_counts, StatusWidget
from . import emg as emg_mod

LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#1f77b4"

# ---------------- helpers ----------------

def _safe_arr(x: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None
        return a
    except Exception:
        return None

def _interpolate_101(arr: np.ndarray) -> Optional[np.ndarray]:
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

def _timenorm(arr: np.ndarray, is_3d: bool) -> Optional[np.ndarray]:
    a = _safe_arr(arr)
    if a is None:
        return None
    if a.ndim == 1:
        return _interpolate_101(a)
    if is_3d and a.ndim == 2 and a.shape[1] == 3:
        out = np.zeros((101, 3), float)
        for k in range(3):
            col = _interpolate_101(a[:, k])
            if col is None:
                return None
            out[:, k] = col
        return out
    return None

def _analog_fs(analog: Dict[str, Any]) -> float:
    try:
        t = np.asarray(analog.get("time")).reshape(-1)
        if t.size >= 2:
            dt = np.median(np.diff(t))
            if dt > 0:
                return float(1.0 / dt)
    except Exception:
        pass
    return 1000.0

def _is_qc_bad(cdict: Dict[str, Any]) -> bool:
    for k in ("reconstruction_ok","IK_RMS_ok","IK_MAX_ok","SO_F_RMS_ok","SO_F_MAX_ok","SO_M_RMS_ok","SO_M_MAX_ok"):
        if k in cdict:
            try:
                if int(cdict[k]) == 0:
                    return True
            except Exception:
                pass
    return False

def _is_left_key(key: str) -> bool:
    s = key.lower()
    return s.startswith("l") or "_l_" in s or s.endswith("_l")

def _is_right_key(key: str) -> bool:
    s = key.lower()
    return s.startswith("r") or "_r_" in s or s.endswith("_r")

def _counterpart_key(key: str, target: str) -> str:
    """Return a likely counterpart key for the other side ('left' or 'right')."""
    s = key
    if target == "left":
        s = re.sub(r"(^R)([A-Z].*)", r"L\2", s)
        s = s.replace("_r_", "_l_").replace("_R_", "_L_").replace("_right_", "_left_")
        s = re.sub(r"(_r$)", "_l", s)
    else:
        s = re.sub(r"(^L)([A-Z].*)", r"R\2", s)
        s = s.replace("_l_", "_r_").replace("_L_", "_R_").replace("_left_", "_right_")
        s = re.sub(r"(_l$)", "_r", s)
    return s

def _passes_stride_side(key: str, stride_side: str) -> bool:
    # Only accept ipsilateral parameters
    if stride_side == "left_stride":
        # Reject clear right-only keys
        if _is_right_key(key):
            return False
        return True
    if stride_side == "right_stride":
        if _is_left_key(key):
            return False
        return True
    return True

# ---------------- key discovery & access ----------------

def discover_keys(files: List["LoadedFile"]) -> List[str]:
    keys: set[str] = set()
    for lf in files:
        d = lf.data
        for stride_side in ("left_stride", "right_stride"):
            block = d.get(stride_side, {})
            if not isinstance(block, dict):
                continue
            for cyc, cdict in block.items():
                # point
                pt = cdict.get("point", {})
                if isinstance(pt, dict):
                    for k, v in pt.items():
                        a = _safe_arr(v)
                        if a is None: continue
                        if a.ndim == 2 and a.shape[1] == 3:
                            if _passes_stride_side(k, stride_side): keys.add(k)
                        elif a.ndim == 1:
                            if _passes_stride_side(k, stride_side): keys.add(k)
                # analog
                analog = cdict.get("analog", {})
                if isinstance(analog, dict):
                    for k, v in analog.items():
                        if k == "time": continue
                        a = _safe_arr(v)
                        if a is not None and a.ndim == 1:
                            keys.add(k)
                # 1D groups
                for grp in ("grf","jrf","residual","JRL"):
                    g = cdict.get(grp, {})
                    if isinstance(g, dict):
                        for k, v in g.items():
                            a = _safe_arr(v)
                            if a is not None and a.ndim == 1:
                                if _passes_stride_side(k, stride_side): keys.add(k)
                # IK & SO names (side-agnostic)
                ik = cdict.get("IK_markerErr", {})
                if isinstance(ik, dict):
                    for nm in ("total_squared_error","marker_error_RMS","marker_error_max"):
                        if nm in ik: keys.add(f"IK_markerErr.{nm}")
                so = cdict.get("SO_forces", {})
                if isinstance(so, dict):
                    for nm in ("FX","FY","FZ","MX","MY","MZ"):
                        if nm in so: keys.add(f"SO_forces.{nm}")
    lst = sorted(keys)
    print(f"[discover] keys={len(lst)}")
    return lst

def resolve_key(cdict: Dict[str, Any], key: str) -> tuple[Optional[np.ndarray], str]:
    # point
    pt = cdict.get("point", {})
    if isinstance(pt, dict) and key in pt:
        a = _safe_arr(pt[key])
        if a is not None:
            if a.ndim == 2 and a.shape[1] == 3:
                return a, "3d"
            if a.ndim == 1:
                return a, "1d"
    # IK
    if key.startswith("IK_markerErr."):
        nm = key.split(".", 1)[1]
        ik = cdict.get("IK_markerErr", {})
        if isinstance(ik, dict) and nm in ik:
            a = _safe_arr(ik[nm])
            return (a.reshape(-1), "1d") if a is not None else (None, "")
    # SO
    if key.startswith("SO_forces."):
        nm = key.split(".", 1)[1]
        so = cdict.get("SO_forces", {})
        if isinstance(so, dict) and nm in so:
            a = _safe_arr(so[nm])
            return (a.reshape(-1), "1d") if a is not None else (None, "")
    # analog
    analog = cdict.get("analog", {})
    if isinstance(analog, dict) and key in analog:
        a = _safe_arr(analog[key])
        if a is not None:
            fs = _analog_fs(analog)
            if emg_mod.is_emg_key(key):
                a = emg_mod.process_emg(a, fs)
            return np.asarray(a).reshape(-1), "1d"
    # 1D groups
    for grp in ("grf","jrf","residual","JRL"):
        g = cdict.get(grp, {})
        if isinstance(g, dict) and key in g:
            a = _safe_arr(g[key])
            return (a.reshape(-1), "1d") if a is not None else (None, "")
    return None, ""

def is_kinetic(cdict: Dict[str, Any]) -> bool:
    try:
        return int(cdict.get("kinetic", 0)) == 1
    except Exception:
        return False


# ---------------- data classes ----------------

@dataclass
class CyclePtr:
    file_idx: int
    stride_side: str  # 'left_stride' or 'right_stride'
    cycle_name: str

@dataclass
class LoadedFile:
    meta: FileMeta
    data: Dict[str, Any]
    cycles: List[CyclePtr]  # pointers into left/right stride dicts


# ---------------- plot widgets ----------------

class PlotCell(QtWidgets.QWidget):
    selectionMade = QtCore.Signal(list, str)  # (items, mode)

    def __init__(self, parent, title: str, coord_index: Optional[int], combo_width_chars: int = 83):
        super().__init__(parent)
        self.title = title
        self.coord_index = coord_index  # 0,1,2 for x/y/z or None for 1D
        self.current_key = ""

        self.fig = Figure(figsize=(3, 2), layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        fm = self.combo.fontMetrics()
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo.setMinimumContentsLength(combo_width_chars)
        self.combo.setMinimumWidth(int(fm.horizontalAdvance("W") * combo_width_chars * 1.05))
        self.combo.currentTextChanged.connect(self._on_param_changed)

        self.btn_ylim = QtWidgets.QToolButton(text="↕︎")
        self.btn_ylim.setToolTip("Autoscale Y")
        self.btn_ylim.clicked.connect(self.autoscale)

        top = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(self.title); lab.setMinimumWidth(100)
        top.addWidget(lab); top.addStretch(1); top.addWidget(self.combo); top.addWidget(self.btn_ylim)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2,2,2,2)
        lay.addLayout(top); lay.addWidget(self.canvas)

        self.selector: Optional[SelectionManager] = None

    def set_options(self, keys: Sequence[str], default_key: Optional[str] = None) -> None:
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
        self.parent().parent().parent().parent().redraw_all()

    def clear(self):
        self.ax.cla()
        if self.selector is not None:
            self.selector.disconnect(); self.selector=None
        self.canvas.draw_idle()

    def autoscale(self):
        lines = [ln for ln in self.ax.lines if ln.get_visible()]
        if not lines: return
        sel = [ln for ln in lines if int(getattr(ln, "_meta", {}).get("manually_selected", 1))==1]
        src = sel if sel else lines
        lo, hi = None, None
        for ln in src:
            y = np.asarray(ln.get_ydata(orig=False))
            y = y[np.isfinite(y)]
            if y.size == 0: continue
            lo = float(np.min(y)) if lo is None else min(lo, float(np.min(y)))
            hi = float(np.max(y)) if hi is None else max(hi, float(np.max(y)))
        if lo is None or hi is None: return
        if hi <= lo:
            pad=1.0; self.ax.set_ylim(lo-pad, hi+pad)
        else:
            pad=0.05*(hi-lo); self.ax.set_ylim(lo-pad, hi+pad)
        self.canvas.draw_idle()

    def plot(self, files: List[LoadedFile], key_left_or_generic: str, show_left: bool, show_right: bool, mode_text: str,
             filter_kinetic: Optional[bool], filter_kinematic: Optional[bool]):
        """key_left_or_generic: for right side we derive counterpart automatically if possible."""
        self.clear()
        self.current_key = key_left_or_generic
        line_map: Dict[Any, SelectionItem] = {}

        # Helper to draw one series
        def draw_series(arr101: np.ndarray, stride_side: str, cdict: Dict[str, Any]):
            xs = np.linspace(0.0, 100.0, arr101.shape[0])
            if arr101.ndim == 1:
                YS = [arr101]
                coords = [""]
            else:
                if self.coord_index is None:
                    # if a 3D series but coord not specified, draw all 3 coords
                    YS = [arr101[:, j] for j in range(3)]
                    coords = ["x","y","z"]
                else:
                    YS = [arr101[:, self.coord_index]]
                    coords = ["xyz"[self.coord_index]]
            for y, coord in zip(YS, coords):
                ln, = self.ax.plot(xs, y, lw=1.2, alpha=1.0)
                if stride_side == "left_stride":
                    ln.set_color(LEFT_COLOR)
                else:
                    ln.set_color(RIGHT_COLOR)
                # de-selected
                ms = int(cdict.get("manually_selected", 1))
                ln._meta = {"manually_selected": ms}
                if ms == 0:
                    ln.set_alpha(0.2)
                # QC dashed
                if _is_qc_bad(cdict):
                    ln.set_linestyle("--")
                return ln

        for fi, lf in enumerate(files):
            dfile = lf.data
            for stride_side in ("left_stride", "right_stride"):
                if stride_side == "left_stride" and not show_left: continue
                if stride_side == "right_stride" and not show_right: continue
                block = dfile.get(stride_side, {})
                if not isinstance(block, dict): continue
                for cyc_name, cdict in block.items():
                    # filter kinetic/kinematic
                    kin = is_kinetic(cdict)
                    if kin and not filter_kinetic: continue
                    if (not kin) and not filter_kinematic: continue

                    # pick side-specific key
                    key = key_left_or_generic
                    if stride_side == "left_stride":
                        # if key looks right-sided, convert to left
                        if _is_right_key(key):
                            key = _counterpart_key(key, "left")
                        if not _passes_stride_side(key, stride_side):
                            continue
                    else:  # right_stride
                        if _is_left_key(key):
                            key = _counterpart_key(key, "right")
                        if not _passes_stride_side(key, stride_side):
                            continue

                    arr, kind = resolve_key(cdict, key)
                    if arr is None: continue
                    arr101 = _timenorm(arr, is_3d=(kind=="3d"))
                    if arr101 is None: continue

                    ln = draw_series(arr101, stride_side, cdict)
                    if ln is not None:
                        line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=key)

        # labels
        base_lab = key_left_or_generic
        if self.coord_index is not None:
            base_lab += f" [{'xyz'[self.coord_index]}]"
        self.ax.set_ylabel(base_lab)
        self.ax.set_xlabel("% cycle")
        self.ax.set_title(f"{self.title}   |   Mode: {mode_text}")
        self.canvas.draw_idle()

        self.selector = SelectionManager(self.ax, line_map, mode="select", on_result=self._on_rect)
        self.selector.set_active(True)

    def _on_rect(self, items: List[SelectionItem], mode: str, bbox):
        self.selectionMade.emit(items, mode)


class PlotPage(QtWidgets.QWidget):
    selectionApplied = QtCore.Signal(list)  # list[(file_idx, stride_side, cycle, new_val)]

    def __init__(self, parent, title: str, rows: int, cols: int, coord_map: List[Optional[int]]):
        super().__init__(parent)
        self.title = title
        self.rows = rows; self.cols = cols
        grid = QtWidgets.QGridLayout(self); grid.setContentsMargins(6,6,6,6); grid.setSpacing(6)
        self.cells: List[PlotCell] = []
        k = 0
        for r in range(rows):
            for c in range(cols):
                cell = PlotCell(self, f"{title} [{r+1},{c+1}]", coord_index=coord_map[k], combo_width_chars=83)
                cell.selectionMade.connect(self._on_cell_sel)
                self.cells.append(cell); grid.addWidget(cell, r, c)
                k += 1

    def set_options_all(self, keys: Sequence[str], defaults: Sequence[str]) -> None:
        for i, cell in enumerate(self.cells):
            default = defaults[i] if i < len(defaults) and defaults[i] in keys else (keys[0] if keys else None)
            cell.set_options(keys, default_key=default)

    def redraw(self, files: List[LoadedFile], show_left: bool, show_right: bool, mode_text: str,
               filter_kinetic: Optional[bool], filter_kinematic: Optional[bool]) -> None:
        for cell in self.cells:
            key = cell.current_key or (cell.combo.currentText().strip() if cell.combo.count() else "")
            if not key and cell.combo.count(): key = cell.combo.itemText(0)
            if key:
                cell.plot(files, key, show_left, show_right, mode_text, filter_kinetic, filter_kinematic)

    @QtCore.Slot(list, str)
    def _on_cell_sel(self, items: List[SelectionItem], mode: str):
        new_val = 1 if mode=="select" else 0
        changes = []
        seen = set()
        for it in items:
            fi, stride_side, cyc = it.cycle_ref
            k = (fi, stride_side, cyc)
            if k in seen: continue
            seen.add(k)
            changes.append((fi, stride_side, cyc, new_val))
        if changes:
            self.selectionApplied.emit(changes)


# ---------------- Main Window ----------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cycle Selection Tool — Step 4")
        self.resize(1600, 950)

        # Toolbar with root, combos, checkboxes, mode
        self.root_edit = QtWidgets.QLineEdit()
        self.btn_root = QtWidgets.QPushButton("Set Root…")
        self.btn_root.clicked.connect(self.on_choose_root)

        self.cmb_pid = QtWidgets.QComboBox(); self.cmb_pid.currentTextChanged.connect(self.on_pid_changed)
        self.cmb_trial = QtWidgets.QComboBox(); self.cmb_trial.currentTextChanged.connect(self.on_trial_changed)

        self.chk_left = QtWidgets.QCheckBox("left_stride data"); self.chk_left.setChecked(True)
        self.chk_right = QtWidgets.QCheckBox("right_stride data"); self.chk_right.setChecked(True)
        self.chk_kinetic = QtWidgets.QCheckBox("show kinetic cycles"); self.chk_kinetic.setChecked(True)
        self.chk_kinematic = QtWidgets.QCheckBox("show kinematic cycles"); self.chk_kinematic.setChecked(False)
        for chk in (self.chk_left, self.chk_right, self.chk_kinetic, self.chk_kinematic):
            chk.toggled.connect(self.on_filters_changed)

        self.lbl_mode = QtWidgets.QLabel("Mode: SELECT")

        tb = QtWidgets.QToolBar()
        tb.addWidget(QtWidgets.QLabel("Root:")); tb.addWidget(self.root_edit); tb.addWidget(self.btn_root)
        tb.addSeparator(); tb.addWidget(QtWidgets.QLabel("PID:")); tb.addWidget(self.cmb_pid)
        tb.addWidget(QtWidgets.QLabel("Trial:")); tb.addWidget(self.cmb_trial)
        tb.addSeparator()
        tb.addWidget(self.chk_left); tb.addWidget(self.chk_right)
        tb.addSeparator()
        tb.addWidget(self.chk_kinetic); tb.addWidget(self.chk_kinematic)
        tb.addSeparator(); tb.addWidget(self.lbl_mode)
        self.addToolBar(Qt.TopToolBarArea, tb)

        # Layouts: Page 1 and Page 2 (3x6)
        # coord_map per cell: rows x,y,z (0,1,2) for 3D series; None for 1D plots
        coord_map = [0]*6 + [1]*6 + [2]*6  # page1 uses x,y,z rows; page2 varies per column
        self.tabs = QtWidgets.QTabWidget()
        self.page1 = PlotPage(self, "Page1", rows=3, cols=6, coord_map=coord_map)

        # Page2: first column rows are 1D (Fx/Fy/Fz), others 1D; coord map None for all
        coord_map2 = [None]*(3*6)
        self.page2 = PlotPage(self, "Page2", rows=3, cols=6, coord_map=coord_map2)

        self.tabs.addTab(self.page1, "Page 1")
        self.tabs.addTab(self.page2, "Page 2")
        self.setCentralWidget(self.tabs)

        # Status
        from .status import StatusWidget
        self.status_widget = StatusWidget(self); self.setStatusBar(self.status_widget)

        # Data state
        self.participants: List[str] = []
        self.trial_types_by_pid: Dict[str, List[str]] = {}
        self.files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}
        self.loaded: List[LoadedFile] = []
        self.all_keys: List[str] = []

        # shortcuts: 's' select, 'd' deselect, Ctrl+U autoscale
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=lambda: self.set_mode("select"))
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self.set_mode("deselect"))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+U"), self, activated=self.autoscale_all)
        self.current_mode = "select"

        # Selection application
        self.page1.selectionApplied.connect(self.apply_selection)
        self.page2.selectionApplied.connect(self.apply_selection)

        # Try env var for root
        env_root = os.getenv("CYCLE_TOOL_ROOT")
        if env_root and os.path.isdir(env_root):
            self.root_edit.setText(env_root); self.set_root(env_root)

        # Set defaults for page layouts
        self.default_keys_page1 = self._default_keys_page1()
        self.default_keys_page2 = self._default_keys_page2()

    # ---------------- Mode ----------------
    def set_mode(self, mode: str):
        self.current_mode = "select" if mode=="select" else "deselect"
        self.lbl_mode.setText(f"Mode: {self.current_mode.upper()}")
        for page in (self.page1, self.page2):
            for cell in page.cells:
                if cell.selector: cell.selector.set_mode(self.current_mode)

    # ---------------- Root / Index ----------------
    @Slot()
    def on_choose_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose root folder")
        if not path: return
        self.root_edit.setText(path); self.set_root(path)

    def set_root(self, root: str):
        print(f"[root] {root}")
        files = scan_root(root)
        print(f"[scan] matched {len(files)} files")
        self.participants, self.trial_types_by_pid, self.files_by_pid_type = build_indices(files)

        self.cmb_pid.blockSignals(True); self.cmb_pid.clear(); self.cmb_pid.addItems(self.participants); self.cmb_pid.blockSignals(False)
        if self.participants:
            self.cmb_pid.setCurrentIndex(0); self.on_pid_changed(self.participants[0])
        else:
            self.cmb_trial.clear(); self.loaded=[]; self.all_keys=[]; self.redraw_all()

    def on_pid_changed(self, pid: str):
        types = self.trial_types_by_pid.get(pid, [])
        print(f"[pid] {pid} types={types}")
        self.cmb_trial.blockSignals(True); self.cmb_trial.clear(); self.cmb_trial.addItems(types); self.cmb_trial.blockSignals(False)
        if types:
            self.cmb_trial.setCurrentIndex(0); self.on_trial_changed(types[0])
        else:
            self.loaded=[]; self.all_keys=[]; self.redraw_all()

    def on_trial_changed(self, trial: str):
        pid = self.cmb_pid.currentText().strip()
        metas = self.files_by_pid_type.get(pid, {}).get(trial, [])
        print(f"[trial] {pid}/{trial}: files={len(metas)}")
        self.loaded = []
        for i, fm in enumerate(metas):
            try:
                d = load_dict(fm.path)
                cycles: List[CyclePtr] = []
                for stride_side in ("left_stride","right_stride"):
                    block = d.get(stride_side, {})
                    if isinstance(block, dict):
                        for cyc in block.keys():
                            cycles.append(CyclePtr(file_idx=i, stride_side=stride_side, cycle_name=cyc))
                self.loaded.append(LoadedFile(meta=fm, data=d, cycles=cycles))
                print(f"  - {os.path.basename(fm.path)} cycles={len(cycles)}")
            except Exception as e:
                print(f"  !! failed {fm.path}: {e!r}")
        self.all_keys = discover_keys(self.loaded)
        # Page1 & Page2 keys
        self.page1.set_options_all(self.all_keys, self.default_keys_page1)
        self.page2.set_options_all(self.all_keys, self.default_keys_page2)

        self._refresh_counts(); self.redraw_all()

    def on_filters_changed(self, _state: bool):
        self.redraw_all()

    def _refresh_counts(self):
        items = []
        for lf in self.loaded:
            d = lf.data
            for stride_side in ("left_stride","right_stride"):
                block = d.get(stride_side, {})
                if not isinstance(block, dict): continue
                for cyc, cdict in block.items():
                    items.append({
                        "manually_selected": int(cdict.get("manually_selected", 1)),
                        "kinetic": 1 if is_kinetic(cdict) else 0
                    })
        totals = compute_counts(items)
        self.status_widget.update_from_counts(totals)

    def redraw_all(self):
        show_left = self.chk_left.isChecked()
        show_right = self.chk_right.isChecked()
        filter_kinetic = self.chk_kinetic.isChecked()
        filter_kinematic = self.chk_kinematic.isChecked()
        mode_text = self.current_mode.upper()
        self.page1.redraw(self.loaded, show_left, show_right, mode_text, filter_kinetic, filter_kinematic)
        self.page2.redraw(self.loaded, show_left, show_right, mode_text, filter_kinetic, filter_kinematic)

    def autoscale_all(self):
        for page in (self.page1, self.page2):
            for cell in page.cells: cell.autoscale()

    @QtCore.Slot(list)
    def apply_selection(self, changes: List[tuple]):
        changed = 0
        for fi, stride_side, cyc, val in changes:
            if fi < 0 or fi >= len(self.loaded): continue
            d = self.loaded[fi].data
            block = d.get(stride_side, {})
            if not isinstance(block, dict): continue
            cdict = block.get(cyc, {})
            try:
                old = int(cdict.get("manually_selected", 1))
            except Exception:
                old = 1
            if old != int(val):
                cdict["manually_selected"] = int(val)
                changed += 1
        if changed:
            print(f"[select] applied {changed} change(s)")
            self._refresh_counts()
            self.redraw_all()

    # ---------------- default layout helpers ----------------

    def _default_keys_page1(self) -> List[str]:
        # Six columns: hip, knee, ankle angles; hip, knee, ankle moments.
        # We seed left-side keys; right counterpart will be derived automatically when plotting right_stride.
        return [
            "LHipAngles","LKneeAngles","LAnkleAngles","LHipMoment","LKneeMoment","LAnkleMoment",  # row 1 (x)
            "LHipAngles","LKneeAngles","LAnkleAngles","LHipMoment","LKneeMoment","LAnkleMoment",  # row 2 (y)
            "LHipAngles","LKneeAngles","LAnkleAngles","LHipMoment","LKneeMoment","LAnkleMoment",  # row 3 (z)
        ]

    def _default_keys_page2(self) -> List[str]:
        # Column 1: GRF per component via analog names pattern; we pick representative keys
        # We'll initialize with the Fx/Fy/Fz channel names if present; otherwise use generic.
        fx = "Force_Fx1"; fy = "Force_Fy1"; fz = "Force_Fz1"
        # Columns 2..4: hip/knee/ankle JRF fx/fy/fz with side-aware names; seed with left keys
        hip_fx = "hip_l_on_femur_l_in_femur_l_fx"
        hip_fy = "hip_l_on_femur_l_in_femur_l_fy"
        hip_fz = "hip_l_on_femur_l_in_femur_l_fz"
        knee_fx = "med_cond_weld_l_on_tibial_plat_l_in_tibial_plat_l_fx"
        knee_fy = "med_cond_weld_l_on_tibial_plat_l_in_tibial_plat_l_fy"
        knee_fz = "med_cond_weld_l_on_tibial_plat_l_in_tibial_plat_l_fz"
        ankle_fx = "ankle_l_on_talus_l_in_talus_l_fx"
        ankle_fy = "ankle_l_on_talus_l_in_talus_l_fy"
        ankle_fz = "ankle_l_on_talus_l_in_talus_l_fz"
        # Column 5: IK error series
        ik_x = "IK_markerErr.total_squared_error"
        ik_y = "IK_markerErr.marker_error_RMS"
        ik_z = "IK_markerErr.marker_error_max"
        # Column 6: SO residual forces
        so_fx = "SO_forces.FX"; so_fy = "SO_forces.FY"; so_fz = "SO_forces.FZ"

        return [
            fx, hip_fx, knee_fx, ankle_fx, ik_x, so_fx,
            fy, hip_fy, knee_fy, ankle_fy, ik_y, so_fy,
            fz, hip_fz, knee_fz, ankle_fz, ik_z, so_fz,
        ]


# --------------- app entry ---------------

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(); w.show(); app.exec()


if __name__ == "__main__":
    main()
