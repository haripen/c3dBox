
# c3dBox/Step4_check/main.py
# Requirements implemented:
# - Print debug to console (no dock)
# - No JSON layouts; use fixed grids (Page1 3x6 with 63-char combos, Page2 3x6)
# - Immediate refresh on root set
# - Dropdowns list all discovered keys (3D and 1D)
# - Ctrl+S select / Ctrl+D deselect with visible mode text
# - Left=red, Right=blue; Y autoscale via button or Ctrl+U
# - Rectangle selection toggles manually_selected and updates counts

from __future__ import annotations

import os
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
from .cycles import collect_cycles, CycleRef
from .selection import SelectionManager, SelectionItem
from .status import compute_counts, StatusWidget
from . import emg as emg_mod

LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#1f77b4"


def _safe_arr(x: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None
        return a
    except Exception:
        return None

def _is_kinetic_cycle(cdict: Dict[str, Any]) -> bool:
    if not isinstance(cdict, dict):
        return False
    if "SO_forces" in cdict or "grf" in cdict or "jrf" in cdict or "residual" in cdict:
        return True
    try:
        return int(cdict.get("kinetic", 0)) == 1
    except Exception:
        return False

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

def _side_name(s: str) -> str:
    s = (s or "").lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return s or "?"

def discover_keys(files: List["LoadedFile"]) -> List[str]:
    keys: set[str] = set()
    for lf in files:
        for cref in lf.cycles:
            cdict = lf.data.get(cref.side, {}).get(cref.cycle, {})
            # point
            pt = cdict.get("point", {})
            if isinstance(pt, dict):
                for k, v in pt.items():
                    a = _safe_arr(v)
                    if a is None:
                        continue
                    if a.ndim == 2 and a.shape[1] == 3:
                        keys.add(k)
                    elif a.ndim == 1:
                        keys.add(k)
            # IK
            ik = cdict.get("IK_markerErr", {})
            if isinstance(ik, dict):
                for nm in ("total_squared_error", "marker_error_RMS", "marker_error_max"):
                    if nm in ik:
                        keys.add(f"IK_markerErr.{nm}")
            # SO
            so = cdict.get("SO_forces", {})
            if isinstance(so, dict):
                for nm in ("FX", "FY", "FZ", "MX", "MY", "MZ"):
                    if nm in so:
                        keys.add(f"SO_forces.{nm}")
            # analog
            analog = cdict.get("analog", {})
            if isinstance(analog, dict):
                for k, v in analog.items():
                    if k == "time":
                        continue
                    a = _safe_arr(v)
                    if a is not None and a.ndim == 1:
                        keys.add(k)
            # other groups
            for grp in ("grf", "jrf", "residual", "JRL"):
                g = cdict.get(grp, {})
                if isinstance(g, dict):
                    for k, v in g.items():
                        a = _safe_arr(v)
                        if a is not None and a.ndim == 1:
                            keys.add(k)
    lst = sorted(keys)
    print(f"[discover] keys={len(lst)}")
    return lst

def resolve_key(cdict: Dict[str, Any], key: str) -> tuple[Optional[np.ndarray], str]:
    print(f"[access] key='{key}' groups={list(cdict.keys())}")
    pt = cdict.get("point", {})
    if isinstance(pt, dict) and key in pt:
        a = _safe_arr(pt[key])
        if a is not None:
            if a.ndim == 2 and a.shape[1] == 3:
                print("  -> point 3D", a.shape)
                return a, "3d"
            if a.ndim == 1:
                print("  -> point 1D", a.shape)
                return a, "1d"
    if key.startswith("IK_markerErr."):
        name = key.split(".", 1)[1]
        ik = cdict.get("IK_markerErr", {})
        if isinstance(ik, dict) and name in ik:
            a = _safe_arr(ik[name])
            if a is not None:
                print("  -> IK 1D", a.shape); return a.reshape(-1), "1d"
    if key.startswith("SO_forces."):
        name = key.split(".", 1)[1]
        so = cdict.get("SO_forces", {})
        if isinstance(so, dict) and name in so:
            a = _safe_arr(so[name])
            if a is not None:
                print("  -> SO 1D", a.shape); return a.reshape(-1), "1d"
    analog = cdict.get("analog", {})
    if isinstance(analog, dict) and key in analog:
        a = _safe_arr(analog[key])
        if a is not None:
            fs = _analog_fs(analog)
            if emg_mod.is_emg_key(key):
                a = emg_mod.process_emg(a, fs)
                print(f"  -> analog EMG (fs~{fs:.1f})", np.shape(a))
            else:
                print("  -> analog 1D", np.shape(a))
            return np.asarray(a).reshape(-1), "1d"
    for grp in ("grf", "jrf", "residual", "JRL"):
        g = cdict.get(grp, {})
        if isinstance(g, dict) and key in g:
            a = _safe_arr(g[key])
            if a is not None:
                print(f"  -> {grp} 1D", np.shape(a)); return np.asarray(a).reshape(-1), "1d"
    print("  !! not found")
    return None, ""


class PlotCell(QtWidgets.QWidget):
    selectionMade = QtCore.Signal(list, str)  # (items, mode)

    def __init__(self, parent, title: str, widen_combo: bool = False):
        super().__init__(parent)
        self.title = title
        self.current_key = ""

        self.fig = Figure(figsize=(3, 2), layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        if widen_combo:
            self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            fm = self.combo.fontMetrics()
            self.combo.setMinimumContentsLength(63)
            self.combo.setMinimumWidth(int(fm.horizontalAdvance("W") * 63 * 1.05))
        self.combo.currentTextChanged.connect(self._on_param_changed)

        self.btn_ylim = QtWidgets.QToolButton(text="↕︎")
        self.btn_ylim.setToolTip("Autoscale Y")
        self.btn_ylim.clicked.connect(self.autoscale)

        top = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(self.title); lab.setMinimumWidth(80)
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

    def plot(self, files: List["LoadedFile"], key: str, mode_text: str):
        self.clear()
        self.current_key = key
        line_map: Dict[Any, SelectionItem] = {}

        for fi, lf in enumerate(files):
            for cref in lf.cycles:
                side = _side_name(cref.side)
                cdict = lf.data.get(cref.side, {}).get(cref.cycle, {})
                arr, kind = resolve_key(cdict, key)
                if arr is None: continue
                if kind == "3d":
                    arr101 = _timenorm(arr, is_3d=True)
                    if arr101 is None: continue
                    xs = np.linspace(0.0, 100.0, arr101.shape[0])
                    for j, coord in enumerate(("x","y","z")):
                        yy = arr101[:, j]
                        ln, = self.ax.plot(xs, yy, lw=1.2, alpha=1.0)
                        ln.set_color(LEFT_COLOR if side=="left" else RIGHT_COLOR)
                        ln._meta = {"manually_selected": int(cdict.get("manually_selected",1))}
                        if ln._meta["manually_selected"]==0: ln.set_alpha(0.2)
                        line_map[ln] = SelectionItem(cycle_ref=(fi, cref.side, cref.cycle), param=key)
                else:
                    arr101 = _timenorm(arr, is_3d=False)
                    if arr101 is None: continue
                    xs = np.linspace(0.0, 100.0, arr101.shape[0])
                    ln, = self.ax.plot(xs, arr101, lw=1.2, alpha=1.0)
                    ln.set_color(LEFT_COLOR if side=="left" else RIGHT_COLOR)
                    ln._meta = {"manually_selected": int(cdict.get("manually_selected",1))}
                    if ln._meta["manually_selected"]==0: ln.set_alpha(0.2)
                    line_map[ln] = SelectionItem(cycle_ref=(fi, cref.side, cref.cycle), param=key)

        self.ax.set_xlabel("% cycle")
        self.ax.set_title(f"{self.title}   |   Mode: {mode_text}")
        self.canvas.draw_idle()

        self.selector = SelectionManager(self.ax, line_map, mode="select", on_result=self._on_rect)
        self.selector.set_active(True)

    def _on_rect(self, items: List[SelectionItem], mode: str, bbox):
        self.selectionMade.emit(items, mode)


class PlotPage(QtWidgets.QWidget):
    selectionApplied = QtCore.Signal(list)  # list[(file_idx, side, cycle, new_val)]
    def __init__(self, parent, rows: int, cols: int, widen_first_page: bool = False):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self); grid.setContentsMargins(6,6,6,6); grid.setSpacing(6)
        self.cells: List[PlotCell] = []
        for r in range(rows):
            for c in range(cols):
                cell = PlotCell(self, f"r{r+1}c{c+1}", widen_combo=widen_first_page)
                cell.selectionMade.connect(self._on_cell_sel)
                self.cells.append(cell); grid.addWidget(cell, r, c)

    def set_options_all(self, keys: Sequence[str]) -> None:
        for i, cell in enumerate(self.cells):
            default = keys[i] if i < len(keys) else (keys[0] if keys else None)
            cell.set_options(keys, default_key=default)

    def redraw(self, files: List["LoadedFile"], mode_text: str) -> None:
        for cell in self.cells:
            key = cell.current_key or (cell.combo.currentText().strip() if cell.combo.count() else "")
            if not key and cell.combo.count(): key = cell.combo.itemText(0)
            if key: cell.plot(files, key, mode_text)

    @QtCore.Slot(list, str)
    def _on_cell_sel(self, items: List[SelectionItem], mode: str):
        new_val = 1 if mode=="select" else 0
        changes = []
        seen = set()
        for it in items:
            fi, side, cyc = it.cycle_ref
            k = (fi, side, cyc)
            if k in seen: continue
            seen.add(k)
            changes.append((fi, side, cyc, new_val))
        if changes:
            self.selectionApplied.emit(changes)


@dataclass
class LoadedFile:
    meta: FileMeta
    data: Dict[str, Any]
    cycles: List[CycleRef]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cycle Selection Tool — Step 4")
        self.resize(1500, 900)

        self.root_edit = QtWidgets.QLineEdit()
        self.btn_root = QtWidgets.QPushButton("Set Root…")
        self.btn_root.clicked.connect(self.on_choose_root)

        self.cmb_pid = QtWidgets.QComboBox(); self.cmb_pid.currentTextChanged.connect(self.on_pid_changed)
        self.cmb_trial = QtWidgets.QComboBox(); self.cmb_trial.currentTextChanged.connect(self.on_trial_changed)

        self.lbl_mode = QtWidgets.QLabel("Mode: SELECT")

        tb = QtWidgets.QToolBar()
        tb.addWidget(QtWidgets.QLabel("Root:")); tb.addWidget(self.root_edit); tb.addWidget(self.btn_root)
        tb.addSeparator(); tb.addWidget(QtWidgets.QLabel("PID:")); tb.addWidget(self.cmb_pid)
        tb.addWidget(QtWidgets.QLabel("Trial:")); tb.addWidget(self.cmb_trial)
        tb.addSeparator(); tb.addWidget(self.lbl_mode)
        self.addToolBar(Qt.TopToolBarArea, tb)

        self.tabs = QtWidgets.QTabWidget()
        self.page1 = PlotPage(self, rows=3, cols=6, widen_first_page=True)
        self.page2 = PlotPage(self, rows=3, cols=6, widen_first_page=False)
        self.tabs.addTab(self.page1, "Page 1"); self.tabs.addTab(self.page2, "Page 2")
        self.setCentralWidget(self.tabs)

        self.status_widget = StatusWidget(self); self.setStatusBar(self.status_widget)

        self.participants: List[str] = []
        self.trial_types_by_pid: Dict[str, List[str]] = {}
        self.files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}
        self.loaded: List[LoadedFile] = []
        self.all_keys: List[str] = []

        # shortcuts (no Save on Ctrl+S)
        self.act_select = QtGui.QAction("Select mode", self); self.act_select.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.act_deselect = QtGui.QAction("Deselect mode", self); self.act_deselect.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        self.act_autoscale = QtGui.QAction("Autoscale Y", self); self.act_autoscale.setShortcut(QtGui.QKeySequence("Ctrl+U"))
        self.act_select.triggered.connect(lambda: self.set_mode("select"))
        self.act_deselect.triggered.connect(lambda: self.set_mode("deselect"))
        self.act_autoscale.triggered.connect(self.autoscale_all)
        for a in (self.act_select, self.act_deselect, self.act_autoscale):
            self.addAction(a)
        self.set_mode("select")

        # apply selection changes
        self.page1.selectionApplied.connect(self.apply_selection)
        self.page2.selectionApplied.connect(self.apply_selection)

        env_root = os.getenv("CYCLE_TOOL_ROOT")
        if env_root and os.path.isdir(env_root):
            self.root_edit.setText(env_root); self.set_root(env_root)

    def set_mode(self, mode: str):
        self.current_mode = "select" if mode=="select" else "deselect"
        self.lbl_mode.setText(f"Mode: {self.current_mode.upper()}")
        for page in (self.page1, self.page2):
            for cell in page.cells:
                if cell.selector: cell.selector.set_mode(self.current_mode)

    @Slot()
    def on_choose_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose root")
        if not path: return
        self.root_edit.setText(path); self.set_root(path)

    def set_root(self, root: str):
        print(f"[root] {root}")
        files = scan_root(root)
        print(f"[scan] matched {len(files)}")
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
        for fm in metas:
            try:
                d = load_dict(fm.path)
                cyc = collect_cycles(d, fm)
                self.loaded.append(LoadedFile(meta=fm, data=d, cycles=cyc))
                print(f"  - {os.path.basename(fm.path)} cycles={len(cyc)}")
            except Exception as e:
                print(f"  !! failed {fm.path}: {e!r}")
        self.all_keys = discover_keys(self.loaded)
        self.page1.set_options_all(self.all_keys); self.page2.set_options_all(self.all_keys)
        self._refresh_counts(); self.redraw_all()

    def _refresh_counts(self):
        items = []
        for lf in self.loaded:
            for cref in lf.cycles:
                cdict = lf.data.get(cref.side, {}).get(cref.cycle, {})
                items.append({
                    "manually_selected": int(cdict.get("manually_selected", 1)),
                    "kinetic": 1 if _is_kinetic_cycle(cdict) else 0
                })
        totals = compute_counts(items)
        self.status_widget.update_from_counts(totals)

    def redraw_all(self):
        mode_text = getattr(self, "current_mode", "select").upper()
        for page in (self.page1, self.page2):
            page.redraw(self.loaded, mode_text)

    def autoscale_all(self):
        for page in (self.page1, self.page2):
            for cell in page.cells:
                cell.autoscale()

    @QtCore.Slot(list)
    def apply_selection(self, changes: List[tuple]):
        changed = 0
        for fi, side, cyc, val in changes:
            if fi < 0 or fi >= len(self.loaded): continue
            d = self.loaded[fi].data
            cdict = d.get(side, {}).get(cyc, {})
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


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(); w.show(); app.exec()


if __name__ == "__main__":
    main()
