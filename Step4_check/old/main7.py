
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot

from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .file_index import scan_root, build_indices, FileMeta
from .io_mat import load_dict
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

def _interpolate_101(arr: np.ndarray) -> Optional[np.ndarray]:
    try: a = np.asarray(arr).reshape(-1)
    except Exception: return None
    if a.size < 2: return None
    n = a.shape[0]; x = np.linspace(0.0,1.0,n); xx = np.linspace(0.0,1.0,101)
    return np.interp(xx, x, a)

def _timenorm(arr: np.ndarray, is_3d: bool) -> Optional[np.ndarray]:
    a = _safe_arr(arr); 
    if a is None: return None
    if a.ndim == 1: return _interpolate_101(a)
    if is_3d and a.ndim==2 and a.shape[1]==3:
        out = np.zeros((101,3), float)
        for k in range(3):
            col = _interpolate_101(a[:,k]); 
            if col is None: return None
            out[:,k] = col
        return out
    return None

def _analog_fs(analog: Dict[str, Any]) -> float:
    try:
        t = np.asarray(analog.get("time")).reshape(-1)
        if t.size>=2:
            dt = np.median(np.diff(t))
            if dt>0: return float(1.0/dt)
    except Exception: pass
    return 1000.0

def is_kinetic(cdict: Dict[str, Any]) -> bool:
    try: return int(cdict.get("kinetic",0))==1
    except Exception: return False

def _is_qc_bad(cdict: Dict[str, Any]) -> bool:
    for k in ("reconstruction_ok","IK_RMS_ok","IK_MAX_ok","SO_F_RMS_ok","SO_F_MAX_ok","SO_M_RMS_ok","SO_M_MAX_ok"):
        if k in cdict:
            try:
                if int(cdict[k])==0: return True
            except Exception: pass
    return False

@dataclass
class KeyTemplate:
    display: str
    pattern: str
    sample: str
    analog_glob: Optional[str] = None
    def to_side(self, stride_side: str) -> str:
        if self.pattern=="none": return self.sample
        side = "left" if stride_side=="left_stride" else "right"
        if self.pattern=="prefix":
            core = re.sub(r"^[LR](?=[A-Z])","", self.sample)
            return ("L" if side=="left" else "R")+core
        if self.pattern=="token":
            tL = "_l_" if "_l_" in self.sample else ("_L_" if "_L_" in self.sample else "_l_")
            tR = "_r_" if "_r_" in self.sample else ("_R_" if "_R_" in self.sample else "_r_")
            return re.sub(r"(_[lLrR]_)", tL if side=="left" else tR, self.sample)
        if self.pattern=="trail":
            return re.sub(r"_(?:l|r)$", "_l" if side=="left" else "_r", self.sample, flags=re.IGNORECASE)
        if self.pattern=="emg":
            return re.sub(r"_(?:L|R)_", "_L_" if side=="left" else "_R_", self.sample)
        if self.pattern=="analog_many":
            return self.analog_glob or self.sample
        return self.sample

def _display_from_key(key: str) -> Tuple[str, KeyTemplate]:
    if "*" in key:
        tpl = KeyTemplate(display=key, pattern="analog_many", sample=key, analog_glob=key); return key, tpl
    if re.search(r"_(?:L|R)_", key):
        disp = re.sub(r"_(?:L|R)_","_", key); return disp, KeyTemplate(display=disp, pattern="emg", sample=key)
    if re.match(r"^[LR][A-Z]", key):
        disp = re.sub(r"^[LR]([A-Z].*)", r"\1", key); return disp, KeyTemplate(display=disp, pattern="prefix", sample=key)
    if re.search(r"_[lLrR]_", key):
        disp = re.sub(r"_[lLrR]_","_", key); return disp, KeyTemplate(display=disp, pattern="token", sample=key)
    if re.search(r"_[lrLR]$", key):
        disp = re.sub(r"_[lrLR]$","", key); return disp, KeyTemplate(display=disp, pattern="trail", sample=key)
    return key, KeyTemplate(display=key, pattern="none", sample=key)

def _analog_many_names(analog: Dict[str, Any], glob: str) -> List[str]:
    if not isinstance(analog, dict) or "*" not in glob: return []
    prefix, _, suffix = glob.partition("*"); out=[]
    for k in sorted(analog.keys()):
        if k=="time": continue
        if k.startswith(prefix) and k.endswith(suffix):
            try:
                if np.asarray(analog[k]).size>0: out.append(k)
            except Exception: continue
    return out

def discover_key_templates(files: List["LoadedFile"]) -> Dict[str, KeyTemplate]:
    templates: Dict[str, KeyTemplate] = {}; analog_wildcards=set()
    for lf in files:
        d = lf.data
        for stride_side in ("left_stride","right_stride"):
            block = d.get(stride_side, {})
            if not isinstance(block, dict): continue
            for cyc, cdict in block.items():
                pt = cdict.get("point", {})
                if isinstance(pt, dict):
                    for k,v in pt.items():
                        if _safe_arr(v) is None: continue
                        disp,tpl = _display_from_key(k); templates.setdefault(disp, tpl)
                analog = cdict.get("analog", {})
                if isinstance(analog, dict):
                    for ak in analog.keys():
                        if ak=="time": continue
                        disp,tpl = _display_from_key(ak); templates.setdefault(disp, tpl)
                        m = re.match(r"^(Force_F[xyz])[0-9]+$", ak, flags=re.IGNORECASE)
                        if m: analog_wildcards.add(m.group(1)+"*")
                for grp in ("grf","jrf","residual","JRL"):
                    g = cdict.get(grp, {})
                    if isinstance(g, dict):
                        for k,v in g.items():
                            if _safe_arr(v) is None: continue
                            disp,tpl = _display_from_key(k); templates.setdefault(disp, tpl)
                ik = cdict.get("IK_markerErr", {})
                if isinstance(ik, dict):
                    for nm in ("total_squared_error","marker_error_RMS","marker_error_max"):
                        if nm in ik:
                            disp=f"IK_markerErr.{nm}"; templates.setdefault(disp, KeyTemplate(display=disp, pattern="none", sample=disp))
                so = cdict.get("SO_forces", {})
                if isinstance(so, dict):
                    for nm in ("FX","FY","FZ","MX","MY","MZ"):
                        if nm in so:
                            disp=f"SO_forces.{nm}"; templates.setdefault(disp, KeyTemplate(display=disp, pattern="none", sample=disp))
    for w in sorted(analog_wildcards):
        templates.setdefault(w, KeyTemplate(display=w, pattern="analog_many", sample=w, analog_glob=w))
    print(f"[discover] display keys={len(templates)}"); return templates

def resolve_key(cdict: Dict[str, Any], actual_key: str):
    if "*" in actual_key:
        names = _analog_many_names(cdict.get("analog", {}), actual_key); pairs=[]
        for nm in names:
            a = _safe_arr(cdict.get("analog", {}).get(nm)); 
            if a is not None: pairs.append((nm, a.reshape(-1)))
        return (pairs, "analog_many") if pairs else (None, "")
    pt = cdict.get("point", {})
    if isinstance(pt, dict) and actual_key in pt:
        a = _safe_arr(pt[actual_key])
        if a is not None:
            if a.ndim==2 and a.shape[1]==3: return a, "3d"
            if a.ndim==1: return a, "1d"
    if actual_key.startswith("IK_markerErr."):
        nm = actual_key.split(".",1)[1]; ik = cdict.get("IK_markerErr", {})
        if isinstance(ik, dict) and nm in ik:
            a = _safe_arr(ik[nm]); 
            if a is not None: return a.reshape(-1), "1d"
    if actual_key.startswith("SO_forces."):
        nm = actual_key.split(".",1)[1]; so = cdict.get("SO_forces", {})
        if isinstance(so, dict) and nm in so:
            a = _safe_arr(so[nm]); 
            if a is not None: return a.reshape(-1), "1d"
    analog = cdict.get("analog", {})
    if isinstance(analog, dict) and actual_key in analog:
        a = _safe_arr(analog[actual_key])
        if a is not None:
            fs = _analog_fs(analog)
            if emg_mod.is_emg_key(actual_key): a = emg_mod.process_emg(a, fs)
            return np.asarray(a).reshape(-1), "1d"
    for grp in ("grf","jrf","residual","JRL"):
        g = cdict.get(grp, {})
        if isinstance(g, dict) and actual_key in g:
            a = _safe_arr(g[actual_key])
            if a is not None: return a.reshape(-1), "1d"
    return None, ""

@dataclass
class CyclePtr:
    file_idx: int; stride_side: str; cycle_name: str

@dataclass
class LoadedFile:
    meta: FileMeta; data: Dict[str, Any]; cycles: List[CyclePtr]

class PlotCell(QtWidgets.QWidget):
    selectionMade = QtCore.Signal(list, str)
    keyChanged = QtCore.Signal()
    def __init__(self, parent, coord_index: Optional[int], combo_width_chars: int = 7,
                 key_resolver: Optional[Callable[[str, str], str]] = None):
        super().__init__(parent)
        self.coord_index = coord_index
        self.current_display_key = ""
        self.key_resolver = key_resolver or (lambda disp, stride: disp)
        self.fig = Figure(figsize=(3,2), layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.nav = NavigationToolbar(self.canvas, self)
        self.nav.setIconSize(QtCore.QSize(14,14))
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(False)
        fm = self.combo.fontMetrics()
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo.setMinimumContentsLength(combo_width_chars)
        self.combo.setMinimumWidth(int(fm.horizontalAdvance("W")*combo_width_chars*1.05))
        self.combo.currentIndexChanged.connect(self._on_param_changed)
        top = QtWidgets.QHBoxLayout()
        top.addStretch(1); top.addWidget(self.combo); top.addWidget(self.nav)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.addLayout(top); lay.addWidget(self.canvas)
        self.selector=None
        self._cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
    def _on_scroll_zoom(self, event):
        ax = self.ax
        if event.inaxes != ax: return
        base_scale = 1.2; scale = (1/base_scale) if event.button=='up' else base_scale
        for axis in (ax.xaxis, ax.yaxis):
            lim = ax.get_xlim() if axis is ax.xaxis else ax.get_ylim()
            xdata = event.xdata if axis is ax.xaxis else event.ydata
            if xdata is None: continue
            left = xdata - (xdata - lim[0]) * scale
            right = xdata + (lim[1] - xdata) * scale
            if axis is ax.xaxis: ax.set_xlim(left, right)
            else: ax.set_ylim(left, right)
        self.canvas.draw_idle()
    def set_options(self, displays: Sequence[str], default_display: Optional[str] = None) -> None:
        prev = self.combo.currentText().strip()
        self.combo.blockSignals(True); self.combo.clear(); self.combo.addItems(list(displays))
        if default_display and default_display in displays: self.combo.setCurrentText(default_display)
        elif prev in displays: self.combo.setCurrentText(prev)
        elif displays: self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False); self.current_display_key = self.combo.currentText().strip()
    def _on_param_changed(self, idx: int):
        self.current_display_key = self.combo.currentText().strip()
        self.keyChanged.emit()
    def clear(self):
        self.ax.cla()
        if self.selector is not None: self.selector.disconnect(); self.selector=None
        self.canvas.draw_idle()
    def autoscale(self):
        lines = [ln for ln in self.ax.lines if ln.get_visible()]
        if not lines: return
        ys = []
        for ln in lines:
            y = np.asarray(ln.get_ydata(orig=False)); y = y[np.isfinite(y)]
            if y.size: ys.append((float(np.min(y)), float(np.max(y))))
        if not ys: return
        ymin = min(lo for lo,hi in ys); ymax = max(hi for lo,hi in ys)
        if not np.isfinite([ymin, ymax]).all(): return
        if ymax <= ymin:
            span = 1.0; self.ax.set_ylim(ymin - 0.5*span, ymax + 0.5*span)
        else:
            span = ymax - ymin; pad = max(0.05*span, 1e-6)
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.canvas.draw_idle()
    def plot(self, files: List[LoadedFile], display_key: str, show_left: bool, show_right: bool,
             filter_kinetic: bool, filter_kinematic: bool):
        self.clear(); self.current_display_key = display_key; line_map: Dict[Any, SelectionItem] = {}
        def style_line(ln, stride_side: str, cdict: Dict[str, Any]):
            ln.set_color(LEFT_COLOR if stride_side=="left_stride" else RIGHT_COLOR)
            ms = int(cdict.get("manually_selected",1)); ln._meta = {"manually_selected": ms}
            if ms==0: ln.set_alpha(0.2)
            if _is_qc_bad(cdict): ln.set_linestyle("--")
        for fi, lf in enumerate(files):
            dfile = lf.data
            for stride_side in ("left_stride","right_stride"):
                if stride_side=="left_stride" and not show_left: continue
                if stride_side=="right_stride" and not show_right: continue
                block = dfile.get(stride_side, {})
                if not isinstance(block, dict): continue
                actual_key = self.key_resolver(display_key, stride_side)
                for cyc_name, cdict in block.items():
                    kin = is_kinetic(cdict)
                    if kin and not filter_kinetic: continue
                    if (not kin) and not filter_kinematic: continue
                    arr, kind = resolve_key(cdict, actual_key)
                    if arr is None: continue
                    if kind=="analog_many":
                        for nm, a in arr:
                            a101 = _timenorm(a, is_3d=False)
                            if a101 is None: continue
                            xs = np.linspace(0.0,100.0,a101.shape[0])
                            ln, = self.ax.plot(xs, a101, lw=1.0, alpha=1.0); style_line(ln, stride_side, cdict)
                            line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=nm)
                    elif kind=="3d":
                        a101 = _timenorm(arr, is_3d=True); 
                        if a101 is None: continue
                        xs = np.linspace(0.0,100.0,a101.shape[0])
                        coords = range(3) if self.coord_index is None else [self.coord_index]
                        for j in coords:
                            yy = a101[:,j]; ln, = self.ax.plot(xs, yy, lw=1.2, alpha=1.0); style_line(ln, stride_side, cdict)
                            line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=f"{actual_key}[{j}]")
                    else:
                        a101 = _timenorm(arr, is_3d=False); 
                        if a101 is None: continue
                        xs = np.linspace(0.0,100.0,a101.shape[0]); ln, = self.ax.plot(xs, a101, lw=1.2, alpha=1.0)
                        style_line(ln, stride_side, cdict)
                        line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=actual_key)
        self.ax.set_ylabel(display_key); self.ax.set_xlabel("% cycle"); self.canvas.draw_idle()
        self.selector = SelectionManager(self.ax, line_map, mode="select", on_result=self._on_rect); self.selector.set_active(True)
    def _on_rect(self, items: List[SelectionItem], mode: str, bbox):
        self.selectionMade.emit(items, mode)

class PlotPage(QtWidgets.QWidget):
    selectionApplied = QtCore.Signal(list)
    def __init__(self, parent, rows: int, cols: int, coord_map: List[Optional[int]],
                 key_resolver: Callable[[str, str], str]):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self); grid.setContentsMargins(6,6,6,6); grid.setSpacing(6)
        self.cells: List[PlotCell] = []; k=0
        for r in range(rows):
            for c in range(cols):
                cell = PlotCell(self, coord_index=coord_map[k], combo_width_chars=7, key_resolver=key_resolver)
                cell.selectionMade.connect(self._on_cell_sel)
                cell.keyChanged.connect(self._on_key_changed)
                self.cells.append(cell); grid.addWidget(cell, r, c); k+=1
    def set_options_all(self, displays: Sequence[str], defaults: Sequence[str]) -> None:
        for i, cell in enumerate(self.cells):
            default = defaults[i] if i < len(defaults) and defaults[i] in displays else (displays[0] if displays else None)
            cell.set_options(displays, default_display=default)
    def redraw(self, files: List[LoadedFile], show_left: bool, show_right: bool,
               filter_kinetic: bool, filter_kinematic: bool) -> None:
        for cell in self.cells:
            disp = cell.current_display_key or (cell.combo.currentText().strip() if cell.combo.count() else "")
            if not disp and cell.combo.count(): disp = cell.combo.itemText(0)
            if disp: cell.plot(files, disp, show_left, show_right, filter_kinetic, filter_kinematic)
    @QtCore.Slot()
    def _on_key_changed(self):
        self.parent().parent().redraw_all(autoscale=True)
    @QtCore.Slot(list, str)
    def _on_cell_sel(self, items: List[SelectionItem], mode: str):
        new_val = 1 if mode=="select" else 0; changes=[]; seen=set()
        for it in items:
            fi, stride_side, cyc = it.cycle_ref; k=(fi, stride_side, cyc)
            if k in seen: continue
            seen.add(k); changes.append((fi, stride_side, cyc, new_val))
        if changes: self.selectionApplied.emit(changes)

@dataclass
class LoadedFile:
    meta: FileMeta; data: Dict[str, Any]; cycles: List[CyclePtr]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cycle Selection Tool — Step 4"); self.resize(1700,980)
        self.root_edit = QtWidgets.QLineEdit(); self.btn_root = QtWidgets.QPushButton("Set Root…"); self.btn_root.clicked.connect(self.on_choose_root)
        self.cmb_pid = QtWidgets.QComboBox(); self.cmb_pid.currentTextChanged.connect(self.on_pid_changed)
        self.cmb_trial = QtWidgets.QComboBox(); self.cmb_trial.currentTextChanged.connect(self.on_trial_changed)
        self.chk_left = QtWidgets.QCheckBox("left_stride data"); self.chk_left.setChecked(True)
        self.chk_right = QtWidgets.QCheckBox("right_stride data"); self.chk_right.setChecked(True)
        self.chk_kinetic = QtWidgets.QCheckBox("show kinetic cycles"); self.chk_kinetic.setChecked(True)
        self.chk_kinematic = QtWidgets.QCheckBox("show kinematic cycles"); self.chk_kinematic.setChecked(False)
        for chk in (self.chk_left,self.chk_right,self.chk_kinetic,self.chk_kinematic):
            chk.toggled.connect(lambda _=None: self.redraw_all(autoscale=True))
        self.lbl_mode = QtWidgets.QLabel("Mode: SELECT")
        self.btn_autoscale = QtWidgets.QPushButton("↕︎ Autoscale Y (all)"); self.btn_autoscale.clicked.connect(lambda: self.autoscale_all())
        tb = QtWidgets.QToolBar()
        tb.addWidget(QtWidgets.QLabel("Root:")); tb.addWidget(self.root_edit); tb.addWidget(self.btn_root)
        tb.addSeparator(); tb.addWidget(QtWidgets.QLabel("PID:")); tb.addWidget(self.cmb_pid)
        tb.addWidget(QtWidgets.QLabel("Trial:")); tb.addWidget(self.cmb_trial)
        tb.addSeparator(); tb.addWidget(self.chk_left); tb.addWidget(self.chk_right)
        tb.addSeparator(); tb.addWidget(self.chk_kinetic); tb.addWidget(self.chk_kinematic)
        tb.addSeparator(); tb.addWidget(self.lbl_mode); tb.addSeparator(); tb.addWidget(self.btn_autoscale)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.key_templates: Dict[str, KeyTemplate] = {}; self.display_keys: List[str] = []
        coord_map1 = [0]*6 + [1]*6 + [2]*6; coord_map2 = [None]*(3*6)
        self.tabs = QtWidgets.QTabWidget()
        self.page1 = PlotPage(self, 3, 6, coord_map1, key_resolver=self.to_actual_key)
        self.page2 = PlotPage(self, 3, 6, coord_map2, key_resolver=self.to_actual_key)
        self.tabs.addTab(self.page1, "Page 1"); self.tabs.addTab(self.page2, "Page 2"); self.setCentralWidget(self.tabs)
        self.status_widget = StatusWidget(self); self.setStatusBar(self.status_widget)
        self.participants: List[str] = []; self.trial_types_by_pid: Dict[str, List[str]] = {}; self.files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}; self.loaded: List[LoadedFile] = []
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=lambda: self.set_mode("select"))
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self.set_mode("deselect"))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+U"), self, activated=self.autoscale_all)
        self.current_mode = "select"
        self.page1.selectionApplied.connect(self.apply_selection); self.page2.selectionApplied.connect(self.apply_selection)
        env_root = os.getenv("CYCLE_TOOL_ROOT")
        if env_root and os.path.isdir(env_root): self.root_edit.setText(env_root); self.set_root(env_root)
        self.default_displays_page1 = self._default_displays_page1(); self.default_displays_page2 = self._default_displays_page2()
    def set_mode(self, mode: str):
        self.current_mode = "select" if mode=="select" else "deselect"; self.lbl_mode.setText(f"Mode: {self.current_mode.upper()}")
        for page in (self.page1,self.page2):
            for cell in page.cells:
                if cell.selector: cell.selector.set_mode(self.current_mode)
        self.autoscale_all()
    @Slot()
    def on_choose_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose root folder")
        if not path: return
        self.root_edit.setText(path); self.set_root(path)
    def set_root(self, root: str):
        print(f"[root] {root}")
        files = scan_root(root); print(f"[scan] matched {len(files)} files")
        self.participants, self.trial_types_by_pid, self.files_by_pid_type = build_indices(files)
        self.cmb_pid.blockSignals(True); self.cmb_pid.clear(); self.cmb_pid.addItems(self.participants); self.cmb_pid.blockSignals(False)
        if self.participants: self.cmb_pid.setCurrentIndex(0); self.on_pid_changed(self.participants[0])
        else:
            self.cmb_trial.clear(); self.loaded=[]; self.display_keys=[]; self._refresh_counts(); self.redraw_all(autoscale=True)
    def on_pid_changed(self, pid: str):
        types = self.trial_types_by_pid.get(pid, []); print(f"[pid] {pid} types={types}")
        self.cmb_trial.blockSignals(True); self.cmb_trial.clear(); self.cmb_trial.addItems(types); self.cmb_trial.blockSignals(False)
        if types: self.cmb_trial.setCurrentIndex(0); self.on_trial_changed(types[0])
        else:
            self.loaded=[]; self.display_keys=[]; self._refresh_counts(); self.redraw_all(autoscale=True)
    def on_trial_changed(self, trial: str):
        pid = self.cmb_pid.currentText().strip(); metas = self.files_by_pid_type.get(pid, {}).get(trial, [])
        print(f"[trial] {pid}/{trial}: files={len(metas)}"); self.loaded = []
        for i, fm in enumerate(metas):
            try:
                d = load_dict(fm.path); cycles: List[CyclePtr] = []
                for stride_side in ("left_stride","right_stride"):
                    block = d.get(stride_side, {})
                    if isinstance(block, dict):
                        for cyc in block.keys(): cycles.append(CyclePtr(file_idx=i, stride_side=stride_side, cycle_name=cyc))
                self.loaded.append(LoadedFile(meta=fm, data=d, cycles=cycles)); print(f"  - {os.path.basename(fm.path)} cycles={len(cycles)}")
            except Exception as e:
                print(f"  !! failed {fm.path}: {e!r}")
        templates = discover_key_templates(self.loaded); self.key_templates = templates; self.display_keys = sorted(templates.keys())
        self.page1.set_options_all(self.display_keys, self._default_displays_page1()); self.page2.set_options_all(self.display_keys, self._default_displays_page2())
        self._refresh_counts(); self.redraw_all(autoscale=True)
    def to_actual_key(self, display: str, stride_side: str) -> str:
        tpl = self.key_templates.get(display); 
        if not tpl: return display
        return tpl.to_side(stride_side)
    def _refresh_counts(self):
        items = []
        for lf in self.loaded:
            d = lf.data
            for stride_side in ("left_stride","right_stride"):
                block = d.get(stride_side, {})
                if not isinstance(block, dict): continue
                for cyc, cdict in block.items():
                    items.append({"manually_selected": int(cdict.get("manually_selected",1)), "kinetic": 1 if is_kinetic(cdict) else 0})
        totals = compute_counts(items); self.status_widget.update_from_counts(totals)
    def redraw_all(self, autoscale: bool = False):
        show_left = self.chk_left.isChecked(); show_right = self.chk_right.isChecked()
        filter_kinetic = self.chk_kinetic.isChecked(); filter_kinematic = self.chk_kinematic.isChecked()
        self.page1.redraw(self.loaded, show_left, show_right, filter_kinetic, filter_kinematic)
        self.page2.redraw(self.loaded, show_left, show_right, filter_kinetic, filter_kinematic)
        if autoscale: self.autoscale_all()
    def autoscale_all(self):
        for page in (self.page1,self.page2):
            for cell in page.cells: cell.autoscale()
    @QtCore.Slot(list)
    def apply_selection(self, changes: List[tuple]):
        changed = 0
        for fi, stride_side, cyc, val in changes:
            if fi<0 or fi>=len(self.loaded): continue
            d = self.loaded[fi].data; block = d.get(stride_side, {})
            if not isinstance(block, dict): continue
            cdict = block.get(cyc, {})
            try: old = int(cdict.get("manually_selected",1))
            except Exception: old = 1
            if old != int(val):
                cdict["manually_selected"] = int(val); changed += 1
        if changed:
            print(f"[select] applied {changed} change(s)"); self._refresh_counts(); self.redraw_all(autoscale=True)
    def _default_displays_page1(self) -> List[str]:
        return ["HipAngles","KneeAngles","AnkleAngles","HipMoment","KneeMoment","AnkleMoment"]*3
    def _default_displays_page2(self) -> List[str]:
        return [
            "Force_Fx*","hip_on_femur_in_femur_fx","med_cond_weld_on_tibial_plat_in_tibial_plat_fx","ankle_on_talus_in_talus_fx","IK_markerErr.total_squared_error","SO_forces.FX",
            "Force_Fy*","hip_on_femur_in_femur_fy","med_cond_weld_on_tibial_plat_in_tibial_plat_fy","ankle_on_talus_in_talus_fy","IK_markerErr.marker_error_RMS","SO_forces.FY",
            "Force_Fz*","hip_on_femur_in_femur_fz","med_cond_weld_on_tibial_plat_in_tibial_plat_fz","ankle_on_talus_in_talus_fz","IK_markerErr.marker_error_max","SO_forces.FZ",
        ]

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(); w.show(); app.exec()

if __name__ == "__main__":
    main()
