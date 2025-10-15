
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QMessageBox, QInputDialog
from pathlib import Path
import json

from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .file_index import scan_root, build_indices, FileMeta
from .io_mat import load_dict, save_dict_check
from .selection import SelectionManager, SelectionItem
from .status import compute_counts, StatusWidget, compute_counts_by_side_and_mode
from . import emg as emg_mod
from . import audit_log
from .history import History, SelectionEdit
from . import plotting
from . import qc

@dataclass
class PageSpec:
    name: str
    rows: int
    cols: int
    coord_map: List[Optional[int]]
    defaults: List[str]
    xlabel: str = "% cycle"

_re_layout_idx = re.compile(r"layout_page(\d+)\.json$", re.IGNORECASE)

def _to_generic_display(opt: str) -> str:
    """
    Convert 'LHipAngles'/'RHipAngles' -> 'HipAngles'. Leave others unchanged.
    """
    if not isinstance(opt, str):
        return ""
    return re.sub(r"^[LR](?=[A-Z])", "", opt).strip()

def _discover_layout_jsons() -> List[Path]:
    here = Path(__file__).resolve().parent
    files = list(here.glob("layout_page*.json"))
    def _idx(p: Path) -> int:
        m = _re_layout_idx.search(p.name)
        return int(m.group(1)) if m else 10_000
    return sorted(files, key=_idx)

def _parse_layout_json(p: Path) -> PageSpec:
    js = json.loads(p.read_text(encoding="utf-8"))
    name = js.get("name") or p.stem.replace("_", " ").title()
    rows = int(js.get("rows", 3))
    cols = int(js.get("cols", 6))
    xlabel = js.get("x_axis_label", "% cycle")
    cells = js.get("cells") or []
    if rows == 3:
        # allow per-row defaults via 'default_by_row': [x, y, z] (or dict with keys 'x','y','z')
        col_defaults_xyz: List[List[str]] = []
        for c in range(cols):
            cell = cells[c] if c < len(cells) else {}
            by_row = cell.get("default_by_row") or cell.get("defaults_by_row")
            if isinstance(by_row, dict):
                triplet = [by_row.get("x") or by_row.get("row1"),
                           by_row.get("y") or by_row.get("row2"),
                           by_row.get("z") or by_row.get("row3")]
            elif isinstance(by_row, (list, tuple)):
                vals = list(by_row)
                while len(vals) < 3:
                    vals.append(vals[-1] if vals else "")
                triplet = vals[:3]
            else:
                base = cell.get("default") or cell.get("title") or ""
                triplet = [base, base, base]
            col_defaults_xyz.append([_to_generic_display(v or "") for v in triplet])

        # flatten into row-major order: row0 all cols, then row1, then row2
        defaults: List[str] = []
        for row_idx in range(3):
            for c in range(cols):
                defaults.append(col_defaults_xyz[c][row_idx])
        coord_map = [0]*cols + [1]*cols + [2]*cols
    else:
        coord_map = [None] * (rows * cols)
        defaults = []
        for r in range(rows):
            for c in range(cols):
                cell = cells[c] if c < len(cells) else {}
                default = cell.get("default") or cell.get("title") or ""
                defaults.append(_to_generic_display(default))
    return PageSpec(name=name, rows=rows, cols=cols, coord_map=coord_map, defaults=defaults, xlabel=xlabel)

def _load_page_specs_from_json() -> List[PageSpec]:
    specs: List[PageSpec] = []
    for p in _discover_layout_jsons():
        try:
            specs.append(_parse_layout_json(p))
        except Exception as e:
            print(f"[layout] skip {p.name}: {e!r}")
    return specs
    
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
    pattern: str     # 'prefix','token','trail','emg','emg_many','none','analog_many'
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
        if self.pattern=="emg_many":
            return re.sub(r"_(?:L|R)_", "_L_" if side=="left" else "_R_", self.sample)
        if self.pattern=="analog_many":
            return self.analog_glob or self.sample
        return self.sample

def _display_from_key(key: str) -> Tuple[str, KeyTemplate]:
    m = re.match(r"^(EMG)_(\d+)_([LR])_(.+)$", key)
    if m:
        base, num, side, rest = m.groups()
        disp = f"{base}_*_{rest}"
        sample = f"{base}_*_{side}_{rest}"
        return disp, KeyTemplate(display=disp, pattern="emg_many", sample=sample)
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
        analog = cdict.get("analog", {}) if isinstance(cdict.get("analog", {}), dict) else {}
        names = _analog_many_names(analog, actual_key); pairs=[]
        fs = _analog_fs(analog) if isinstance(analog, dict) else 1000.0
        for nm in names:
            a = _safe_arr(analog.get(nm))
            if a is None: continue
            try:
                if emg_mod.is_emg_key(nm):
                    a = emg_mod.process_emg(a, fs)
            except Exception:
                pass
            pairs.append((nm, np.asarray(a).reshape(-1)))
        #print(f"[analog_many] key='{actual_key}' matched={len(pairs)}")
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
        self.fig = Figure(figsize=(3,2), layout="tight"); self.canvas = FigureCanvas(self.fig); self.ax = self.fig.add_subplot(111)
        self.nav = NavigationToolbar(self.canvas, self); self._strip_nav_ctrl_s(); self.nav.setIconSize(QtCore.QSize(14,14))
        self._hover_pick_tol = 6  # pixels: picking tolerance for hover tooltips
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._last_hovered = None  # remember last hovered line to reduce flicker
        self.combo = QtWidgets.QComboBox(); self.combo.setEditable(False)
        fm = self.combo.fontMetrics(); self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo.setMinimumContentsLength(combo_width_chars)
        self.combo.setMinimumWidth(int(fm.horizontalAdvance("W")*combo_width_chars*1.05))
        self.combo.currentIndexChanged.connect(self._on_param_changed)
        top = QtWidgets.QHBoxLayout(); top.addStretch(1); top.addWidget(self.combo); top.addWidget(self.nav)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.addLayout(top); lay.addWidget(self.canvas)
        self.selector=None
        self._cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
    def _strip_nav_ctrl_s(self):
        """Remove Ctrl+S from the Matplotlib NavigationToolbar to avoid conflicts."""
        for act in self.nav.actions():
            try:
                txt = (act.text() or "").lower()
            except Exception:
                txt = ""
            if "save" in txt:  # matches "&Save", "Save the figure", etc.
                act.setShortcuts([])
                act.setShortcut(QtGui.QKeySequence())  # clear
                act.setShortcutVisibleInContextMenu(False)
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

    def _format_tooltip(self, meta: Dict[str, Any]) -> str:
        """Build a compact tooltip from a line's metadata."""
        fn = meta.get("filename", "?")
        trial = meta.get("trial_type", "?")
        side = meta.get("side", "?")
        cyc = meta.get("cycle_no", "?")
        param = meta.get("param", "")
        coord = meta.get("coord", "")
        # --- insert below line above ---
        time_info = meta.get("time", "")
        first_line = str(fn)
        bits = [str(trial), str(side), f"{cyc}"]
        if param:
            bits.append(str(param))
        if coord:
            bits.append(str(coord))
        if time_info:
            bits.append(str(time_info))
        return first_line + "\n" + " • ".join(bits)

    def _on_motion(self, event) -> None:
        """Hover handler: show a tooltip when the cursor is close to a line."""
        # Only for our axes
        if event.inaxes is not self.ax:
            self._hide_tooltip()
            return
        # Find first visible line under the cursor
        hit = None
        for ln in self.ax.lines:
            if not ln.get_visible():
                continue
            try:
                contains, _ = ln.contains(event)
            except Exception:
                contains = False
            if contains:
                hit = ln
                break
        if hit is None:
            self._hide_tooltip()
            return
        if self._last_hovered is hit:
            return  # unchanged
        self._last_hovered = hit
        meta = getattr(hit, "_meta", {}) or {}
        tip = self._format_tooltip(meta)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), tip)

    def _hide_tooltip(self) -> None:
        if self._last_hovered is not None:
            self._last_hovered = None
            QtWidgets.QToolTip.hideText()
    def _time_meta_from_cycle(self, cdict: Dict[str, Any], group_hint: Optional[str], npts: int) -> str:
        """
        Build a compact time/index string for tooltips.
        - If a 'time' vector exists under the hinted group (analog/point), use its min/max (s).
        - Otherwise, fall back to indices only.
        """
        try:
            if group_hint == "analog":
                tvec = np.asarray(cdict.get("analog", {}).get("time", []))
                fvec = np.asarray(cdict.get("analog", {}).get("frames", []))
                idx0, idx1 = int(np.nanmin(fvec)), int(np.nanmax(fvec))
            elif group_hint == "point":
                # Some datasets store point time too; if missing we’ll fall back gracefully.
                tvec = np.asarray(cdict.get("point", {}).get("time", []))
                fvec = np.asarray(cdict.get("point", {}).get("frames", []))
                idx0, idx1 = int(np.nanmin(fvec)), int(np.nanmax(fvec))
        except Exception:
            tvec = None
            idx0 = 0
            idx1 = max(0, int(npts) - 1)

        if isinstance(tvec, np.ndarray) and tvec.size > 0 and np.isfinite(tvec).any():
            t0 = float(np.nanmin(tvec))
            t1 = float(np.nanmax(tvec))
            return f"{t0:.3f} s (idx {idx0})—{t1:.3f} s (idx {idx1})"
        return f"idx {idx0}—{idx1}"
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
        # Use helper that autoscale both X & Y for visible & selected lines
        lines = list(self.ax.lines)
        updated = plotting.autoscale_y_from_selected(lines, margin_pct=0.05)
        if not updated:
            # Fallback: at least refresh
            self.canvas.draw_idle()
    def plot(self, files: List[LoadedFile], display_key: str, show_left: bool, show_right: bool,
             filter_kinetic: bool, filter_kinematic: bool, current_mode: str,
             hide_deselected: bool = False):
        self.clear(); self.current_display_key = display_key; line_map: Dict[Any, SelectionItem] = {}
        def style_line(ln, stride_side: str, cdict: Dict[str, Any]):
            ln.set_color(LEFT_COLOR if stride_side=="left_stride" else RIGHT_COLOR)
            ms = int(cdict.get("manually_selected",1)); ln._meta = {"manually_selected": ms}
            if ms==0: ln.set_alpha(0.2)
            if _is_qc_bad(cdict): ln.set_linestyle("--")
            # NEW: hide de-selected lines completely when toggle is on
            ln.set_visible(False if (hide_deselected and int(ln._meta.get("manually_selected", 1)) == 0) else True)
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
                            ln._meta = getattr(ln, "_meta", {}) or {}
                            ln._meta.update({
                                "filename": Path(lf.meta.path).name,
                                "trial_type": lf.meta.original_trial_type,
                                "side": stride_side,
                                "cycle_no": cyc_name,
                                "param": nm,
                                "coord": "",
                                "time": self._time_meta_from_cycle(cdict, group_hint="analog", npts=a101.shape[0]),
                                "reconstruction_ok": int(cdict.get("reconstruction_ok", 1)) == 1,
                                "IK_RMS_ok": bool(cdict.get("IK_RMS_ok", True)),
                                "IK_MAX_ok": bool(cdict.get("IK_MAX_ok", True)),
                                "SO_F_RMS_ok": bool(cdict.get("SO_F_RMS_ok", True)),
                                "SO_F_MAX_ok": bool(cdict.get("SO_F_MAX_ok", True)),
                                "SO_M_RMS_ok": bool(cdict.get("SO_M_RMS_ok", True)),
                                "SO_M_MAX_ok": bool(cdict.get("SO_M_MAX_ok", True)),
                            })
                            ln.set_pickradius(self._hover_pick_tol)
                            line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=nm)
                    elif kind=="3d":
                        a101 = _timenorm(arr, is_3d=True); 
                        if a101 is None: continue
                        xs = np.linspace(0.0,100.0,a101.shape[0])
                        coords = range(3) if self.coord_index is None else [self.coord_index]
                        time_meta = self._time_meta_from_cycle(cdict, group_hint="point", npts=a101.shape[0])
                        for j in coords:
                            yy = a101[:,j]; ln, = self.ax.plot(xs, yy, lw=1.2, alpha=1.0); style_line(ln, stride_side, cdict)
                            coord_label = ("x", "y", "z")[j]
                            ln._meta = getattr(ln, "_meta", {}) or {}
                            ln._meta.update({
                                "filename": Path(lf.meta.path).name,
                                "trial_type": lf.meta.original_trial_type,
                                "side": stride_side,
                                "cycle_no": cyc_name,
                                "param": actual_key,                        # base key (e.g., "LHipAngles")
                                "coord": coord_label,                       # "x"/"y"/"z"
                                "time": time_meta,
                                "reconstruction_ok": int(cdict.get("reconstruction_ok", 1)) == 1,
                                "IK_RMS_ok": bool(cdict.get("IK_RMS_ok", True)),
                                "IK_MAX_ok": bool(cdict.get("IK_MAX_ok", True)),
                                "SO_F_RMS_ok": bool(cdict.get("SO_F_RMS_ok", True)),
                                "SO_F_MAX_ok": bool(cdict.get("SO_F_MAX_ok", True)),
                                "SO_M_RMS_ok": bool(cdict.get("SO_M_RMS_ok", True)),
                                "SO_M_MAX_ok": bool(cdict.get("SO_M_MAX_ok", True)),

                            })
                            ln.set_pickradius(self._hover_pick_tol)
                            line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=f"{actual_key}[{j}]")
                    else:
                        a101 = _timenorm(arr, is_3d=False); 
                        if a101 is None: continue
                        xs = np.linspace(0.0,100.0,a101.shape[0]); ln, = self.ax.plot(xs, a101, lw=1.2, alpha=1.0)
                        style_line(ln, stride_side, cdict)
                        ln._meta = getattr(ln, "_meta", {}) or {}
                        ln._meta.update({
                            "filename": Path(lf.meta.path).name,
                            "trial_type": lf.meta.original_trial_type,
                            "side": stride_side,
                            "cycle_no": cyc_name,
                            "param": actual_key,                        # the selected signal key
                            "coord": "",                                # 1-D signal
                            "time": self._time_meta_from_cycle(cdict, group_hint="point", npts=a101.shape[0]),
                            "reconstruction_ok": int(cdict.get("reconstruction_ok", 1)) == 1,
                            "IK_RMS_ok": bool(cdict.get("IK_RMS_ok", True)),
                            "IK_MAX_ok": bool(cdict.get("IK_MAX_ok", True)),
                            "SO_F_RMS_ok": bool(cdict.get("SO_F_RMS_ok", True)),
                            "SO_F_MAX_ok": bool(cdict.get("SO_F_MAX_ok", True)),
                            "SO_M_RMS_ok": bool(cdict.get("SO_M_RMS_ok", True)),
                            "SO_M_MAX_ok": bool(cdict.get("SO_M_MAX_ok", True)),
                        })
                        ln.set_pickradius(self._hover_pick_tol)
                        line_map[ln] = SelectionItem(cycle_ref=(fi, stride_side, cyc_name), param=actual_key)
        self.ax.set_ylabel(display_key); self.ax.set_xlabel("% cycle"); self.canvas.draw_idle()
        self.selector = SelectionManager(self.ax, line_map, mode=("deselect" if current_mode=="deselect" else "select"), on_result=self._on_rect); self.selector.set_active(True)
    def _on_rect(self, items: List[SelectionItem], mode: str, bbox):
        #print(f"[select] emitting {len(items)} items, mode={mode}")
        self.selectionMade.emit(items, mode)

class PlotPage(QtWidgets.QWidget):
    selectionApplied = QtCore.Signal(list); keyChanged = QtCore.Signal()
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
               filter_kinetic: bool, filter_kinematic: bool, current_mode: str,
               hide_deselected: bool = False) -> None:
        for cell in self.cells:
            disp = cell.current_display_key or (cell.combo.currentText().strip() if cell.combo.count() else "")
            if not disp and cell.combo.count(): disp = cell.combo.itemText(0)
            if disp: cell.plot(files, disp, show_left, show_right, filter_kinetic, filter_kinematic, current_mode, hide_deselected)
    @QtCore.Slot()
    def _on_key_changed(self):
        print("[page] key changed; requesting window redraw+autoscale")
        self.keyChanged.emit()
    @QtCore.Slot(list, str)
    def _on_cell_sel(self, items: List[SelectionItem], mode: str):
        new_val = 1 if mode=="select" else 0; changes=[]; seen=set()
        for it in items:
            fi, stride_side, cyc = it.cycle_ref; k=(fi, stride_side, cyc)
            if k in seen: continue
            seen.add(k); changes.append((fi, stride_side, cyc, new_val))
        if changes:
            #print(f"[page] selection changes={len(changes)}"); 
            self.selectionApplied.emit(changes)

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
        self.lbl_mode.setToolTip("Mode Select (S) or Deselect (D)")
        self.btn_autoscale = QtWidgets.QPushButton("↔︎↕︎ Autoscale")
        self.btn_autoscale.clicked.connect(lambda: self.autoscale_all())
        self.btn_autoscale.setToolTip("Autoscale X & Y (Ctrl+U)")
        self.act_hide_deselected = QtGui.QAction("Hide Deselected", self)
        self.act_hide_deselected.setCheckable(True)
        self.act_hide_deselected.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        self.act_hide_deselected.setShortcutContext(Qt.ApplicationShortcut)
        self.act_hide_deselected.setToolTip("Hide deselected (Ctrl+H)")
        # On toggle, replot (and autoscale to currently visible lines)
        self.act_hide_deselected.toggled.connect(lambda _: self.redraw_all(autoscale=True))
        self.addAction(self.act_hide_deselected)  # ensure shortcut works app-wide
        tb = QtWidgets.QToolBar()
        tb.addWidget(QtWidgets.QLabel("Root:")); tb.addWidget(self.root_edit); tb.addWidget(self.btn_root)
        tb.addSeparator(); tb.addWidget(QtWidgets.QLabel("PID:")); tb.addWidget(self.cmb_pid)
        tb.addWidget(QtWidgets.QLabel("Trial:")); tb.addWidget(self.cmb_trial)
        tb.addSeparator(); tb.addWidget(self.chk_left); tb.addWidget(self.chk_right)
        tb.addSeparator(); tb.addWidget(self.chk_kinetic); tb.addWidget(self.chk_kinematic)
        tb.addSeparator(); tb.addWidget(self.lbl_mode)
        tb.addSeparator(); tb.addAction(self.act_hide_deselected)
        tb.addSeparator(); tb.addWidget(self.btn_autoscale)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.key_templates: Dict[str, KeyTemplate] = {}; self.display_keys: List[str] = []
        coord_map1 = [0]*6 + [1]*6 + [2]*6; coord_map2 = [None]*(3*6)
        self.tabs = QtWidgets.QTabWidget()
        self.pages: List[Tuple["PlotPage", "PageSpec"]] = []
        specs = _load_page_specs_from_json()
        if not specs:
            specs = [
                PageSpec(
                    name="Page 1",
                    rows=3, cols=6,
                    coord_map=[0]*6 + [1]*6 + [2]*6,
                    defaults=["HipAngles","KneeAngles","AnkleAngles","HipMoment","KneeMoment","AnkleMoment"]*3,
                    xlabel="% cycle",
                ),
                PageSpec(
                    name="Page 2",
                    rows=3, cols=6,
                    coord_map=[None]*(3*6),
                    defaults=[
                        "Force_Fx*","hip_on_femur_in_femur_fx","med_cond_weld_on_tibial_plat_in_tibial_plat_fx","ankle_on_talus_in_talus_fx","IK_markerErr.total_squared_error","SO_forces.FX",
                        "Force_Fy*","hip_on_femur_in_femur_fy","med_cond_weld_on_tibial_plat_in_tibial_plat_fy","ankle_on_talus_in_talus_fy","IK_markerErr.marker_error_RMS","SO_forces.FY",
                        "Force_Fz*","hip_on_femur_in_femur_fz","med_cond_weld_on_tibial_plat_in_tibial_plat_fz","ankle_on_talus_in_talus_fz","IK_markerErr.marker_error_max","SO_forces.FZ",
                    ],
                    xlabel="% cycle",
                ),
            ]
        for spec in specs:
            page = PlotPage(self, spec.rows, spec.cols, spec.coord_map, key_resolver=self.to_actual_key)
            self.pages.append((page, spec))
            self.tabs.addTab(page, spec.name)
        self.setCentralWidget(self.tabs)
        for page, _ in self.pages:
            page.keyChanged.connect(lambda: self.redraw_all(autoscale=True))
        self.status_widget = StatusWidget(self); self.setStatusBar(self.status_widget)
        self.participants: List[str] = []; self.trial_types_by_pid: Dict[str, List[str]] = {}; self.files_by_pid_type: Dict[str, Dict[str, List[FileMeta]]] = {}; self.loaded: List[LoadedFile] = []
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=lambda: self.set_mode("select"))
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self.set_mode("deselect"))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+U"), self, activated=self.autoscale_all)
        self.current_mode = "select"
        for page, _ in self.pages:
            page.selectionApplied.connect(self.apply_selection)
        env_root = os.getenv("CYCLE_TOOL_ROOT")
        if env_root and os.path.isdir(env_root): self.root_edit.setText(env_root); self.set_root(env_root)
        # --- Saving / audit state ---
        self._username: str = ""
        self._baseline: dict[tuple, int] = {}   # (file_idx, side, cycle_name) -> manually_selected (0/1)
        self._navigating_programmatic = False   # suppress prompts during programmatic combo changes
        # Ctrl+S shortcut
        self.act_save = QtGui.QAction("Save _check.mat", self)
        self.act_save.setShortcut(QtGui.QKeySequence.Save)     # Ctrl+S / Cmd+S
        self.act_save.setShortcutContext(Qt.ApplicationShortcut)  # optional, works app-wide
        self.act_save.triggered.connect(self.save_now)
        self.addAction(self.act_save)  # enables the shortcut even without focusing the menu
        # --- Menu ---
        m = self.menuBar().addMenu("&File")
        m.addAction(self.act_save)
        m.addSeparator()
        act_user = m.addAction("Set Username…")
        act_user.triggered.connect(self._prompt_and_set_username)
        edit = self.menuBar().addMenu("&Edit")
        self.act_undo = edit.addAction("Undo")
        self.act_undo.setShortcuts(QtGui.QKeySequence.Undo)   # Ctrl+Z / Cmd+Z
        self.act_undo.triggered.connect(self._on_undo)

        self.act_redo = edit.addAction("Redo")
        self.act_redo.setShortcuts(QtGui.QKeySequence.Redo)   # Ctrl+Shift+Z (Win/Linux), Cmd+Shift+Z (macOS)
        self.act_redo.triggered.connect(self._on_redo)

        # >>> Make shortcuts application-wide and register on the window <<<
        self.act_undo.setShortcutContext(Qt.ApplicationShortcut)
        self.act_redo.setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.act_undo)
        self.addAction(self.act_redo)
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)

        self.history = History()
        # a store the SelectionEdit can mutate
        self._selection_store = {}  # { (fi, side, cyc): 0/1 }
    
    def _apply_qc_flags(self, data_dict: Dict[str, Any]) -> None:
        """
        Evaluate per-cycle QC flags (IK and SO) in-place on `data_dict`.

        - Adds/keeps: reconstruction_ok (placeholder True), IK_RMS_ok, IK_MAX_ok,
                      SO_F_RMS_ok, SO_F_MAX_ok, SO_M_RMS_ok, SO_M_MAX_ok
        - Flags are consumed by plotting (dashed linestyle if any fail).
        """
        meta = data_dict.get("meta", {})  # for SO timing & sampling info
        for side in ("left_stride", "right_stride"):
            block = data_dict.get(side, {})
            if not isinstance(block, dict):
                continue
            for cyc_name, cyc in block.items():
                if not isinstance(cyc, dict):
                    continue

                # Ensure baseline keys exist
                cyc.setdefault("manually_selected", 1)
                cyc.setdefault("reconstruction_ok", True)  # placeholder pass

                # IK flags (max over time vs thresholds)
                try:
                    cyc.update(qc.eval_ik_flags(cyc))
                except Exception:
                    # missing or malformed IK -> evaluator treats as pass, but guard anyway
                    cyc.setdefault("IK_RMS_ok", True)
                    cyc.setdefault("IK_MAX_ok", True)

                # SO flags (RMS & MAX in contralateral SLS window)
                try:
                    # evaluator needs the cycle dict, the indexing side, and meta for sampling
                    cyc.update(qc.eval_so_flags(cyc, side, meta))
                except Exception:
                    cyc.setdefault("SO_F_RMS_ok", True)
                    cyc.setdefault("SO_F_MAX_ok", True)
                    cyc.setdefault("SO_M_RMS_ok", True)
                    cyc.setdefault("SO_M_MAX_ok", True)
    def set_mode(self, mode: str):
        self.current_mode = "select" if mode=="select" else "deselect"; self.lbl_mode.setText(f"Mode: {self.current_mode.upper()}")
        for page, _ in self.pages:
            for cell in page.cells:
                if cell.selector: cell.selector.set_mode(self.current_mode)
        fm = self.loaded[0].meta if self.loaded else None
        if fm:
            fname = Path(fm.path).name
            self.status_widget.showMessage(
                f"Mode: {self.current_mode.upper()} — {fname} • {fm.original_trial_type}", 1500
            )
        else:
            self.status_widget.showMessage(f"Mode: {self.current_mode.upper()}", 1500)
        self.autoscale_all()
    @Slot()
    def on_choose_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose root folder")
        if not path: return
        self.root_edit.setText(path); self.set_root(path)
    def set_root(self, root: str):
        #print(f"[root] {root}")
        files = scan_root(root); #print(f"[scan] matched {len(files)} files")
        self.participants, self.trial_types_by_pid, self.files_by_pid_type = build_indices(files)
        self.cmb_pid.blockSignals(True); self.cmb_pid.clear(); self.cmb_pid.addItems(self.participants); self.cmb_pid.blockSignals(False)
        if self.participants: self.cmb_pid.setCurrentIndex(0); self.on_pid_changed(self.participants[0])
        else:
            self.cmb_trial.clear(); self.loaded=[]; self.display_keys=[]; self._refresh_counts(); self.redraw_all(autoscale=True)
    def on_pid_changed(self, pid: str):
        if self._navigating_programmatic is False and self.loaded:
            if not self._maybe_prompt_save():
                # Revert the combobox change
                self._navigating_programmatic = True
                # put back previous PID (the one in the baseline context)
                cur = self.cmb_pid.findText(self._last_pid or "", Qt.MatchExactly)
                if cur >= 0:
                    self.cmb_pid.setCurrentIndex(cur)
                self._navigating_programmatic = False
                return
        self._last_pid = pid  # remember for potential revert next time

        types = self.trial_types_by_pid.get(pid, [])
        #print(f"[pid] {pid} types={types}")
        self.cmb_trial.blockSignals(True)
        self.cmb_trial.clear()
        self.cmb_trial.addItems(types)
        self.cmb_trial.blockSignals(False)
        if types:
            self.cmb_trial.setCurrentIndex(0)
            self.on_trial_changed(types[0])
        else:
            self.loaded = []
            self.display_keys = []
            self._refresh_counts()
            self.redraw_all(autoscale=True)
            self._rebuild_baseline()  # nothing loaded but keep state consistent

    def on_trial_changed(self, trial: str):
        if self._navigating_programmatic is False and self.loaded:
            if not self._maybe_prompt_save():
                # Revert the combobox change
                self._navigating_programmatic = True
                cur = self.cmb_trial.findText(self._last_trial or "", Qt.MatchExactly)
                if cur >= 0:
                    self.cmb_trial.setCurrentIndex(cur)
                self._navigating_programmatic = False
                return
        self._last_trial = trial  # remember for potential revert next time

        pid = self.cmb_pid.currentText().strip()
        metas = self.files_by_pid_type.get(pid, {}).get(trial, [])
        #print(f"[trial] {pid}/{trial}: files={len(metas)}")
        self.loaded = []
        for i, fm in enumerate(metas):
            try:
                d = load_dict(fm.path)
                self._apply_qc_flags(d) # populate IK/SO flags per cycle so lines can style themselves
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
        self._ensure_manually_selected_defaults()
        templates = discover_key_templates(self.loaded)
        self.key_templates = templates
        self.display_keys = sorted(templates.keys())
        for page, spec in self.pages:
            page.set_options_all(self.display_keys, spec.defaults)
        self._refresh_counts()
        self.redraw_all(autoscale=True)

        # NEW: establish a fresh baseline for “dirty” detection after loading
        self._rebuild_baseline()
    def _ensure_manually_selected_defaults(self) -> None:
        """Guarantee that every cycle has 'manually_selected' set to 1."""
        for lf in self.loaded:
            for side in ("left_stride", "right_stride"):
                block = lf.data.get(side, {})
                if not isinstance(block, dict):
                    continue
                for cyc_name, cdict in block.items():
                    # only set if missing, do not overwrite user decisions
                    if "manually_selected" not in cdict:
                        cdict["manually_selected"] = 1

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
        hide_deselected = bool(getattr(self, "act_hide_deselected", None) and self.act_hide_deselected.isChecked())
        for page, _ in self.pages:
            page.redraw(self.loaded, show_left, show_right, filter_kinetic, filter_kinematic, self.current_mode, hide_deselected)
        if autoscale: self.autoscale_all()
    def autoscale_all(self):
        for page, _ in self.pages:
            for cell in page.cells:
                cell.autoscale()
    @QtCore.Slot(list)
    def apply_selection(self, changes: List[tuple]):
        """
        changes: list of (fi, stride_side, cyc, new_val) gathered from rectangle selection.
        Convert to History command so we can Undo/Redo.
        """
        cmd_changes = []
        for fi, stride_side, cyc, new_val in changes:
            if fi < 0 or fi >= len(self.loaded):
                continue
            block = self.loaded[fi].data.get(stride_side, {})
            if not isinstance(block, dict):
                continue
            cdict = block.get(cyc, {})
            try:
                old_val = int(cdict.get("manually_selected", 1))
            except Exception:
                old_val = 1
            new_val = int(new_val)
            if old_val == new_val:
                continue
            key = (fi, stride_side, cyc)  # use a tuple as the stable key
            cmd_changes.append((key, old_val, new_val))

        if not cmd_changes:
            return

        cmd = SelectionEdit(
            store=self._selection_store,
            changes=cmd_changes,
            description="Rectangle select",
            on_apply=self._apply_selection_by_key,
        )
        self.history.push(cmd)          # applies .do() and resets redo branch
        self._post_edit_refresh()

    def _default_displays_page1(self) -> List[str]:
        return ["HipAngles","KneeAngles","AnkleAngles","HipMoment","KneeMoment","AnkleMoment"]*3
    def _default_displays_page2(self) -> List[str]:
        return [
            "Force_Fx*","hip_on_femur_in_femur_fx","med_cond_weld_on_tibial_plat_in_tibial_plat_fx","ankle_on_talus_in_talus_fx","IK_markerErr.total_squared_error","SO_forces.FX",
            "Force_Fy*","hip_on_femur_in_femur_fy","med_cond_weld_on_tibial_plat_in_tibial_plat_fy","ankle_on_talus_in_talus_fy","IK_markerErr.marker_error_RMS","SO_forces.FY",
            "Force_Fz*","hip_on_femur_in_femur_fz","med_cond_weld_on_tibial_plat_in_tibial_plat_fz","ankle_on_talus_in_talus_fz","IK_markerErr.marker_error_max","SO_forces.FZ",
        ]
    def _snapshot_selection(self) -> dict[tuple, int]:
        """
        Return a mapping (file_idx, side, cycle_name) -> manually_selected(0/1)
        across all currently loaded files.
        """
        snap: dict[tuple, int] = {}
        for fi, lf in enumerate(self.loaded):
            d = lf.data
            for side in ("left_stride", "right_stride"):
                block = d.get(side, {})
                if not isinstance(block, dict):
                    continue
                for cyc_name, cdict in block.items():
                    try:
                        val = int(cdict.get("manually_selected", 1))
                    except Exception:
                        val = 1
                    snap[(fi, side, str(cyc_name))] = 1 if val == 1 else 0
        return snap

    def _rebuild_baseline(self) -> None:
        self._baseline = self._snapshot_selection()
        self._seed_selection_store()

    def _seed_selection_store(self) -> None:
        """Mirror current selection state into the command store for History."""
        self._selection_store = dict(self._snapshot_selection())  # keys: (fi, side, cyc)

    def _apply_selection_by_key(self, key: tuple, val: int) -> None:
        """Callback used by SelectionEdit to apply a single change."""
        fi, side, cyc = key
        if fi < 0 or fi >= len(self.loaded):
            return
        cdict = self.loaded[fi].data.get(side, {}).get(cyc)
        if not isinstance(cdict, dict):
            return
        cdict["manually_selected"] = int(val)

    def _post_edit_refresh(self) -> None:
        """Refresh counts and plots after a batch of edits/undo/redo."""
        self._refresh_counts()
        self.redraw_all(autoscale=False)
        self._update_undo_redo_enabled()

    def _is_dirty(self) -> bool:
        if not self._baseline:
            return False
        current = self._snapshot_selection()
        # if any key differs (value or new/removed key), treat as dirty
        keys = set(self._baseline.keys()) | set(current.keys())
        for k in keys:
            if self._baseline.get(k, -1) != current.get(k, -1):
                return True
        return False

    def _changes_breakdown(self) -> dict[str, dict[str, int]]:
        """
        Count cycles whose *final selection state changed* since baseline,
        broken down by side ('left'/'right') and mode ('kinetic'/'kinematic').
        """
        breakdown = {
            "left": {"kinetic": 0, "kinematic": 0},
            "right": {"kinetic": 0, "kinematic": 0},
        }
        cur = self._snapshot_selection()
        for (fi, side, cyc_name), cur_val in cur.items():
            base_val = self._baseline.get((fi, side, cyc_name), cur_val)
            if cur_val == base_val:
                continue
            # which mode is this cycle?
            try:
                cdict = self.loaded[fi].data.get(side, {}).get(cyc_name, {})
            except Exception:
                cdict = {}
            mode = "kinetic" if is_kinetic(cdict) else "kinematic"
            lr = "left" if "left" in side.lower() else "right"
            breakdown[lr][mode] += 1
        return breakdown

    def _final_counts_by_side_mode(self) -> dict[str, dict[str, dict[str, int]]]:
        """
        Use status.compute_counts_by_side_and_mode to produce the final counts for logging.
        """
        items = []
        for lf in self.loaded:
            d = lf.data
            for side in ("left_stride", "right_stride"):
                block = d.get(side, {})
                if not isinstance(block, dict):
                    continue
                for cyc_name, cdict in block.items():
                    items.append({
                        "manually_selected": int(cdict.get("manually_selected", 1)),
                        "kinetic": 1 if is_kinetic(cdict) else 0,
                        "side": side,
                    })
        return compute_counts_by_side_and_mode(items)

    def _ensure_username(self) -> str:
        if self._username.strip():
            return self._username.strip()
        return self._prompt_and_set_username()

    def _prompt_and_set_username(self) -> str:
        root = self.root_edit.text().strip()
        try:
            known = audit_log.list_usernames(root) if root else []
        except Exception:
            known = []
        # Use an editable dropdown so they can pick or type a new one
        name, ok = QInputDialog.getItem(self, "Select Username",
                                        "Choose or enter your username:",
                                        known or [], 0, True)
        if ok:
            self._username = (name or "").strip()
        return self._username
    @Slot()
    def save_now(self) -> None:
        """
        Write sibling *_check.mat for each loaded file and append one audit row
        (per PID×trial-type) if any selection changes occurred since last baseline.
        """
        self._ensure_manually_selected_defaults()

        if not self.loaded:
            QMessageBox.information(self, "Save", "Nothing loaded to save.")
            return

        # Username (one-time prompt)
        user = self._ensure_username() or "unknown"

        # Compute changes + final counts before we mutate baseline
        changes_breakdown = self._changes_breakdown()
        final_counts = self._final_counts_by_side_mode()

        saved_paths = []
        for lf in self.loaded:
            try:
                outp = save_dict_check(lf.meta.path, lf.data)
                saved_paths.append(str(outp))
            except Exception as e:
                QMessageBox.warning(self, "Save error", f"Failed to save:\n{lf.meta.path}\n\n{e!r}")

        # Audit log only if something actually changed
        any_change = any(v for side in changes_breakdown.values() for v in side.values())
        if any_change:
            try:
                audit_log.log_selection_change(
                    root=self.root_edit.text().strip(),
                    pid=self.cmb_pid.currentText().strip(),
                    trial_type=self.cmb_trial.currentText().strip(),
                    username=user,
                    changes_breakdown=changes_breakdown,
                    final_counts=final_counts,
                )
            except Exception as e:
                # Non-fatal; saving must not be blocked by logging
                print(f"[audit_log] warning: {e!r}")

        # New baseline after successful save
        self._rebuild_baseline()

        # Inform user
        if saved_paths:
            msg = "\n".join(os.path.basename(p) for p in saved_paths)
            QMessageBox.information(self, "Saved", f"Wrote:\n{msg}")
    def _maybe_prompt_save(self) -> bool:
        """
        Ask user to save if dirty. Returns True to proceed, False to cancel.
        """
        if not self._is_dirty():
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Unsaved changes")
        box.setText("You have unsaved selection edits.")
        box.setInformativeText("Save changes before switching?")
        btn_save = box.addButton("Save", QMessageBox.AcceptRole)
        btn_discard = box.addButton("Don't Save", QMessageBox.DestructiveRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_save)
        box.exec()

        clicked = box.clickedButton()
        if clicked is btn_save:
            self.save_now()
            return True
        if clicked is btn_discard:
            # Drop edits by adopting current state as new baseline
            self._rebuild_baseline()
            return True
        return False  # cancel
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._maybe_prompt_save():
            event.accept()
        else:
            event.ignore()
    # helper: build a stable key; e.g., "left::cycle7"
    def _cycle_id(self, side: str, name: str) -> str:
        s = "left" if "left" in side.lower() or side.lower()=="l" else "right"
        return f"{s}::{name}"


    #def _apply_selection(self, cycle_id: str, val: int) -> None:
    #    # update model
    #    cyc = self._cycles_by_id[cycle_id]              # your own mapping: id -> cycle dict
    #    cyc["manually_selected"] = int(val)

        # update any cached store so redo/undo stays consistent
    #    self._selection_store[cycle_id] = int(val)

        # refresh visuals for lines that belong to this cycle (restyle alpha/dashes)
    #    self._restyle_lines_for_cycle(cycle_id)

        # refresh counts/status bar
        #self._update_status_counts()
    # during plot construction, when you create SelectionManager per-axes:
    #def _on_rect_result(items, mode, bbox):
    #    new_val = 1 if mode == "select" else 0
    #    changes = []
    #    for it in items:                      # it: SelectionItem(cycle_ref=..., param=...)
    #        side = it.cycle_ref.side         # or however you track side on your ref
    #        name = it.cycle_ref.cycle        # the cycle name/number
    #        cid = self._cycle_id(side, name)
    #        old = int(self._cycles_by_id[cid].get("manually_selected", 1))
    #        if old != new_val:
    #            changes.append((cid, old, new_val))

    #    if not changes:
    #        return

    #    cmd = SelectionEdit(
    #        store=self._selection_store,
    #        changes=changes,
    #        description="Rectangle select",
    #        on_apply=lambda k, v: self._apply_selection(k, v),
    #    )
    #    self.history.push(cmd)        # applies .do() and truncates redo branch
    #    self._update_undo_redo_enabled()
    #def _update_status_counts(self):
    #    # recompute using your Status helpers across all visible cycles
    #    counts = compute_counts(self._iter_all_cycles())  # or your current source of cycles
    #    self.status_widget.update_counts(counts)          # if you added such method
    #    # or just update labels directly
    @QtCore.Slot()
    def _on_undo(self):
        print("[undo] triggered")
        if self.history.undo() is not False:   # treat None as success
            self._post_edit_refresh()
        else:
            self._update_undo_redo_enabled()

    @QtCore.Slot()
    def _on_redo(self):
        print("[redo] triggered")
        if self.history.redo() is not False:   # treat None as success
            self._post_edit_refresh()
        else:
            self._update_undo_redo_enabled()

    def _update_undo_redo_enabled(self):
        def _as_bool(obj, stack_hint_names=("undo_stack", "_undo_stack")):
            # 1) method?
            if callable(obj):
                try:
                    return bool(obj())
                except TypeError:
                    pass
            # 2) bool/int property?
            if isinstance(obj, (bool, int)):
                return bool(obj)
            # 3) introspect stacks if present
            for nm in stack_hint_names:
                st = getattr(self.history, nm, None)
                if st is not None:
                    try:
                        return len(st) > 0
                    except Exception:
                        pass
            # 4) last resort (optimistic)
            return True

        can_undo_attr = getattr(self.history, "can_undo", None)
        can_redo_attr = getattr(self.history, "can_redo", None)

        # Try common redo stack names too
        can_undo = _as_bool(can_undo_attr, ("undo_stack", "_undo_stack"))
        can_redo = _as_bool(can_redo_attr, ("redo_stack", "_redo_stack"))

        self.act_undo.setEnabled(bool(can_undo))
        self.act_redo.setEnabled(bool(can_redo))



def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(); w.show(); app.exec()

if __name__ == "__main__":
    main()
