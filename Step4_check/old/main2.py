from __future__ import annotations

import os, json, re, itertools, functools
from typing import Dict, Any, List, Tuple, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

# matplotlib (no pyplot)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as MplToolbar

# Project modules
from . import settings as settings_mod          # << use function API (no Settings class)
from .io_mat import load_dict, save_dict_check, derive_check_path
from .file_index import scan_root, build_indices
from .cycles import collect_cycles              # returns CycleRef(side, cycle, groups, flags)
from .time_norm import resample_to_101
from .emg import is_emg_key, process_emg
from . import qc
from . import status as status_mod
from .audit_log import list_usernames as audit_list_usernames, log_selection_change as audit_log_selection_change

# -------------------------- Small helpers --------------------------

def deep_get(d: Dict[str,Any], dotted: str) -> Optional[Any]:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def side_name_from_key(side_key: str) -> str:
    return "left" if "left" in side_key else "right"

# -------------------------- Plot page/widgets (unchanged API) --------------------------
# NOTE: These are trimmed for brevity; keep your existing PlotCell/PlotPage classes.

class PlotCell(QtWidgets.QWidget):
    linePicked = QtCore.Signal(object)  # (cycle_ref)
    selectionChanged = QtCore.Signal()

    def __init__(self, parent, row_idx: int, col_idx: int, options: List[str], default_key: str, title: str):
        super().__init__(parent)
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.options = options
        self.current_key = default_key
        self.title = title

        self.fig = Figure(figsize=(3,2), layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(self.options)
        if default_key in self.options:
            self.combo.setCurrentText(default_key)
        self.combo.currentTextChanged.connect(self._on_param_changed)

        self.btn_ylim = QtWidgets.QToolButton()
        self.btn_ylim.setText("↕︎")
        self.btn_ylim.setToolTip("New y-lim from selected lines (Ctrl+U)")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(title))
        top.addStretch(1)
        top.addWidget(self.combo)
        top.addWidget(self.btn_ylim)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2,2,2,2)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

        self.lines = []

    def _on_param_changed(self, key: str):
        self.current_key = key
        self.selectionChanged.emit()

class PlotPage(QtWidgets.QWidget):
    requestYAutoscale = QtCore.Signal()
    def __init__(self, parent, layout_json_path: str):
        super().__init__(parent)
        with open(layout_json_path, "r") as f:
            self.cfg = json.load(f)
        self.rows = int(self.cfg.get("rows",3))
        self.cols = int(self.cfg.get("cols",6))
        self.cells_cfg = self.cfg.get("cells", [])
        assert len(self.cells_cfg) in (self.cols, self.rows*self.cols), "layout cells must be cols or rows*cols long"

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(6,6,6,6)
        grid.setSpacing(6)

        self.cells: List[PlotCell] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if len(self.cells_cfg) == self.cols:
                    cell_cfg = self.cells_cfg[c]
                else:
                    cell_cfg = self.cells_cfg[r*self.cols + c]
                title = cell_cfg.get("title", f"r{r+1}c{c+1}")
                options = cell_cfg.get("options", [])
                default = cell_cfg.get("default", options[0] if options else "")
                pc = PlotCell(self, r, c, options, default, title)
                self.cells.append(pc)
                grid.addWidget(pc, r, c)

        for cell in self.cells:
            cell.btn_ylim.clicked.connect(self.requestYAutoscale)

# -------------------------- Status bar wrapper --------------------------

class StatusBar(status_mod.StatusWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

# -------------------------- Username dialog --------------------------

class UsernameDialog(QtWidgets.QDialog):
    def __init__(self, root_dir: str, initial: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Username")
        self.setModal(True)
        v = QtWidgets.QVBoxLayout(self)
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Known usernames:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(audit_list_usernames(root_dir))
        h1.addWidget(self.combo)
        v.addLayout(h1)

        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Username:"))
        self.edit = QtWidgets.QLineEdit()
        self.edit.setText(initial)
        h2.addWidget(self.edit)
        v.addLayout(h2)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self.combo.currentTextChanged.connect(self.edit.setText)

    def value(self) -> str:
        txt = self.edit.text().strip()
        if not txt:
            txt = self.combo.currentText().strip()
        return txt

# -------------------------- Main window --------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycle Selection Tool — Step4 Check")
        self.setObjectName("MainWindow")
        self.setMinimumSize(1200, 700)

        # settings: use function API, not a non-existent Settings class
        self.settings_dict: Dict[str, Any] = settings_mod.load()
        if settings_mod.get("ui.start_fullscreen", True):
            self.showMaximized()   # value comes from settings.json :contentReference[oaicite:5]{index=5}

        self.username: Optional[str] = None
        self.root_dir: Optional[str] = None
        self.is_dirty: bool = False
        self.saved_once: bool = False

        # file indices (fix: nested mapping pid -> trial_type -> [FileMeta])
        self.participants: List[str] = []
        self.trial_types_by_pid: Dict[str, List[str]] = {}
        self.files_by_pid_type: Dict[str, Dict[str, list]] = {}  # matches build_indices(...) :contentReference[oaicite:6]{index=6}

        # current selection
        self.current_pid: Optional[str] = None
        self.current_trial_type: Optional[str] = None

        # loaded data
        self.file_dicts: List[Dict[str,Any]] = []
        # cycles: list of (CycleRef-from-cycles.py, pointer to cycle dict)
        self.cycles: List[Tuple[Any, Dict[str,Any]]] = []
        self.initial_snapshot: Dict[Tuple[int,str,str], int] = {}  # (file_idx, side, cycle) -> manually_selected

        # central UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(6,6,6,6)

        # top toolbar: PID, TrialType, checkboxes
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Participant ID:"))
        self.combo_pid = QtWidgets.QComboBox()
        self.combo_pid.currentTextChanged.connect(self._on_pid_changed)
        top.addWidget(self.combo_pid, 2)

        top.addWidget(QtWidgets.QLabel("Trial-type:"))
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        top.addWidget(self.combo_type, 2)

        top.addSpacing(16)
        self.cb_left = QtWidgets.QCheckBox("Left");    self.cb_left.setChecked(True)
        self.cb_right = QtWidgets.QCheckBox("Right");  self.cb_right.setChecked(True)
        self.cb_kin = QtWidgets.QCheckBox("Kinetic");  self.cb_kin.setChecked(True)
        self.cb_kinem = QtWidgets.QCheckBox("Kinematic"); self.cb_kinem.setChecked(False)
        for cb in (self.cb_left, self.cb_right, self.cb_kin, self.cb_kinem):
            cb.stateChanged.connect(self.redraw_all)
        top.addWidget(self.cb_left); top.addWidget(self.cb_right)
        top.addWidget(self.cb_kin);  top.addWidget(self.cb_kinem)
        top.addStretch(1)
        v.addLayout(top)

        # Tabs with two pages
        here = os.path.dirname(__file__)
        self.tabs = QtWidgets.QTabWidget()
        self.page1 = PlotPage(self, os.path.join(here, "layout_page1.json"))
        self.page2 = PlotPage(self, os.path.join(here, "layout_page2.json"))
        self.tabs.addTab(self.page1, "Page 1")
        self.tabs.addTab(self.page2, "Page 2")
        v.addWidget(self.tabs, 1)

        # shared mpl toolbar with New y-lim button
        self.current_toolbar: Optional[MplToolbar] = None
        self._attach_toolbar(self.page1)
        self.tabs.currentChanged.connect(lambda idx: self._attach_toolbar(self.tabs.widget(idx)))

        # Status bar (counts)
        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)

        # actions / menu
        self._build_menu()

        # keyboard shortcuts: KEEP ONLY non-duplicated ones (S/D), move the rest to menu actions
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=lambda: self._set_selection_mode("select"))
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self._set_selection_mode("deselect"))
        # DO NOT add QShortcuts for Ctrl+S / Ctrl+Z / Ctrl+Shift+Z / Ctrl+U here — that caused the ambiguous warnings. :contentReference[oaicite:7]{index=7}

        # selection mode
        self.selection_mode: Optional[str] = None
        self._set_selection_mode(None)

        # connect per-page autoscale buttons
        self.page1.requestYAutoscale.connect(self.autoscale_all_y)
        self.page2.requestYAutoscale.connect(self.autoscale_all_y)

        # history (unchanged)
        from .history import History, SelectionEdit
        self.History = History
        self.SelectionEdit = SelectionEdit
        self.history = History()

        # wire updates when parameter combos change
        for page in (self.page1, self.page2):
            for cell in page.cells:
                cell.selectionChanged.connect(self.redraw_all)

        self._closing = False

    # ---------- Menu ----------

    def _build_menu(self):
        bar = self.menuBar()

        m_file = bar.addMenu("&File")
        act_open = m_file.addAction("Open &Root…")
        act_open.triggered.connect(self.choose_root)
        m_file.addSeparator()
        act_save = m_file.addAction("&Save")
        act_save.setShortcut("Ctrl+S")               # keep Save on menu
        act_save.triggered.connect(self.save)
        m_file.addSeparator()
        act_quit = m_file.addAction("Quit")
        act_quit.triggered.connect(self.close)

        m_edit = bar.addMenu("&Edit")
        act_undo = m_edit.addAction("Undo")
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self.undo)
        act_redo = m_edit.addAction("Redo")
        act_redo.setShortcut("Ctrl+Shift+Z")
        act_redo.triggered.connect(self.redo)

        m_view = bar.addMenu("&View")
        act_ylim = m_view.addAction("New y-lim from selected")
        act_ylim.setShortcut("Ctrl+U")
        act_ylim.triggered.connect(self.autoscale_all_y)

        m_settings = bar.addMenu("&Settings")
        act_open_settings = m_settings.addAction("Open Settings…")
        act_open_settings.triggered.connect(self.open_settings_dialog)
        act_reset_defaults = m_settings.addAction("Reset Defaults")
        act_reset_defaults.triggered.connect(self.reset_settings)

        m_user = bar.addMenu("&User")
        act_set_user = m_user.addAction("Set Username…")
        act_set_user.triggered.connect(self.prompt_username)

    def _attach_toolbar(self, page: PlotPage):
        # Remove existing toolbar if any
        if self.current_toolbar is not None:
            self.current_toolbar.setParent(None)
            self.current_toolbar.deleteLater()
            self.current_toolbar = None
        # Attach toolbar to the first cell's canvas on the page
        if page.cells:
            tb = MplToolbar(page.cells[0].canvas, self)

            # Avoid shortcut collisions with our app's menu actions:
            # strip conflicting shortcuts from the Matplotlib toolbar (notably Ctrl+S).
            for act in tb.actions():
                try:
                    seq = act.shortcut().toString()
                except Exception:
                    seq = ""
                if seq in ("Ctrl+S", "Ctrl+Z", "Ctrl+Shift+Z", "Ctrl+U"):
                    act.setShortcut(QtGui.QKeySequence())  # clear

            # Add a custom y-lim action WITHOUT a shortcut (menu already owns Ctrl+U)
            custom = QtGui.QAction("New y-lim", self)
            custom.setToolTip("Compute y-limits from selected lines for all subplots")
            custom.triggered.connect(self.autoscale_all_y)
            tb.addAction(custom)

            self.addToolBar(Qt.BottomToolBarArea, tb)
            self.current_toolbar = tb

    # ---------- Root & indices ----------

    def choose_root(self):
        if self.is_dirty and not self.saved_once:
            if not self._confirm_discard_or_save():
                return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Root Folder", self.root_dir or os.path.expanduser("~"))
        if d:
            self.root_dir = d
            self._scan_indices()
            self._populate_pid_type_combos()  # will auto-select first PID/type

    def _scan_indices(self):
        files = scan_root(self.root_dir)
        # build_indices returns: participants, trial_types_by_pid, files_by_pid_type (nested)
        self.participants, self.trial_types_by_pid, self.files_by_pid_type = build_indices(files)  # :contentReference[oaicite:8]{index=8}

    def _populate_pid_type_combos(self):
        self.combo_pid.blockSignals(True)
        self.combo_type.blockSignals(True)
        self.combo_pid.clear()
        self.combo_type.clear()
        self.combo_pid.addItems(list(self.participants))
        self.combo_pid.blockSignals(False)
        self.combo_type.blockSignals(False)

        # Auto-select first PID (will cascade to types)
        if self.participants:
            self.combo_pid.setCurrentIndex(0)

    def _on_pid_changed(self, pid: str):
        if not pid:
            return
        if self.is_dirty and not self.saved_once and pid != self.current_pid:
            if not self._confirm_discard_or_save():
                idx = self.combo_pid.findText(self.current_pid or "", Qt.MatchExactly)
                if idx >= 0:
                    self.combo_pid.setCurrentIndex(idx)
                return
        self.current_pid = pid
        types = self.trial_types_by_pid.get(pid, [])
        self.combo_type.blockSignals(True)
        self.combo_type.clear()
        self.combo_type.addItems(list(types))
        self.combo_type.blockSignals(False)
        # Auto-select first trial type so the plots appear immediately
        if types:
            self.combo_type.setCurrentIndex(0)

    def _on_type_changed(self, ttype: str):
        if not ttype:
            return
        if self.is_dirty and not self.saved_once and ttype != self.current_trial_type:
            if not self._confirm_discard_or_save():
                idx = self.combo_type.findText(self.current_trial_type or "", Qt.MatchExactly)
                if idx >= 0:
                    self.combo_type.setCurrentIndex(idx)
                return
        self.current_trial_type = ttype
        if self.current_pid and self.current_trial_type:
            self.load_pid_type(self.current_pid, self.current_trial_type)

    # ---------- Load & prepare ----------

    def load_pid_type(self, pid: str, ttype: str):
        # FIX: files_by_pid_type is nested: pid -> trial_type -> [FileMeta]
        files = self.files_by_pid_type.get(pid, {}).get(ttype, [])

        self.file_dicts = []
        for fm in files:
            try:
                self.file_dicts.append(load_dict(fm.path))
            except Exception as e:
                print("Failed to load", getattr(fm, "path", "<unknown>"), e)

        # collect cycles
        self.cycles.clear()
        self.initial_snapshot.clear()

        for fi, fd in enumerate(self.file_dicts):
            file_cycles = collect_cycles(fd, files[fi])  # returns cycles.CycleRef (with .side and .cycle) :contentReference[oaicite:9]{index=9}
            for cref in file_cycles:
                side_dict = fd.get(cref.side, {})  # 'left_stride' or 'right_stride'
                cyc = side_dict.get(cref.cycle, {})
                key = (fi, cref.side, cref.cycle)   # use .cycle, not .cycle_key
                init_val = int(cyc.get("manually_selected", 1))
                self.initial_snapshot[key] = init_val

                # Append pair (CycleRef, pointer to actual dict) for downstream code
                self.cycles.append((cref, cyc))

                # apply QC flags if available
                try:
                    qc.apply_qc_flags(cyc, fd.get("meta", {}), self.settings_dict)
                except Exception:
                    pass

        self.is_dirty = False
        self.saved_once = False
        self._update_title_dirty()
        self.redraw_all()
        self._refresh_counts()

    # ---------- Redraw / counts / selection mode ----------
    # (keep your existing implementations; signatures below are placeholders)

    def redraw_all(self):
        show_left = self.cb_left.isChecked()
        show_right = self.cb_right.isChecked()
        show_kin = self.cb_kin.isChecked()
        show_kinem = self.cb_kinem.isChecked()

        # page iteration
        for page in (self.page1, self.page2):
            for cell in page.cells:
                ax = cell.ax
                ax.cla()
                lines = []
                key = cell.current_key
                # loop cycles and draw
                for (cref, cyc) in self.cycles:
                    side = side_name_from_key(cref.side)
                    if side=="left" and not show_left: 
                        continue
                    if side=="right" and not show_right:
                        continue
                    mode = "kinetic" if int(cref.kinetic)==1 else "kinematic"
                    if mode=="kinetic" and not show_kin:
                        continue
                    if mode=="kinematic" and not show_kinem:
                        continue

                    arr, kind = find_param_array(cyc, key)
                    if arr is None:
                        continue

                    # 3d vs 1d vs analog_many
                    fs_analog = float(self.file_dicts[cref.file_idx].get("meta",{}).get("header",{}).get("analogs",{}).get("frame_rate", 1000.0))
                    if kind=="analog_many":
                        for name,val in arr:
                            yy = timenorm_array(apply_emg_if_needed(name, val, fs_analog), is_3d=False)
                            if yy is None: 
                                continue
                            xx = [i for i in range(len(yy))]
                            ln, = ax.plot(xx, yy, linewidth=1.0, alpha=1.0)
                            ln._cycle_ref = cref
                            ln._meta = {"file": os.path.basename(self.files_by_pid_type[(self.current_pid,self.current_trial_type)][cref.file_idx].path),
                                        "trial_type": self.current_trial_type,
                                        "side": side,
                                        "cycle": cref.cycle_key,
                                        "param": name}
                            lines.append((ln, cref, ln._meta))
                        continue

                    if kind=="3d":
                        # choose coord by row index: 0=x,1=y,2=z
                        arr = timenorm_array(arr, True)
                        yy = arr[:, min(cell.row_idx, 2)]
                    else:
                        arr = apply_emg_if_needed(key, arr, fs_analog)
                        yy = timenorm_array(arr, False)
                    if yy is None:
                        continue
                    xx = [i for i in range(len(yy))]

                    # style by side & selection
                    color = "red" if side=="left" else "blue"
                    alpha = 1.0 if int(cyc.get("manually_selected",1))==1 else 0.2
                    linestyle = "-"  # could be dashed if QC flags fail
                    # QC flags: dashed if any *_ok flag is False
                    flags = [v for k,v in cyc.items() if isinstance(v,bool) and k.endswith("_ok")]
                    if any(f is False for f in flags):
                        linestyle = "--"

                    ln, = ax.plot(xx, yy, linestyle=linestyle, alpha=alpha, color=color, linewidth=1.5)
                    # hover tooltip meta
                    ln._cycle_ref = cref
                    ln._meta = {"file": os.path.basename(self.files_by_pid_type[(self.current_pid,self.current_trial_type)][cref.file_idx].path),
                                "trial_type": self.current_trial_type,
                                "side": side,
                                "cycle": cref.cycle_key,
                                "param": key}
                    lines.append((ln, cref, ln._meta))

                # labels
                ax.set_xlabel(self.page1.cfg.get("x_axis_label","% cycle"))
                ax.set_ylabel(cell.current_key)

                # store lines for selection use
                cell.lines = lines
                cell.canvas.draw_idle()

        self._refresh_counts()

    def autoscale_all_y(self):
        margin_pct = float(self.settings.get("ui","ylim_margin_pct", default=0.05))
        for page in (self.page1, self.page2):
            for cell in page.cells:
                ax = cell.ax
                # compute min/max over selected visible lines
                ymin, ymax = None, None
                for (ln, cref, meta) in cell.lines:
                    cyc_dict = self.file_dicts[cref.file_idx].get(cref.side,{}).get(cref.cycle_key,{})
                    if int(cyc_dict.get("manually_selected",1)) != 1:
                        continue
                    ydata = ln.get_ydata()
                    y0, y1 = float(min(ydata)), float(max(ydata))
                    if not math.isfinite(y0) or not math.isfinite(y1):
                        continue
                    ymin = y0 if ymin is None else min(ymin, y0)
                    ymax = y1 if ymax is None else max(ymax, y1)
                if ymin is not None and ymax is not None and ymin!=ymax:
                    rng = ymax - ymin
                    pad = rng * margin_pct
                    ax.set_ylim(ymin - pad, ymax + pad)
                cell.canvas.draw_idle()

    def _refresh_counts(self):
        # Build a cycles list in the shape expected by status.py helpers
        cycles_repr = []
        for (cref, cyc) in self.cycles:
            side = side_name_from_key(cref.side)
            mode = "kinetic" if int(cyc.get("kinetic",1))==1 else "kinematic"
            sel = int(cyc.get("manually_selected",1))==1
            cycles_repr.append({"side": side, "mode": mode, "selected": sel})
        try:
            counts = status_mod.compute_counts(cycles_repr)
            self.status_bar.update_counts(counts)
        except Exception:
            pass

    def _set_selection_mode(self, mode: Optional[str]):
        self.selection_mode = mode

    # ---------- Save / Undo / Redo ----------
    def save(self):
        if not self.file_dicts:
            return
        if not self.username:
            who = self.prompt_username()
            if not who:
                return

        # Save each file as *_check.mat (always persist data)
        files = self.files_by_pid_type.get((self.current_pid, self.current_trial_type), [])
        for i, fd in enumerate(self.file_dicts):
            try:
                src = files[i].path
            except Exception:
                continue
            try:
                save_dict_check(src, fd)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save for:\n{os.path.basename(src)}\n\n{e}")

        # Build change breakdown (per side×mode, count cycles whose final selection state changed)
        changes_breakdown: Dict[str, Dict[str, int]] = {"left": {"kinetic": 0, "kinematic": 0},
                                                        "right": {"kinetic": 0, "kinematic": 0}}
        for (cref, cyc) in self.cycles:
            key = (cref.file_idx, cref.side, cref.cycle_key)
            before = int(self.initial_snapshot.get(key,1))
            after = int(cyc.get("manually_selected",1))
            if before != after:
                side = side_name_from_key(cref.side)
                mode = "kinetic" if int(cyc.get("kinetic",1))==1 else "kinematic"
                changes_breakdown[side][mode] += 1

        # Final counts by side×mode using status helpers
        cycles_repr = []
        for (cref, cyc) in self.cycles:
            side = side_name_from_key(cref.side)
            mode = "kinetic" if int(cyc.get("kinetic",1))==1 else "kinematic"
            sel = int(cyc.get("manually_selected",1))==1
            cycles_repr.append({"side": side, "mode": mode, "selected": sel})
        try:
            final_counts = status_mod.compute_counts_by_side_and_mode(cycles_repr)
        except Exception:
            final_counts = {}

        # Write audit log only if there were any changes
        any_changes = any(changes_breakdown[s][m] > 0 for s in changes_breakdown for m in changes_breakdown[s])
        if any_changes:
            try:
                audit_log_selection_change(self.root_dir,
                                           self.current_pid,
                                           self.current_trial_type,
                                           self.username,
                                           changes_breakdown,
                                           final_counts)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Audit Log Error", f"Could not write audit log:\n{e}")


        # Update initial snapshot
        for (cref, cyc) in self.cycles:
            key = (cref.file_idx, cref.side, cref.cycle_key)
            self.initial_snapshot[key] = int(cyc.get("manually_selected",1))

        self.is_dirty = False
        self.saved_once = True
        self._update_title_dirty()
        QtWidgets.QMessageBox.information(self, "Saved", "Saved _check files (audit logged if selections changed).")
    def undo(self):
        if self.history.undo(self.file_dicts):
            self.is_dirty = True
            self._update_title_dirty()
            self.redraw_all()

    def redo(self):
        if self.history.redo(self.file_dicts):
            self.is_dirty = True
            self._update_title_dirty()
            self.redraw_all()

    # ---------- Settings / user ----------
    def open_settings_dialog(self):
        s = self.settings
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Settings")
        layout = QtWidgets.QFormLayout(dlg)

        # IK thresholds
        ik_rms = QtWidgets.QDoubleSpinBox(); ik_rms.setRange(0, 1); ik_rms.setDecimals(6); ik_rms.setValue(float(s.get("ik","marker_error_RMS_max")))
        ik_max = QtWidgets.QDoubleSpinBox(); ik_max.setRange(0, 1); ik_max.setDecimals(6); ik_max.setValue(float(s.get("ik","marker_error_max_max")))
        layout.addRow("IK: marker_error_RMS_max (m)", ik_rms)
        layout.addRow("IK: marker_error_max_max (m)", ik_max)

        # SO thresholds
        so_frms = QtWidgets.QDoubleSpinBox(); so_frms.setRange(0, 1000); so_frms.setValue(float(s.get("so","force_rms_max")))
        so_fmax = QtWidgets.QDoubleSpinBox(); so_fmax.setRange(0, 1000); so_fmax.setValue(float(s.get("so","force_max_max")))
        so_mrms = QtWidgets.QDoubleSpinBox(); so_mrms.setRange(0, 1000); so_mrms.setValue(float(s.get("so","moment_rms_max")))
        so_mmax = QtWidgets.QDoubleSpinBox(); so_mmax.setRange(0, 1000); so_mmax.setValue(float(s.get("so","moment_max_max")))
        so_tol = QtWidgets.QSpinBox(); so_tol.setRange(1, 1000); so_tol.setValue(int(s.get("so","so_frames_tol")))
        layout.addRow("SO: force_rms_max (N)", so_frms)
        layout.addRow("SO: force_max_max (N)", so_fmax)
        layout.addRow("SO: moment_rms_max (Nm)", so_mrms)
        layout.addRow("SO: moment_max_max (Nm)", so_mmax)
        layout.addRow("SO: so_frames_tol", so_tol)

        # UI settings
        ui_margin = QtWidgets.QDoubleSpinBox(); ui_margin.setRange(0, 1); ui_margin.setDecimals(3); ui_margin.setSingleStep(0.01); ui_margin.setValue(float(s.get("ui","ylim_margin_pct")))
        layout.addRow("UI: y-lim margin (%)", ui_margin)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        layout.addRow(btns)

        def do_save():
            s.update_and_save({
                "ik": {"marker_error_RMS_max": ik_rms.value(), "marker_error_max_max": ik_max.value()},
                "so": {"force_rms_max": so_frms.value(), "force_max_max": so_fmax.value(),
                       "moment_rms_max": so_mrms.value(), "moment_max_max": so_mmax.value(), "so_frames_tol": so_tol.value()},
                "ui": {"ylim_margin_pct": ui_margin.value()}
            })
            dlg.accept()

        btns.accepted.connect(do_save)
        btns.rejected.connect(dlg.reject)
        dlg.exec()

    def reset_settings(self):
        self.settings.reset_to_defaults()
        QtWidgets.QMessageBox.information(self, "Settings", "Settings reset to defaults.")
    def prompt_username(self) -> Optional[str]:
        if not self.root_dir:
            QtWidgets.QMessageBox.information(self, "Set Username", "Choose a root folder first.")
            return None
        dlg = UsernameDialog(self.root_dir, self.username or "", self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.username = dlg.value()
            return self.username
        return None
    def _confirm_discard_or_save(self): return True

# Optional: app entry point if you run this module directly
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
