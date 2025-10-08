#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI — Add OpenSim Outputs to Cycle-Split MAT Files
-------------------------------------------------
Subsequent step after the "MAT to Cycle-Split MAT Converter".
"""

import os
import sys
from pathlib import Path
from typing import Dict

from PyQt5 import QtWidgets, QtCore, QtGui

# Ensure we can import the engine sitting in the same folder
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR  = CURRENT_DIR.parent
for p in (PARENT_DIR, CURRENT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import step3_addosim as engine  # the engine you just saved

# ------------------------
# Worker (runs in QThread)
# ------------------------

class AddOsimWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)       # current, total
    log      = QtCore.pyqtSignal(str)
    done     = QtCore.pyqtSignal(int, int, str)  # success, total, missing_log_path

    def __init__(self, mat_root: Path, osim_root: Path,
                 config_path: Path, shortlabels_path: Path):
        super().__init__()
        self.mat_root = mat_root
        self.osim_root = osim_root
        self.config_path = config_path
        self.shortlabels_path = shortlabels_path
        self._cancel = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if not self.config_path.exists():
                self.log.emit(f"[FATAL] Config not found: {self.config_path}")
                self.done.emit(0, 0, "")
                return
            if not self.mat_root.exists():
                self.log.emit(f"[FATAL] MAT root not found: {self.mat_root}")
                self.done.emit(0, 0, "")
                return
            if not self.osim_root.exists():
                self.log.emit(f"[FATAL] OpenSim root not found: {self.osim_root}")
                self.done.emit(0, 0, "")
                return

            osim_cfg = engine.load_osim_config(self.config_path)
            patterns = engine.compile_patterns(osim_cfg)
            translation: Dict[str, str] = engine.load_shortlabels_json(self.shortlabels_path)

            self.log.emit(f"[INFO] Short labels: {len(translation)} from {self.shortlabels_path.name}")
            self.log.emit(f"[INFO] Indexing OpenSim outputs under: {self.osim_root}")
            osim_index = engine.build_osim_index(self.osim_root, patterns)
            self.log.emit(f"[INFO] Indexed {sum(len(v) for v in osim_index.values())} files across {len(osim_index)} trials.")

            missing_log_path = self.mat_root / engine.MISSING_LOG_NAME
            mats = []
            for dp, _, files in os.walk(self.mat_root):
                for f in files:
                    if engine.is_splitcycles_mat(f):
                        mats.append(Path(dp) / f)
            total = len(mats)
            self.log.emit(f"[INFO] Found {total} '*_splitCycles.mat' files under: {self.mat_root}")
            self.progress.emit(0, total)

            success = 0
            with open(missing_log_path, "w", encoding="utf-8") as logf:
                for i, mat_file in enumerate(mats, 1):
                    if self._cancel:
                        self.log.emit("[INFO] Cancellation requested. Stopping…")
                        break
                    self.log.emit(f"[RUN ] {i}/{total}  {mat_file.name}")
                    ok = engine.add_osim_to_mat(mat_file, osim_index, osim_cfg, translation, logf)
                    success += int(ok)
                    self.progress.emit(i, total)

            self.done.emit(success, total, str(missing_log_path))

        except Exception as e:
            self.log.emit(f"[FATAL] Unhandled error: {e}")
            self.done.emit(0, 0, "")

    def cancel(self):
        self._cancel = True

# ------------------------
# Main Window
# ------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add OpenSim Data to Cycle-Split MAT")
        self.resize(900, 600)

        # Paths
        self.mat_root  = Path(engine.DEFAULT_MAT_ROOT)
        self.osim_root = Path(engine.DEFAULT_OSIM_ROOT)
        self.config_path = engine.resolve_app_dir() / engine.OSIM_JSON_NAME
        self.shortlabels_path = engine.resolve_app_dir() / engine.SHORTLABELS_JSONNAME

        self.worker = None
        self.thread = None

        self._build_ui()

    # ---- UI layout ----
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("Add OpenSim Outputs to Cycle-Split MAT Files")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # MAT root
        self.matLineEdit = QtWidgets.QLineEdit(str(self.mat_root))
        btn_mat = QtWidgets.QPushButton("Browse…")
        btn_mat.clicked.connect(self.pick_mat_root)
        row1 = self._labeled_row("Cycle‑split root:", self.matLineEdit, btn_mat)
        layout.addLayout(row1)

        # OSIM root
        self.osimLineEdit = QtWidgets.QLineEdit(str(self.osim_root))
        btn_osim = QtWidgets.QPushButton("Browse…")
        btn_osim.clicked.connect(self.pick_osim_root)
        row2 = self._labeled_row("OpenSim root:", self.osimLineEdit, btn_osim)
        layout.addLayout(row2)

        # Config json
        self.cfgLineEdit = QtWidgets.QLineEdit(str(self.config_path))
        btn_cfg = QtWidgets.QPushButton("Load…")
        btn_cfg.clicked.connect(self.pick_config)
        row3 = self._labeled_row("Config (osim_outputs.json):", self.cfgLineEdit, btn_cfg)
        layout.addLayout(row3)

        # Shortlabels json
        self.shortLineEdit = QtWidgets.QLineEdit(str(self.shortlabels_path))
        btn_short = QtWidgets.QPushButton("Load…")
        btn_short.clicked.connect(self.pick_shortlabels)
        row4 = self._labeled_row("Short labels (shortlabels_osim_outputs.json):", self.shortLineEdit, btn_short)
        layout.addLayout(row4)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.runBtn = QtWidgets.QPushButton("Add OpenSim Data")
        self.runBtn.setStyleSheet("background-color: #007BFF; color: white; font-size: 14px;")
        self.runBtn.clicked.connect(self.start)
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.cancelBtn.clicked.connect(self.cancel)
        self.openLogBtn = QtWidgets.QPushButton("Open Missing Log")
        self.openLogBtn.clicked.connect(self.open_log)
        btns.addWidget(self.runBtn)
        btns.addWidget(self.cancelBtn)
        btns.addWidget(self.openLogBtn)
        layout.addLayout(btns)

        # Progress
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Log window
        self.logEdit = QtWidgets.QTextEdit()
        self.logEdit.setReadOnly(True)
        self.logEdit.setStyleSheet("background-color: #f7f7f7;")
        layout.addWidget(self.logEdit)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Menu → About
        aboutAction = QtWidgets.QAction("About", self)
        aboutAction.triggered.connect(self.show_about)
        menu = self.menuBar().addMenu("Help")
        menu.addAction(aboutAction)

    def _labeled_row(self, label, lineedit, button):
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(label)
        lab.setMinimumWidth(240)
        row.addWidget(lab)
        row.addWidget(lineedit)
        row.addWidget(button)
        return row

    # ---- Actions ----

    def pick_mat_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select cycle‑split root")
        if folder: self.matLineEdit.setText(folder)

    def pick_osim_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select OpenSim root")
        if folder: self.osimLineEdit.setText(folder)

    def pick_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select osim_outputs.json", "", "JSON Files (*.json)")
        if path: self.cfgLineEdit.setText(path)

    def pick_shortlabels(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select shortlabels_osim_outputs.json", "", "JSON Files (*.json)")
        if path: self.shortLineEdit.setText(path)

    def open_log(self):
        mat_root = Path(self.matLineEdit.text().strip())
        log_path = mat_root / engine.MISSING_LOG_NAME
        if log_path.exists():
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(log_path))
                elif sys.platform == "darwin":
                    os.system(f'open "{log_path}"')
                else:
                    os.system(f'xdg-open "{log_path}"')
            except Exception:
                QtWidgets.QMessageBox.information(self, "Open Missing Log", str(log_path))
        else:
            QtWidgets.QMessageBox.information(self, "Open Missing Log", "Log file not found yet.")

    def start(self):
        mat_root  = Path(self.matLineEdit.text().strip())
        osim_root = Path(self.osimLineEdit.text().strip())
        cfg_path  = Path(self.cfgLineEdit.text().strip())
        short_path= Path(self.shortLineEdit.text().strip())

        # reset UI
        self.logEdit.clear()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.statusBar().showMessage("Running…")
        self.runBtn.setEnabled(False)

        # thread
        self.thread = QtCore.QThread(self)
        self.worker = AddOsimWorker(mat_root, osim_root, cfg_path, short_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.done.connect(self.on_done)
        self.thread.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.on_log("[UI  ] Cancel requested; finishing current file…")

    # ---- slots ----

    @QtCore.pyqtSlot(int, int)
    def on_progress(self, cur, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(cur)

    @QtCore.pyqtSlot(str)
    def on_log(self, msg):
        self.logEdit.append(msg)
        print(msg)

    @QtCore.pyqtSlot(int, int, str)
    def on_done(self, success, total, missing_log_path):
        self.statusBar().showMessage(f"Done — {success}/{total} files saved.")
        self.on_log(f"[DONE] Saved {success}/{total} files.")
        if missing_log_path:
            self.on_log(f"[INFO] Missing/log file: {missing_log_path}")
        self.runBtn.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None

    def show_about(self):
        QtWidgets.QMessageBox.about(
            self,
            "About — Add OpenSim Data",
            "<h3>Add OpenSim Outputs to Cycle-Split MAT Files</h3>"
            "<p>Subsequent step after the MAT to Cycle‑Split converter.</p>"
            "<p>Compression is enabled for .mat saves; long field names are translated or auto-shortened.</p>"
            "<p>Author: Harald Penasso with ChatGPT assistance</p>"
        )

# ---- run ----

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QToolTip { color: black; background-color: #ffffe0; border: 1px solid black; }")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
