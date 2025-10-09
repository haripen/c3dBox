#!/usr/bin/env python3
"""
MAT to Cycle-Split MAT Converter
---------------------------------
This PyQt5-based GUI tool converts MATLAB .mat files into a cycle‐split version.
It recursively searches for .mat files in the source folder and, for each file,
extracts event times/labels and uses cycle definitions (loaded from a JSON file)
to split the data into cycles. The output is saved as a new .mat file with the 
"_splitCycles" suffix.

The JSON file should define a dictionary mapping cycle names to event label sequences,
for example:
  {
      "left_stride": ["Left Foot Strike", "Left Foot Off", "Left Foot Strike"],
      "right_stride": ["Right Foot Strike", "Right Foot Off", "Right Foot Strike"]
  }

If a file named "cycle_config.json" exists in the working directory, it will be loaded automatically.
You may load a different JSON file via the Settings menu.
  
Author: Harald Penasso with ChatGPT assistance  
Date: 2025-03-14  
License: MIT License  
Packages used: PyQt5, numpy, scipy
"""

import os, sys, json, warnings, gc
from pathlib import Path
import numpy as np
import scipy.io
from PyQt5 import QtWidgets, QtCore, QtGui

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import conversion functions
from utils_py.mat2dict import loadmat_to_dict
from utils_py.cycle_help import extract_cycle_periods, split_data_by_cycles

# Global logger (will be set by the UI)
logger = print

def log_crash(message):
    crash_log = os.path.join(current_dir, "crash_report.log")
    try:
        with open(crash_log, "a") as f:
            f.write(message + "\n")
    except Exception as e:
        logger("Could not write crash report: " + str(e))

# --- Helpers to normalize MATLAB-loaded data to Python-native types ---
def _as_matlab_cellstr(seq):
    """Return Nx1 object array so scipy.io.savemat writes a MATLAB cell array of strings."""
    return np.array([str(s) for s in seq], dtype=object).reshape(-1, 1)

def _to_1d_float_list(x):
    """Coerce MATLAB numeric vectors (row/col) to Python list[float]."""
    arr = np.asarray(x)
    # MATLAB structs via scipy can wrap scalars in arrays-of-arrays; unwrap gently
    if arr.dtype == object:
        # e.g., array([[array([1.0]), array([2.0])]], dtype=object)
        flat = []
        for elem in arr.ravel():
            flat.extend(np.asarray(elem).ravel().tolist())
        return [float(v) for v in flat]
    return arr.astype(float).ravel().tolist()

def _labels_to_list(labels):
    """
    Convert event_labels from various MATLAB encodings to list[str]:
      - cell array of char: dtype=object, shapes (N,1), (1,N), (N,)
      - char matrix: dtype kind in {'U','S'}, shape (N,M)
      - nested object arrays containing 1xM char arrays
    """
    # Already a Python list/tuple of strings?
    if isinstance(labels, (list, tuple)):
        return [str(s).strip() for s in labels]

    arr = np.asarray(labels)

    # Case 1: char matrix (NxM of single characters)
    if arr.ndim == 2 and arr.dtype.kind in ("U", "S"):
        return ["".join(row).strip() for row in arr.tolist()]

    # Case 2: object array (typical for MATLAB cell arrays)
    if arr.dtype == object:
        out = []
        for elem in arr.ravel():
            # elem can itself be a small char array or a scalar string/bytes
            if isinstance(elem, (str, np.str_)):
                out.append(str(elem).strip())
            elif isinstance(elem, bytes):
                out.append(elem.decode("utf-8", "ignore").strip())
            elif isinstance(elem, np.ndarray):
                if elem.dtype.kind in ("U", "S"):            # small char array
                    out.append("".join(elem.ravel().tolist()).strip())
                elif elem.dtype == object and elem.size:     # nested cell
                    # recurse once to unwrap inner cell/char content
                    inner = _labels_to_list(elem)
                    if inner:
                        out.append(inner[0].strip())
                else:
                    out.append(str(elem).strip())
            else:
                out.append(str(elem).strip())
        return out

    # Fallback: best-effort string conversion
    return [str(s).strip() for s in arr.ravel().tolist()]

# -------------------------------
# Conversion Functions for MAT Files
# -------------------------------

def process_mat_file(mat_path, cycles_from_to, show_keys=False):
    """
    Process a single MAT file:
      - Loads the .mat file as a dictionary.
      - Extracts event_times and event_labels from data_dict['events'].
      - Detects if the events follow an exercise or locomotion scheme.
          * For exercise: if at least 50% of left events (e.g. "Left Foot Strike"/"Off")
            have a corresponding right event with the same action occurring within 0.02 sec,
            then use only the left_stride cycle (renaming it to "exercise") and ignore right_stride.
          * For locomotion: export left_stride and right_stride as in the original.
      - Computes cycle periods using extract_cycle_periods.
      - Splits point and analog data by cycles using split_data_by_cycles.
      - Merges meta and events back into the output.
      - Optionally logs available keys and event labels in the MAT file.
      - Saves the new dictionary as a MAT file with a "_splitCycles" suffix.
    """
    try:
        data_dict = loadmat_to_dict(mat_path)
    except Exception as e:
        logger(f"Error loading {mat_path}: {e}")
        log_crash(f"Error loading {mat_path}: {e}")
        return False
    
    event_times  = _to_1d_float_list(data_dict['events']['event_times'])
    event_labels = _labels_to_list(data_dict['events']['event_labels'])
    data_dict['events']['event_times']  = event_times
    data_dict['events']['event_labels'] = event_labels
    
    if show_keys:
        logger("AVAILABLE KEYS: Loaded MAT file keys:")
        for key, value in data_dict.items():
            if isinstance(value, dict):
                logger(f"AVAILABLE KEYS: {key}: {list(value.keys())}")
            else:
                logger(f"AVAILABLE KEYS: {key}: {type(value)}")
        if 'events' in data_dict and 'event_labels' in data_dict['events']:
            logger("AVAILABLE EVENT LABELS: " + str(data_dict['events']['event_labels']))

    if ('events' not in data_dict or 
        'event_times' not in data_dict['events'] or 
        'event_labels' not in data_dict['events']):
        logger(f"File {mat_path} does not contain valid 'events' data. Skipping.")
        return False
    
    # --- Detect scheme ---
    threshold_simultaneous = 0.02  # tolerance for matching events in exercise mode
    left_simultaneous_count = 0
    total_left_count = 0
    for i, lab in enumerate(event_labels):
        if lab.startswith("Left Foot"):
            total_left_count += 1
            try:
                action = lab.split("Left Foot ")[1]
            except IndexError:
                continue
            for j, lab2 in enumerate(event_labels):
                if lab2.startswith("Right Foot"):
                    try:
                        action2 = lab2.split("Right Foot ")[1]
                    except IndexError:
                        continue
                    if action2 == action and abs(event_times[i] - event_times[j]) < threshold_simultaneous:
                        left_simultaneous_count += 1
                        break
    if total_left_count > 0 and (left_simultaneous_count / total_left_count) >= 0.5:
        scheme = 'exercise'
        logger("Exercise scheme detected: using left_stride as exercise cycle and ignoring right_stride.")
        cycles_from_to = {"exercise": cycles_from_to["left_stride"]}
    else:
        scheme = 'locomotion'
        logger("Locomotion scheme detected: using left_stride and right_stride cycles.")
    # --- End scheme detection ---

    try:
        cycle_periods = extract_cycle_periods(event_times, event_labels, cycles_from_to)
        cycle_data = split_data_by_cycles(data_dict, cycle_periods, cycles_from_to,add_contralateral=(scheme == 'locomotion'))
    except Exception as e:
        logger(f"Error processing cycles in {mat_path}: {e}")
        log_crash(f"Error processing cycles in {mat_path}: {e}")
        return False

    # --- Scheme-specific post-processing ---
    if scheme == 'exercise':
        # Replace left_stride key with exercise and remove right_stride.
        if "left_stride" in cycle_data:
            cycle_data["exercise"] = cycle_data.pop("left_stride")
        if "right_stride" in cycle_data:
            cycle_data.pop("right_stride")
        # Now, go one level deeper: if cycle_data["exercise"] is a dict (e.g., keys like 'cycle1', 'cycle2', ...)
        # iterate through each cycle and remove the "Left_" prefix from its keys.
        if "exercise" in cycle_data:
            ex = cycle_data["exercise"]
            if isinstance(ex, dict):
                for cycle_name, cycle in ex.items():
                    if isinstance(cycle, dict):
                        new_cycle = {}
                        for k, v in cycle.items():
                            if k.startswith("Left_"):
                                new_key = k[len("Left_"):]
                            else:
                                new_key = k
                            new_cycle[new_key] = v
                        ex[cycle_name] = new_cycle
            elif isinstance(ex, list):
                for idx, cycle in enumerate(ex):
                    if isinstance(cycle, dict):
                        new_cycle = {}
                        for k, v in cycle.items():
                            if k.startswith("Left_"):
                                new_key = k[len("Left_"):]
                            else:
                                new_key = k
                            new_cycle[new_key] = v
                        ex[idx] = new_cycle
    # --- End scheme-specific post-processing ---

    if 'meta' in data_dict:
        cycle_data['meta'] = data_dict['meta']
    if 'events' in data_dict:     
        cycle_data['events'] = data_dict['events']

        # --- ensure MATLAB-friendly types ---
        ev = dict(cycle_data['events'])  # shallow copy
        ev_labels_list = _labels_to_list(ev.get('event_labels', []))
        ev_times_list  = _to_1d_float_list(ev.get('event_times', []))

        ev['event_labels'] = _as_matlab_cellstr(ev_labels_list)                # -> Nx1 cellstr
        ev['event_times']  = np.asarray(ev_times_list, dtype=float).reshape(-1, 1)  # -> Nx1 double

        cycle_data['events'] = ev
  
    folder, filename = os.path.split(mat_path)
    base, ext = os.path.splitext(filename)
    out_filename = base + "_splitCycles.mat"
    out_file = os.path.join(folder, out_filename)
    
    try:
        scipy.io.savemat(out_file, cycle_data, oned_as='column')
        logger(f"Saved split cycles to {out_file}")
    except Exception as e:
        logger(f"Error saving {out_file}: {e}")
        log_crash(f"Error saving {out_file}: {e}")
        return False
    finally:
        # Free memory by deleting large objects and calling garbage collector.
        del data_dict, cycle_data
        gc.collect()
    return True

def find_mat_files(root_folder):
    """
    Recursively search for .mat files under root_folder.
    Returns a list of full file paths.
    """
    mat_files = []
    for dirpath, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".mat"):
                mat_files.append(os.path.join(dirpath, file))
    return mat_files

def process_folder(source_root, output_root, cycles_from_to, show_keys=False):
    """
    Process all .mat files in source_root recursively:
      - For each file, load and process the file.
      - In case of a crash while processing a file, log the error to a crash report.
      - Save output files under output_root preserving the relative directory structure.
      - Call garbage collection after each file to free memory.
    """
    logger(f"Processing MAT files in: {source_root}")
    logger(f"Output will be saved in: {output_root}")
    mat_files = find_mat_files(source_root)
    logger(f"Found {len(mat_files)} .mat files.")
    processed_count = 0
    for file_path in mat_files:
        try:
            if process_mat_file(file_path, cycles_from_to, show_keys):
                folder, filename = os.path.split(file_path)
                base, ext = os.path.splitext(filename)
                out_filename = base + "_splitCycles.mat"
                original_out_path = os.path.join(folder, out_filename)
                rel_path = os.path.relpath(os.path.dirname(file_path), source_root)
                out_folder = os.path.join(output_root, rel_path)
                os.makedirs(out_folder, exist_ok=True)
                new_out_path = os.path.join(out_folder, out_filename)
                try:
                    os.replace(original_out_path, new_out_path)
                    logger(f"Saved processed file to {new_out_path}")
                except Exception as e:
                    logger(f"Error moving file {original_out_path} to {new_out_path}: {e}")
                    log_crash(f"Error moving file {original_out_path} to {new_out_path}: {e}")
                processed_count += 1
        except Exception as e:
            logger(f"Error processing file {file_path}: {e}")
            log_crash(f"Error processing file {file_path}: {e}")
        finally:
            gc.collect()
    logger(f"Finished processing. {processed_count} files were converted.")

# -------------------------------
# PyQt5 UI for MAT to Cycle-Split MAT Converter
# -------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAT to Cycle-Split MAT Converter")
        self.resize(800, 600)
        self.cycles_from_to = {}
        self.initUI()
        self.loadCycleConfig()

    def initUI(self):
        menubar = self.menuBar()
        settingsMenu = menubar.addMenu("Settings")
        loadJsonAction = QtWidgets.QAction("Load JSON", self)
        loadJsonAction.setToolTip("Click to load a cycle configuration JSON file from disk. This file defines the event sequences for cycle splitting.")
        loadJsonAction.triggered.connect(self.loadJson)
        settingsMenu.addAction(loadJsonAction)
        aboutAction = QtWidgets.QAction("About", self)
        aboutAction.setToolTip("Show information about this tool, including author, date, and license details.")
        aboutAction.triggered.connect(self.showAbout)
        settingsMenu.addAction(aboutAction)
        
        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)
        layout = QtWidgets.QVBoxLayout()
        centralWidget.setLayout(layout)
        
        titleLabel = QtWidgets.QLabel("MAT to Cycle-Split MAT Converter")
        titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        titleLabel.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(titleLabel)
        
        hboxSource = QtWidgets.QHBoxLayout()
        self.sourceLineEdit = QtWidgets.QLineEdit()
        self.sourceLineEdit.setPlaceholderText("Select source folder containing MAT files")
        self.sourceBrowseButton = QtWidgets.QPushButton("Browse...")
        self.sourceBrowseButton.setStyleSheet("background-color: #007BFF; color: white;")
        self.sourceBrowseButton.setToolTip("Select the folder that contains the MAT files to be processed.")
        self.sourceBrowseButton.clicked.connect(self.browseSource)
        hboxSource.addWidget(QtWidgets.QLabel("Source Folder:"))
        hboxSource.addWidget(self.sourceLineEdit)
        hboxSource.addWidget(self.sourceBrowseButton)
        layout.addLayout(hboxSource)
        
        hboxOutput = QtWidgets.QHBoxLayout()
        self.outputLineEdit = QtWidgets.QLineEdit()
        self.outputLineEdit.setPlaceholderText("Select output folder")
        self.outputBrowseButton = QtWidgets.QPushButton("Browse...")
        self.outputBrowseButton.setStyleSheet("background-color: #007BFF; color: white;")
        self.outputBrowseButton.setToolTip("Select the folder where the processed MAT files will be saved.")
        self.outputBrowseButton.clicked.connect(self.browseOutput)
        hboxOutput.addWidget(QtWidgets.QLabel("Output Folder:"))
        hboxOutput.addWidget(self.outputLineEdit)
        hboxOutput.addWidget(self.outputBrowseButton)
        layout.addLayout(hboxOutput)
        
        self.showKeysCheckBox = QtWidgets.QCheckBox("Show all available keys and event labels")
        self.showKeysCheckBox.setToolTip("When checked, logs all available keys and event labels from the loaded MAT file at 9pt font size in the log window.")
        self.showKeysCheckBox.setChecked(False)
        layout.addWidget(self.showKeysCheckBox)
        
        self.runButton = QtWidgets.QPushButton("Split MAT to Cycles")
        self.runButton.setStyleSheet("background-color: #007BFF; color: white; font-size: 16px;")
        self.runButton.setToolTip("Click to split the MAT files into cycles using the loaded cycle configuration.")
        self.runButton.clicked.connect(self.runConversion)
        layout.addWidget(self.runButton)
        
        self.logTextEdit = QtWidgets.QTextEdit()
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setStyleSheet("background-color: #f0f0f0;")
        self.logTextEdit.setToolTip("This log displays progress messages, errors, and available keys and event labels from processed MAT files.")
        layout.addWidget(self.logTextEdit)

    def loadCycleConfig(self):
        """
        Load cycle_config.json from the folder where this GUI's .py file lives.
        Falls back to the executable's folder when frozen (e.g., PyInstaller).
        """
        # Resolve the app directory robustly
        if getattr(sys, "frozen", False):  # bundled executable
            app_dir = Path(sys.executable).resolve().parent
        else:
            try:
                app_dir = Path(__file__).resolve().parent
            except NameError:
                # __file__ can be missing in some interactive contexts
                app_dir = Path.cwd()

        json_path = app_dir / "cycle_config.json"

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.cycles_from_to = json.load(f)
                self.log(f"Automatically loaded cycle configuration from {json_path}")
            except Exception as e:
                self.log(f"Error loading {json_path}: {e}. Using empty cycle configuration.")
                self.cycles_from_to = {}
        else:
            self.log(f"{json_path.name} not found at {json_path}. Using empty cycle configuration.")
            self.cycles_from_to = {}

    def loadJson(self):
        json_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Cycle Config JSON File", "", "JSON Files (*.json)")
        if json_path:
            try:
                with open(json_path, "r") as f:
                    self.cycles_from_to = json.load(f)
                self.log(f"Loaded cycle configuration from {json_path}")
            except Exception as e:
                self.log(f"Error loading {json_path}: {e}")
        else:
            self.log("No JSON file selected.")

    def browseSource(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.sourceLineEdit.setText(folder)

    def browseOutput(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.outputLineEdit.setText(folder)

    def log(self, message):
        if message.startswith("AVAILABLE KEYS:") or message.startswith("AVAILABLE EVENT LABELS:"):
            self.logTextEdit.append(f"<span style='font-size:9pt;'>{message}</span>")
        else:
            self.logTextEdit.append(message)
        print(message)

    def showAbout(self):
        about_text = (
            "<h2>MAT to Cycle-Split MAT Converter</h2>"
            "<p>Author: Harald Penasso with ChatGPT assistance</p>"
            "<p>Date: 2025-03-14</p>"
            "<p>License: MIT License</p>"
            "<p>Packages used: PyQt5, numpy, scipy</p>"
        )
        QtWidgets.QMessageBox.about(self, "About MAT to Cycle-Split MAT Converter", about_text)

    def runConversion(self):
        source = self.sourceLineEdit.text().strip()
        output = self.outputLineEdit.text().strip()
        if not source or not output:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select both source and output folders.")
            return
        if not self.cycles_from_to:
            QtWidgets.QMessageBox.warning(self, "Error", "Cycle configuration is empty. Please load a valid JSON file.")
            return
        global logger
        logger = self.log
        show_keys = self.showKeysCheckBox.isChecked()
        self.log("Starting conversion...")
        process_folder(source, output, self.cycles_from_to, show_keys)
        self.log("Conversion finished.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QToolTip { color: black; background-color: #ffffe0; border: 1px solid black; }")
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
