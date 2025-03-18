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

import os, sys, json, warnings
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

# -------------------------------
# Conversion Functions for MAT Files
# -------------------------------

def process_mat_file(mat_path, cycles_from_to, show_keys=False):
    """
    Process a single MAT file:
      - Loads the .mat file as a dictionary.
      - Extracts event_times and event_labels from data_dict['events'].
      - Detects if the events follow an exercise or locomotion scheme.
        For exercise: take only left_stride, rename the field to exercise, and ignore right_stride.
        For locomotion: export left_stride and right_stride as in the original.
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
        return False

    # Optionally log available keys from the loaded MAT file.
    if show_keys:
        logger("AVAILABLE KEYS: Loaded MAT file keys:")
        for key, value in data_dict.items():
            if isinstance(value, dict):
                logger(f"AVAILABLE KEYS: {key}: {list(value.keys())}")
            else:
                logger(f"AVAILABLE KEYS: {key}: {type(value)}")
        # Additionally log event labels if available.
        if 'events' in data_dict and 'event_labels' in data_dict['events']:
            logger("AVAILABLE EVENT LABELS: " + str(data_dict['events']['event_labels']))
    
    # Check that events exist
    if 'events' not in data_dict or 'event_times' not in data_dict['events'] or 'event_labels' not in data_dict['events']:
        logger(f"File {mat_path} does not contain valid 'events' data. Skipping.")
        return False

    event_times = data_dict['events']['event_times']
    event_labels = data_dict['events']['event_labels']
    
    # --- NEW: Detect if events follow an exercise scheme ---
    # If both left_stride and right_stride are defined, check for at least one pair of
    # left/right events occurring within 0.1 sec. In that case, use only left_stride,
    # rename it to exercise, and ignore right_stride.
    scheme = 'locomotion'
    if "left_stride" in cycles_from_to and "right_stride" in cycles_from_to:
        left_indices = [i for i, lab in enumerate(event_labels) if "Left" in lab]
        right_indices = [i for i, lab in enumerate(event_labels) if "Right" in lab]
        simultaneous_found = False
        for li in left_indices:
            for ri in right_indices:
                if abs(event_times[li] - event_times[ri]) <= 0.1:
                    simultaneous_found = True
                    break
            if simultaneous_found:
                break
        if simultaneous_found:
            logger("Exercise scheme detected: using left_stride cycles as exercise cycles and ignoring right_stride.")
            scheme = 'exercise'
            cycles_from_to = {"exercise": cycles_from_to["left_stride"]}
    # --- End of new logic ---

    # Compute cycle periods using provided cycle definitions from JSON.
    cycle_periods = extract_cycle_periods(event_times, event_labels, cycles_from_to)
    
    # Split the data into cycles.
    cycle_data = split_data_by_cycles(data_dict, cycle_periods, cycles_from_to)
    
    # If the scheme is exercise, ensure that only the exercise cycle is saved.
    if scheme == 'exercise':
        # In case split_data_by_cycles still returns a key "left_stride", rename it to "exercise"
        if "left_stride" in cycle_data:
            cycle_data["exercise"] = cycle_data.pop("left_stride")
        # Remove any right_stride data if present.
        if "right_stride" in cycle_data:
            cycle_data.pop("right_stride")
    
    # Merge meta and events if available.
    if 'meta' in data_dict:
        cycle_data['meta'] = data_dict['meta']
    if 'events' in data_dict:
        cycle_data['events'] = data_dict['events']
    
    # Determine output file name (append _splitCycles before .mat)
    folder, filename = os.path.split(mat_path)
    base, ext = os.path.splitext(filename)
    out_filename = base + "_splitCycles.mat"
    out_file = os.path.join(folder, out_filename)
    
    try:
        scipy.io.savemat(out_file, cycle_data)
        logger(f"Saved split cycles to {out_file}")
        return True
    except Exception as e:
        logger(f"Error saving {out_file}: {e}")
        return False

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
      - Save output files under output_root preserving the relative directory structure.
    """
    logger(f"Processing MAT files in: {source_root}")
    logger(f"Output will be saved in: {output_root}")
    mat_files = find_mat_files(source_root)
    logger(f"Found {len(mat_files)} .mat files.")
    processed_count = 0
    for file_path in mat_files:
        # Compute relative folder and create corresponding output folder.
        rel_path = os.path.relpath(os.path.dirname(file_path), source_root)
        out_folder = os.path.join(output_root, rel_path)
        os.makedirs(out_folder, exist_ok=True)
        
        # Process file
        if process_mat_file(file_path, cycles_from_to, show_keys):
            # Move the output file to the new output folder.
            folder, filename = os.path.split(file_path)
            base, ext = os.path.splitext(filename)
            out_filename = base + "_splitCycles.mat"
            original_out_path = os.path.join(folder, out_filename)
            new_out_path = os.path.join(out_folder, out_filename)
            try:
                os.replace(original_out_path, new_out_path)
                logger(f"Saved processed file to {new_out_path}")
            except Exception as e:
                logger(f"Error moving file {original_out_path} to {new_out_path}: {e}")
            processed_count += 1
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
        self.loadCycleConfig()  # Try to load cycle configuration from JSON automatically.

    def initUI(self):
        # Menu Bar with Settings Menu.
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
        
        # Source folder selection.
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
        
        # Output folder selection.
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
        
        # Checkbox: Show All Available Keys and Event Labels (unchecked by default).
        self.showKeysCheckBox = QtWidgets.QCheckBox("Show all available keys and event labels")
        self.showKeysCheckBox.setToolTip("When checked, logs all available keys and event labels from the loaded MAT file at 9pt font size in the log window.")
        self.showKeysCheckBox.setChecked(False)
        layout.addWidget(self.showKeysCheckBox)
        
        # Button: Split MAT to Cycles.
        self.runButton = QtWidgets.QPushButton("Split MAT to Cycles")
        self.runButton.setStyleSheet("background-color: #007BFF; color: white; font-size: 16px;")
        self.runButton.setToolTip("Click to split the MAT files into cycles using the loaded cycle configuration.")
        self.runButton.clicked.connect(self.runConversion)
        layout.addWidget(self.runButton)
        
        # Log output text area.
        self.logTextEdit = QtWidgets.QTextEdit()
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setStyleSheet("background-color: #f0f0f0;")
        self.logTextEdit.setToolTip("This log displays progress messages, errors, and available keys and event labels from processed MAT files.")
        layout.addWidget(self.logTextEdit)

    def loadCycleConfig(self):
        # Automatically try to load "cycle_config.json" if it exists.
        json_path = "cycle_config.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    self.cycles_from_to = json.load(f)
                self.log(f"Automatically loaded cycle configuration from {json_path}")
            except Exception as e:
                self.log(f"Error loading {json_path}: {e}. Using empty cycle configuration.")
                self.cycles_from_to = {}
        else:
            self.log("cycle_config.json not found. Using empty cycle configuration.")
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
        # Format keys and event labels messages at 9pt font.
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
        # Set the global logger to the UI log function.
        global logger
        logger = self.log
        show_keys = self.showKeysCheckBox.isChecked()
        self.log("Starting conversion...")
        process_folder(source, output, self.cycles_from_to, show_keys)
        self.log("Conversion finished.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Set global tooltip style with neutral background and black text.
    app.setStyleSheet("QToolTip { color: black; background-color: #ffffe0; border: 1px solid black; }")
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
