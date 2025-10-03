#!/usr/bin/env python3
"""
PyQt5 C3D to MAT Converter
---------------------------
This is a PyQt5-based GUI tool for converting C3D files to MATLAB MAT files.
It reuses conversion functions from your existing c3d_to_mat.py script.

Features:
  - Choose source and output folders.
  - Select filter type (default "enf") with dynamic explanation for keywords.
  - Option to print raw point/analog labels (displayed in 9pt in the log).
  - Automatically loads a filtering dictionary from data_filter.json if present.
  - You can load a different JSON via the Settings menu.
  - Meta data is enriched with point_rate, analog_rate, analog_first, analog_last, and the full C3D header.
  - In case of any crash processing a file, a crash report log is saved to the app folder.
  - Info menu shows credits, author, date, and MIT license info.

Author: Harald Penasso with ChatGPT assistance  
Date: 2025-03-14  
License: MIT License  
Packages used: PyQt5, ezc3d, numpy, scipy
"""

import os, sys, json, re, numpy as np, ezc3d, gc, traceback
from pathlib import Path
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.io import savemat

# Global flags and logger
PRINT_LABELS = False
DEBUG_KINETIC = False
logger = print

# Default filtering dictionary (loaded via UI)
data_filter_dict = {"meta": [], "events": [], "point": [], "analog": []}

# -------------------------------
# Helper: Flatten kinetic target lists
# -------------------------------
def flatten_targets(targets):
    flattened = []
    for t in targets:
        if isinstance(t, list):
            flattened.extend(t)
        else:
            flattened.append(t)
    return flattened

# -------------------------------
# Kinetic Validity Check Function (unchanged)
# -------------------------------
def check_kinetic_validity(extracted, data_filter_dict):
    debug = DEBUG_KINETIC
    try:
        fp_validity = extracted["meta"]["FORCE_PLATFORM"]["VALID"]
        fp_corners = extracted["meta"]["FORCE_PLATFORM"]["CORNERS"]["value"]
    except KeyError:
        if debug:
            logger("Warning: FORCE_PLATFORM data missing. Marking all events as non-kinetic.")
        extracted["events"]["kinetic"] = [False] * len(extracted["events"]["event_times"])
        return

    num_fp = fp_corners.shape[2]
    fp_centers = {}
    for i in range(num_fp):
        fp_key = f"FP{i+1}"
        corners = fp_corners[:, :, i]
        center = np.mean(corners, axis=1)
        fp_centers[fp_key] = center
        if debug:
            logger(f"Forceplate {fp_key}: center = {center}")

    point_time = np.array(extracted["point"]["time"])
    kinetic_validity = []
    for evt_idx, (event_time, event_label_chars) in enumerate(zip(extracted["events"]["event_times"],
                                                                   extracted["events"]["event_labels"])):
        event_label = "".join(event_label_chars).strip().lower()
        valid = False
        if "left" in event_label:
            side = "Left"
            marker_list = data_filter_dict.get("left_kinetic_target", [])
            thresh_list = data_filter_dict.get("left_critDist_xyz", [])
        elif "right" in event_label:
            side = "Right"
            marker_list = data_filter_dict.get("right_kinetic_target", [])
            thresh_list = data_filter_dict.get("right_critDist_xyz", [])
        else:
            if debug:
                logger(f"Event {evt_idx+1}: '{event_label}' does not specify left/right. Marked as non-kinetic.")
            kinetic_validity.append(False)
            continue

        frame_idx = int(np.argmin(np.abs(point_time - event_time)))
        if debug:
            logger(f"\nEvent {evt_idx+1}: time = {event_time}, label = '{event_label}', side = {side}, frame_idx = {frame_idx}")

        valid_fp_list = []
        for fp_key, fp_side in fp_validity.items():
            if fp_side.lower() == side.lower() and fp_key in fp_centers:
                valid_fp_list.append(fp_key)
        if not valid_fp_list:
            if debug:
                logger(f"  No valid forceplates found for side {side}.")
            kinetic_validity.append(False)
            continue

        marker_list = flatten_targets(marker_list)
        for marker, thresh in zip(marker_list, thresh_list):
            if marker not in extracted["point"]:
                if debug:
                    logger(f"  Marker '{marker}' not found in point data.")
                continue
            marker_data = extracted["point"][marker]
            if marker_data.shape[0] <= frame_idx:
                if debug:
                    logger(f"  Marker '{marker}' does not have data for frame index {frame_idx}.")
                continue
            marker_pos = marker_data[frame_idx, :]
            if debug:
                logger(f"  Kinetic target '{marker}' position at frame {frame_idx}: {marker_pos}")
            for fp_key in valid_fp_list:
                center = fp_centers[fp_key]
                corners = fp_corners[:, :, int(fp_key[2:]) - 1]
                half_range_x = np.max(np.abs(corners[0, :] - center[0]))
                half_range_y = np.max(np.abs(corners[1, :] - center[1]))
                allowed_x = half_range_x + thresh[0]
                allowed_y = half_range_y + thresh[1]
                allowed_z = thresh[2]
                dist_x = np.abs(marker_pos[0] - center[0])
                dist_y = np.abs(marker_pos[1] - center[1])
                dist_z = np.abs(marker_pos[2] - center[2])
                if debug:
                    logger(f"    Checking forceplate {fp_key}:")
                    logger(f"      Center: {center}")
                    logger(f"      Half-range x: {half_range_x}, allowed_x = {allowed_x} (threshold_x = {thresh[0]})")
                    logger(f"      Half-range y: {half_range_y}, allowed_y = {allowed_y} (threshold_y = {thresh[1]})")
                    logger(f"      Allowed z = {allowed_z} (threshold_z = {thresh[2]})")
                    logger(f"      Distances: x = {dist_x}, y = {dist_y}, z = {dist_z}")
                if (dist_x <= allowed_x) and (dist_y <= allowed_y) and (dist_z <= allowed_z):
                    if debug:
                        logger(f"      Marker '{marker}' is within allowed range of forceplate {fp_key}.")
                    valid = True
                    break
                else:
                    if debug:
                        logger(f"      Marker '{marker}' is NOT within allowed range of forceplate {fp_key}.")
            if valid:
                break
        kinetic_validity.append(valid)
        if debug:
            logger(f"  --> Event {evt_idx+1} kinetic validity: {valid}")
    extracted["events"]["kinetic"] = kinetic_validity

# -------------------------------
# Conversion Functions (c3d to mat)
# -------------------------------
def extract_event_times_and_labels(c3d_path):
    c3d_file = ezc3d.c3d(c3d_path)
    try:
        event_times = c3d_file['parameters']['EVENT']['TIMES']['value'][1]
        event_labels = c3d_file['parameters']['EVENT']['LABELS']['value']
        event_contexts = c3d_file['parameters']['EVENT']['CONTEXTS']['value']
        for i, ctx in enumerate(event_contexts):
            event_labels[i] = ctx + ' ' + event_labels[i]
        events = sorted(zip(event_times, event_labels))
        if events:
            sorted_times, sorted_labels = zip(*events)
        else:
            sorted_times, sorted_labels = [], []
    except Exception as e:
        logger(f"Warning: could not extract events from {c3d_path}: {e}")
        sorted_times, sorted_labels = [], []
    trimmed_labels = [lab.strip() for lab in sorted_labels]
    if trimmed_labels:
        max_len = max(len(lab) for lab in trimmed_labels)
        padded = np.array([list(lab.ljust(max_len)) for lab in trimmed_labels])
    else:
        padded = np.empty((0,0))
    return {"event_times": list(sorted_times), "event_labels": padded}

def process_field_label(label):
    label = label.strip()
    tokens = re.split(r'[ \._]+', label)
    tokens = [t for t in tokens if t]
    new_tokens = []
    for token in tokens:
        if not new_tokens or token.lower() != new_tokens[-1].lower():
            new_tokens.append(token)
    return "_".join(new_tokens)

def get_unique_field_label(raw_label, used_labels):
    candidate = process_field_label(raw_label)
    if len(candidate) > 31:
        tokens = re.split(r'[_]+', candidate)
        tokens = [token[:3] for token in tokens]
        candidate = "_".join(tokens)
    original_candidate = candidate
    suffix = 1
    while candidate in used_labels:
        suffix_str = f"_{suffix}"
        if len(original_candidate) + len(suffix_str) > 31:
            candidate = original_candidate[:31 - len(suffix_str)] + suffix_str
        else:
            candidate = original_candidate + suffix_str
        suffix += 1
    used_labels.add(candidate)
    return candidate

def find_enf_file(c3d_path):
    folder = os.path.dirname(c3d_path)
    c3d_base = os.path.splitext(os.path.basename(c3d_path))[0].lower()
    for file in os.listdir(folder):
        if file.lower().endswith(".enf"):
            candidate_base = os.path.splitext(file)[0].replace(".Trial", "").lower()
            if candidate_base == c3d_base:
                return os.path.join(folder, file)
    return None

def parse_forceplate_validity(enf_path):
    fp_validity = {}
    try:
        with open(enf_path, 'r') as f:
            for line in f:
                m = re.match(r'^(FP\d{1,2})\s*=\s*(.+)', line)
                if m:
                    key = m.group(1).strip()
                    value = m.group(2).strip()
                    fp_validity[key] = value
    except Exception as e:
        logger(f"Error parsing forceplate validity from {enf_path}: {e}")
    return fp_validity

def get_unique_mp_basenames(folder):
    files = os.listdir(folder)
    mp_files = [f for f in files if f.lower().endswith(".mp") or f.lower().endswith(".vsk")]
    basenames = {}
    for f in mp_files:
        base = os.path.splitext(f)[0].strip()
        basenames.setdefault(base, []).append(f)
    return basenames

# --- Original MP info function ---
def get_mp_info(folder):
    basenames = get_unique_mp_basenames(folder)
    if not basenames:
        return {}
    if len(basenames) > 1:
        logger("Error: Multiple unique MP/VSK file sets detected in folder:")
        for base, files in basenames.items():
            logger(f"  {base}: {files}")
        logger("Please remove all additional MP/VSK file sets and try again.")
        sys.exit(1)
    mp_info = {}
    for base, files in basenames.items():
        for f in files:
            path = os.path.join(folder, f)
            info = parse_mp_file(path)
            mp_info.update(info)
    return mp_info

# --- New MP info function that cross-checks with SUBJECTS= ---
def get_mp_info_with_subject(folder, subject_list):
    basenames = get_unique_mp_basenames(folder)
    if subject_list:
        # Keep only MP files whose basename contains any of the subjects.
        basenames = {k: v for k, v in basenames.items() if any(sub in k.lower() for sub in subject_list)}
        # If multiple candidates exist and one has basename exactly "kiste", remove it.
        if len(basenames) > 1:
            for k in list(basenames.keys()):
                if k.lower() == "kiste" and len(basenames) > 1:
                    logger(f"Ignoring MP/VSK file set '{k}' as it matches 'kiste'.")
                    del basenames[k]
    if not basenames:
        logger(f"No MP/VSK file set matching subjects '{subject_list}' found in folder.")
        return {}
    if len(basenames) > 1:
        logger("Error: Multiple unique MP/VSK file sets detected in folder:")
        for base, files in basenames.items():
            logger(f"  {base}: {files}")
        logger("Please remove all additional MP/VSK file sets and try again.")
        sys.exit(1)
    mp_info = {}
    for base, files in basenames.items():
        for f in files:
            path = os.path.join(folder, f)
            info = parse_mp_file(path)
            mp_info.update(info)
    return mp_info

def parse_mp_file(mp_path):
    mp_params = {}
    try:
        with open(mp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("$"):
                    parts = line[1:].split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        mp_params[key] = value
    except Exception as e:
        logger(f"Error parsing MP file {mp_path}: {e}")
    return mp_params

def filter_dict_by_keywords(data_dict, keywords):
    if not keywords:
        return data_dict
    filtered = {}
    for key, value in data_dict.items():
        for kw in keywords:
            if kw.lower() in key.lower():
                filtered[key] = value
                break
    return filtered

def filter_list_by_keywords(data_list, keywords):
    if not keywords:
        return data_list
    return [item for item in data_list if any(kw.lower() in item.lower() for kw in keywords)]

def process_c3d_file(c3d_path, data_filter_dict):
    logger(f"Processing {c3d_path} ...")
    c3d_file = ezc3d.c3d(c3d_path)
    meta = {}
    for key in c3d_file['parameters']:
        if key.upper() in ['SUBJECT', 'PROCESSING', 'FORCE_PLATFORM']:
            meta[key] = c3d_file['parameters'][key]
    if "PROCESSING" in meta and isinstance(meta["PROCESSING"], dict):
        for proc_key in list(meta["PROCESSING"].keys()):
            item = meta["PROCESSING"][proc_key]
            if isinstance(item, dict) and "value" in item:
                meta["PROCESSING"][proc_key] = item["value"]
    enf_path = find_enf_file(c3d_path)
    if enf_path:
        fp_validity = parse_forceplate_validity(enf_path)
        if fp_validity:
            if "FORCE_PLATFORM" in meta and isinstance(meta["FORCE_PLATFORM"], dict):
                meta["FORCE_PLATFORM"]["VALID"] = fp_validity
            else:
                meta["FORCE_PLATFORM"] = {"VALID": fp_validity}
    else:
        logger(f"Warning: No matching ENF file found for {c3d_path}.")

    folder = os.path.dirname(c3d_path)
    # --- Extract subjects from ENF file (strictly from SUBJECTS=) ---
    subject_list = []
    if enf_path:
        try:
            with open(enf_path, 'r') as f:
                for line in f:
                    if line.startswith('SUBJECTS='):
                        subjects_line = line.split('=', 1)[1].strip()
                        # Split on commas and whitespace to handle lines like "Rd0001 Rd0001,Kiste"
                        subject_list = [s.strip().lower() for s in re.split(r'[,\s]+', subjects_line) if s.strip()]
                        break
        except Exception as e:
            logger(f"Error reading subjects from {enf_path}: {e}")
    # Use the new helper if subjects are provided; otherwise, fallback to original.
    if subject_list:
        mp_info = get_mp_info_with_subject(folder, subject_list)
        # If no MP info is found using the subject filter, fallback to original.
        if not mp_info:
            mp_info = get_mp_info(folder)
    else:
        mp_info = get_mp_info(folder)
    if mp_info:
        processing_keys = set()
        if "PROCESSING" in c3d_file['parameters']:
            processing_keys = set(c3d_file['parameters']['PROCESSING'].keys())
        filtered_mp_info = {key: value for key, value in mp_info.items() if key not in processing_keys}
        if filtered_mp_info:
            meta["mp_info"] = filtered_mp_info
    meta = filter_dict_by_keywords(meta, data_filter_dict.get("meta", []))
    point_rate = c3d_file['parameters']['POINT']['RATE']['value'][0]
    analog_rate = c3d_file['parameters']['ANALOG']['RATE']['value'][0]
    meta["header"] = c3d_file.get("header", {})

    events = extract_event_times_and_labels(c3d_path)
    ev_keywords = data_filter_dict.get("events", [])
    if ev_keywords:
        filtered_times = []
        filtered_labels = []
        for t, lab in zip(events["event_times"], events["event_labels"]):
            lab_str = "".join(lab).strip()
            if any(kw.lower() in lab_str.lower() for kw in ev_keywords):
                filtered_times.append(t)
                filtered_labels.append(lab_str)
        if filtered_labels:
            max_len = max(len(lab) for lab in filtered_labels)
            padded = np.array([list(lab.ljust(max_len)) for lab in filtered_labels])
        else:
            padded = np.empty((0,0))
        events = {"event_times": filtered_times, "event_labels": padded}

    # --- Updated Point (marker) data extraction ---
    raw_point_labels = c3d_file['parameters']['POINT']['LABELS']['value']
    if "LABELS2" in c3d_file['parameters']['POINT']:
        raw_point_labels += c3d_file['parameters']['POINT']['LABELS2']['value']
    points = c3d_file['data']['points'][0:3, :, :]
    points = np.transpose(points, (2, 1, 0))
    actual_start = c3d_file['parameters']['TRIAL']['ACTUAL_START_FIELD']['value'][0]
    n_frames = points.shape[0]
    frames_vector = np.arange(0, n_frames) + actual_start
    timeC3D = frames_vector / point_rate - 1/point_rate
    # Build the full point dictionary (internal use)
    point_struct = {"time": timeC3D.tolist(), "frames": frames_vector.tolist()}
    used_point_labels = set()
    point_filter = data_filter_dict.get("point", [])
    kinetic_targets = flatten_targets(data_filter_dict.get("left_kinetic_target", []) + data_filter_dict.get("right_kinetic_target", []))
    kinetic_targets_lower = [t.lower() for t in kinetic_targets]

    for i, lab in enumerate(raw_point_labels):
        lab_stripped = lab.strip()
        # --- If SUBJECTS= was found, remove a prepended subject if its prefix matches any subject ---
        if subject_list and ":" in lab_stripped:
            prefix, remainder = lab_stripped.split(":", 1)
            if any(sub in prefix.strip().lower() for sub in subject_list):
                lab_stripped = remainder.strip()
        if lab_stripped.startswith("*"):
            continue
        # Always include a marker if it is a kinetic target,
        # even if it is not in the JSON "point" list.
        if point_filter and (lab_stripped.lower() not in set(pt.lower() for pt in point_filter)) and (lab_stripped.lower() not in kinetic_targets_lower):
            continue
        unique_label = get_unique_field_label(lab_stripped, used_point_labels)
        point_struct[unique_label] = points[:, i, :]
    if PRINT_LABELS:
        logger("[label]Raw Point Labels for " + c3d_path + " : " + str(raw_point_labels))

    analog_header = c3d_file['header']['analogs']
    analog_first = analog_header['first_frame']
    analog_last = analog_header['last_frame']
    n_analog_frames = analog_last - analog_first + 1
    timeAnalog = np.arange(0, n_analog_frames) / analog_rate + timeC3D[0]
    analog_frames = np.arange(analog_first, analog_last + 1)
    raw_analog_labels = c3d_file['parameters']['ANALOG']['LABELS']['value']
    an_keywords = data_filter_dict.get("analog", [])
    if an_keywords:
        indices = [i for i, lab in enumerate(raw_analog_labels) if any(kw.lower() in lab.lower() for kw in an_keywords)]
        raw_analog_labels = [raw_analog_labels[i] for i in indices]
        analogue = c3d_file['data']['analogs'][0, :, :][indices, :]
    else:
        analogue = c3d_file['data']['analogs'][0, :, :]
    analog_struct = {"time": timeAnalog.tolist(), "frames": analog_frames.tolist()}
    used_analog_labels = set()
    for i, lab in enumerate(raw_analog_labels):
        unique_label = get_unique_field_label(lab, used_analog_labels)
        analog_struct[unique_label] = analogue[i, :]
    if PRINT_LABELS:
        logger("[label]Raw Analog Labels for " + c3d_path + " : " + str(raw_analog_labels))

    extracted = {"meta": meta, "events": events, "point": point_struct, "analog": analog_struct}
    # Run kinetic validity check using the full point dictionary.
    check_kinetic_validity(extracted, data_filter_dict)
    
    # --- Final Filtering for Export ---
    # If the JSON "point" list is provided, remove any markers from the exported point data
    # that are not part of that list (except for 'time' and 'frames').
    allowed_points = data_filter_dict.get("point", [])
    if allowed_points:
        allowed_set = set(pt.lower() for pt in allowed_points)
        filtered_points = {"time": extracted["point"]["time"], "frames": extracted["point"]["frames"]}
        for key, value in extracted["point"].items():
            if key in ("time", "frames"):
                continue
            if key.lower() in allowed_set:
                filtered_points[key] = value
        extracted["point"] = filtered_points

    return extracted

def find_c3d_files(root_folder):
    c3d_files = []
    for dirpath, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".c3d"):
                c3d_files.append(os.path.join(dirpath, file))
    return c3d_files

def file_matches_filter(c3d_path, filter_type, keywords):
    if filter_type == "all":
        return True
    if filter_type == "enf":
        enf_path = find_enf_file(c3d_path)
        if not enf_path:
            return False
        description = None
        try:
            with open(enf_path, 'r') as f:
                for line in f:
                    if line.startswith('DESCRIPTION='):
                        description = line.split('=', 1)[1].strip().replace(' ', '_')
                        break
        except Exception as e:
            logger(f"Error reading {enf_path}: {e}")
            return False
        if description is None:
            return False
        return any(description.lower().startswith(kw.lower()) for kw in keywords)
    elif filter_type == "filename":
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        return any(kw.lower() in base.lower() for kw in keywords)
    else:
        return False

def process_folder(filter_type, keywords, data_filter_dict, source_root, output_root):
    logger(f"Processing C3D files in: {source_root}")
    logger(f"Output will be saved in: {output_root}")
    c3d_files = find_c3d_files(source_root)
    logger(f"Found {len(c3d_files)} C3D files.")
    processed_count = 0
    crash_log_path = os.path.join(os.getcwd(), "crash_report.log")
    for file_path in c3d_files:
        if not file_matches_filter(file_path, filter_type, keywords):
            continue
        try:
            extracted = process_c3d_file(file_path, data_filter_dict)
            rel_path = os.path.relpath(os.path.dirname(file_path), source_root)
            out_folder = os.path.join(output_root, rel_path)
            os.makedirs(out_folder, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_file = os.path.join(out_folder, base_name + ".mat")
            savemat(out_file, extracted)
            logger(f"Saved extracted data to {out_file}")
            processed_count += 1
        except Exception as e:
            error_message = f"Error processing {file_path}: {str(e)}\n" + traceback.format_exc() + "\n"
            logger(error_message)
            with open(crash_log_path, "a") as crash_log:
                crash_log.write(error_message)
        finally:
            # Force garbage collection to free memory after each file.
            gc.collect()
    logger(f"Finished processing. {processed_count} files were extracted and saved.")

# -------------------------------
# PyQt5 UI Version (unchanged)
# -------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C3D to MAT Converter")
        self.resize(800, 600)
        self.initUI()
        self.loadDataFilterDict()

    def initUI(self):
        menubar = self.menuBar()
        settingsMenu = menubar.addMenu("Settings")
        loadJsonAction = QtWidgets.QAction("Load JSON", self)
        loadJsonAction.setToolTip("Load a filtering JSON file from disk.")
        loadJsonAction.triggered.connect(self.loadJson)
        settingsMenu.addAction(loadJsonAction)
        aboutAction = QtWidgets.QAction("About", self)
        aboutAction.triggered.connect(self.showAbout)
        settingsMenu.addAction(aboutAction)
        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)
        layout = QtWidgets.QVBoxLayout()
        centralWidget.setLayout(layout)
        titleLabel = QtWidgets.QLabel("C3D to MAT Converter")
        titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        titleLabel.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(titleLabel)
        hboxSource = QtWidgets.QHBoxLayout()
        self.sourceLineEdit = QtWidgets.QLineEdit()
        self.sourceLineEdit.setPlaceholderText("Select source folder")
        self.sourceBrowseButton = QtWidgets.QPushButton("Browse...")
        self.sourceBrowseButton.setStyleSheet("background-color: #007BFF; color: white;")
        self.sourceBrowseButton.setToolTip("Click to select the folder containing C3D files.")
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
        self.outputBrowseButton.setToolTip("Click to select the folder where MAT files will be saved.")
        self.outputBrowseButton.clicked.connect(self.browseOutput)
        hboxOutput.addWidget(QtWidgets.QLabel("Output Folder:"))
        hboxOutput.addWidget(self.outputLineEdit)
        hboxOutput.addWidget(self.outputBrowseButton)
        layout.addLayout(hboxOutput)
        hboxFilter = QtWidgets.QHBoxLayout()
        self.filterComboBox = QtWidgets.QComboBox()
        self.filterComboBox.addItems(["enf", "all", "filename"])
        self.filterComboBox.setToolTip("Select the filtering method:\n'enf' - filter based on ENF DESCRIPTION (default)\n'all' - process all files\n'filename' - filter based on filename.")
        self.filterComboBox.currentIndexChanged.connect(self.updateKeywordsPlaceholder)
        hboxFilter.addWidget(QtWidgets.QLabel("Filter Type:"))
        hboxFilter.addWidget(self.filterComboBox)
        layout.addLayout(hboxFilter)
        hboxKeywords = QtWidgets.QHBoxLayout()
        self.keywordsLineEdit = QtWidgets.QLineEdit()
        self.keywordsLineEdit.setPlaceholderText("Enter keywords for ENF description")
        self.keywordsLineEdit.setToolTip("Keywords will be matched against the ENF DESCRIPTION if 'enf' is selected, or filename if 'filename' is selected. Ignored if filter type is 'all'.")
        hboxKeywords.addWidget(QtWidgets.QLabel("Keywords:"))
        hboxKeywords.addWidget(self.keywordsLineEdit)
        layout.addLayout(hboxKeywords)
        self.printCheckBox = QtWidgets.QCheckBox("Print raw point and analog labels (also enables kinetic debug)")
        self.printCheckBox.setToolTip("If checked, raw labels and kinetic debug output will be printed to the log in 9pt font.")
        layout.addWidget(self.printCheckBox)
        self.runButton = QtWidgets.QPushButton("Run Conversion")
        self.runButton.setStyleSheet("background-color: #007BFF; color: white; font-size: 16px;")
        self.runButton.setToolTip("Click to start the conversion process.")
        self.runButton.clicked.connect(self.runConversion)
        layout.addWidget(self.runButton)
        self.logTextEdit = QtWidgets.QTextEdit()
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setStyleSheet("background-color: #f0f0f0;")
        layout.addWidget(self.logTextEdit)

    def updateKeywordsPlaceholder(self):
        filter_type = self.filterComboBox.currentText().lower()
        if filter_type == "enf":
            self.keywordsLineEdit.setPlaceholderText("Enter keywords for ENF description")
            self.keywordsLineEdit.setToolTip("Keywords will be matched against the DESCRIPTION in the ENF file.")
        elif filter_type == "filename":
            self.keywordsLineEdit.setPlaceholderText("Enter keywords for filename")
            self.keywordsLineEdit.setToolTip("Keywords will be matched against the C3D filename.")
        else:
            self.keywordsLineEdit.setPlaceholderText("No keywords needed for 'all'")
            self.keywordsLineEdit.setToolTip("All files will be processed; keywords are ignored.")

    def loadDataFilterDict(self):
        """
        Load data_filter.json from the folder where this GUI's .py file lives.
        Falls back to the executable's folder when frozen (e.g., PyInstaller).
        Ensures keys: meta, events, point, analog (all lists).
        """
        default_filter = {"meta": [], "events": [], "point": [], "analog": []}

        # Resolve the app directory robustly
        if getattr(sys, "frozen", False):  # bundled executable
            app_dir = Path(sys.executable).resolve().parent
        else:
            try:
                app_dir = Path(__file__).resolve().parent
            except NameError:
                # __file__ can be missing in some interactive contexts
                app_dir = Path.cwd()

        json_path = app_dir / "data_filter.json"

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    raise ValueError("Root JSON must be an object/dict.")

                # Merge with defaults and validate list types
                merged = default_filter.copy()
                for key in merged.keys():
                    val = data.get(key, merged[key])
                    merged[key] = val if isinstance(val, list) else merged[key]

                self.data_filter_dict = merged
                self.log(f"Automatically loaded data_filter_dict from {json_path}")
            except Exception as e:
                self.log(f"Error loading {json_path}: {e}. Using default empty filter dict.")
                self.data_filter_dict = default_filter
        else:
            self.log(f"{json_path.name} not found at {json_path}. Using default empty filter dict.")
            self.data_filter_dict = default_filter

    def loadJson(self):
        json_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load JSON Filter File", "", "JSON Files (*.json)")
        if json_path:
            try:
                with open(json_path, "r") as f:
                    self.data_filter_dict = json.load(f)
                self.log(f"Loaded data_filter_dict from {json_path}")
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
        if message.startswith("[label]"):
            message = message.replace("[label]", "")
            self.logTextEdit.append(f"<span style='font-size:9pt;'>{message}</span>")
        else:
            self.logTextEdit.append(message)
        print(message)

    def showAbout(self):
        about_text = (
            "<h2>C3D to MAT Converter</h2>"
            "<p>Author: Harald Penasso with ChatGPT assistance</p>"
            "<p>Date: 2025-03-14</p>"
            "<p>License: MIT License</p>"
            "<p>Packages used: PyQt5, ezc3d, numpy, scipy</p>"
        )
        QtWidgets.QMessageBox.about(self, "About C3D to MAT Converter", about_text)

    def runConversion(self):
        global DEBUG_KINETIC
        DEBUG_KINETIC = self.printCheckBox.isChecked()
        source = self.sourceLineEdit.text().strip()
        output = self.outputLineEdit.text().strip()
        if not source or not output:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select both source and output folders.")
            return
        keywords_str = self.keywordsLineEdit.text().strip()
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []
        filter_type = self.filterComboBox.currentText().lower()
        global PRINT_LABELS, logger
        PRINT_LABELS = self.printCheckBox.isChecked()
        logger = self.log
        self.log("Starting conversion...")
        process_folder(filter_type, keywords, self.data_filter_dict, self.sourceLineEdit.text(), self.outputLineEdit.text())
        self.log("Conversion finished.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QToolTip { color: black; background-color: #f0f0f0; border: 1px solid black; }")
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
