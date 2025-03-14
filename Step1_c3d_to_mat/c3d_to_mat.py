#!/usr/bin/env python3
"""
C3D Data Extractor
------------------
This script recursively searches a user–selected root folder for .c3d files,
applies one of three filtering strategies ("all", "enf", "filename"), extracts
data from the C3D files (metadata, events, point, analog) including time and frame vectors
that correspond exactly to the exports used for TRC/MOT files, and writes additional
information (e.g. point and analog rates, header data) into the meta.
Point and analog data are restructured so that users can access, for example,
struct.point.<CleanLabel> or struct.analog.<CleanLabel>.

Additional modifications:
 - Forceplate validity is extracted from the accompanying ENF file and stored in meta["FORCE_PLATFORM"]["VALID"].
 - MP/VSK files (if present) are merged (if only one unique set exists) and added under meta["mp_info"].
 - In meta["PROCESSING"], if a field is a dict containing a "value", it is replaced directly by that "value".
 - Labels are processed by stripping leading/trailing spaces, splitting by dots, spaces, and underscores,
   removing duplicate consecutive tokens, and (if necessary) shortening tokens to ensure the final field name
   does not exceed 31 characters. Uniqueness is ensured by appending a numeric suffix if needed.
 - Event labels are trimmed and padded to form a 2D char array (MATLAB conversion with cellstr is still necessary).
 - The analog time vector is computed using the header’s analog frame range (shifted by the point time offset),
   and the analog frames are taken directly from the header.
 - The meta information now includes point_rate, analog_rate, analog_first, analog_last, and the complete header.
 - A prompt asks the user whether to print the raw point and analog labels.
 
Author: Harald Penasso with ChatGPT assistance  
Date: 2025-03-14  
License: MIT License  
Packages used: PyQt5, ezc3d, numpy, scipy
"""

import os
import re
import sys
import numpy as np
import ezc3d
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, simpledialog
from scipy.io import savemat

# Module-level flag for printing labels.
PRINT_LABELS = False

# -------------------------------
# Helper functions for GUI prompts
# -------------------------------

def select_folder(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path

def ask_string(title, prompt, initial=""):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    ans = simpledialog.askstring(title, prompt, initialvalue=initial)
    root.destroy()
    return ans

# -------------------------------
# Event Extraction Function
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
        print(f"Warning: could not extract events from {c3d_path}: {e}")
        sorted_times, sorted_labels = [], []
    trimmed_labels = [lab.strip() for lab in sorted_labels]
    if trimmed_labels:
        max_len = max(len(lab) for lab in trimmed_labels)
        padded = np.array([list(lab.ljust(max_len)) for lab in trimmed_labels])
    else:
        padded = np.empty((0,0))
    return {"event_times": list(sorted_times), "event_labels": padded}

# -------------------------------
# Field Label Processing Functions
# -------------------------------

def process_field_label(label):
    label = label.strip()
    tokens = re.split(r'[ \._]+', label)
    tokens = [t for t in tokens if t]
    new_tokens = []
    for token in tokens:
        if not new_tokens or token.lower() != new_tokens[-1].lower():
            new_tokens.append(token)
    candidate = "_".join(new_tokens)
    return candidate

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

# -------------------------------
# ENF File Search and Parsing Functions
# -------------------------------

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
        print(f"Error parsing forceplate validity from {enf_path}: {e}")
    return fp_validity

# -------------------------------
# MP/VSK File Grouping and Parsing Functions
# -------------------------------

def get_unique_mp_basenames(folder):
    files = os.listdir(folder)
    mp_files = [f for f in files if f.lower().endswith(".mp") or f.lower().endswith(".vsk")]
    basenames = {}
    for f in mp_files:
        base = os.path.splitext(f)[0].strip()
        basenames.setdefault(base, []).append(f)
    return basenames

def get_mp_info(folder):
    basenames = get_unique_mp_basenames(folder)
    if not basenames:
        return {}
    if len(basenames) > 1:
        print("Error: Multiple unique MP/VSK file sets detected in folder:")
        for base, files in basenames.items():
            print(f"  {base}: {files}")
        print("Please remove all additional MP/VSK file sets and try again.")
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
        print(f"Error parsing MP file {mp_path}: {e}")
    return mp_params

# -------------------------------
# Optional Filtering on Extracted Data
# -------------------------------

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

# -------------------------------
# Main Processing Function for a C3D File
# -------------------------------

def process_c3d_file(c3d_path, data_filter_dict):
    print(f"Processing {c3d_path} ...")
    c3d_file = ezc3d.c3d(c3d_path)
    # --- Metadata extraction ---
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
        print(f"Warning: No matching ENF file found for {c3d_path}.")
    folder = os.path.dirname(c3d_path)
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
    analog_first = c3d_file['header']['analogs']['first_frame']
    analog_last = c3d_file['header']['analogs']['last_frame']
    meta["point_rate"] = point_rate
    meta["analog_rate"] = analog_rate
    meta["analog_first"] = analog_first
    meta["analog_last"] = analog_last
    meta["header"] = c3d_file.get("header", {})

    # --- Events extraction ---
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

    # --- Point (marker) data extraction ---
    raw_point_labels = c3d_file['parameters']['POINT']['LABELS']['value']
    if "LABELS2" in c3d_file['parameters']['POINT']:
        raw_point_labels += c3d_file['parameters']['POINT']['LABELS2']['value']
    points = c3d_file['data']['points'][0:3, :, :]
    points = np.transpose(points, (2, 1, 0))
    actual_start = c3d_file['parameters']['TRIAL']['ACTUAL_START_FIELD']['value'][0]
    n_frames = points.shape[0]
    frames_vector = np.arange(0, n_frames) + actual_start
    timeC3D = frames_vector / point_rate - 1/point_rate
    point_struct = {"time": timeC3D.tolist(), "frames": frames_vector.tolist()}
    used_point_labels = set()
    point_filter = data_filter_dict.get("point", [])
    for i, lab in enumerate(raw_point_labels):
        lab_stripped = lab.strip()
        if lab_stripped.startswith("*"):
            continue
        # If filtering is specified, only export channels matching one of the keywords.
        if point_filter and not any(kw.lower() in lab_stripped.lower() for kw in point_filter):
            continue
        unique_label = get_unique_field_label(lab_stripped, used_point_labels)
        point_struct[unique_label] = points[:, i, :]
    if PRINT_LABELS:
        print("Raw Point Labels for", c3d_path, ":", raw_point_labels)

    # --- Analog data extraction ---
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
        print("Raw Analog Labels for", c3d_path, ":", raw_analog_labels)

    extracted = {
        "meta": meta,
        "events": events,
        "point": point_struct,
        "analog": analog_struct
    }
    return extracted

# -------------------------------
# File Search and Filtering Functions
# -------------------------------

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
                        description = line.split('=')[1].strip().replace(' ', '_')
                        break
        except Exception as e:
            print(f"Error reading {enf_path}: {e}")
            return False
        if description is None:
            return False
        return any(description.lower().startswith(kw.lower()) for kw in keywords)
    elif filter_type == "filename":
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        return any(kw.lower() in base.lower() for kw in keywords)
    else:
        return False

# -------------------------------
# Main Folder Processing
# -------------------------------

def process_folder(filter_type, keywords, data_filter_dict):
    print("Select the root folder to search for C3D files:")
    source_root = select_folder("Select the root folder to search for C3D files")
    print("Select the output folder for MAT files:")
    output_root = select_folder("Select the output folder for MAT files")
    print(f"Searching in: {source_root}")
    print(f"Output will be saved in: {output_root}")
    c3d_files = find_c3d_files(source_root)
    print(f"Found {len(c3d_files)} C3D files.")
    processed_count = 0
    for file_path in c3d_files:
        if not file_matches_filter(file_path, filter_type, keywords):
            continue
        extracted = process_c3d_file(file_path, data_filter_dict)
        rel_path = os.path.relpath(os.path.dirname(file_path), source_root)
        out_folder = os.path.join(output_root, rel_path)
        os.makedirs(out_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(out_folder, base_name + ".mat")
        savemat(out_file, extracted)
        print(f"Saved extracted data to {out_file}")
        processed_count += 1
    print(f"Finished processing. {processed_count} files were extracted and saved.")

# -------------------------------
# Main Script Execution
# -------------------------------

if __name__ == "__main__":
    print("Starting c3d_to_mat.py ...")
    data_filter_dict = {
        "meta": [],
        "events": [],
        "point": ['RNeckAngles', 'LNeckAngles', 'LShoulderAngles', 'LElbowAngles', 'LWristAngles', 'RShoulderAngles', 'RElbowAngles', 'RWristAngles', 'RThoraxAngles', 'LThoraxAngles', 'RHeadAngles', 'LHeadAngles', 'LPelvisAngles', 'RPelvisAngles', 'LHipAngles', 'RHipAngles', 'LKneeAngles', 'RKneeAngles', 'LAnkleAngles', 'RAnkleAngles'],
        "analog": ['Force.Fx7', 'Force.Fy7', 'Force.Fz7', 'Force.Fx8', 'Force.Fy8', 'Force.Fz8', 'Force.Fx9', 'Force.Fy9', 'Force.Fz9', 'EMG.EMG_01_R M.tibialis anterior', 'EMG.EMG_02_R M.peroneus long.', 'EMG.EMG_03_R M.gastro. lat.', 'EMG.EMG_04_R M.gastroc.med.', 'EMG.EMG_05_R M.soleus', 'EMG.EMG_06_R M.rectus fem.', 'EMG.EMG_07_R M.gluteus maximus', 'EMG.EMG_08_R M. gluteus medius', 'EMG.EMG_09_R M.vastus lat.', 'EMG.EMG_10_R M.vastus med.', 'EMG.EMG_11_R M.semitendinosus', 'EMG.EMG_12_R M.biceps femoris', 'EMG.EMG_13_R M.erector spine', 'EMG.EMG_17_L M.tibialis anterior', 'EMG.EMG_18_L M.peroneus long.', 'EMG.EMG_19_L M.gastro. lat.', 'EMG.EMG_20_L M.gastroc.med.', 'EMG.EMG_21_L M.vastus lat.', 'EMG.EMG_22_L M.vastus med.', 'EMG.EMG_23_L M.semitendinosus', 'EMG.EMG_24_L M.biceps femoris', 'EMG.EMG_25_L M.soleus', 'EMG.EMG_26_L M.rectus fem.', 'EMG.EMG_27_L M.erector spine', 'EMG.EMG_31_L M.gluteus maximus', 'EMG.EMG_32_L M. gluteus medius']
    }
    kw_str = ask_string("Keywords", "Enter keywords (comma separated) for filtering (default: stand, walk):", "stand, walk")
    keywords = [k.strip() for k in kw_str.split(",") if k.strip()] if kw_str else []
    filter_type = ask_string("Filter type", "Enter filter type (all, enf, filename):", "enf")
    print_labels_response = ask_string("Print Labels", "Print raw point and analog labels? (yes/no):", "no")
    PRINT_LABELS = print_labels_response.lower() in ["yes", "y"]
    process_folder(filter_type.lower(), keywords, data_filter_dict)
    print("c3d_to_mat.py finished.")
