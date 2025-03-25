import os
import ezc3d
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime
import numpy as np

def contains_sequence(events, sequence):
    """
    Return True if the list 'events' contains the contiguous subsequence 'sequence'.
    """
    n = len(events)
    m = len(sequence)
    if m == 0:
        return True
    for i in range(n - m + 1):
        if events[i:i+m] == sequence:
            return True
    return False

def extract_event_times_and_labels(c3d_file):
    """
    Extract event times and labels from the C3D file, ordered by event times.
    Combines each event's CONTEXT with its LABEL (trimming extra spaces).
    
    Returns:
        dict: Dictionary with keys 'event_times' and 'event_labels'
    
    Raises:
        ValueError: if any required EVENT key is missing.
    """
    try:
        times = c3d_file['parameters']['EVENT']['TIMES']['value'][1]
        labels = c3d_file['parameters']['EVENT']['LABELS']['value']
        contexts = c3d_file['parameters']['EVENT']['CONTEXTS']['value']
    except Exception as e:
        raise ValueError("Missing EVENT keys: " + str(e))
    
    combined_labels = []
    for i, ctx in enumerate(contexts):
        combined = f"{ctx.strip()} {labels[i].strip()}"
        combined_labels.append(combined)
    events = sorted(zip(times, combined_labels))
    if not events:
        return {'event_times': [], 'event_labels': []}
    sorted_times, sorted_labels = zip(*events)
    return {'event_times': list(sorted_times), 'event_labels': list(sorted_labels)}

def extract_description(enf_file_path):
    """
    Extract the trial description from an ENF file.
    Looks for a line starting with 'DESCRIPTION='.
    
    Returns:
        str: The extracted description (in lowercase), or None if not found.
    """
    try:
        with open(enf_file_path, 'r') as file:
            for line in file:
                if line.startswith('DESCRIPTION='):
                    return line.split('=')[1].strip().lower()
    except Exception as e:
        # Debug print commented out.
        # print(f"DEBUG: Error reading ENF file '{enf_file_path}': {e}")
        pass
    return None

def select_folder(prompt):
    """Prompt the user to select a folder and return its normalized path."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return os.path.normpath(folder_path)

def get_keywords():
    """
    Ask the user to enter a comma-separated list of keywords for filtering files.
    Files with a trial description (from the ENF file) that starts with one of these keywords will be processed.
    Enter '__all' to process all files.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    kw_str = simpledialog.askstring("Keyword Filter",
                                    "Enter keywords (comma separated) for processing files.\n"
                                    "Files with a trial description (from the ENF file) that starts with one of these keywords will be processed.\n"
                                    "Enter '__all' to process all files:",
                                    initialvalue="__all")
    root.destroy()
    if not kw_str or kw_str.strip() == "":
        return ["__all"]
    else:
        return [kw.strip() for kw in kw_str.split(",")]

def get_static_keyword(default="stand"):
    """Ask the user for the static trial keyword; default is 'stand'."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    keyword = simpledialog.askstring("Static Trial Keyword",
                                     "Enter the keyword for static trials (to exempt from event checking):",
                                     initialvalue=default)
    root.destroy()
    return keyword.lower() if keyword else default.lower()

def get_extra_marker_list(default_list=None):
    """
    Ask the user to enter a comma-separated list of extra marker names to check.
    If left empty, returns an empty list.
    (This will be labeled as "Point Data Checked" in the header.)
    """
    if default_list is None:
        default_list = ["RThoraxAngles", "LThoraxAngles", "LPelvisAngles", "RPelvisAngles",
                        "LFootProgressAngles", "RFootProgressAngles", "LHipAngles", "RHipAngles",
                        "LKneeAngles", "RKneeAngles", "LAnkleAngles", "RAnkleAngles"]
    default_str = ",".join(default_list)
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    marker_str = simpledialog.askstring("Extra Marker Data Check",
                                        "Enter extra marker names to check (comma separated). Leave blank if not needed:",
                                        initialvalue=default_str)
    root.destroy()
    if marker_str is None or marker_str.strip() == "":
        return []
    else:
        return [m.strip() for m in marker_str.split(",") if m.strip()]

def get_event_sequence(default_left, default_right):
    """
    Ask the user to input desired event sequences for left and right feet, 
    as comma-separated lists.
    
    Returns:
        tuple: (left_sequence, right_sequence) as lists.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    left_seq_str = simpledialog.askstring("Left Foot Event Sequence",
                                          "Enter desired left foot event sequence (comma separated):",
                                          initialvalue=",".join(default_left))
    right_seq_str = simpledialog.askstring("Right Foot Event Sequence",
                                           "Enter desired right foot event sequence (comma separated):",
                                           initialvalue=",".join(default_right))
    root.destroy()
    left_seq = [s.strip() for s in left_seq_str.split(",")] if left_seq_str and left_seq_str.strip() else default_left
    right_seq = [s.strip() for s in right_seq_str.split(",")] if right_seq_str and right_seq_str.strip() else default_right
    return left_seq, right_seq

def get_additional_event_label():
    """
    Ask the user if they want to check for an additional event label.
    
    Returns:
        str: the label or an empty string.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    label = simpledialog.askstring("Additional Event Label",
                                   "Enter an additional event label to check for (leave empty if none):",
                                   initialvalue="")
    root.destroy()
    return label.strip()

def log_message(log_list, message):
    """Append a message with a timestamp to the given log list."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_list.append(f"[{timestamp}] {message}")

def create_log_header(user_inputs, source_root, output_folder, start_time, end_time):
    """Return a header string listing all user inputs, root folder, output folder, start and end time."""
    header_lines = [
        "User Inputs:",
        f"  Keywords: {user_inputs['keywords']}",
        f"  Static Trial Keyword: {user_inputs['static_keyword']}",
        f"  Point Data Checked: {user_inputs['extra_markers']}",
        f"  Left Foot Sequence: {user_inputs['left_seq']}",
        f"  Right Foot Sequence: {user_inputs['right_seq']}",
        f"  Additional Event Label: {user_inputs['additional_event_label']}",
        f"Root Folder: {source_root}",
        f"Output Folder: {output_folder}",
        f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 60
    ]
    return "\n".join(header_lines)

def check_c3d_consistency():
    """
    Traverse a user-selected folder for .c3d files and perform the following checks:
    
    - Filter files based on a list of keywords (from the ENF trial description):
         Only files with a trial description (from the ENF file) that starts with one of the specified keywords will be processed.
    - Exempt files identified as static trials using the static trial keyword.
    - Verify that the EVENT keys ('TIMES', 'LABELS', 'CONTEXTS') and POINT LABELS exist.
    - Extract and sort event times and labels.
    - Separate merged event labels into left and right events.
    - Check that the left events contain the desired left foot sequence and the right events contain the desired right foot sequence.
    - Optionally check for the presence of an additional event label.
    - Check for extra marker labels.
    
    The script prints each folder path once and for each file (that is processed) prints:
         "   ... <filename> with ENF description: '<description>'"
         
    Two log files are produced:
       1. Files Checked Log – shows which files were checked or skipped (only one message per file).
       2. Consistency Log – shows, for each processed file, the consistency issues.
    Both log files include a header with all user inputs, the source and output folders, and the start and end times.
    If the Consistency Log contains any messages (beyond the header), they are printed to the console with a "      ... " indent.
    """
    files_checked_log = []
    consistency_log = []
    
    source_root = select_folder("Select source folder containing C3D files")
    if not source_root:
        print("No source folder selected. Exiting.")
        return
    output_folder = select_folder("Select folder to save the log files")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return

    # Capture start time.
    start_time = datetime.now()

    # Get user inputs.
    keywords = get_keywords()  # E.g., ["walk", "ramp"] or ["__all"]
    static_keyword = get_static_keyword(default="stand")
    extra_markers = get_extra_marker_list()
    default_left_seq = ["Left Foot Strike", "Left Foot Off", "Left Foot Strike"]
    default_right_seq = ["Right Foot Strike", "Right Foot Off", "Right Foot Strike"]
    left_seq, right_seq = get_event_sequence(default_left_seq, default_right_seq)
    additional_event_label = get_additional_event_label()
    
    user_inputs = {
        "keywords": keywords,
        "static_keyword": static_keyword,
        "extra_markers": extra_markers,
        "left_seq": left_seq,
        "right_seq": right_seq,
        "additional_event_label": additional_event_label
    }
    
    # Initially add a temporary header (without end time).
    temp_header = "User Inputs Header (end time not set yet)"
    files_checked_log.append(temp_header)
    consistency_log.append(temp_header)
    
    last_printed_folder = None
    for root_dir, _, files in os.walk(source_root):
        c3d_files = [f for f in files if f.lower().endswith(".c3d")]
        if not c3d_files:
            continue
        if root_dir != last_printed_folder:
            print(root_dir)
            last_printed_folder = root_dir
        for file_name in c3d_files:
            file_path = os.path.join(root_dir, file_name)
            # Locate corresponding ENF file; try ".Trial.enf" first, then ".enf".
            base_name = os.path.splitext(file_name)[0]
            candidate_paths = [os.path.join(root_dir, base_name + ".Trial.enf"),
                               os.path.join(root_dir, base_name + ".enf")]
            enf_file_path = None
            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    enf_file_path = candidate
                    break
            if not enf_file_path:
                log_message(files_checked_log, f"ENF file for '{file_path}' not found; skipping.")
                continue
            
            enf_desc = extract_description(enf_file_path)
            if enf_desc is None:
                log_message(files_checked_log, f"Could not extract trial description from ENF file '{enf_file_path}'; skipping.")
                continue
            
            trial_desc = enf_desc  # Already in lowercase.
            
            # Apply filtering based on keywords.
            if keywords != ["__all"] and not any(trial_desc.startswith(k.lower()) for k in keywords):
                log_message(files_checked_log, f"File '{file_path}' does not match keyword filter; skipping.")
                continue
            
            # Skip static trials.
            if static_keyword in trial_desc:
                log_message(files_checked_log, f"File '{file_path}' identified as static trial; skipping event check.")
                continue
            
            # File passes filtering; print to console and log as "checked".
            print(f"   ... {file_name} with ENF description: '{trial_desc}'")
            log_message(files_checked_log, f"File '{file_path}' with ENF description: '{trial_desc}' checked.")
            
            try:
                c3d = ezc3d.c3d(file_path)
            except Exception as e:
                log_message(files_checked_log, f"Error loading file '{file_path}': {e}")
                continue
            
            if "POINT" not in c3d['parameters'] or "LABELS" not in c3d['parameters']["POINT"]:
                log_message(consistency_log, f"Missing POINT or POINT LABELS in file: {file_path}")
            
            if "EVENT" not in c3d['parameters']:
                log_message(consistency_log, f"Missing 'EVENT' parameter in file: {file_path}")
            else:
                event_params = c3d['parameters']['EVENT']
                required_keys = ["TIMES", "LABELS", "CONTEXTS"]
                missing_keys = [key for key in required_keys if key not in event_params]
                if missing_keys:
                    log_message(consistency_log, f"Missing EVENT keys {missing_keys} in file: {file_path}")
                else:
                    try:
                        events_dict = extract_event_times_and_labels(c3d)
                    except Exception as e:
                        log_message(consistency_log, f"Error extracting events from file '{file_path}': {e}")
                        events_dict = {'event_times': [], 'event_labels': []}
                    
                    sorted_labels = events_dict.get('event_labels', [])
                    # Separate merged event labels into left and right events.
                    left_events = [label for label in sorted_labels if "left" in label.lower()]
                    right_events = [label for label in sorted_labels if "right" in label.lower()]
                    
                    valid_left = contains_sequence(left_events, left_seq)
                    valid_right = contains_sequence(right_events, right_seq)
                    if not (valid_left or valid_right):
                        log_message(consistency_log, f"Missing valid foot event sequence in file: {file_path}. Expected left: {left_seq} in left events {left_events} or right: {right_seq} in right events {right_events}.")
                    
                    if additional_event_label and additional_event_label not in sorted_labels:
                        log_message(consistency_log, f"Additional event label '{additional_event_label}' not found in file: {file_path}")
            
            if extra_markers:
                try:
                    point_labels = c3d['parameters']['POINT']['LABELS']['value']
                except Exception:
                    point_labels = []
                missing_markers = [marker for marker in extra_markers if marker not in point_labels]
                if missing_markers:
                    log_message(consistency_log, f"Missing point data {missing_markers} in file: {file_path}")
    
    # Capture end time.
    end_time = datetime.now()
    
    # Create final header with all details.
    final_header = create_log_header(user_inputs, source_root, output_folder, start_time, end_time)
    # Replace the temporary header with the final one.
    files_checked_log[0] = final_header
    consistency_log[0] = final_header
    
    # Write the files checked log.
    files_checked_log_filename = os.path.join(output_folder, f"files_checked_log_{end_time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(files_checked_log_filename, "w") as f_log:
        f_log.write("\n".join(files_checked_log))
    
    # Write the consistency log.
    consistency_log_filename = os.path.join(output_folder, f"consistency_log_{end_time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(consistency_log_filename, "w") as c_log:
        c_log.write("\n".join(consistency_log))
    
    # If there are consistency issues beyond the header, print them with an indent.
    if len(consistency_log) > 1:
        print("Consistency issues:")
        for message in consistency_log[1:]:
            print("      ... " + message)
    
    messagebox.showinfo("Consistency Check Complete", f"Files Checked Log saved as:\n{files_checked_log_filename}\n\nConsistency Log saved as:\n{consistency_log_filename}")
    print(f"Files Checked Log saved as: {files_checked_log_filename}")
    print(f"Consistency Log saved as: {consistency_log_filename}")

if __name__ == "__main__":
    check_c3d_consistency()
