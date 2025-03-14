import os, sys
import scipy.io

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils_py.mat2dict import loadmat_to_dict
from utils_py.cycle_help import extract_cycle_periods, split_data_by_cycles

# Add the repository root (one level up) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Build the absolute path to the .mat file
mat_file = os.path.abspath(os.path.join(current_dir, '..', 'example', 'mat', 'from_C3D', 'walk.mat'))

# Load the .mat file as a Python dictionary
data_dict = loadmat_to_dict(mat_file)

# Get the event times and labels from the loaded data
event_times = data_dict['events']['event_times']
event_labels = data_dict['events']['event_labels']

# Define cycle boundaries using a three-event sequence.
# For left_stride, for example, a cycle is defined as:
#   start: 'Left Foot Strike'
#   central: 'Left Foot Off'
#   end: 'Left Foot Strike'
cycles_from_to = {
    'left_stride': ['Left Foot Strike', 'Left Foot Off', 'Left Foot Strike'],
    'right_stride': ['Right Foot Strike', 'Right Foot Off', 'Right Foot Strike']
}

# Extract cycle periods based on the defined boundaries.
# The extract_cycle_periods function will look for the central event in each cycle.
cycle_periods = extract_cycle_periods(event_times, event_labels, cycles_from_to)

# Split the point and analog data by cycles, adding central event info if available.
cycle_data = split_data_by_cycles(data_dict, cycle_periods, cycles_from_to)

# Build the output dictionary by merging meta and events from the original,
# and adding the cycle data under a new key 'cycles'.
if 'meta' in data_dict:
    cycle_data['meta'] = data_dict['meta']
if 'events' in data_dict:
    cycle_data['events'] = data_dict['events']

# Determine the directory of the original .mat file
mat_dir = os.path.dirname(mat_file.replace('from_C3D','split_to_cycles'))
# Build the output file path next to the original
out_file = os.path.join(mat_dir, 'walk_splitCycles.mat')

# Save the cycle_data dictionary to a new .mat file.
scipy.io.savemat(out_file, cycle_data)