# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:27:08 2025

@author: harald.penasso
"""
%matplotlib qt
#%%
import os
import sys
import pandas as pd 
import pickle, types, io, dill # save workspace
import textwrap
import itertools # Added for grid search
import numpy as np
import seaborn as sns
import spm1d
# Monkey‐patch _reml.traceMV to accept (V, X, c) but only call the original with (V, X)
from spm1d.stats import _reml
_orig_traceMV = _reml.traceMV
def _patch_traceMV(V, X, *args, **kwargs):
    # ignore extra args (e.g. c), pass only V and X to the original
    return _orig_traceMV(V, X)
_reml.traceMV = _patch_traceMV
# End Monkey‐patch _reml.traceMV
import time 
from io import StringIO
from scipy.stats import shapiro, kruskal
from scipy.signal import butter, filtfilt, firwin
from scipy.interpolate import Akima1DInterpolator
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, soft_dtw #, dtw 
from tslearn.clustering import silhouette_score as dtw_silhouette_score
from skopt import gp_minimize
from skopt.space import Real
#from skopt.utils import use_named_args
#from tslearn.utils import to_time_series_dataset
#from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from ezc3d import c3d
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
#from scipy.cluster.hierarchy import linkage, fcluster
#from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
#%% 00. USER-SETTINGS: Choose below...
#=========================================================================
# General settings
cluster_method = "agglo_ward" # ts_kmeans or agglo_ward
time_warping_method = "softdtw" # soft_dtw or dtw !!!! not yet working, still hard coded, fix!!!

# $ Path Hard-Codes
if os.name == 'nt':  # Windows
    username = "harald.penasso" # harald.penasso User
    onedrivePath = os.path.join(r"C:\Users", username, r"OneDrive - FH Campus Wien")
else:
    onedrivePath = r"/Users/User/OneDrive - FH Campus Wien"
#C:\Data_Local\opensim_WalkA_extracted_upd
version = 'simResults_opensim_WalkA_extracted_upd_v2'
data_folder = os.path.join(onedrivePath, "Documents", "Projekte", "KneeSim", "kin2KJRL", "data")

metaData_fullFilePath = os.path.join(data_folder, "refDataRef.xlsx")
output_folder = os.path.join(os.path.split(metaData_fullFilePath)[0], "level_walking")
workspace_dump_path = os.path.join(output_folder, "full_workspace.pkl") # dump workspace there for debugging

n_comparisons = 18
corrected_alpha = 0.05/n_comparisons

# openSim RRA thresholds
th_rms_residual = 0.09
th_max_residual = 0.265

# --- Grid-search parameters ---
coarse_ks = [2,3] # Example: test k=2 and k=3 initially
coarse_gammas = [0.0005, 0.1, 1] # Coarse gamma values
initial_dm_method = 'softdtw' # Focus on softDTW for gamma optimization
use_scaler = '_scMnVar' # Or '' or your specific suffix

# Step 5 - Grid Search Parameters
# --- Define Search Space for Gamma ---
# Adjust bounds based on initial grid results and expectations
# Using log-uniform is good if you expect optimal gamma to span orders of magnitude
GAMMA_SPACE = Real(5e-5, 1e1, prior='log-uniform', name='gamma')

# Global variable to store clustering data (or pass it explicitly)
# This avoids reloading/recalculating it in each objective function call
CLUSTERING_DATA_CACHE = None
RANDOM_STATE_CACHE = 42 # Ensure consistency NEEDS TO BE FIXED: IS HARDCODED NOW

# Step 6.4 - Bayesian optimization settings
n_gamma_opt_calls = 5 #15 # Number of optimization iterations (adjust as needed)

# Define sub-phases with their corresponding percentage intervals.
sub_phases = {
    "Loading_Response": (0, 10),
    "Midstance": (10, 30),
    "First_Stance_Phase_Half": (0, 30),
    "Terminal_Stance": (30, 50),
    "Pre_Swing": (50, 60),
    "Second_Stance_Phase_Half": (30, 60),
    "Initial_Swing": (60, 73),
    "Mid_Swing": (73, 87),
    "Terminal_Swing": (87, 100),
    "Swing_Phase": (60, 100)
}

#%% 0. SET-UP: Define OneDrive paths and input data files 
#=========================================================================
simResults_fullFilePath = os.path.join(data_folder, version + ".csv")

# Read simulation results and remove duplicates
df_simResults = pd.read_csv(simResults_fullFilePath)
df_simResults = df_simResults.drop_duplicates(subset=['ID', 'ikFilt_MOT_fullFilePath', 'event_number', 'sideExerciseLeg'])

# Keep only walking cycles based on stride labels.
subset = ((df_simResults["sideExerciseLeg"] == "left_stride") | (df_simResults["sideExerciseLeg"] == "right_stride"))
df_simResults = df_simResults.loc[subset, :].reset_index(drop=True)

# Load and add participant meta-data from Excel.
meta_df = pd.read_excel(metaData_fullFilePath)

# Convert IDs to uppercase and join the data.
df_simResults.ID = df_simResults.ID.str.upper()
meta_df.ID = meta_df.ID.str.upper()
df_simResults = df_simResults.join(meta_df.set_index('ID'), on='ID').reset_index()

#%% 0.a. Replace file root paths if needed (example provided)
oldRoot = r'D:\Data_local\opensim_WalkA_extracted_upd'
newRoot = r'C:\Data_Local\opensim_WalkA_extracted_upd'
df_simResults = df_simResults.map(lambda x: x.replace(oldRoot, newRoot) if isinstance(x, str) else x)

#=========================================================================
#%% Dump Workspace In/Out functions for debugging
#=========================================================================
def workspace_out(output_folder, filename="full_workspace.pkl", use_dill=True):
    """
    Dumps all picklable globals (no modules, no functions, no open files)
    into `output_folder/filename`. Returns the full path to the file.
    """
    # allow dill for more coverage
    try:
        import dill
    except ImportError:
        dill = None
    import pickle

    # ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)

    # snapshot names so dict won't change underneath us
    names = list(globals().keys())
    serial = {}
    for name in names:
        if name.startswith("__") or name in ("pickle","os","types","io","dill"):
            continue
        val = globals()[name]
        if isinstance(val, types.ModuleType) or callable(val) or isinstance(val, io.IOBase, pd.ExcelWriter):
            continue
        try:
            # test serializability
            if use_dill and dill:
                dill.dumps(val)
            else:
                pickle.dumps(val)
        except Exception:
            continue
        serial[name] = val

    # write it out
    with open(path, "wb") as f:
        if use_dill and dill:
            dill.dump(serial, f)
        else:
            pickle.dump(serial, f)

    print(f"[workspace_out] Saved {len(serial)} items → {path}")
    return path
#=========================================================================    
def workspace_in(path, inject=False, use_dill=True):
    """
    Loads a workspace file created by workspace_out().
    If inject=True, updates globals() with its contents.
    Returns the dict of loaded variables.
    """
    try:
        import dill
    except ImportError:
        dill = None
    import pickle

    with open(path, "rb") as f:
        if use_dill and dill:
            data = dill.load(f, ignore = True)
        else:
            data = pickle.load(f, ignore = True)

    print(f"[workspace_in] Loaded {len(data)} items from {path}")
    if inject:
        globals().update(data)
        print(f"[workspace_in] Injected into globals(): {list(data.keys())}")
    return data
#=========================================================================    
def safe_workspace_in(path, inject=False, prefer_dill=True):
    """
    Load a workspace while intercepting any classes from `pandas.io.excel*`
    so the unpickler never calls their constructors. Afterwards, drop any
    such objects from the dict.
    """
    import io, pickle

    # universal harmless stub to stand in for Excel writers
    class _ExcelWriterStub:
        def __init__(self, *a, **k): pass

    # --- Custom unpicklers that redirect pandas.io.excel* classes to stub ---
    class _SafePickleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("pandas.io.excel"):
                return _ExcelWriterStub
            return super().find_class(module, name)

    try:
        import dill
        class _SafeDillUnpickler(dill.Unpickler):
            def find_class(self, module, name):
                if module.startswith("pandas.io.excel"):
                    return _ExcelWriterStub
                return super().find_class(module, name)
    except Exception:
        dill = None
        _SafeDillUnpickler = None

    # --- Try dill-first (if requested and available), else pickle ---
    with open(path, "rb") as f:
        if prefer_dill and dill is not None:
            try:
                data = _SafeDillUnpickler(f).load()
            except Exception:
                # fall back to pickle with our safe unpickler
                f.seek(0)
                data = _SafePickleUnpickler(f).load()
        else:
            try:
                data = _SafePickleUnpickler(f).load()
            except Exception:
                if dill is None:
                    raise
                f.seek(0)
                data = _SafeDillUnpickler(f).load()

    # --- Purge any stubbed writer objects from the loaded dict ---
    if isinstance(data, dict):
        to_drop = [k for k, v in data.items() if isinstance(v, _ExcelWriterStub)]
        for k in to_drop:
            data.pop(k, None)

    print(f"[safe_workspace_in] Loaded {len(data) if isinstance(data, dict) else '?'} items from {path}")
    if inject and isinstance(data, dict):
        globals().update(data)
        print(f"[safe_workspace_in] Injected into globals(): {list(data.keys())}")
    return data


#=========================================================================
#%% Update numerical cluster labels
#=========================================================================
def digit_replace(match):
    d = match.group()
    return chr(ord('A') + int(d))
#=========================================================================
#%% 1. Helper Function: readTRC
#=========================================================================
def readTRC(filepath):
    df = pd.read_csv(
        filepath,
        delimiter='\t',
        skiprows=3,
        header=[0, 1]
    )
    header_0 = df.columns.get_level_values(0).tolist()
    header_1 = df.columns.get_level_values(1).tolist()
    updated_header_0 = []
    last_label = None
    for i, col in enumerate(header_0):
        if i < 2:
            updated_header_0.append(col)
            last_label = col
        else:
            last_label = col if col != f"Unnamed: {i}_level_0" else last_label
            updated_header_0.append(last_label)
    xyz_cycle = ['X', 'Y', 'Z']
    updated_header_1 = ["" if i < 2 else xyz_cycle[(i - 2) % 3] for i in range(len(header_1))]
    updated_headers = pd.MultiIndex.from_tuples(zip(updated_header_0, updated_header_1))
    df.columns = updated_headers
    return df.set_index(('Time', ''))

#%% 2. Preprocessing and Valid Cycle Selection
#=========================================================================

# (Optional visualization of residuals)
plt.hist(np.sort(df_simResults["rms_residual_norm_maxF"]), bins=200, label='RMS Residual', alpha=0.5)
plt.hist(np.sort(df_simResults["max_residual_norm_maxF"]), bins=200, label='Max Residual', alpha=0.5)
plt.legend()
plt.show()

# Define residual thresholds and select valid cycles.
exclbyRMS_pc = np.mean(df_simResults["rms_residual_norm_maxF"] >= th_rms_residual) * 100
exclbyMAX_pc = np.mean(df_simResults["max_residual_norm_maxF"] >= th_max_residual) * 100
selValidCycles = (df_simResults["rms_residual_norm_maxF"] < th_rms_residual) & (df_simResults["max_residual_norm_maxF"] < th_max_residual)
exclbyRMSandMAX_pc = np.mean(selValidCycles) * 100
keep_cycles = np.sum(selValidCycles)

# Keep valid cycles only.
df_simResults = df_simResults.loc[selValidCycles, :].reset_index(drop=True)

# Quick and dirty fix - remove
xxx = np.ones((652,))
xxx[621] = 0 #err
xxx[632] = 0 #err
df_simResults = df_simResults.loc[xxx==1, :].reset_index(drop=True)
# Quick and dirty fix end - remove

# Compute additional participant statistics like BMI.
df_simResults['BMI'] = df_simResults['weight_kg'] / ((df_simResults['height_mm'] / 1000) ** 2)

# Group by Sex and compute descriptive statistics.
grouped = df_simResults.drop_duplicates(subset=['ID']).groupby(['Sex'])
participant_metaInfo = grouped.agg({
    'Sex': ['count'],
    'height_mm': ['mean', 'std', 'median', 'min', 'max'],
    'weight_kg': ['mean', 'std', 'median', 'min', 'max'],
    'BMI': ['mean', 'std', 'median', 'min', 'max'],
    'age': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

modelAST_metaInfo = grouped.agg({
    'Sex': ['count'],
    'AST_nCycles': ['mean', 'std', 'median', 'min', 'max'],
    'AST_rmse_cm': ['mean', 'std', 'median', 'min', 'max'],
    'AST_maxe_cm': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

df_simResults["ik_rmsErr_cm"] = 100 * df_simResults.ik_rmsErr_mm
df_simResults["ik_maxErr_cm"] = 100 * df_simResults.ik_maxErr_mm
dynamic_trial_stats = df_simResults.drop_duplicates(subset=['ID','ikFilt_MOT_fullFilePath']).groupby(['Sex']).agg({
    'Sex': ['count'],
    'ik_rmsErr_cm': ['mean', 'std', 'median', 'min', 'max'],
    'ik_maxErr_cm': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

df_simResults["rms_residual_norm_maxF_pc"] = 100 * df_simResults.rms_residual_norm_maxF
df_simResults["max_residual_norm_maxF_pc"] = 100 * df_simResults.max_residual_norm_maxF
cycle_stats = df_simResults.drop_duplicates(subset=['ID','ikFilt_MOT_fullFilePath','event_number','sideExerciseLeg']).groupby('Sex').agg({
    'Sex': ['count'],
    'rms_residual_norm_maxF_pc': ['mean', 'std', 'median', 'min', 'max'],
    'max_residual_norm_maxF_pc': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

#%% 3. Save Descriptive Tables to Excel (Initial Output)
#=========================================================================
os.makedirs(output_folder, exist_ok=True)
excel_path = os.path.join(output_folder, 'tables.xlsx')

with pd.ExcelWriter(excel_path) as writer:
    participant_metaInfo.to_excel(writer, sheet_name='participant_metaInfo', index=True)
    modelAST_metaInfo.to_excel(writer, sheet_name='modelAST_metaInfo', index=True)
    dynamic_trial_stats.to_excel(writer, sheet_name='dynamic_trial_stats', index=True)
    cycle_stats.to_excel(writer, sheet_name='cycle_stats', index=True)

captions = {
    'participant_metaInfo': 'Participant meta information grouped by sex. ' +
                             f"Percentage of cycles excluded by RMS was {exclbyRMS_pc:.2f}%, " +
                             f"and those excluded by MAX was {exclbyMAX_pc:.2f}%, thus valid cycles: {exclbyRMSandMAX_pc:.2f}% ({keep_cycles} cycles).",
    'modelAST_metaInfo': 'Automatic scaling tool metrics information grouped by sex.',
    'dynamic_trial_stats': 'Dynamic trial based descriptive statistics grouped by exercise, including RMS and max errors.',
    'cycle_stats': 'Cycle-based descriptive statistics of final data, including RMS and max residual norms.'
}
for sheet, caption in captions.items():
    print(f"Caption for {sheet}: {caption}")

#%% 4. Additional Functions: Filtering, Joint Reaction Reading, and Processing
#=========================================================================

def butter_lowpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    corrected_cutoff = cutoff / 0.802
    normal_cutoff = corrected_cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(signal, cutoff_frequency, frame_rate, order=2, package="scipy", filter_type="butter"):
    if (package == "scipy") and (filter_type == "butter"):
        b, a = butter_lowpass(cutoff_frequency, frame_rate, order)
        non_zero_indices = np.nonzero(signal)[0]
        if len(non_zero_indices) > 3 * (max(len(a), len(b))) - 1:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1]
            filtered_segment = filtfilt(b, a, signal[start_idx:end_idx+1])
            filt_signal = np.zeros_like(signal)
            filt_signal[start_idx:end_idx+1] = filtered_segment
            if start_idx > 0:
                filt_signal[:start_idx] = signal[:start_idx]
            if end_idx < len(signal) - 1:
                filt_signal[end_idx+1:] = signal[end_idx+1:]
        else:
            filt_signal = signal
    elif (package == "scipy") and (filter_type == "FIR"):
        numtaps = 13
        pad_width = numtaps
        signal = np.pad(signal, pad_width, mode='reflect')
        window = "hamming"
        fir_coefficients = firwin(numtaps, cutoff_frequency, pass_zero="lowpass", fs=frame_rate, window=window)
        filtered_signal = np.convolve(signal, fir_coefficients, mode='same')
        filtered_signal = filtered_signal[pad_width:-pad_width]
        return filtered_signal
    elif package == "opensim":
        import opensim as osim
        table = osim.TimeSeriesTable()
        table.setColumnLabels(["Signal"])
        time = np.arange(len(signal)) / frame_rate
        for t, value in zip(time, signal):
            row_vector = osim.RowVector([value])
            table.appendRow(t, row_vector)
        non_zero_indices = np.nonzero(signal)[0]
        if len(non_zero_indices) > 0:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1]
            pad_data = False
            filtered_segment = table.clone()
            osim.TableUtilities.filterLowpass(filtered_segment, cutoff_frequency, pad_data)
            filt_signal = np.zeros_like(signal)
            for i in range(start_idx, end_idx + 1):
                filt_signal[i] = filtered_segment.getRowAtIndex(i)[0]
            if start_idx > 0:
                filt_signal[:start_idx] = signal[:start_idx]
            if end_idx < len(signal) - 1:
                filt_signal[end_idx+1:] = signal[end_idx+1:]
    else:
        raise ValueError("Unsupported lpf_method. Use 'scipy' or 'opensim'.")
    return filt_signal

def read_joint_reaction_file(file_path, cols2return):
    if isinstance(cols2return, str):
        cols2return = (cols2return,)
    with open(file_path, 'r') as file:
        while True:
            line = file.readline().strip()
            if line == 'endheader':
                break
        header_line = file.readline().replace('\t', ' ').strip()
        header_line = ' '.join(header_line.split())
        data_lines = file.readlines()
    cleaned_data = [' '.join(line.split()) for line in data_lines]
    cleaned_file_content = header_line + '\n' + '\n'.join(cleaned_data)
    df = pd.read_csv(StringIO(cleaned_file_content), sep=' ')
    cols2return = ('time',) + cols2return if 'time' not in cols2return else cols2return
    df = df[list(cols2return)]
    df.set_index('time', inplace=True)
    return df

def extract_jrl_data(jrl_fullFilePath, side="l", cutoff_hz=15, sampling_freq_hz=200):
    base_cols = (
        f"med_cond_weld_{side}_on_tibial_plat_{side}_in_tibial_plat_{side}",
        f"lat_cond_joint_{side}_on_lat_cond_{side}_in_lat_cond_{side}",
        f"fem_pat_{side}_on_patella_{side}_in_patella_{side}"
    )
    components = ("fx", "fy", "fz")
    cols2extract = tuple(f"{base}_{comp}" for base in base_cols for comp in components)
    df_jrl = read_joint_reaction_file(jrl_fullFilePath, cols2extract)
    for col in cols2extract:
        df_jrl[col] = apply_lowpass_filter(df_jrl[col].to_numpy(), cutoff_hz, sampling_freq_hz)
    for base in base_cols:
        df_jrl[f"{base}_resultant"] = np.linalg.norm(df_jrl[[f"{base}_{comp}" for comp in components]].values, axis=1)
    df_jrl[f"mediototal_{side}_resultant_ratio_pc"] = (
        df_jrl[f"{base_cols[0]}_resultant"] /
        (df_jrl[f"{base_cols[0]}_resultant"]+df_jrl[f"{base_cols[1]}_resultant"]) * 100
    )
    df_jrl[f"total_{side}_resultant_force"] = (
        df_jrl[f"{base_cols[0]}_resultant"] +
        df_jrl[f"{base_cols[1]}_resultant"]
    )
    return df_jrl

def process_sim_results(df_simResults, side="l"):
    """
    Process simulation results to obtain both:
      - A time-normalized full cycle dataset for spm analysis and clustering.
        (Full cycle: from row['cycle_from_s'] to row['cycle_to_s'])
      - Aggregated single-value metrics computed from the raw stance phase.
        (Stance phase: from row['cycle_from_s'] to row['foot_off_s'])
        
    Returns:
      dataset: xarray.Dataset containing interpolated full cycle data.
      agg_metrics_df: DataFrame containing aggregated metrics from the stance phase.
    """
    results_full = []         # For full-cycle (interpolated) data.
    agg_metrics_list = []     # For stance-phase aggregated metrics.
    interp_time_grid = np.linspace(0, 100, 101)
    headers = []
    metadata = df_simResults.drop(columns=['jrl_fullFilePath', 'marker_fullFilePath', 'C3D_fullFilePath', 'ENF_fullFilePath', 'grf_MOT_fullFilePath'], errors='ignore')
    
    for _, row in df_simResults.iterrows():
        weight_newtons = row['weight_kg'] * 9.81
        height_meters = row['height_mm'] / 1000
        
        # Full cycle parameters.
        full_start = row['cycle_from_s']
        full_end   = row['cycle_to_s']
        # Stance phase: from cycle start to foot_off_s.
        stance_end = row['foot_off_s']
        
        # ---- FULL CYCLE EXTRACTION (for spm analysis) ----
        # Joint Reaction Data full cycle.
        df_jrl = extract_jrl_data(row['jrl_fullFilePath'], side=side)
        df_jrl_full = df_jrl.loc[(df_jrl.index >= full_start) & (df_jrl.index <= full_end)].copy()
        
        # Marker data full cycle.
        df_trc = readTRC(row['marker_fullFilePath'])
        df_trc_full = df_trc.loc[(df_trc.index >= full_start) & (df_trc.index <= full_end)].copy()
        # Determine frame indices based on full cycle.
        from_frame_full = df_trc_full["Frame#"].min() - int(df_trc.loc[df_trc.index[0], "Frame#"].iloc[0])
        to_frame_full = df_trc_full["Frame#"].max() - int(df_trc.loc[df_trc.index[0], "Frame#"].iloc[0])
        df_trc_full = df_trc_full.drop(columns=[('Frame#','')], errors='ignore')
        trc_data_full = df_trc_full.to_numpy() / height_meters * 100
        trc_columns_full = [f"{level0}_{level1}_pcMeters" if level1 else level0 for level0, level1 in df_trc_full.columns]
        
        # Compute force-related data from joint reaction full cycle.
        medlat_ratioCol = [f'mediototal_{side}_resultant_ratio_pc']
        medial_lateral_full = df_jrl_full[medlat_ratioCol].to_numpy()
        total_KneeForceCol = [f'total_{side}_resultant_force']
        base_cols = (
            f"med_cond_weld_{side}_on_tibial_plat_{side}_in_tibial_plat_{side}",
            f"lat_cond_joint_{side}_on_lat_cond_{side}_in_lat_cond_{side}",
            f"fem_pat_{side}_on_patella_{side}_in_patella_{side}"
        )
        components = ("fx","fy","fz")
        # Ensure resultant forces are computed.
        for base in base_cols:
            df_jrl_full.loc[:, f"{base}_resultant"] = np.linalg.norm(
                df_jrl_full[[f"{base}_{comp}" for comp in components]].values, axis=1)
        force_full = df_jrl_full[[f"{base}_resultant" for base in base_cols] + total_KneeForceCol].to_numpy() / weight_newtons * 100
        
        # Combine full cycle data.
        combined_data_full = np.column_stack((medial_lateral_full, trc_data_full, force_full))
        # Joint angles extraction for full cycle from C3D.
        c3d_file = c3d(row['marker_fullFilePath'].replace('.trc','.c3d'))
        angles = ['ThoraxAngles','PelvisAngles','FootProgressAngles','HipAngles','KneeAngles','AnkleAngles']
        angle_header_full = []
        for angle in angles:
            if angle in ["ThoraxAngles", "PelvisAngles"]:
                label = "L" + angle
            else:
                label = side.upper() + angle
            if label in c3d_file['parameters']['POINT']['LABELS']['value']:
                index = c3d_file['parameters']['POINT']['LABELS']['value'].index(label)
                # Use full cycle: from full_start to full_end.
                data = c3d_file['data']['points'][:, index, :][0:3, from_frame_full:to_frame_full+1].transpose()
                angle_header_full.extend([label + a for a in ['X','Y','Z']])
                combined_data_full = np.column_stack((combined_data_full, data))
        headers = medlat_ratioCol + trc_columns_full + \
                  [f"{base}_resultant_pcNewtons" for base in base_cols] + [total_KneeForceCol[0] + "_pcNewtons"] + angle_header_full
        
        # Interpolate full cycle data onto a 101-point grid.
        original_time_full = np.linspace(full_start, full_end, combined_data_full.shape[0])
        interpolated_full = np.zeros((101, combined_data_full.shape[1]))
        for i in range(combined_data_full.shape[1]):
            try:
                interpolator = Akima1DInterpolator(original_time_full, combined_data_full[:, i])
            except Exception as e:
                print(f'Full cycle interpolation error for {headers[i]}: {row["ID"]}, error: {e}')
                interpolated_full[:, i] = combined_data_full[:, i][:101]
            else:
                interpolated_full[:, i] = interpolator(np.linspace(full_start, full_end, 101))
        results_full.append(interpolated_full)
        
        # ---- STANCE PHASE EXTRACTION (for aggregated metrics) ----
        # Use stance phase: from full_start to row['foot_off_s'].
        df_jrl_stance = df_jrl.loc[(df_jrl.index >= full_start) & (df_jrl.index <= stance_end)].copy()
        df_trc_stance = df_trc.loc[(df_trc.index >= full_start) & (df_trc.index <= stance_end)].copy()
        # Compute stance-phase frame indices.
        from_frame_stance = df_trc_stance["Frame#"].min() - int(df_trc.loc[df_trc.index[0], "Frame#"].iloc[0])
        to_frame_stance = df_trc_stance["Frame#"].max() - int(df_trc.loc[df_trc.index[0], "Frame#"].iloc[0])
        df_trc_stance = df_trc_stance.drop(columns=[('Frame#','')], errors='ignore')
        trc_data_stance = df_trc_stance.to_numpy() / height_meters * 100
        trc_columns_stance = [f"{level0}_{level1}_pcMeters" if level1 else level0 for level0, level1 in df_trc_stance.columns]
        
        # Force and joint reaction for stance phase.
        medial_lateral_stance = df_jrl_stance[medlat_ratioCol].to_numpy()
        for base in base_cols:
            df_jrl_stance.loc[:, f"{base}_resultant"] = np.linalg.norm(
                df_jrl_stance[[f"{base}_{comp}" for comp in components]].values, axis=1)
        force_stance = df_jrl_stance[[f"{base}_resultant" for base in base_cols] + total_KneeForceCol].to_numpy() / weight_newtons * 100
        
        combined_data_stance = np.column_stack((medial_lateral_stance, trc_data_stance, force_stance))
        angle_header_stance = []
        for angle in angles:
            if angle in ["ThoraxAngles", "PelvisAngles"]:
                label = "L" + angle
            else:
                label = side.upper() + angle
            if label in c3d_file['parameters']['POINT']['LABELS']['value']:
                index = c3d_file['parameters']['POINT']['LABELS']['value'].index(label)
                # For stance, use full_start to stance_end.
                data = c3d_file['data']['points'][:, index, :][0:3, from_frame_stance:to_frame_stance+1].transpose()
                angle_header_stance.extend([label + a for a in ['X','Y','Z']])
                combined_data_stance = np.column_stack((combined_data_stance, data))
        # Aggregated metrics are computed over the raw time vector of the stance phase.
        raw_time_stance = df_jrl_stance.index.values
        trial_agg = {}
        for i, feature in enumerate(medlat_ratioCol + trc_columns_stance + \
                                      [f"{base}_resultant_pcNewtons" for base in base_cols] + 
                                      [total_KneeForceCol[0] + "_pcNewtons"] + angle_header_stance):
            # Use corresponding column from combined_data_stance.
            col_data = combined_data_stance[:, i]
            trial_agg[feature + "_mean"]   = np.mean(col_data)
            trial_agg[feature + "_std"]    = np.std(col_data)
            trial_agg[feature + "_median"] = np.median(col_data)
            trial_agg[feature + "_min"]    = np.min(col_data)
            trial_agg[feature + "_max"]    = np.max(col_data)
            trial_agg[feature + "_auc"]    = np.trapz(col_data, x=raw_time_stance)
        trial_agg['ID'] = row['ID']
        trial_agg['sideExerciseLeg'] = row['sideExerciseLeg']
        agg_metrics_list.append(trial_agg)
    
    final_data_full = np.array(results_full)
    dataset = xr.Dataset(
        {headers[i]: (('sample', 'time'), final_data_full[:, :, i]) for i in range(len(headers))},
        coords={'sample': np.arange(len(df_simResults)), 'time': interp_time_grid}
    )
    for col in metadata.columns:
        dataset[col] = ('sample', metadata[col].values)
    dataset['sideExerciseLeg'] = ('sample', df_simResults['sideExerciseLeg'].values)
    scaler = StandardScaler()
    for var in [f"{base}_resultant_pcNewtons" for base in base_cols] + medlat_ratioCol + [total_KneeForceCol[0] + "_pcNewtons"]:
        dataset[f"{var}_scStd"] = (('sample', 'time'), scaler.fit_transform(dataset[var].values))
    scaler = MinMaxScaler()
    for var in [f"{base}_resultant_pcNewtons" for base in base_cols] + medlat_ratioCol + [total_KneeForceCol[0] + "_pcNewtons"]:
        dataset[f"{var}_scMinMax"] = (('sample', 'time'), scaler.fit_transform(dataset[var].values))
    
    agg_metrics_df = pd.DataFrame(agg_metrics_list)
    return dataset, agg_metrics_df

def precompute_distance_matrix(data, method='softdtw', gamma=1.0, n_jobs=-1):
    """
    Precomputes the pairwise distance matrix using either DTW or soft-DTW.
    Adjusts soft-DTW output to be non-negative with a zero diagonal for metric compatibility.

    Args:
        data (np.ndarray): The time series dataset.
                           Shape: (n_samples, n_timesteps, n_features)
                           Should be C-contiguous and float64 for optimal performance.
        method (str): The distance metric to use ('dtw' or 'softdtw'). Default: 'softdtw'.
        gamma (float): The gamma parameter for soft-DTW. Ignored if method is 'dtw'. Default: 1.0.
        n_jobs (int): Number of CPU cores to use for parallel computation (for 'dtw'). Default: -1 (use all).

    Returns:
        tuple: A tuple containing:
            - distance_matrix (np.ndarray or None): The computed pairwise distance matrix
              (n_samples, n_samples), or None if an error occurred.
            - computation_time (float or np.nan): The time taken for computation in seconds,
              or np.nan if an error occurred.
    """
    n_samples = data.shape[0]
    print(f"\n--- Precomputing Distance Matrix ({method.upper()}) ---")
    print(f"Calculating pairwise distances for {n_samples} samples...")
    if method == 'softdtw':
        print(f"  Using soft-DTW with gamma={gamma}")
    elif method == 'dtw':
        print(f"  Using classic DTW (n_jobs={n_jobs})")
    else:
        print(f"ERROR: Unknown distance matrix method '{method}'. Use 'dtw' or 'softdtw'.")
        return None, np.nan

    start_time = time.time()
    distance_matrix = None
    computation_time = np.nan

    try:
        data_prepared = np.ascontiguousarray(data, dtype=np.float64)

        if method == 'softdtw':
            # Calculate raw soft-DTW similarity matrix
            distance_matrix = cdist_soft_dtw(data_prepared, gamma=gamma)
            # 1. Shift the matrix so the minimum value is 0.0
            min_val = np.min(distance_matrix)
            print(f"  Adjusting soft-DTW matrix: Subtracting min value ({min_val:.4f}) to ensure non-negativity.")
            distance_matrix = distance_matrix - min_val # Element-wise subtraction
            # 2. Fill the diagonal with zeros
            print("  Adjusting soft-DTW matrix: Setting diagonal to 0.")
            np.fill_diagonal(distance_matrix, 0)
        elif method == 'dtw':
            distance_matrix = cdist_dtw(data_prepared, n_jobs=n_jobs)
            # Classic DTW already has zeros on the diagonal and is non-negative

        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Pairwise distances calculated and adjusted (if softDTW) in {computation_time:.2f} seconds.")

    except MemoryError:
        # ... (error handling remains the same) ...
        print("\nERROR: MemoryError calculating the full distance matrix!")
        distance_matrix = None
        computation_time = np.nan
    except Exception as e:
        # ... (error handling remains the same) ...
        print(f"\nERROR: An unexpected error occurred during distance calculation: {e}")
        distance_matrix = None
        computation_time = np.nan

    return distance_matrix, computation_time

def calculate_dtw_clustering_metrics(distance_matrix, computation_time, cluster_labels):
    """
    Calculates internal clustering evaluation metrics suitable for time series,
    using a PRECOMPUTED DTW or soft-DTW distance matrix.

    Args:
        distance_matrix (np.ndarray): The precomputed pairwise distance matrix.
                                      Shape: (n_samples, n_samples). Assumes DTW or soft-DTW.
        computation_time (float): The time taken to compute the distance_matrix (in seconds).
        cluster_labels (np.ndarray): The cluster assignments for each sample.
                                      Shape: (n_samples,)

    Returns:
        dict: A dictionary containing the calculated metrics and summary info.
              Keys: 'silhouette_dtw', 'davies_bouldin', 'cluster_distribution',
                    'dtw_within', 'dtw_between', 'dtw_computation_time'
              Returns NaNs for metrics if calculation is not possible (e.g., < 2 clusters,
              distance matrix computation failed).
    """
    print("\n--- Calculating Clustering Metrics from Precomputed Distances ---")

    if distance_matrix is None:
        print("ERROR: Distance matrix is None. Cannot calculate metrics.")
        # Attempt to get distribution if possible
        try:
            distribution = dict(zip(*np.unique(cluster_labels, return_counts=True)))
        except Exception:
            distribution = {}
        return {
            'silhouette_dtw': np.nan,
            'davies_bouldin': np.nan,
            'cluster_distribution': distribution,
            'dtw_within': {},
            'dtw_between': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'dtw_computation_time': computation_time if not np.isnan(computation_time) else 0.0 # Use provided time if available
        }

    n_samples = distance_matrix.shape[0]
    unique_labels = np.unique(cluster_labels)
    n_clusters_found = len(unique_labels)

    # --- Basic Info ---
    distribution = dict(zip(*np.unique(cluster_labels, return_counts=True)))
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of clusters found: {n_clusters_found}")
    print(f"  Cluster Distribution: {distribution}")
    print(f"  Distance Matrix Computation Time: {computation_time:.2f} seconds")


    # Handle cases where silhouette score cannot be calculated
    if n_samples <= 1 or n_clusters_found <= 1:
        print(f"WARN: Cannot calculate Silhouette or Davies-Bouldin. Need > 1 sample and > 1 cluster.")
        print(f"      (Found {n_samples} samples and {n_clusters_found} unique cluster labels: {unique_labels})")
        return {
            'silhouette_dtw': np.nan,
            'davies_bouldin': np.nan,
            'cluster_distribution': distribution,
            'dtw_within': {}, # Still try to compute if possible, but likely empty
            'dtw_between': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'dtw_computation_time': computation_time
        }

    # --- Calculate DTW Silhouette Score ---
    print("Calculating DTW Silhouette Score...")
    silhouette_dtw = np.nan # Default to NaN
    try:
        silhouette_dtw = dtw_silhouette_score(
            X=distance_matrix, # Use the precomputed matrix
            labels=cluster_labels,
            metric='precomputed', # Specify that X is a distance matrix
            verbose=0
        )
        print(f"  DTW Silhouette Score: {silhouette_dtw:.4f}")
        # Interpretation Guidance
        print("  Silhouette Interpretation Guide:")
        if silhouette_dtw > 0.7:
            print("    - Strong structure detected (clusters are dense and well-separated).")
        elif silhouette_dtw > 0.5:
            print("    - Reasonable structure detected.")
        elif silhouette_dtw > 0.25:
            print("    - Weak structure detected (could be artificial, consider different k).")
        else:
            print("    - No substantial structure detected (clustering may not be meaningful).")
    except ValueError as ve:
         # Specifically catch ValueError which often occurs with invalid inputs (e.g. all points in one cluster)
        print(f"\nWARN: Could not calculate DTW Silhouette Score: {ve}")
        print(f"      This often happens if the number of labels is invalid (e.g., only 1 cluster found for {n_samples} samples).")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during Silhouette Score calculation: {e}")


    # --- Compute Davies-Bouldin index using DTW distances ---
    print("Calculating Davies-Bouldin Index (DTW-based)...")
    davies_bouldin = np.nan # Default to NaN
    medoids = {}
    intra_cluster_distances = {}
    calculation_possible = True

    for cluster in unique_labels:
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            print(f"  WARN: Cluster {cluster} has no members. Skipping for DB index.")
            calculation_possible = False
            break # Cannot compute if a cluster is empty
        if len(indices) == 1:
             print(f"  WARN: Cluster {cluster} has only one member. Intra-cluster distance is 0.")
             medoids[cluster] = indices[0]
             intra_cluster_distances[cluster] = 0.0
             continue # Continue to next cluster

        # Extract the submatrix for the cluster
        submatrix = distance_matrix[np.ix_(indices, indices)]
        # Average distance per candidate within the cluster
        avg_distances = np.mean(submatrix, axis=1)
        # Medoid: sample with the smallest average distance to others in the cluster
        medoid_index_in_cluster = np.argmin(avg_distances)
        medoid_global_index = indices[medoid_index_in_cluster]
        medoids[cluster] = medoid_global_index
        # S_i: average distance from all samples in the cluster to the medoid
        intra_cluster_distances[cluster] = np.mean(distance_matrix[indices, medoid_global_index])
        # Alternative S_i: average distance among all pairs in the cluster (more common but slower)
        # triu_idx = np.triu_indices_from(submatrix, k=1)
        # intra_cluster_distances[cluster] = np.mean(submatrix[triu_idx]) if triu_idx[0].size > 0 else 0.0


    if calculation_possible and len(unique_labels) > 1:
        try:
            R_list = []
            for i in unique_labels:
                if i not in medoids: continue # Skip if medoid wasn't found (e.g., empty cluster)
                max_R_i = 0.0 # Initialize max ratio for cluster i
                for j in unique_labels:
                    if i == j or j not in medoids: continue # Skip self-comparison or if medoid missing

                    # Distance between medoids (M_ij)
                    M_ij = distance_matrix[medoids[i], medoids[j]]

                    if M_ij == 0:
                         # Avoid division by zero. Happens if medoids are identical (e.g., duplicate time series)
                         # Or if distance is exactly zero for distinct series (possible with some DTW variants/data)
                         print(f"  WARN: Distance between medoids of cluster {i} and {j} is 0. Skipping this pair for DB index.")
                         # In this scenario, you might assign a large penalty or handle based on context.
                         # For simplicity here, we skip calculating this specific R_ij, which might slightly
                         # underestimate the DB index if other R_ij values are non-zero. If all M_ij are 0, DB will be 0.
                         continue # Skip this pair

                    S_i = intra_cluster_distances.get(i, 0.0) # Get intra-cluster distance, default 0
                    S_j = intra_cluster_distances.get(j, 0.0)
                    R_ij = (S_i + S_j) / M_ij
                    if R_ij > max_R_i:
                        max_R_i = R_ij

                if max_R_i > 0: # Only add if a valid comparison was made
                    R_list.append(max_R_i)

            if R_list: # Check if any ratios were calculated
                davies_bouldin = np.mean(R_list)
                print(f"  Davies-Bouldin (DTW): {davies_bouldin:.4f}")
            else:
                print("  WARN: Could not compute Davies-Bouldin index (no valid inter-cluster comparisons possible).")
                davies_bouldin = np.nan

        except Exception as e:
            print(f"\nERROR: Could not calculate Davies-Bouldin (DTW): {e}")
            davies_bouldin = np.nan
    elif len(unique_labels) <= 1:
         print("  Skipping Davies-Bouldin: Only one cluster found.")
         davies_bouldin = np.nan


    # --- Compute within-cluster DTW distance statistics ---
    print("Calculating DTW distance statistics...")
    dtw_within = {}
    for cluster in unique_labels:
        indices = np.where(cluster_labels == cluster)[0]
        values = np.array([]) # Default to empty array
        if len(indices) > 1:
            submatrix = distance_matrix[np.ix_(indices, indices)]
            # Extract only the upper triangle (excluding the diagonal)
            triu_idx = np.triu_indices_from(submatrix, k=1)
            if triu_idx[0].size > 0:
                values = submatrix[triu_idx]
        # Store stats even if values is empty (will result in NaNs)
        dtw_within[cluster] = {
            'mean': np.mean(values) if values.size > 0 else np.nan,
            'std': np.std(values) if values.size > 0 else np.nan,
            'min': np.min(values) if values.size > 0 else np.nan,
            'max': np.max(values) if values.size > 0 else np.nan,
        }
    print(' - within:')
    if dtw_within:
        # Convert dict of dicts to DataFrame for nice printing
        df_within = pd.DataFrame(dtw_within).T # Transpose to get clusters as rows
        df_within.index.name = 'Cluster'
        print(textwrap.indent(df_within.to_string(), '    '))
    else:
        print("    No within-cluster distances to report.")

    # --- Compute between-cluster DTW distance statistics ---
    between_values = []
    if len(unique_labels) > 1:
        for i, cluster_i in enumerate(unique_labels):
            indices_i = np.where(cluster_labels == cluster_i)[0]
            if len(indices_i) == 0: continue # Skip empty clusters

            for cluster_j in unique_labels[i+1:]: # Compare with subsequent clusters only
                 indices_j = np.where(cluster_labels == cluster_j)[0]
                 if len(indices_j) == 0: continue # Skip empty clusters

                 # Extract the rectangular submatrix of distances between clusters i and j
                 submatrix = distance_matrix[np.ix_(indices_i, indices_j)]
                 if submatrix.size > 0:
                     between_values.extend(submatrix.flatten()) # Add all pairwise distances

    between_values = np.array(between_values)
    dtw_between = {
        'mean': np.mean(between_values) if between_values.size > 0 else np.nan,
        'std': np.std(between_values) if between_values.size > 0 else np.nan,
        'min': np.min(between_values) if between_values.size > 0 else np.nan,
        'max': np.max(between_values) if between_values.size > 0 else np.nan,
    }
    print(' - between (overall):')
    if between_values.size > 0:
         # Create a DataFrame for consistent printing
         df_between = pd.DataFrame([dtw_between])
         print(textwrap.indent(df_between.to_string(index=False), '    '))
    else:
         print("    No between-cluster distances to report (need > 1 non-empty cluster).")

    print("-------------------------------------------------")

    results = {
        'silhouette_dtw': silhouette_dtw,
        'davies_bouldin': davies_bouldin,
        'cluster_distribution': distribution,
        'dtw_within': dtw_within,       # dict of stats per cluster
        'dtw_between': dtw_between,     # dict of overall stats
        'dtw_computation_time': computation_time
    }
    return results

def cluster_timeseries(dataset, output_folder, excel_path,
                       method="ts_kmeans", # Clustering method: "ts_kmeans" or "agglo_ward"
                       use_scaler='_scMnVar',
                       # --- Grid Search / Parameter Settings ---
                       grid_search="off", # "on" to run grid search, "off" to run with specified params
                       ks=[2],            # List of k values for grid search (or single value if grid_search="off")
                       gammas=[1.0],      # List of gamma values for grid search (or single value)
                       dm_method='softdtw',# 'dtw' or 'softdtw' (used if grid_search="off" or as default in grid search)
                       n_clusters=2,     # Default k if grid_search="off"
                       gamma=1.0,        # Default gamma if grid_search="off"
                       random_state=42,
                       # --- Other settings ---
                       n_jobs=-1         # For parallel DTW calculation
                       ):
    """
    Performs time series clustering on time series data, optionally running a grid search
    over k and gamma, precomputing the distance matrix, and saving results.

    Args:
        dataset (xr.Dataset): Input data containing time series features.
        output_folder (str): Path to save figures.
        excel_path (str): Path to save summary and results (.xlsx).
        method (str): Clustering method (currently only "ts_kmeans" and "agglo_ward").
        use_scaler (str): Suffix indicating scaling applied or to apply ('_scMnVar' or '').
        grid_search (str): "on" to perform grid search over ks and gammas, "off" to run once.
        ks (list): List of integers for the number of clusters (k) to try in grid search.
        gammas (list): List of floats for the gamma parameter (soft-DTW) to try in grid search.
        dm_method (str): Distance metric 'dtw' or 'softdtw' for the run (if grid_search="off")
                         or as the method during grid search.
        n_clusters (int): Number of clusters if grid_search="off".
        gamma (float): Gamma value for soft-DTW if grid_search="off".
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of jobs for parallel DTW calculation.

    Returns:
        xr.Dataset or None: The dataset with added 'cluster_labels' if grid_search="off",
                            otherwise None (as grid search just saves results).
    """

    # --- 1. Prepare Data ---
    feature_vars = [
        f"fem_pat_{'EXERC'}_on_patella_{'EXERC'}_in_patella_{'EXERC'}_resultant_pcNewtons",
        f"total_{'EXERC'}_resultant_force_pcNewtons",
        f"mediototal_{'EXERC'}_resultant_ratio_pc"
    ]
    # Add scaler suffix if needed (assuming data is already scaled if suffix present)
    feature_vars_scaled = [f + use_scaler if use_scaler and use_scaler != "_scMnVar" else f for f in feature_vars]

    # Select relevant data and reshape for tslearn (n_samples, n_timesteps, n_features)
    try:
        clustering_data_list = [dataset[var].values for var in feature_vars_scaled]
        # Stack along the last axis to create the feature dimension
        clustering_data = np.stack(clustering_data_list, axis=-1)
        # Ensure it's a 3D array (it should be, but double-check)
        if clustering_data.ndim == 2: # If only one feature was selected
             clustering_data = clustering_data[:, :, np.newaxis]

    except KeyError as e:
        print(f"ERROR: Feature variable not found in dataset: {e}")
        print(f"Available variables: {list(dataset.data_vars)}")
        return None if grid_search == "off" else False # Indicate failure

    # Apply scaling if specified
    if use_scaler == "_scMnVar":
        print("Applying TimeSeriesScalerMeanVariance...")
        scaler = TimeSeriesScalerMeanVariance()
        # Reshape for scaler (n_samples, n_timesteps * n_features) if needed, then back
        # Or apply feature-wise? tslearn scaler handles 3D data correctly.
        clustering_data = scaler.fit_transform(clustering_data) # Modifies data in place potentially
        print("Scaling complete.")

    # Ensure data is contiguous and float64 for C extensions efficiency
    clustering_data = np.ascontiguousarray(clustering_data, dtype=np.float64)
    print(f"Clustering Data prepared with shape: {clustering_data.shape}") # (n_samples, n_timesteps, n_features)

    # --- 2. Grid Search Logic ---
    if grid_search.lower() == "on":
        print("\n=== Starting Grid Search ===")
        results_list = []
        # For agglo_ward we only grid‐search gamma; k is determined internally
        if method == "agglo_ward":
            param_grid = [(None, g) for g in gammas]
        else:
            param_grid = list(itertools.product(ks, gammas if dm_method == 'softdtw' else [None]))

        # --- Precompute Distance Matrix ONCE per gamma (if softDTW) or just ONCE (if DTW) ---
        # Store precomputed matrices to avoid recalculation within the k-loop
        precomputed_matrices = {} # Key: gamma (or 'dtw'), Value: (matrix, time)

        for k_val, gamma_val in param_grid:
            print(f"\n--- Grid Search: k={k_val}, gamma={gamma_val if gamma_val is not None else 'N/A'} ---")

            current_dm_method = dm_method # Use the overall method specified
            current_gamma = gamma_val if current_dm_method == 'softdtw' else None

            # --- Get or Compute Distance Matrix ---
            matrix_key = current_gamma if current_dm_method == 'softdtw' else 'dtw'
            if matrix_key not in precomputed_matrices:
                print(f"Calculating distance matrix for gamma={current_gamma}" if current_dm_method == 'softdtw' else "Calculating DTW distance matrix")
                precomputed_matrix, precomputation_time = precompute_distance_matrix(
                    clustering_data,
                    method=current_dm_method,
                    gamma=current_gamma if current_gamma is not None else 1.0, # Pass default if None, although ignored by dtw
                    n_jobs=n_jobs
                )
                if precomputed_matrix is None:
                    print(f"WARN: Failed to compute distance matrix for {matrix_key}. Skipping this grid point.")
                    continue # Skip to next grid combination
                precomputed_matrices[matrix_key] = (precomputed_matrix, precomputation_time)
            else:
                print(f"Using precomputed distance matrix for {matrix_key}")
                precomputed_matrix, precomputation_time = precomputed_matrices[matrix_key]


            # --- Run Clustering ---
            if method == "ts_kmeans":
                print(f"Running TimeSeriesKMeans (k={k_val}, metric={current_dm_method}, gamma={current_gamma})...")
                model_params = {
                    "n_clusters": k_val,
                    "random_state": random_state,
                    "n_init": 3, # Add n_init for stability
                    "verbose": 0 # Set to 1 for more details
                }
                if current_dm_method == 'softdtw':
                    model_params["metric"] = "softdtw"
                    model_params["metric_params"] = {"gamma": current_gamma}
                else: # 'dtw'
                    model_params["metric"] = "dtw"
                    # No metric_params needed for standard dtw

                tskmeans_start_time = time.time()
                model = TimeSeriesKMeans(**model_params)
                tskmeans_end_time = time.time()
                eval_time = tskmeans_end_time - tskmeans_start_time
                print(f"  TimeSeriesKMeans time: {eval_time:.2f}s")

                try:
                    cluster_labels = model.fit_predict(clustering_data)
                except Exception as e:
                    print(f"ERROR during TimeSeriesKMeans fitting: {e}")
                    print("Skipping metric calculation for this grid point.")
                    continue # Skip to next grid point

            elif method == "agglo_ward":
                # We ignore k_val here, only gamma_val matters:
                print(f"Running Agglomerative (Ward) with Soft‑DTW γ={gamma_val}…")

                # 1) We already have precomputed_matrix for this gamma_val
                dist_mat = precomputed_matrix

                # 2) Build linkage tree
                condensed = squareform(dist_mat, checks=False)
                Z = linkage(condensed, method='ward')

                # 3) Evaluate cuts k=2…4, pick best silhouette
                best_sil, best_k = -1, None
                for cut_k in range(2, 5):
                    labels = fcluster(Z, t=cut_k, criterion='maxclust')
                    sil = dtw_silhouette_score(
                        X=dist_mat,
                        labels=labels,
                        metric='precomputed'
                    )
                    print(f"  k={cut_k} → silhouette={sil:.4f}")
                    if sil > best_sil:
                        best_sil, best_k = sil, cut_k

                print(f"  → best silhouette {best_sil:.4f} at k={best_k}")

                # 4) Final cut & metrics
                final_labels = fcluster(Z, t=best_k, criterion='maxclust')
                metrics = calculate_dtw_clustering_metrics(
                    distance_matrix=dist_mat,
                    computation_time=precomputation_time,
                    cluster_labels=final_labels
                )

                # 5) Store result for this gamma
                result_row = {
                    "k":        best_k,
                    "gamma":    gamma_val,
                    "dm_method": dm_method,
                    "silhouette_dtw": metrics['silhouette_dtw'],
                    "davies_bouldin": metrics['davies_bouldin'],
                    "n_clusters_found": best_k,
                    "dtw_comp_time_s": metrics['dtw_computation_time']
                }
                results_list.append(result_row)
                continue

            else:
                print(f"ERROR: Method {method} not implemented in this version.")
                continue

            # --- Calculate Metrics using Precomputed Matrix ---
            metrics_start_time = time.time()
            metrics = calculate_dtw_clustering_metrics(
                distance_matrix=precomputed_matrix,
                computation_time=precomputation_time,
                cluster_labels=cluster_labels
            )
            metrics_end_time = time.time()
            eval_time = metrics_end_time - metrics_start_time
            print(f"  calculate_dtw_clustering_metrics time: {eval_time:.2f}s")

            # --- Store Results ---
            result_row = {
                "k": k_val,
                "gamma": current_gamma if current_dm_method == 'softdtw' else np.nan,
                "dm_method": current_dm_method,
                "silhouette_dtw": metrics['silhouette_dtw'],
                "davies_bouldin": metrics['davies_bouldin'],
                "n_clusters_found": len(metrics.get('cluster_distribution', {})),
                "cluster_distribution": str(metrics.get('cluster_distribution', {})), # Store as string for Excel
                "dtw_comp_time_s": metrics['dtw_computation_time']
            }
            results_list.append(result_row)

        # --- Save Grid Search Results ---
        if results_list:
            grid_results_df = pd.DataFrame(results_list)
            print("\n=== Grid Search Complete ===")
            print(grid_results_df.to_string())
            try:
                # Use mode 'a' and if_sheet_exists='replace' to add/overwrite sheet
                with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                    grid_results_df.to_excel(writer, sheet_name='grid_search_results', index=False)
                print(f"\nGrid search results saved to sheet 'grid_search_results' in '{excel_path}'")
            except Exception as e:
                print(f"\nERROR: Could not save grid search results to Excel: {e}")
                print("Saving to CSV as fallback: grid_search_results.csv")
                grid_results_df.to_csv("grid_search_results.csv", index=False)

            print("\nGrid search finished. Set grid_search='off' and update parameters (k, gamma, dm_method) based on results.")
            # sys.exit() # Stop execution after grid search
            # return None # Alternative to sys.exit()

        else:
            print("\n=== Grid Search Complete (No valid results obtained) ===")
            sys.exit()
            # return None


    # --- 3. Single Run Logic (grid_search="off") ---
    else:
        print(f"\n=== Running Single Clustering (k={n_clusters}, method={dm_method}, gamma={gamma if dm_method=='softdtw' else 'N/A'}) ===")

        # --- Precompute Distance Matrix ---
        #precomputed_matrix, precomputation_time = precompute_distance_matrix(
        #    clustering_data,
        #    method=dm_method,
        #    gamma=gamma,
        #    n_jobs=n_jobs
        #)

        #if precomputed_matrix is None:
        #     print("ERROR: Failed to compute distance matrix. Aborting clustering.")
        #     return None # Indicate failure

        # --- Run Clustering ---
        if method == "ts_kmeans":
             print(f"Running TimeSeriesKMeans (k={n_clusters}, metric={dm_method}, gamma={gamma if dm_method=='softdtw' else 'N/A'})...")
             model_params = {
                 "n_clusters": n_clusters,
                 "random_state": random_state,
                 "n_init": 5, # Use more inits for final run
                 "verbose": 0
             }
             if dm_method == 'softdtw':
                 model_params["metric"] = "softdtw"
                 model_params["metric_params"] = {"gamma": gamma}
             else: # 'dtw'
                 model_params["metric"] = "dtw"
             tskmeans_start_time = time.time()
             model = TimeSeriesKMeans(**model_params)
             tskmeans_end_time = time.time()
             eval_time = tskmeans_end_time - tskmeans_start_time
             print(f"  TimeSeriesKMeans time: {eval_time:.2f}s")

             try:
                 cluster_labels = model.fit_predict(clustering_data)
             except Exception as e:
                 print(f"ERROR during TimeSeriesKMeans fitting: {e}")
                 return None

        elif method == "agglo_ward":
            # 1) Precompute DTW / Soft‑DTW distance matrix
            precomputed_matrix, dm_time = precompute_distance_matrix(
                clustering_data,
                method=dm_method,
                gamma=gamma,
                n_jobs=n_jobs
            )
            if precomputed_matrix is None:
                print("ERROR: could not compute distance matrix for agglo_ward. Aborting.")
                return None

            # 2) Perform Ward’s hierarchical clustering
            #    SciPy linkage expects a condensed distance vector:
            condensed = squareform(precomputed_matrix, checks=False)
            Z = linkage(condensed, method='ward')

            # 3) (Optional) plot dendrogram
            fig, ax = plt.subplots(figsize=(9, 5))
            dendrogram(
                Z,
                ax=ax,
                no_labels=True,           # set False if you want leaf labels
                distance_sort='descending',
                show_leaf_counts=True,    # small counts under leaves help readability
                above_threshold_color='gray',
                color_threshold=None,     # color every merge uniquely unless cut line used
                leaf_rotation=90,
                leaf_font_size=8,
                truncate_mode=None        # or 'lastp' to show only the last p merges
            )
            ax.set_title("Ward Linkage Dendrogram", pad=10)
            ax.set_ylabel("Linkage distance (Ward / ΔSSE)")
            ax.set_xlabel("Samples / merged clusters (ordered)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.grid(True, axis='y', linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_folder, "dendrogram_agglo_ward.pdf"))
            plt.close(fig)
            
            # 4) Evaluate cluster‐cuts from k=2…4
            hier_results = []
            for k_cut in range(2, 5):
                labels = fcluster(Z, t=k_cut, criterion='maxclust')
                metrics = calculate_dtw_clustering_metrics(
                    distance_matrix=precomputed_matrix,
                    computation_time=dm_time,
                    cluster_labels=labels
                )
                hier_results.append({
                    "k":        k_cut,
                    "silhouette_dtw": metrics['silhouette_dtw'],
                    "davies_bouldin": metrics['davies_bouldin'],
                    "distribution":    metrics['cluster_distribution']
                })

            # 5) Save the k=2…4 results to Excel
            hier_df = pd.DataFrame(hier_results)
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                hier_df.to_excel(writer, sheet_name='hierarchy_metrics', index=False)

            # 6) pick the best k by max silhouette (tie→smallest k)
            best = max(hier_results, key=lambda r: (r['silhouette_dtw'] or -999, -r['k']))
            best_k = best['k']
            print(f"→ Recommended cluster count (highest silhouette): k={best_k}")

            # 7) re‑cut tree at best_k and assign labels
            final_labels = fcluster(Z, t=best_k, criterion='maxclust')

            # 8) attach to dataset exactly like k‑means does
            dataset = dataset.assign(cluster_labels=(('sample',), final_labels))
            for var in feature_vars:
                dataset[f"{var}_cluster"] = ('sample', final_labels)

            # 9) reuse your existing plotting loop (it will pick up .cluster on each var)
            print("Generating hierarchical cluster overlay plots…")
            # (You already have the code below that generates & saves plots)
            # … so just let your normal plotting section run.
            return dataset

        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # --- Calculate Metrics ---
        metrics_start_time = time.time()
        metrics = calculate_dtw_clustering_metrics(
            distance_matrix=precomputed_matrix,
            computation_time=precomputation_time,
            cluster_labels=cluster_labels
        )
        metrics_end_time = time.time()
        eval_time = metrics_end_time - metrics_start_time
        print(f"  calculate_dtw_clustering_metrics time: {eval_time:.2f}s")

        # Extract metrics for summary
        silhouette_dtw = metrics['silhouette_dtw']
        davies_bouldin = metrics['davies_bouldin']
        cluster_distribution_dict = metrics['cluster_distribution']
        dtw_within = metrics['dtw_within']
        dtw_between = metrics['dtw_between']
        dtw_comp_time = metrics['dtw_computation_time'] # Use the time from metrics dict

        # --- Assign Labels and Plot ---
        print("\nAssigning cluster labels to dataset...")
        dataset = dataset.assign(cluster_labels=(('sample',), cluster_labels))

        # Add cluster labels as coordinates for easier selection/plotting if needed
        # Also helpful to associate unscaled data with clusters if scaling was applied
        for var in feature_vars: # Iterate through original base names
             scaled_var = var + use_scaler if use_scaler and use_scaler != "_scMnVar" else var
             unscaled_var = var # Original name without suffix
             # Add cluster coord based on scaled var name (if exists)
             if scaled_var in dataset:
                  dataset[f"{scaled_var}_cluster"] = ('sample', cluster_labels)
             # Always add cluster coord based on unscaled var name
             if unscaled_var in dataset:
                  dataset[f"{unscaled_var}_cluster"] = ('sample', cluster_labels)


        print("Generating cluster plots...")
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        num_clusters_found = len(unique_clusters) - (1 if -1 in unique_clusters else 0) # Exclude noise if present
        print(f"Identified {num_clusters_found} clusters (excluding noise if any). Distribution: {cluster_distribution_dict}")

        # Determine palette size based on actual clusters found (excluding -1)
        plot_palette_size = max(1, num_clusters_found) # Need at least 1 color
        palette = sns.color_palette("tab10", plot_palette_size)

        num_plot_vars = len(feature_vars) # Plot unscaled variables
        num_cols = 3
        num_rows = (num_plot_vars + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, squeeze=False) # Ensure axes is 2D
        axes = axes.flatten() # Flatten for easy iteration

        # Sort clusters by size for potentially clearer plotting (largest first)
        # Handle potential noise cluster (-1) separately or sort ignoring it
        cluster_items = sorted(cluster_distribution_dict.items(), key=lambda item: item[1], reverse=True)

        # Map cluster labels to palette indices (0, 1, 2...) handling noise (-1)
        cluster_to_palette_idx = {label: idx for idx, (label, count) in enumerate(
                                   item for item in cluster_items if item[0] != -1)}

        for i, var in enumerate(feature_vars): # Plot original variables
            ax = axes[i]
            # Plot individual traces first with low alpha
            for cluster_label, count in cluster_items:
                color = 'gray' if cluster_label == -1 else palette[cluster_to_palette_idx.get(cluster_label, 0)] # Default to first color if issue
                alpha = 0.05 if cluster_label == -1 else 0.1
                cluster_indices = np.where(cluster_labels == cluster_label)[0]

                if len(cluster_indices) > 0:
                    # Plot individual time series for this cluster
                    # Use .isel instead of direct numpy indexing for xarray DataArrays
                    cluster_data = dataset[var].isel(sample=cluster_indices)
                    # Plot each sample's time series
                    for sample_idx in range(len(cluster_indices)):
                         # Plotting the transpose (.T) assumes time is the second dimension (index 1)
                         ax.plot(cluster_data.coords[cluster_data.dims[1]], # Get time coordinates
                                 cluster_data.isel(sample=sample_idx).values, # Get data for one sample
                                 color=color, alpha=alpha, linestyle='-')


            # Plot means on top
            plot_legend_handles = {} # To store one handle per cluster mean
            for cluster_label, count in cluster_items:
                color = 'gray' if cluster_label == -1 else palette[cluster_to_palette_idx.get(cluster_label, 0)]
                label_str = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}" # Use original label
                cluster_indices = np.where(cluster_labels == cluster_label)[0]

                if len(cluster_indices) > 0:
                    # Calculate mean across samples for this cluster
                    mean_values = dataset[var].isel(sample=cluster_indices).mean(dim='sample')
                    time_coords = mean_values.coords[mean_values.dims[0]] # Get time coordinates for the mean plot

                    # Plot mean line
                    line, = ax.plot(time_coords, mean_values.values, color=color, linewidth=3, linestyle='-',
                                    label=f"{label_str} Mean ({count} samples)", # Add count to legend
                                    zorder=1000, # Ensure mean is plotted on top
                                    path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()]) # Outline effect
                    if cluster_label not in plot_legend_handles:
                        plot_legend_handles[cluster_label] = line


            ax.set_title(f"Clustering: {var}")
            ax.set_xlabel("Time / Cycle (%)") # Adjust label as needed
            ax.set_ylabel(var.replace("_pcNewtons", " (N)").replace("_pc", " (%)")) # Basic unit cleaning

        # Remove empty subplots
        for j in range(num_plot_vars, len(axes)):
            fig.delaxes(axes[j])

        # Create a single legend for the figure
        # Extract handles and labels from the last populated axis or build manually
        handles = list(plot_legend_handles.values())
        labels = [h.get_label() for h in handles]
        if handles:
             # Sort legend by cluster label (numerically, handling -1)
             handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: int(x[1].split(' ')[1]) if 'Cluster' in x[1] else -1)
             handles_sorted, labels_sorted = zip(*handles_labels_sorted)
             fig.legend(handles_sorted, labels_sorted, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(len(labels), 4)) # Place legend above plots

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for legend
        clustering_fig_path = os.path.join(output_folder, f'clustering_{dm_method}_k{n_clusters}_gamma{gamma if dm_method=="softdtw" else "na"}_figure.pdf')
        try:
            fig.savefig(clustering_fig_path, format='pdf', bbox_inches='tight')
            print(f"Clustering figure saved to '{clustering_fig_path}'")
        except Exception as e:
            print(f"ERROR saving clustering figure: {e}")
        plt.show() # Display the plot


        # --- Save Summary and Detailed Metrics to Excel ---
        print(f"Saving results summary to '{excel_path}'...")
        summary_df = pd.DataFrame({
            "Method": [method],
            "Scaler": [use_scaler if use_scaler else "None"],
            "Distance_Metric": [dm_method],
            "Gamma_SoftDTW": [gamma if dm_method == 'softdtw' else np.nan],
            "Num_Clusters_Requested": [n_clusters],
            "Num_Clusters_Found": [num_clusters_found], # Excludes noise
            "Silhouette_DTW": [silhouette_dtw],
            "Davies_Bouldin_DTW": [davies_bouldin],
            "DTW_Distance_Time_s": [dtw_comp_time],
            "Random_State": [random_state]
        })

        dist_data = [{"Cluster": k, "Sample_Count": v} for k, v in cluster_distribution_dict.items()]
        distribution_df = pd.DataFrame(dist_data).sort_values(by="Cluster").reset_index(drop=True)

        within_data = []
        # Check if dtw_within is populated correctly
        if isinstance(dtw_within, dict) and dtw_within:
             for cluster, stats in dtw_within.items():
                 # stats should be a dict like {'mean': val, 'std': val,...}
                 row = {"Cluster": cluster}
                 row.update(stats) # Add mean, std, min, max keys/values
                 within_data.append(row)
             dtw_within_df = pd.DataFrame(within_data).sort_values(by="Cluster").reset_index(drop=True)
             # Rename columns for clarity
             dtw_within_df.columns = ['Cluster', 'DTW_Within_Mean', 'DTW_Within_Std', 'DTW_Within_Min', 'DTW_Within_Max']
        else:
            dtw_within_df = pd.DataFrame(columns=['Cluster', 'DTW_Within_Mean', 'DTW_Within_Std', 'DTW_Within_Min', 'DTW_Within_Max'])


        # Check if dtw_between is populated
        if isinstance(dtw_between, dict) and not all(np.isnan(v) for v in dtw_between.values()):
             dtw_between_df = pd.DataFrame([dtw_between]) # Convert single dict to DataFrame
              # Rename columns for clarity
             dtw_between_df.columns = ['DTW_Between_Mean', 'DTW_Between_Std', 'DTW_Between_Min', 'DTW_Between_Max']
        else:
            dtw_between_df = pd.DataFrame(columns=['DTW_Between_Mean', 'DTW_Between_Std', 'DTW_Between_Min', 'DTW_Between_Max'])


        try:
            # Use mode 'a' and if_sheet_exists='replace' to add/overwrite sheets
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                summary_df.to_excel(writer, sheet_name='clustering_summary', index=False)
                distribution_df.to_excel(writer, sheet_name='cluster_distribution', index=False)
                dtw_within_df.to_excel(writer, sheet_name='dtw_within_distribution', index=False)
                dtw_between_df.to_excel(writer, sheet_name='dtw_between_distribution', index=False)
            print(f"Clustering information successfully appended/updated in '{excel_path}'.")
        except Exception as e:
             print(f"\nERROR: Could not save results to Excel: {e}")
             print("Attempting to save sheets as separate CSV files...")
             try:
                  summary_df.to_csv("clustering_summary.csv", index=False)
                  distribution_df.to_csv("cluster_distribution.csv", index=False)
                  dtw_within_df.to_csv("dtw_within_distribution.csv", index=False)
                  dtw_between_df.to_csv("dtw_between_distribution.csv", index=False)
                  print("Successfully saved results to CSV files.")
             except Exception as csv_e:
                  print(f"ERROR saving to CSV files: {csv_e}")


        return dataset # Return the dataset with labels added


# Process simulation results for left and right sides, then merge.
interpData_xr_l, agg_metrics_df_l = process_sim_results(df_simResults, side='l')
interpData_xr_r, agg_metrics_df_r = process_sim_results(df_simResults, side='r')
interpData_xr_r.mediototal_r_resultant_ratio_pc.max()
interpData_xr = interpData_xr_l.merge(interpData_xr_r, join='outer')
agg_metrics_df = pd.concat([agg_metrics_df_l, agg_metrics_df_r], ignore_index=True)
print('Extraction complete: Full cycle for spm and aggregated stance-phase metrics computed.')
#%% 5. Grid Search parameter value optimization
# =============================================================================

# Assuming previous functions (precompute_distance_matrix,
# calculate_dtw_clustering_metrics, TimeSeriesKMeans etc.) are defined


def objective_function_silhouette(gamma, k_value, data, random_state):
    """
    Objective function for Bayesian Optimization.
    Calculates -Silhouette score for a given gamma and fixed k.
    NOTE: Assumes 'data' is already scaled and prepared!
    """
    print(f"  Optimizing: Evaluating gamma={gamma:.4f} for k={k_value}...")
    start_time = time.time()

    # --- 1. Precompute Distance Matrix ---
    # Using softDTW as gamma optimization is relevant for it
    dm_method = 'softdtw'
    print("line 1358 sets fixed softdtw")
    precomputed_matrix, precomputation_time = precompute_distance_matrix(
        data,
        method=dm_method,
        gamma=gamma,
        # n_jobs = -1 # Set as needed
    )

    if precomputed_matrix is None:
        print(f"  WARN: Failed distance matrix computation for gamma={gamma:.4f}. Returning worst score.")
        # Return a value indicating failure (worse than any expected score)
        # Since we minimize -SIL, return a large positive number
        return 10.0 # Or larger if needed

    # --- 2. Run Clustering ---
    model_params = {
        "n_clusters": k_value,
        "random_state": random_state,
        "metric": dm_method,
        "metric_params": {"gamma": gamma},
        "n_init": 3, # Fewer inits during optimization for speed
        "verbose": 0
    }
    tskmeans_start_time = time.time()
    model = TimeSeriesKMeans(**model_params)
    tskmeans_end_time = time.time()
    eval_time = tskmeans_end_time - tskmeans_start_time
    print(f"  TimeSeriesKMeans time: {eval_time:.2f}s")
    try:
        cluster_labels = model.fit_predict(data)
    except Exception as e:
        print(f"  WARN: TimeSeriesKMeans failed for gamma={gamma:.4f}, k={k_value}: {e}. Returning worst score.")
        return 10.0 # Indicate failure

    # --- 3. Calculate Metrics ---
    metrics_start_time = time.time()
    metrics = calculate_dtw_clustering_metrics(
        distance_matrix=precomputed_matrix,
        computation_time=precomputation_time,
        cluster_labels=cluster_labels
    )
    metrics_end_time = time.time()
    eval_time = metrics_end_time - metrics_start_time
    print(f"  calculate_dtw_clustering_metrics time: {eval_time:.2f}s")

    silhouette = metrics.get('silhouette_dtw', np.nan)
    end_time = time.time()
    eval_time = end_time - start_time
    print(f"  Optimizing: gamma={gamma:.4f}, k={k_value} -> Silhouette={silhouette:.4f} (Eval time: {eval_time:.2f}s)")


    # Handle NaN or calculation failures
    if np.isnan(silhouette):
        # If silhouette fails, return a very bad score to deter optimizer
        return 10.0
    else:
        # We want to MAXIMIZE silhouette, gp_minimize MINIMIZES, so return NEGATIVE silhouette
        return -silhouette

def objective_function_hac(gamma, data):
    """
    Bayesian objective for agglo_ward: returns -best silhouette over cuts k=2..10
    """
    # 1) Build the soft‑DTW distance matrix
    dist_mat, _ = precompute_distance_matrix(
        data,
        method="softdtw",
        gamma=gamma,
        n_jobs=-1
    )
    if dist_mat is None:
        return 10.0  # worst possible

    # 2) Build Ward linkage tree
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(dist_mat, checks=False), method="ward")

    # 3) Scan k=2..4, track best silhouette
    best_sil = -np.inf
    for k in range(2, 5):
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = dtw_silhouette_score(
            X=dist_mat,
            labels=labels,
            metric="precomputed"
        )
        if sil > best_sil:
            best_sil = sil

    # 4) Return negative so gp_minimize *maximizes* silhouette
    return -best_sil if not np.isnan(best_sil) else 10.0

def optimize_gamma_for_k(k_value, data, initial_results_df, n_calls=20, random_state=42, method="ts_kmeans"):
    """
    Uses Bayesian Optimization to find the best gamma for a specific k.

    Args:
        k_value (int): The fixed number of clusters k.
        data (np.ndarray): The prepared clustering data.
        initial_results_df (pd.DataFrame): DataFrame of initial grid search results.
        n_calls (int): Number of optimization iterations (evaluations of objective).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (best_gamma, best_score) found for this k. Score is the metric value (e.g. Silhouette).
               Returns (None, None) if optimization cannot proceed.
    """
    global RANDOM_STATE_CACHE
    RANDOM_STATE_CACHE = random_state # Store for objective function

    print(f"\n--- Optimizing Gamma for k = {k_value} ---")

    # Filter initial results for this k and softDTW
    initial_k_df = initial_results_df[
        (initial_results_df['k'] == k_value) &
        (initial_results_df['dm_method'] == 'softdtw') &
        (initial_results_df['silhouette_dtw'].notna()) # Only use valid starting points
    ].copy()

    if initial_k_df.empty:
        print(f"WARN: No valid initial grid search results found for k={k_value} with softDTW. Skipping optimization.")
        # Optionally, run gp_minimize without initial points, but it might take longer
        # For now, we skip if no starting point
        # Or, you could provide default initial points if none are found
        # x0 = [1.0] # Default gamma guess
        # y0 = [objective_function_silhouette(1.0, k_value, data, random_state)]
        return None, None # Indicate failure/skip


    # Prepare initial points for the optimizer
    # Ensure correct column names from your grid search excel/df
    x0 = [[g] for g in initial_k_df['gamma'].tolist()] # Create a list of lists for x0
    y0 = (-initial_k_df['silhouette_dtw']).tolist() # Objective minimizes -SIL
    
    print(f"Starting Bayesian Optimization with {len(x0)} initial points and {n_calls} total calls.")
    print(f"Initial gammas (x0): {x0}")
    print(f"Initial objectives (-SIL): {y0}")

    # Define the objective function specific to this k
    # Using a lambda captures the current k_value
    if method == "ts_kmeans":
        # optimize gamma for soft‑DTW k‑means with fixed k_value
        objective_for_this_k = lambda gamma_list: objective_function_silhouette(
            gamma=gamma_list[0],
            k_value=k_value,
            data=data,
            random_state=RANDOM_STATE_CACHE
        )
    else:  # method == "agglo_ward"
        # optimize gamma for agglomerative Ward (no fixed k_value used)
        objective_for_this_k = lambda gamma_list: objective_function_hac(
            gamma=gamma_list[0],
            data=data
        )

    try:
         # Run Bayesian Optimization
         result = gp_minimize(
             func=objective_for_this_k,
             dimensions=[GAMMA_SPACE], # Pass the defined gamma search space
             x0=x0,                 # Initial gamma values
             y0=y0,                 # Initial objective function values (-SIL)
             n_calls=n_calls,       # Total number of evaluations (including initial points)
             random_state=random_state,
             noise=1e-10 # Add slight noise assumption for numerical stability if needed
         )

         best_gamma = result.x[0]
         best_neg_silhouette = result.fun
         best_silhouette = -best_neg_silhouette # Convert back to actual Silhouette

         print(f"--- Optimization Complete for k = {k_value} ---")
         print(f"  Best Gamma found: {best_gamma:.4f}")
         print(f"  Best Silhouette score found: {best_silhouette:.4f}")

         return best_gamma, best_silhouette

    except Exception as e:
         print(f"ERROR during Bayesian Optimization for k={k_value}: {e}")
         return None, None # Indicate failure
     
#%% 6. Merge left and right: Take data from the active leg only.
# =============================================================================
exercKey = "EXERC"
feature_EXERCvars = [
    f"total_{exercKey}_resultant_force_pcNewtons",
    f"fem_pat_{exercKey}_on_patella_{exercKey}_in_patella_{exercKey}_resultant_pcNewtons",
    f"mediototal_{exercKey}_resultant_ratio_pc",
    f"med_cond_weld_{exercKey}_on_tibial_plat_{exercKey}_in_tibial_plat_{exercKey}_resultant_pcNewtons",
    f"lat_cond_joint_{exercKey}_on_lat_cond_{exercKey}_in_lat_cond_{exercKey}_resultant_pcNewtons"
]
feature_EXERCvars_scStd = [f + "_scStd" for f in feature_EXERCvars]
feature_EXERCvars_scMinMax = [f + "_scMinMax" for f in feature_EXERCvars]

angles = ['ThoraxAngles', 'PelvisAngles','FootProgressAngles',
          'HipAngles', 'KneeAngles', 'AnkleAngles']
angles_exerc = [exercKey.upper() + a + c for a in angles for c in ["X", "Y", "Z"]]
feature_EXERCvars_all = feature_EXERCvars + feature_EXERCvars_scStd+feature_EXERCvars_scMinMax+angles_exerc

for col in feature_EXERCvars_all:
    interpData_xr[col] = (('sample', 'time'), np.empty((interpData_xr.sizes['sample'], interpData_xr.sizes['time'])))

for i_row, exLeg in enumerate(interpData_xr['sideExerciseLeg'].values):
    for exercVar in feature_EXERCvars_all:
        if exercVar[len(exercKey):-1] in ["ThoraxAngles", "PelvisAngles"]:
            side = "l"
        else:
            side = exLeg[0]
        if exercVar in angles_exerc:
            interpData_xr[exercVar].loc[{'sample': i_row}] = interpData_xr[exercVar.replace('EXERC', side.upper())].loc[{'sample': i_row}]
        else:
            interpData_xr[exercVar].loc[{'sample': i_row}] = interpData_xr[exercVar.replace('EXERC', side)].loc[{'sample': i_row}]
"""
# Cluster the dataset using ts_kmeans.
# --- Define Grid Search Parameters ---
ks_to_search = [2]
gammas_to_search = [1.0] # Only used if dm_method is 'softdtw'
distance_method_for_search = 'softdtw' # Or 'dtw'

# --- Run the grid search ---
cluster_timeseries(
    dataset=interpData_xr, # Replace with your actual dataset variable
    output_folder=output_folder,
    excel_path=excel_path,
    method="ts_kmeans",
    use_scaler='_scMnVar', # Or '' or your specific suffix
    grid_search="on",      # Turn grid search ON
    ks=ks_to_search,
    gammas=gammas_to_search,
    dm_method=distance_method_for_search,
    random_state=42
)
#"""  #uncomment/fix 2025-04-15
# --- Optimize grid-search parameters ---
# --- Step 1: Initial Coarse Grid Search ---
print("\n=== Step 1: Running Initial Coarse Grid Search ===")

# --- === Validation Check === ---
gamma_lower_bound = GAMMA_SPACE.low
gamma_upper_bound = GAMMA_SPACE.high
print("\nValidating coarse_gammas against GAMMA_SPACE bounds...")
validation_passed = True
for gamma_val in coarse_gammas:
    if not (gamma_lower_bound <= gamma_val <= gamma_upper_bound):
        validation_passed = False
        error_message = (
            f"\n--- CONFIGURATION ERROR ---"
            f"\nValue {gamma_val} in 'coarse_gammas' list is outside the defined "
            f"GAMMA_SPACE bounds [{gamma_lower_bound}, {gamma_upper_bound}]."
            f"\nPlease adjust either the 'coarse_gammas' list or the 'GAMMA_SPACE' definition."
            f"\n---------------------------\n"
        )
        print(error_message)
        # Stop the script
        # raise ValueError(error_message) # Option 1: Raise an error
        sys.exit(1) # Option 2: Exit with a non-zero status code

if validation_passed:
    print("Validation successful: All coarse_gammas are within GAMMA_SPACE bounds.")

# Check if results already exist to potentially skip this step
grid_search_needed = True
if os.path.exists(excel_path):
    try:
        existing_sheets = pd.ExcelFile(excel_path).sheet_names
        if 'grid_search_results' in existing_sheets:
             print("Found existing grid search results. Skipping initial grid search.")
             grid_search_needed = False
        else:
             print("Grid search results sheet not found. Running initial grid search.")
    except Exception as e:
        print(f"Error reading existing Excel file: {e}. Running initial grid search.")

if grid_search_needed:
     # Temporarily set grid_search="on" for the function call
     # Note: cluster_timeseries will sys.exit() if grid_search="on" by default.
     # We need to run it without exiting for this workflow.
     # Modification needed in cluster_timeseries: remove or comment out sys.exit() when grid_search="on"
     # For demonstration, we'll assume it runs and returns None or similar indication
     print("Calling cluster_timeseries for initial grid search...")
     # --- IMPORTANT: Modify cluster_timeseries to NOT exit when grid_search='on' for this workflow ---
     # Remove or comment out the 'sys.exit()' line within the grid_search=="on" block in cluster_timeseries
     cluster_timeseries(
         dataset=interpData_xr,
         output_folder=output_folder,
         excel_path=excel_path,
         method=cluster_method, # ts_kmeans or agglo_ward
         use_scaler=use_scaler,
         grid_search="on",
         ks=coarse_ks,
         gammas=coarse_gammas,
         dm_method=initial_dm_method,
         random_state=42
     )
     print("Initial grid search function call completed (assuming no exit).")
else:
     pass # Results exist, proceed to next step


# --- Step 2: Load Grid Results ---
print("\n=== Step 2: Loading Grid Search Results ===")
try:
    grid_results_df = pd.read_excel(excel_path, sheet_name='grid_search_results')
    print("Loaded grid search results:")
    print(grid_results_df)
except FileNotFoundError:
    print(f"ERROR: Excel file '{excel_path}' not found. Cannot proceed.")
    sys.exit()
except Exception as e:
    print(f"ERROR: Could not read 'grid_search_results' sheet from '{excel_path}': {e}")
    sys.exit()


# --- Step 3: Prepare Data (once) ---
# This replicates the data preparation part from cluster_timeseries
# Ensure this matches exactly how data is prepared in your main function
print("\n=== Step 3: Preparing Clustering Data ===")
feature_vars = [
    f"fem_pat_{'EXERC'}_on_patella_{'EXERC'}_in_patella_{'EXERC'}_resultant_pcNewtons",
    f"total_{'EXERC'}_resultant_force_pcNewtons",
    f"mediototal_{'EXERC'}_resultant_ratio_pc"
]
feature_vars_scaled = [f + use_scaler if use_scaler and use_scaler != "_scMnVar" else f for f in feature_vars]
try:
    clustering_data_list = [interpData_xr[var].values for var in feature_vars_scaled]
    CLUSTERING_DATA_CACHE = np.stack(clustering_data_list, axis=-1)
    if CLUSTERING_DATA_CACHE.ndim == 2:
         CLUSTERING_DATA_CACHE = CLUSTERING_DATA_CACHE[:, :, np.newaxis]
    if use_scaler == "_scMnVar":
        print("Applying TimeSeriesScalerMeanVariance...")
        scaler = TimeSeriesScalerMeanVariance()
        CLUSTERING_DATA_CACHE = scaler.fit_transform(CLUSTERING_DATA_CACHE)
    CLUSTERING_DATA_CACHE = np.ascontiguousarray(CLUSTERING_DATA_CACHE, dtype=np.float64)
    print(f"Clustering data prepared with shape: {CLUSTERING_DATA_CACHE.shape}")
except KeyError as e:
    print(f"ERROR: Feature variable not found in dataset during data preparation: {e}")
    sys.exit()
except Exception as e:
     print(f"ERROR during data preparation: {e}")
     sys.exit()


# --- Step 4: Optimize Gamma per K ---
print("\n=== Step 4: Optimizing Gamma for each K using Bayesian Optimization ===")
optimization_results = []
unique_ks = sorted(grid_results_df['k'].unique())
print(f"--- - unique k values: {unique_ks}")

# Determine if gamma optimization is needed
optimize_gamma = True
if 'gamma' not in grid_results_df.columns or grid_results_df['gamma'].nunique() <= 1:
     print("Only one gamma value found in grid search or gamma column missing. Skipping gamma optimization.")
     print("Optimal k will be chosen based on the single gamma value.")
     optimize_gamma = False
     # Populate optimization_results directly from grid_results if needed
     if not grid_results_df.empty:
          # Find best k based on silhouette for the single gamma
          best_idx = grid_results_df['silhouette_dtw'].idxmax() # Or use Davies-Bouldin: .idxmin()
          best_row = grid_results_df.loc[best_idx]
          optimization_results.append({
               'k': best_row['k'],
               'optimal_gamma': best_row.get('gamma', np.nan), # Get gamma if exists
               'best_silhouette': best_row['silhouette_dtw'],
               'optimization_method': 'From Grid (Single Gamma)'
          })


if optimize_gamma:
     for k in unique_ks:
          # Run optimization for this k value
          best_gamma_k, best_score_k = optimize_gamma_for_k(
              k_value=k,
              data=CLUSTERING_DATA_CACHE,
              initial_results_df=grid_results_df,
              n_calls=n_gamma_opt_calls, 
              random_state=42,
              method=cluster_method 
          )
          if best_gamma_k is not None:
              optimization_results.append({
                  'k': k,
                  'optimal_gamma': best_gamma_k,
                  'best_silhouette': best_score_k, # Assumes optimize_gamma_for_k returns SIL
                  'optimization_method': 'Bayesian Optimization'
              })
          else:
              print(f"Optimization failed or was skipped for k={k}.")


# --- Step 5: Select Overall Optimal ---
print("\n=== Step 5: Selecting Overall Optimal (k, gamma) ===")
if not optimization_results:
    print("ERROR: No optimization results available. Cannot determine optimal parameters.")
    # Fallback: Maybe pick best from grid search directly?
    if not grid_results_df.empty:
         print("Falling back to best result from initial grid search.")
         best_idx = grid_results_df['silhouette_dtw'].idxmax() # Or use Davies-Bouldin
         best_grid_row = grid_results_df.loc[best_idx]
         optimal_k_final = int(best_grid_row['k'])
         # Handle cases where gamma might not be applicable (e.g., dtw) or only one was tested
         optimal_gamma_final = best_grid_row.get('gamma', np.nan)
         if pd.isna(optimal_gamma_final):
             optimal_dm_method_final = 'dtw' # Infer if gamma is NaN
             optimal_gamma_final = 1.0 # Default placeholder if needed
         else:
              optimal_dm_method_final = 'softdtw'
         best_metric_final = best_grid_row['silhouette_dtw']
         print(f"Selected from grid: k={optimal_k_final}, gamma={optimal_gamma_final:.4f}, method={optimal_dm_method_final}, Silhouette={best_metric_final:.4f}")
    else:
         print("ERROR: No grid search results either. Exiting.")
         sys.exit()

else:
     optimal_df = pd.DataFrame(optimization_results)
     print("Optimization results summary:")
     print(optimal_df)

     # Select the k and gamma that gave the best silhouette score across all k optimizations
     best_idx_opt = optimal_df['best_silhouette'].idxmax()
     best_opt_row = optimal_df.loc[best_idx_opt]

     optimal_k_final = int(best_opt_row['k'])
     optimal_gamma_final = best_opt_row['optimal_gamma']
     optimal_dm_method_final = 'softdtw' # Since we optimized gamma
     best_metric_final = best_opt_row['best_silhouette']
     print(f"\nSelected Overall Optimal Parameters:")
     print(f"  Optimal k = {optimal_k_final}")
     print(f"  Optimal gamma = {optimal_gamma_final:.4f}")
     print(f"  Best Silhouette Score = {best_metric_final:.4f}")


# --- Step 6: Save Optimal Parameters ---
print("\n=== Step 6: Saving Optimal Parameters ===")
summary_df = pd.DataFrame([{
    "optimal_k": optimal_k_final,
    "optimal_gamma": optimal_gamma_final if optimal_dm_method_final == 'softdtw' else np.nan,
    "optimal_distance_metric": optimal_dm_method_final,
    "metric_optimized": "Silhouette", # Or Davies-Bouldin if you changed it
    "optimal_metric_score": best_metric_final,
    "optimization_details": "From Bayesian Opt." if optimize_gamma and optimization_results else "From Grid Search"
}])

try:
    with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        summary_df.to_excel(writer, sheet_name='optimal_params', index=False)
        # Optionally save the detailed optimization results per k
        if optimization_results:
             optimal_df.to_excel(writer, sheet_name='optimization_details', index=False)
    print(f"Optimal parameters saved to sheet 'optimal_params' in '{excel_path}'")
except Exception as e:
    print(f"ERROR: Could not save optimal parameters to Excel: {e}")


# --- Step 7: Final Run (Optional) ---
print("\n=== Step 7: Running Final Clustering with Optimal Parameters ===")
# Set grid_search="off" and use the determined optimal parameters
interpData_wClusters_xr = cluster_timeseries(
    dataset=interpData_xr,
    output_folder=output_folder,
    excel_path=excel_path, # It will overwrite sheets like 'clustering_summary' etc.
    method=cluster_method,
    use_scaler=use_scaler,
    grid_search="off", # Ensure grid search is off
    n_clusters=optimal_k_final,
    gamma=optimal_gamma_final,
    dm_method=optimal_dm_method_final,
    random_state=42
)

if interpData_wClusters_xr is not None:
    print("\nFinal clustering run completed successfully. Labels added to dataset.")
else:
    print("\nFinal clustering run failed.")
"""
# --- Define Optimal Parameters Found from Grid Search ---
optimal_k = 2
optimal_gamma = 0.262067091278236 # Relevant only if dm_method='softdtw'
optimal_dm_method = 'softdtw' # 'softdtw' Or 'dtw'

# --- Run the final clustering ---
interpData_wClusters_xr = cluster_timeseries(
    dataset=interpData_xr,
    output_folder=output_folder,
    excel_path=excel_path,
    method="ts_kmeans",
    use_scaler='_scMnVar',
    grid_search="off", # Turn grid search OFF
    n_clusters=optimal_k,
    gamma=optimal_gamma,
    dm_method=optimal_dm_method,
    random_state=42
)

if interpData_wClusters_xr is not None:
    print("\nFinal clustering complete. Labels added to dataset.")
    # You can now use 'final_clustered_dataset' which includes 'cluster_labels'
    # print(final_clustered_dataset['cluster_labels'])
else:
    print("\nFinal clustering failed.")
""" #uncomment 2025-04-15
#%% 7. Post-Clustering Grouping and Exporting Cluster Statistics
#=========================================================================

cluster_col = 'total_EXERC_resultant_force_pcNewtons_cluster'
df_clusters = interpData_wClusters_xr.to_dataframe().reset_index()

grouped_clusters = df_clusters.drop_duplicates(subset=['ID']).groupby(cluster_col)
participant_metaInfo_clusters = grouped_clusters.agg({
    'Sex': ['count'],
    'height_mm': ['mean', 'std', 'median', 'min', 'max'],
    'weight_kg': ['mean', 'std', 'median', 'min', 'max'],
    'BMI': ['mean', 'std', 'median', 'min', 'max'],
    'age': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

grouped_clusters_sex = df_clusters.drop_duplicates(subset=['ID']).groupby(['Sex',cluster_col])
participant_metaInfo_clusters_sex = grouped_clusters_sex.agg({
    'Sex': ['count'],
    'height_mm': ['mean', 'std', 'median', 'min', 'max'],
    'weight_kg': ['mean', 'std', 'median', 'min', 'max'],
    'BMI': ['mean', 'std', 'median', 'min', 'max'],
    'age': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

grouped_clusters_pro = df_clusters.drop_duplicates(subset=['ID']).groupby(['Pro',cluster_col])
participant_metaInfo_clusters_pro = grouped_clusters_pro.agg({
    'Pro': ['count'],
    'height_mm': ['mean', 'std', 'median', 'min', 'max'],
    'weight_kg': ['mean', 'std', 'median', 'min', 'max'],
    'BMI': ['mean', 'std', 'median', 'min', 'max'],
    'age': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()

grouped_clusters_ast = df_clusters.drop_duplicates(subset=['ID']).groupby(cluster_col)
modelAST_metaInfo_clusters = grouped_clusters_ast.agg({
    'Sex': ['count'],
    'AST_nCycles': ['mean', 'std', 'median', 'min', 'max'],
    'AST_rmse_cm': ['mean', 'std', 'median', 'min', 'max'],
    'AST_maxe_cm': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()
modelAST_metaInfo_clusters.columns = ['_'.join(col).strip('_') for col in modelAST_metaInfo_clusters.columns.values]

df_clusters["ik_rmsErr_cm"] = 100 * df_clusters["ik_rmsErr_mm"]
df_clusters["ik_maxErr_cm"] = 100 * df_clusters["ik_maxErr_mm"]

grouped_clusters_dynamic = df_clusters.drop_duplicates(subset=['ID', 'ikFilt_MOT_fullFilePath']).groupby(cluster_col)
dynamic_trial_stats_clusters = grouped_clusters_dynamic.agg({
    'Sex': ['count'],
    'ik_rmsErr_cm': ['mean', 'std', 'median', 'min', 'max'],
    'ik_maxErr_cm': ['mean', 'std', 'median', 'min', 'max']
}).reset_index()
dynamic_trial_stats_clusters.columns = ['_'.join(col).strip('_') for col in dynamic_trial_stats_clusters.columns.values]

df_clusters["rms_residual_norm_maxF_pc"] = 100 * df_clusters["rms_residual_norm_maxF"]
df_clusters["max_residual_norm_maxF_pc"] = 100 * df_clusters["max_residual_norm_maxF"]

cycle_stats_clusters = (interpData_wClusters_xr.isel(time=0).to_dataframe().reset_index()
                        .drop_duplicates(subset=['ID', 'event_number', 'sideExerciseLeg'])
                        .groupby(cluster_col)
                        .agg({
                            'Sex': ['count'],
                            'rms_residual_norm_maxF_pc': ['mean', 'std', 'median', 'min', 'max'],
                            'max_residual_norm_maxF_pc': ['mean', 'std', 'median', 'min', 'max']
                        })
                        .reset_index())
cycle_stats_clusters.columns = ['_'.join(col).strip('_') for col in cycle_stats_clusters.columns.values]

with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    participant_metaInfo_clusters.to_excel(writer, sheet_name='participant_metaInfo_clusters', index=True)
    participant_metaInfo_clusters_sex.to_excel(writer, sheet_name='participant_metaInfo_clusters_sex', index=True)
    participant_metaInfo_clusters_pro.to_excel(writer, sheet_name='participant_metaInfo_clusters_pro', index=True)
    modelAST_metaInfo_clusters.to_excel(writer, sheet_name='modelAST_metaInfo_clusters', index=False)
    dynamic_trial_stats_clusters.to_excel(writer, sheet_name='dynamic_trial_stats_clusters', index=False)
    cycle_stats_clusters.to_excel(writer, sheet_name='cycle_stats_clusters', index=False)

print("Excel file has been updated with the cluster-based summary sheets.")

#%% 8. Split Clusters and spm1d Analysis for Joint Angles
#=========================================================================

def split_clusters(dataset):
    clusters = np.unique(dataset["cluster_labels"].values)
    cluster_dict = {}
    for cluster in clusters:
        if cluster == -1:
            continue  # Skip noise points
        cluster_indices = np.where(dataset["cluster_labels"].values == cluster)[0]
        cluster_dict[cluster] = dataset.isel(sample=cluster_indices)
    return cluster_dict

data_per_cluster = split_clusters(interpData_wClusters_xr)

# Save the entire global namespace (excluding built-ins) for post-mortem
#ws_path = workspace_out(output_folder)
#ws = workspace_in(r"path\to\level_walking\full_workspace.pkl", inject=True)
#ws = safe_workspace_in(r"path\to\level_walking - ISB\full_workspace.pkl", inject=True)
#%% plot_angles_cluster_means_spm1d
def plot_angles_cluster_means_spm1d(ds, output_folder, angles_of_interest, main_font_size=14, star_axis_fraction=0.015, corrected_alpha = 0.05):
    """
    Plots cluster mean ± SD curves for joint angles using spm1d ANOVA.
    
    This version:
      - Uses the "°" symbol in the y-axis label.
      - Allows a customizable main font size.
      - Places spm1d significance stars a bit lower (controlled by star_axis_fraction).
      - Sets subplot titles using clear names from angle_label_info.
      - Comments out FootProgressAngles.
      - Places anatomical direction annotations to the RIGHT of the y-axis.
      
    Parameters
    ----------
    ds : xarray.Dataset
        The clustered dataset containing variables named like "EXERC{Angle}{Coord}".
    output_folder : str
        Folder path for saving the PDF figure.
    angles_of_interest : dict
        Dictionary defining the angles and corresponding coordinate labels.
        For example:
            {
              "ThoraxAngles": ["X", "Y", "Z"],
              "PelvisAngles": ["X", "Y", "Z"],
              "HipAngles": ["X", "Y", "Z"],
              "KneeAngles": ["X", "Y", "Z"],
              "AnkleAngles": ["X", "Y", "Z"],
              "FootProgressAngles": ["X", "Z"]
            }
    main_font_size : float, optional
        Base font size (default 14).
    star_axis_fraction : float, optional
        Vertical position (axis fraction) for placing spm1d significance stars (default 0.015).
    """
    # Small font for annotations and legend (60% of main font size)
    small_font_size = main_font_size * 0.6

    # Define the angle label information dictionary.
    angle_label_info = {
        ("PelvisAngles","X"):  {"clear_name": "Pelvis sagittal",    "dir_start": "Post", "dir_end": "Ant"},
        ("PelvisAngles","Y"):  {"clear_name": "Pelvis Frontal",     "dir_start": "Down", "dir_end": "Up"},
        ("PelvisAngles","Z"):  {"clear_name": "Pelvis Transversal", "dir_start": "Ext",  "dir_end": "Int"},
        ("HipAngles","X"):     {"clear_name": "Hip Sagittal",       "dir_start": "Ext",  "dir_end": "Flex"},
        ("HipAngles","Y"):     {"clear_name": "Hip Frontal",        "dir_start": "Abd",  "dir_end": "Add"},
        ("HipAngles","Z"):     {"clear_name": "Hip Transversal",    "dir_start": "Ext",  "dir_end": "Int"},
        ("KneeAngles","X"):    {"clear_name": "Knee Sagittal",      "dir_start": "Ext",  "dir_end": "Flex"},
        ("KneeAngles","Y"):    {"clear_name": "Knee Frontal",       "dir_start": "Val",  "dir_end": "Var"},
        ("KneeAngles","Z"):    {"clear_name": "Knee Transversal",   "dir_start": "Ext",  "dir_end": "Int"},
        ("AnkleAngles","X"):   {"clear_name": "Ankle Sagittal",     "dir_start": "PF",   "dir_end": "DF"},
        ("AnkleAngles","Y"):   {"clear_name": "Ankle Frontal",      "dir_start": "Ever", "dir_end": "Inve"},
        ("AnkleAngles","Z"):   {"clear_name": "Ankle Transversal",  "dir_start": "Ext",  "dir_end": "Int"},
        # ("FootProgressAngles","X"): {"clear_name": "Sole Angle",         "dir_start": "-",   "dir_end": "+"},
        # ("FootProgressAngles","Z"): {"clear_name": "Foot Progression",   "dir_start": "Ext", "dir_end": "Int"},
        ("ThoraxAngles","X"):  {"clear_name": "Thorax Sagittal",    "dir_start": "Post", "dir_end": "Ant"},
        ("ThoraxAngles","Y"):  {"clear_name": "Thorax Frontal",     "dir_start": "Ipsi", "dir_end": "Contra"},
        ("ThoraxAngles","Z"):  {"clear_name": "Thorax Transversal", "dir_start": "Ext",  "dir_end": "Int"}
    }
    
    # Get unique clusters (excluding noise if present)
    clusters = np.unique(ds["cluster_labels"].values)
    clusters = clusters[clusters != -1]
    
    # Determine grid dimensions.
    num_angles = len(angles_of_interest)
    max_coords = max(len(coords) for coords in angles_of_interest.values())
    fig, axes = plt.subplots(nrows=num_angles, ncols=max_coords, figsize=(max_coords * 5, num_angles * 3), sharex=True)
    fig.suptitle("Comparison of Cluster Means (±SD) with spm1d ANOVA", fontsize=main_font_size)
    
    # Ensure axes is 2D.
    if num_angles == 1 and max_coords == 1:
        axes = np.array([[axes]])
    elif num_angles == 1:
        axes = np.array([axes])
    elif max_coords == 1:
        axes = axes[:, np.newaxis]
    
    custom_handles = {}
    palette = plt.get_cmap("tab10")
    time = ds['time'].values
    
    # Loop through each angle and coordinate.
    for row_idx, (angle_key, coords) in enumerate(angles_of_interest.items()):
        for col_idx, coord in enumerate(coords):
            ax = axes[row_idx, col_idx]
            var_name = f"EXERC{angle_key}{coord}"
            if var_name not in ds.variables:
                ax.set_visible(False)
                continue
            # Set subplot title using clear name if available.
            label_info = angle_label_info.get((angle_key, coord), None)
            if label_info:
                ax.set_title(label_info["clear_name"], fontsize=main_font_size)
            else:
                ax.set_title(f"{angle_key} {coord}", fontsize=main_font_size)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=main_font_size-2)
            
            if col_idx == 0:
                ax.set_ylabel("Angle (°)", fontsize=main_font_size)
            if row_idx == num_angles - 1:
                ax.set_xlabel("Gait cycle (%)", fontsize=main_font_size)
            
            data_for_spm = []
            for cluster in clusters:
                ds_cluster = ds.where(ds["cluster_labels"] == cluster, drop=True)
                cluster_vals = ds_cluster[var_name].values  # shape: (n_samples, n_time)
                data_for_spm.append(cluster_vals)
                print("Data fed to spm1d ANOVA:")
                for idx, arr in enumerate(data_for_spm):
                    print(f"  Cluster {clusters[idx]} → array shape {arr.shape}")
                mean_vals = np.mean(cluster_vals, axis=0)
                std_vals = np.std(cluster_vals, axis=0)
                color = palette(cluster)
                ax.plot(time, mean_vals, color=color, lw=2)
                ax.fill_between(time, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
                if cluster not in custom_handles:
                    custom_handles[cluster] = plt.Line2D([0], [0], color=color, lw=2, label=f"Cluster {cluster} (±SD)")
            print(type(data_for_spm))
            print(data_for_spm)
            plt.show()
            if len(clusters) > 1:
                anova = spm1d.stats.anova1(tuple(data_for_spm))
                inference = anova.inference(alpha=corrected_alpha)
                for clus in inference.clusters:
                    if clus.P < 0.05:
                        start, end = clus.endpoints
                        ax.axvspan(start, end, color='dimgray', alpha=0.3, lw=0, ymin=0, ymax=0.05)
                        if clus.P < 0.001:
                            significance_symbol = '***'
                        elif clus.P < 0.01:
                            significance_symbol = '**'
                        else:
                            significance_symbol = '*'
                        center = (start + end) / 2
                        ax.text(center, star_axis_fraction, significance_symbol,
                                transform=ax.get_xaxis_transform(), ha='center', va='center',
                                fontsize=main_font_size/2, color='black')
            
            # Add anatomical direction annotations to the RIGHT of the y-axis.
            if label_info:
                y_min, y_max = ax.get_ylim()
                offset_fraction = 0.13  # Adjust vertical spacing as desired.
                y_start = y_min + offset_fraction * (y_max - y_min)
                y_end = y_max - offset_fraction * (y_max - y_min)
                # Place annotation to the right by taking time[-1] plus an offset.
                x_dir_label = time[0] - 4  # Adjust the offset as necessary.
                ax.text(x_dir_label, y_start, label_info["dir_start"],
                        ha='left', va='center', fontsize=small_font_size,
                        rotation=90, color='black')
                ax.text(x_dir_label, y_end, label_info["dir_end"],
                        ha='left', va='center', fontsize=small_font_size,
                        rotation=90, color='black')
    
    handles = list(custom_handles.values())
    labels = [h.get_label() for h in handles]
    #labels = [label.replace('0', 'A').replace('1', 'B') for label in labels]
    trans = str.maketrans({str(i): chr(ord('A') + i-1) for i in range(10)}) # remove -1 for non-zero padded version
    labels = [label.translate(trans) for label in labels]
    #labels = [re.sub(r"\d", digit_replace, label) for label in labels]
    fig.legend(handles, labels, loc='upper right', fontsize=small_font_size, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(output_folder, "cluster_means_spm1d_custom.pdf")
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()
    print(f"Figure saved to: {save_path}")

angles_of_interest = {
    "ThoraxAngles": ["X", "Y", "Z"],
    "PelvisAngles": ["X", "Y", "Z"],
    "HipAngles": ["X", "Y", "Z"],
    "KneeAngles": ["X", "Y", "Z"],
    "AnkleAngles": ["X", "Y", "Z"]#,
    #"FootProgressAngles": ["X", "Z"]
}

plot_angles_cluster_means_spm1d(interpData_wClusters_xr,
                                output_folder,
                                angles_of_interest,
                                main_font_size=14,
                                star_axis_fraction=0.015)

#%% 9. Plotting Cluster Means with spm1d ANOVA and Saving the Figure
#=========================================================================

def plot_clustering_meanSD(ds, output_folder, corrected_alpha=0.05, font_size=14):
    var_names = [
        "fem_pat_EXERC_on_patella_EXERC_in_patella_EXERC_resultant_pcNewtons",
        "total_EXERC_resultant_force_pcNewtons",
        "mediototal_EXERC_resultant_ratio_pc"
    ]
    y_labels = [
        "Patellofemoral load (% bodyweight)",
        "Total tibiofemoral load (% bodyweight)",
        "Medial tibiofemoral load (% total)"
    ]
    time = ds['time'].values  
    clusters = np.unique(ds["cluster_labels"].values)
    clusters = clusters[clusters != -1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    custom_handles = {}
    for i, (var, ylabel) in enumerate(zip(var_names, y_labels)):
        ax = axes[i]
        ax.set_xlabel("Gait cycle (%)", fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=font_size-2)
        group_data = {}
        means = {}
        stds = {}
        for cluster in clusters:
            ds_cluster = ds.where(ds["cluster_labels"] == cluster, drop=True)
            data = ds_cluster[var].values
            group_data[cluster] = data
            means[cluster] = np.mean(data, axis=0)
            stds[cluster] = np.std(data, axis=0)
            color = plt.get_cmap("tab10")(cluster)
            ax.plot(time, means[cluster], color=color, lw=2)
            ax.fill_between(time, means[cluster]-stds[cluster], means[cluster]+stds[cluster],
                            color=color, alpha=0.2)
            if i == 0 and cluster not in custom_handles:
                custom_handles[cluster] = Line2D([0], [0], color=color, lw=2,
                                                  label=f"Cluster {cluster} Mean and SD")
        data_groups = [group_data[cluster] for cluster in clusters]
        anova = spm1d.stats.anova1(data_groups)
        inference = anova.inference(alpha=corrected_alpha)
        if inference.clusters:
            for clus in inference.clusters:
                start, end = clus.endpoints
                ax.axvspan(start, end, color='dimgray', alpha=0.3, lw=0, ymin=0, ymax=0.03)
                if clus.P < 0.001:
                    significance_symbol = '***'
                elif clus.P < 0.01:
                    significance_symbol = '**'
                else:
                    significance_symbol = '*'
                center = (start + end) / 2
                ax.text(center, 0.01, significance_symbol, transform=ax.get_xaxis_transform(),
                        ha='center', va='center', fontsize=font_size, color='black')
        ax.set_xlim([0, 100])
    handles = list(custom_handles.values())
    labels = [h.get_label() for h in handles]
    trans = str.maketrans({str(i): chr(ord('A') + i-1) for i in range(10)}) # remove -1 for non-zero padded version
    labels = [label.translate(trans) for label in labels]
    axes[0].legend(handles, labels, loc='upper right', fontsize=font_size-2)
    fig.tight_layout()
    output_path = os.path.join(output_folder, "clustering_kjrl_figure_meanSD.pdf")
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Figure saved to: {output_path}")

plot_clustering_meanSD(interpData_wClusters_xr, output_folder, corrected_alpha=corrected_alpha, font_size=14)

#%% 10. GENERATE MISSING EXCEL TABLES BASED ON NORMALITY AND DIFFERENCES
#    (Using ONLY the STANCE PHASE period from the normalized 101-point cycle.)
#===============================================================================
# Retrieve the normalized time vector (0 to 100, 101 points)
norm_time = interpData_wClusters_xr['time'].values # Shape: (101,)

# Retrieve the metadata for each trial.
cycle_from = interpData_wClusters_xr["cycle_from_s"].values
cycle_to   = interpData_wClusters_xr["cycle_to_s"].values
foot_off   = interpData_wClusters_xr["foot_off_s"].values

n_samples = interpData_wClusters_xr.sizes["sample"]

# Get cluster labels (Ensure this key is correct for your data)
cluster_label_key = "total_EXERC_resultant_force_pcNewtons_cluster"
if cluster_label_key not in interpData_wClusters_xr:
    raise KeyError(f"Cluster label key '{cluster_label_key}' not found in the dataset.")
cluster_labels = interpData_wClusters_xr[cluster_label_key].values # Shape: (n_samples,)

# Define all features of interest
features_of_interest = {
    # Angles
    "ThoraxAngles_X": "EXERCThoraxAnglesX", "ThoraxAngles_Y": "EXERCThoraxAnglesY", "ThoraxAngles_Z": "EXERCThoraxAnglesZ",
    "PelvisAngles_X": "EXERCPelvisAnglesX", "PelvisAngles_Y": "EXERCPelvisAnglesY", "PelvisAngles_Z": "EXERCPelvisAnglesZ",
    "HipAngles_X": "EXERCHipAnglesX", "HipAngles_Y": "EXERCHipAnglesY", "HipAngles_Z": "EXERCHipAnglesZ",
    "KneeAngles_X": "EXERCKneeAnglesX", "KneeAngles_Y": "EXERCKneeAnglesY", "KneeAngles_Z": "EXERCKneeAnglesZ",
    "AnkleAngles_X": "EXERCAnkleAnglesX", "AnkleAngles_Y": "EXERCAnkleAnglesY", "AnkleAngles_Z": "EXERCAnkleAnglesZ",
    # Joint Reaction Loads (JRL)
    "FemPat_Resultant": "fem_pat_EXERC_on_patella_EXERC_in_patella_EXERC_resultant_pcNewtons",
    "Total_Resultant": "total_EXERC_resultant_force_pcNewtons",
    "Medial/Total_Ratio": "mediototal_EXERC_resultant_ratio_pc"
}

# Define the metrics to calculate for each feature over the stance phase
metrics = ["mean", "median", "std", "min", "max", "auc"]

# --- Calculation ---

all_results = []

# Loop 1: Calculate stance-phase metrics for each sample for each feature
sample_wise_feature_metrics = {} # Dict to store results: {feature_key: {metric: [sample_values]}}

for feature_key, feature_var_name in features_of_interest.items():
    if feature_var_name in interpData_wClusters_xr:
        print(f"Processing feature: {feature_key} ({feature_var_name})")
        var_data = interpData_wClusters_xr[feature_var_name].values # Shape: (n_samples, 101)

        # Initialize lists to store metric values for each sample for this feature
        sample_metrics = {metric: [] for metric in metrics}

        for i in range(n_samples):
            # Compute stance percentage and corresponding index
            cycle_dur = cycle_to[i] - cycle_from[i]
            if cycle_dur != 0:
                stance_pct = ((foot_off[i] - cycle_from[i]) / cycle_dur) * 100
            else:
                stance_pct = 0.0 # Handle division by zero

            # Determine the index up to which data is considered (stance phase end)
            idx = int(np.round(stance_pct))
            idx = max(1, min(idx, len(norm_time))) # Ensures we have at least one time point

            # Extract data and time for the stance phase of this sample
            stance_data = var_data[i, :idx]
            stance_time = norm_time[:idx]

            # Calculate metrics for the stance phase of this sample
            # These metrics (mean, median, std, min, max) handle positive/negative values correctly based on their definitions.
            if len(stance_data) > 0:
                sample_metrics["mean"].append(np.mean(stance_data))
                sample_metrics["median"].append(np.median(stance_data))
                # Std deviation measures spread around the mean, works correctly with +/- values.
                sample_metrics["std"].append(np.std(stance_data) if len(stance_data) > 0 else 0.0)
                sample_metrics["min"].append(np.min(stance_data))
                sample_metrics["max"].append(np.max(stance_data))

                # AUC (Area Under Curve) using trapz calculates SIGNED area by default.
                # Negative values contribute negatively.
                # If total area magnitude is needed, use np.trapz(np.abs(stance_data), x=stance_time)
                if len(stance_data) >= 2:
                   sample_metrics["auc"].append(np.trapz(stance_data, x=stance_time))
                elif len(stance_data) == 1:
                   # AUC for a single point is typically 0 or ill-defined for trapz
                   sample_metrics["auc"].append(stance_data[0] * (stance_time[0] if len(stance_time)>0 else 1.0) ) # Or simply 0.0 or np.nan
                else: # len(stance_data) == 0
                   sample_metrics["auc"].append(np.nan)
            else:
                # Handle cases where stance phase might be empty
                for metric in metrics:
                    sample_metrics[metric].append(np.nan)

        # Store the calculated metrics for this feature (converting lists to arrays)
        sample_wise_feature_metrics[feature_key] = {
            metric: np.array(values) for metric, values in sample_metrics.items()
        }
    else:
        print(f"Warning: Feature variable '{feature_var_name}' for key '{feature_key}' not found in dataset. Skipping.")


# Loop 2: Compare metrics between clusters for each feature
for feature_key, feature_metrics_data in sample_wise_feature_metrics.items():
    print(f"\nComparing clusters for feature: {feature_key}")
    for metric in metrics:
        try:
            # Get the array of sample-wise values for this specific metric
            metric_values = feature_metrics_data[metric]  # Shape: (n_samples,)
        except KeyError:
            print(f"Warning: Metric '{metric}' not found for feature '{feature_key}'. Skipping...")
            continue

        # Separate values based on cluster labels
        try:
            group0_raw = metric_values[cluster_labels == 1]
            group1_raw = metric_values[cluster_labels == 2]
        except Exception as e:
            raise ValueError(
                f"Failed to separate cluster groups for feature '{feature_key}', "
                f"metric '{metric}'. Error: {e}"
            )

        # Validate cluster assignments
        if len(group0_raw) == 0 or len(group1_raw) == 0:
            raise ValueError(
                f"Invalid cluster assignment detected for feature '{feature_key}', "
                f"metric '{metric}'. One of the clusters has no samples.\n"
                f"Cluster distribution: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}"
            )
        
        # Remove potential NaNs that might have occurred
        group0 = group0_raw[~np.isnan(group0_raw)]
        group1 = group1_raw[~np.isnan(group1_raw)]

        # Calculate descriptive stats for each cluster for this metric
        n0, n1 = len(group0), len(group1)
        stats0 = {
            "mean": np.mean(group0) if n0 > 0 else np.nan,
            "std": np.std(group0) if n0 > 1 else 0.0 if n0 == 1 else np.nan, # Std requires >1 point
            "median": np.median(group0) if n0 > 0 else np.nan,
            "min": np.min(group0) if n0 > 0 else np.nan,
            "max": np.max(group0) if n0 > 0 else np.nan,
            "count": n0
        }
        stats1 = {
            "mean": np.mean(group1) if n1 > 0 else np.nan,
            "std": np.std(group1) if n1 > 1 else 0.0 if n1 == 1 else np.nan, # Std requires >1 point
            "median": np.median(group1) if n1 > 0 else np.nan,
            "min": np.min(group1) if n1 > 0 else np.nan,
            "max": np.max(group1) if n1 > 0 else np.nan,
            "count": n1
        }

        # Calculate Standard Error of the Difference between Means (SED)
        sed = np.nan # Default to NaN
        s0, s1 = stats0["std"], stats1["std"]
        # Ensure we have counts > 0 and valid std devs to calculate SED
        if n0 > 0 and n1 > 0 and not np.isnan(s0) and not np.isnan(s1):
             # Avoid division by zero if counts are 0 (already handled by n0>0, n1>0 check)
             # Note: Using sample standard deviation (ddof=1) might be technically more correct
             # for SED formula, but np.std uses ddof=0 by default. Let's be consistent.
             # If using sample std (ddof=1), ensure stats are calculated with it.
             # Here we use population std (ddof=0) as calculated by np.std by default.
             term0 = (s0**2) / n0 if n0 > 0 else 0
             term1 = (s1**2) / n1 if n1 > 0 else 0
             if term0 + term1 >= 0: # Ensure variance sum is non-negative
                 sed = np.sqrt(term0 + term1)

        # Perform statistical test
        p_thresh = 0.05 # Significance level before correction
        test_used = "Insufficient data"
        test_stat = np.nan
        p_value_uncorrected = np.nan
        p_value_corrected = np.nan
        significant = "N/A"

        # Only perform test if both groups have enough data points (e.g., > 1 for Kruskal)
        if n0 > 1 and n1 > 1: # Kruskal-Wallis needs at least 2 per group effectively
            # Check normality (optional, requires >= 3 samples for Shapiro)
            normal_group0 = False
            normal_group1 = False
            if n0 >= 3:
                try: _, p0 = shapiro(group0); normal_group0 = p0 >= 0.05
                except Exception: pass
            if n1 >= 3:
                try: _, p1 = shapiro(group1); normal_group1 = p1 >= 0.05
                except Exception: pass

            # Defaulting to Kruskal-Wallis as often appropriate for non-normal biomechanics data
            test_used = "Kruskal-Wallis"
            try:
                test_stat, p_value_uncorrected = kruskal(group0, group1)
            except ValueError: # Handle cases like identical groups
                 test_stat, p_value_uncorrected = np.nan, np.nan # Or 0.0 and 1.0 depending on interpretation
            except Exception as e:
                print(f"Error running Kruskal-Wallis for {feature_key} - {metric}: {e}")
                test_stat, p_value_uncorrected = np.nan, np.nan

            # Apply Bonferroni correction
            if not np.isnan(p_value_uncorrected) and n_comparisons > 0:
                 p_value_corrected = min(p_value_uncorrected * n_comparisons, 1.0)
                 significant = "SIGNIFICANT" if p_value_corrected < p_thresh else "NOT SIGNIFICANT"
            elif np.isnan(p_value_uncorrected):
                 significant = "Test Failed"
            else:
                 significant = "Correction Error"
        else:
            significant = "Insufficient Data for Test"


        # Store results for this feature-metric combination
        row = {
            "Feature": feature_key,
            "Metric": metric,
            "0_count": stats0["count"],
            "0_mean": stats0["mean"],
            "0_std": stats0["std"],
            "0_median": stats0["median"],
            "0_min": stats0["min"],
            "0_max": stats0["max"],
            "1_count": stats1["count"],
            "1_mean": stats1["mean"],
            "1_std": stats1["std"],
            "1_median": stats1["median"],
            "1_min": stats1["min"],
            "1_max": stats1["max"],
            "Diff_Mean": stats0["mean"] - stats1["mean"] if not (np.isnan(stats0["mean"]) or np.isnan(stats1["mean"])) else np.nan,
            "StdError_Diff_Mean": sed, # Added Standard Error of the Difference
            "Test_Used": test_used,
            "Test_Statistic": test_stat,
            "P_Value_Uncorrected": p_value_uncorrected,
            "P_Value_Bonferroni": p_value_corrected,
            "Significance_Threshold": p_thresh,
            "Significant_After_Correction": significant,
        }
        all_results.append(row)

# Create the final DataFrame
results_df = pd.DataFrame(all_results)

# Optional: Save to CSV
# results_df.to_csv("cluster_comparison_stats.csv", index=False)
#--- Write the new tables to the same Excel file ---
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    results_df.to_excel(writer, sheet_name='time_collapsed_stats', index=False)
   # jrl_compare_total_means_df.to_excel(writer, sheet_name='jrl_compare_total_means', index=False)
   # spm_anova_df.to_excel(writer, sheet_name='jrl_spm_anova', index=False)

#%% 11. Sub-Phase Analysis ---


# Dictionary to store sub-phase metrics per feature:
sample_wise_feature_metrics_subphases = {}

for feature_key, feature_var_name in features_of_interest.items():
    if feature_var_name in interpData_wClusters_xr:
        print(f"Processing sub-phase metrics for feature: {feature_key} ({feature_var_name})")
        var_data = interpData_wClusters_xr[feature_var_name].values  # Shape: (n_samples, 101)
        # Initialize a nested dictionary to hold metrics per sub-phase
        feature_phase_metrics = {}
        for phase_name, (start_pct, end_pct) in sub_phases.items():
            feature_phase_metrics[phase_name] = {metric: [] for metric in metrics}

        # Loop over each sample to calculate metrics for every sub-phase
        for i in range(n_samples):
            for phase_name, (start_pct, end_pct) in sub_phases.items():
                # Identify the indices corresponding to the current sub-phase window
                phase_mask = (norm_time >= start_pct) & (norm_time <= end_pct)
                indices = np.where(phase_mask)[0]
                phase_data = var_data[i, indices]
                phase_time = norm_time[indices]

                # Calculate metrics if data exists for this sub-phase
                if phase_data.size > 0:
                    feature_phase_metrics[phase_name]["mean"].append(np.mean(phase_data))
                    feature_phase_metrics[phase_name]["median"].append(np.median(phase_data))
                    feature_phase_metrics[phase_name]["std"].append(np.std(phase_data))
                    feature_phase_metrics[phase_name]["min"].append(np.min(phase_data))
                    feature_phase_metrics[phase_name]["max"].append(np.max(phase_data))
                    if phase_data.size >= 2:
                        feature_phase_metrics[phase_name]["auc"].append(np.trapz(phase_data, x=phase_time))
                    elif phase_data.size == 1:
                        feature_phase_metrics[phase_name]["auc"].append(phase_data[0] * (phase_time[0] if phase_time.size > 0 else 1.0))
                    else:
                        feature_phase_metrics[phase_name]["auc"].append(np.nan)
                else:
                    # In case there is no data within the sub-phase window
                    for metric in metrics:
                        feature_phase_metrics[phase_name][metric].append(np.nan)

        # Convert lists to numpy arrays for each sub-phase
        sample_wise_feature_metrics_subphases[feature_key] = {
            phase_name: {metric: np.array(values) for metric, values in phase_metrics.items()}
            for phase_name, phase_metrics in feature_phase_metrics.items()
        }
    else:
        print(f"Warning: Feature variable '{feature_var_name}' not found in dataset. Skipping sub-phase analysis for this feature.")


# Now compare clusters for each sub-phase in a similar manner to the full stance-phase analysis
all_subphase_results = []

for feature_key, phase_metrics_dict in sample_wise_feature_metrics_subphases.items():
    print(f"\nComparing clusters for feature: {feature_key}")
    for phase_name, metrics_data in phase_metrics_dict.items():
        print(f"  Phase: {phase_name}")
        for metric in metrics:
            # 1) Metric presence
            try:
                metric_values = metrics_data[metric]  # Expected shape: (n_samples,)
            except KeyError:
                print(f"Warning: Metric '{metric}' not found for feature '{feature_key}' / phase '{phase_name}'. Skipping...")
                continue

            # 2) Validate cluster labels (must be exactly two)
            labels, counts = np.unique(cluster_labels, return_counts=True)
            if labels.size != 2:
                raise ValueError(
                    f"Expected exactly 2 cluster labels for feature '{feature_key}', phase '{phase_name}', "
                    f"metric '{metric}', but got {labels.tolist()} with counts {counts.tolist()}."
                )

            # 3) Separate values by the detected labels
            try:
                label_a, label_b = labels[0], labels[1]
                group_a_raw = metric_values[cluster_labels == label_a]
                group_b_raw = metric_values[cluster_labels == label_b]
            except Exception as e:
                raise ValueError(
                    f"Failed to separate clusters for feature '{feature_key}', phase '{phase_name}', "
                    f"metric '{metric}'. Error: {e}"
                )

            # 4) Check for empty clusters
            if len(group_a_raw) == 0 or len(group_b_raw) == 0:
                raise ValueError(
                    f"Invalid cluster assignment for feature '{feature_key}', phase '{phase_name}', "
                    f"metric '{metric}': one of the clusters has no samples. "
                    f"Cluster distribution: {dict(zip(labels.tolist(), counts.tolist()))}"
                )

            group0_raw, group1_raw = group_a_raw, group_b_raw

            group0 = group0_raw[~np.isnan(group0_raw)]
            group1 = group1_raw[~np.isnan(group1_raw)]
            n0, n1 = len(group0), len(group1)
            stats0 = {
                "mean": np.mean(group0) if n0 > 0 else np.nan,
                "std": np.std(group0) if n0 > 1 else 0.0 if n0 == 1 else np.nan,
                "median": np.median(group0) if n0 > 0 else np.nan,
                "min": np.min(group0) if n0 > 0 else np.nan,
                "max": np.max(group0) if n0 > 0 else np.nan,
                "count": n0
            }
            stats1 = {
                "mean": np.mean(group1) if n1 > 0 else np.nan,
                "std": np.std(group1) if n1 > 1 else 0.0 if n1 == 1 else np.nan,
                "median": np.median(group1) if n1 > 0 else np.nan,
                "min": np.min(group1) if n1 > 0 else np.nan,
                "max": np.max(group1) if n1 > 0 else np.nan,
                "count": n1
            }
            sed = np.nan
            s0, s1 = stats0["std"], stats1["std"]
            if n0 > 0 and n1 > 0 and not np.isnan(s0) and not np.isnan(s1):
                term0 = (s0**2) / n0
                term1 = (s1**2) / n1
                if term0 + term1 >= 0:
                    sed = np.sqrt(term0 + term1)
            p_thresh = 0.05
            test_used = "Insufficient data"
            test_stat = np.nan
            p_value_uncorrected = np.nan
            p_value_corrected = np.nan
            significant = "N/A"
            if n0 > 1 and n1 > 1:
                test_used = "Kruskal-Wallis"
                try:
                    test_stat, p_value_uncorrected = kruskal(group0, group1)
                except Exception as e:
                    test_stat, p_value_uncorrected = np.nan, np.nan
                if not np.isnan(p_value_uncorrected) and n_comparisons > 0:
                    p_value_corrected = min(p_value_uncorrected * n_comparisons, 1.0)
                    significant = "SIGNIFICANT" if p_value_corrected < p_thresh else "NOT SIGNIFICANT"
                elif np.isnan(p_value_uncorrected):
                    significant = "Test Failed"
                else:
                    significant = "Correction Error"
            else:
                significant = "Insufficient Data for Test"

            row = {
                "Feature": feature_key,
                "SubPhase": phase_name,
                "Metric": metric,
                "0_count": stats0["count"],
                "0_mean": stats0["mean"],
                "0_std": stats0["std"],
                "0_median": stats0["median"],
                "0_min": stats0["min"],
                "0_max": stats0["max"],
                "1_count": stats1["count"],
                "1_mean": stats1["mean"],
                "1_std": stats1["std"],
                "1_median": stats1["median"],
                "1_min": stats1["min"],
                "1_max": stats1["max"],
                "Diff_Mean": stats0["mean"] - stats1["mean"] if not (np.isnan(stats0["mean"]) or np.isnan(stats1["mean"])) else np.nan,
                "StdError_Diff_Mean": sed,
                "Test_Used": test_used,
                "Test_Statistic": test_stat,
                "P_Value_Uncorrected": p_value_uncorrected,
                "P_Value_Bonferroni": p_value_corrected,
                "Significance_Threshold": p_thresh,
                "Significant_After_Correction": significant,
            }
            all_subphase_results.append(row)

# Create DataFrame for sub-phase analysis and optionally save it to the Excel file.
subphase_results_df = pd.DataFrame(all_subphase_results)
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    subphase_results_df.to_excel(writer, sheet_name='subphase_stats', index=False)

print("Additional Excel sheets ('time_collapsed_stats', 'subphase_stats') have been added successfully.") #, 'jrl_compare_total_means', and 'jrl_spm_anova'

#%% 12. Merging plots for distribution
#==========================================================================
def assemble_merged_knee_plots(ds, output_folder, filename="merged_knee_cluster_output", fmt="jpg",
                               corrected_alpha=0.05, font_size=14):
    """
    Merges two rows of plots into one combined figure:
      - Upper row (subplots A–C): Knee angle spm1d analysis using variables
           "EXERCKneeAnglesX", "EXERCKneeAnglesY", "EXERCKneeAnglesZ".
         The y-axis label is set to the clear name (with unit " (°)") from the original dictionary.
         Anatomical direction annotations (dir_start and dir_end) are added to the right of the y-axis.
         X-axis tick labels are hidden, and the top/right spines are removed.
         Spm1d ANOVA is performed and significant clusters are annotated with gray bars and stars.
      - Lower row (subplots D–F): Force-related clustering mean ± SD plots using variables:
           "fem_pat_EXERC_on_patella_EXERC_in_patella_EXERC_resultant_pcNewtons",
           "total_EXERC_resultant_force_pcNewtons",
           "mediototal_EXERC_resultant_ratio_pc".
         These subplots display the mean and SD curves and are also annotated with spm1d significance.
         Additionally, anatomical direction annotations are added according to:
            • Subplots D and E: "Unload" / "Load"
            • Subplot F: "Med" / "Lat"
         The force annotation font size is increased.
    
    All six subplots are annotated with letters A–F at the top of the y-axis.
    The legend (derived from the upper row) is placed inside subplot A and its labels are remapped:
         "0" → "A" and "1" → "B".
         
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing:
          - A 'time' coordinate (normalized 0–100).
          - A 'cluster_labels' variable.
          - Knee angles as "EXERCKneeAnglesX", "EXERCKneeAnglesY", "EXERCKneeAnglesZ".
          - Force-related variables:
            "fem_pat_EXERC_on_patella_EXERC_in_patella_EXERC_resultant_pcNewtons",
            "total_EXERC_resultant_force_pcNewtons",
            "mediototal_EXERC_resultant_ratio_pc".
    output_folder : str
        Folder where the combined figure will be saved.
    filename : str, optional
        Base file name (default "merged_knee_cluster_output").
    fmt : str, optional
        Output file format ("jpg" by default; "pdf" is also supported).
    corrected_alpha : float, optional
        Significance level for spm1d inference.
    font_size : int, optional
        Base font size for labels and annotations.
    """
    # --- Original full dictionary for anatomical direction annotations ---
    angle_label_info = {
        ("PelvisAngles","X"):  {"clear_name": "Pelvis sagittal",    "dir_start": "Posterior", "dir_end": "Anterior"},
        ("PelvisAngles","Y"):  {"clear_name": "Pelvis Frontal",     "dir_start": "Down", "dir_end": "Up"},
        ("PelvisAngles","Z"):  {"clear_name": "Pelvis Transversal", "dir_start": "External",  "dir_end": "Internal"},
        ("HipAngles","X"):     {"clear_name": "Hip Sagittal",       "dir_start": "Extension",  "dir_end": "Flexion"},
        ("HipAngles","Y"):     {"clear_name": "Hip Frontal",        "dir_start": "Abduction",  "dir_end": "Adduction"},
        ("HipAngles","Z"):     {"clear_name": "Hip Transversal",    "dir_start": "External",  "dir_end": "Internal"},
        ("KneeAngles","X"):    {"clear_name": "Knee Sagittal",      "dir_start": "Extension",  "dir_end": "Flexion"},
        ("KneeAngles","Y"):    {"clear_name": "Knee Frontal",       "dir_start": "Valgus",  "dir_end": "Varus"},
        ("KneeAngles","Z"):    {"clear_name": "Knee Transversal",   "dir_start": "External",  "dir_end": "Internal"},
        ("AnkleAngles","X"):   {"clear_name": "Ankle Sagittal",     "dir_start": "Plantar Flexion",   "dir_end": "Dorsal Flexion"},
        ("AnkleAngles","Y"):   {"clear_name": "Ankle Frontal",      "dir_start": "Eversion", "dir_end": "Inversion"},
        ("AnkleAngles","Z"):   {"clear_name": "Ankle Transversal",  "dir_start": "External",  "dir_end": "Internal"},
        ("ThoraxAngles","X"):  {"clear_name": "Thorax Sagittal",    "dir_start": "Posterior", "dir_end": "Anterior"},
        ("ThoraxAngles","Y"):  {"clear_name": "Thorax Frontal",     "dir_start": "Ipsilateral", "dir_end": "Contralateral"},
        ("ThoraxAngles","Z"):  {"clear_name": "Thorax Transversal", "dir_start": "External",  "dir_end": "Internal"},
    }
    # For the upper row we use keys ("KneeAngles", coord) for coord in ["X", "Y", "Z"].
    
    # --- Force annotation for lower row ---
    # Mapping: lower row, col 0 and col 1 → "Unload"/"Load", col 2 → "Med"/"Lat"
    force_label_info = {
        0: {"dir_start": "Unload", "dir_end": "Load"},
        1: {"dir_start": "Unload", "dir_end": "Load"},
        2: {"dir_start": "Lateral",    "dir_end": "Medial"}
    }
    
    # Common settings
    time = ds['time'].values
    clusters = np.unique(ds["cluster_labels"].values)
    clusters = clusters[clusters != -1]
    cmap = plt.get_cmap("tab10")
    # Set x-axis limits to (-5, 100) for all subplots.
    x_limits = (-5, 100)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=True)
    # No overall title.
    
    # Define subplot letters.
    letters_top = ["A", "B", "C"]
    letters_bottom = ["D", "E", "F"]
    # Define common star annotation settings.
    star_y_pos = 0.00   # For both rows
    star_font_size = font_size * 0.7  # For both rows
    
    # Ensure consistent tick label sizes for both rows.
    tick_label_size = font_size - 2
    
    #############################
    # Upper Row: Knee Angles spm1d Analysis
    #############################
    custom_handles = {}  # For legend inside subplot A.
    knee_coords = ["X", "Y", "Z"]
    for col_idx, coord in enumerate(knee_coords):
        ax = axes[0, col_idx]
        #if col_idx == 2:
        #    coord = "Z"
        #    varname = f"EXERCHipAngles{coord}"
        #    label_info = angle_label_info.get(("HipAngles", coord))
        #else:
        varname = f"EXERCKneeAngles{coord}"
        label_info = angle_label_info.get(("KneeAngles", coord))
        if label_info:
            ax.set_ylabel(f"{label_info['clear_name']} (angle °)", fontsize=font_size)
        else:
            ax.set_ylabel(f"KneeAngles {coord} (°)", fontsize=font_size)
        # Set tick label size for both axes.
        ax.tick_params(axis="y", labelsize=tick_label_size)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")
        # Remove top and right spines.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(x_limits)
        # Annotate subplot letter at top of y-axis.
        ax.text(-0.10, 1.02, letters_top[col_idx],
                transform=ax.transAxes,
                fontsize=font_size+2,
                fontweight="bold",
                va="bottom", ha="left")
        
        data_for_spm = []
        for cluster in clusters:
            ds_cluster = ds.where(ds["cluster_labels"] == cluster, drop=True)
            if varname not in ds_cluster:
                continue
            cluster_data = ds_cluster[varname].values  # (n_samples, time)
            if cluster_data.size == 0:
                continue
            mean_vals = np.mean(cluster_data, axis=0)
            std_vals = np.std(cluster_data, axis=0)
            color = cmap(cluster % 10)
            ax.plot(time, mean_vals, color=color, lw=2)
            ax.fill_between(time, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
            data_for_spm.append(cluster_data)
            if cluster not in custom_handles:
                custom_handles[cluster] = Line2D([0], [0], color=color, lw=2, label=f"Cluster {cluster}")
        # Set x-axis limit.
        ax.set_xlim(x_limits)
        # Add anatomical direction annotations (from dictionary) to the right of the y-axis.
        if label_info:
            y_min, y_max = ax.get_ylim()
            offset_fraction = 0.125  # Adjust as desired.
            y_start = y_min + offset_fraction * (y_max - y_min)
            y_end = y_max - offset_fraction * (y_max - y_min)
            x_dir_label = time[0] - 4  # Using -4 offset.
            ax.text(x_dir_label, y_start, label_info["dir_start"],
                    ha='left', va='center', fontsize=font_size, rotation=90, color='black')
            ax.text(x_dir_label, y_end, label_info["dir_end"],
                    ha='left', va='center', fontsize=font_size, rotation=90, color='black')
        # spm1d ANOVA and annotate significance.
        if len(data_for_spm) > 1:
            anova = spm1d.stats.anova1(data_for_spm)
            inference = anova.inference(alpha=corrected_alpha)
            for clus in inference.clusters:
                if clus.P < 0.05:
                    start, end = clus.endpoints
                    ax.axvspan(start, end, color='dimgray', alpha=0.3, lw=0, ymin=0, ymax=0.03)
                    if clus.P < 0.001:
                        symbol = '***'
                    elif clus.P < 0.01:
                        symbol = '**'
                    else:
                        symbol = '*'
                    center = (start + end) / 2
                    ax.text(center, star_y_pos, symbol,
                            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                            fontsize=star_font_size, color='black')
    
    #############################
    # Lower Row: Force-related Mean ± SD with spm1d Annotations
    #############################
    force_vars = [
        "fem_pat_EXERC_on_patella_EXERC_in_patella_EXERC_resultant_pcNewtons",
        "total_EXERC_resultant_force_pcNewtons",
        "mediototal_EXERC_resultant_ratio_pc"
    ]
    y_labels = [
        "Patellofemoral load (% bodyweight)",
        "Total tibiofemoral load (% bodyweight)",
        "Medial tibiofemoral load (% total)"
    ]
    for col_idx, (var, ylabel) in enumerate(zip(force_vars, y_labels)):
        ax = axes[1, col_idx]
        ax.set_xlabel("Gait cycle (%)", fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis="y", labelsize=tick_label_size)
        ax.tick_params(axis="x", labelsize=tick_label_size)
        ax.set_xlim(x_limits)
        ax.text(-0.10, 1.02, letters_bottom[col_idx],
                transform=ax.transAxes,
                fontsize=font_size+2,
                fontweight="bold",
                va="bottom", ha="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        data_groups = []
        for cluster in clusters:
            mask = (ds["cluster_labels"].values == cluster)
            if var not in ds:
                continue
            data = ds[var].values[mask]  # (n_samples, time)
            if data.size == 0:
                continue
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)
            color = cmap(cluster % 10)
            ax.plot(time, mean_vals, color=color, lw=2)
            ax.fill_between(time, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
            data_groups.append(data)
        ax.set_xlim(x_limits)
        if len(data_groups) > 1:
            anova = spm1d.stats.anova1(data_groups)
            inference = anova.inference(alpha=corrected_alpha)
            for clus in inference.clusters:
                if clus.P < 0.05:
                    start, end = clus.endpoints
                    ax.axvspan(start, end, color='dimgray', alpha=0.3, lw=0, ymin=0, ymax=0.03)
                    if clus.P < 0.001:
                        symbol = '***'
                    elif clus.P < 0.01:
                        symbol = '**'
                    else:
                        symbol = '*'
                    center = (start + end) / 2
                    ax.text(center, star_y_pos, symbol,
                            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                            fontsize=star_font_size, color='black')
        # Add anatomical direction annotations for force plots.
        if col_idx in force_label_info:
            info = force_label_info[col_idx]
            y_min, y_max = ax.get_ylim()
            offset_fraction = 0.125
            y_start = y_min + offset_fraction*(y_max - y_min)
            y_end = y_max - offset_fraction*(y_max - y_min)
            x_dir_label = time[0] - 4
            # Use base font size (increased) for force annotations.
            ax.text(x_dir_label, y_start, info["dir_start"],
                    ha='left', va='center', fontsize=font_size, rotation=90, color='black')
            ax.text(x_dir_label, y_end, info["dir_end"],
                    ha='left', va='center', fontsize=font_size, rotation=90, color='black')
    
    #############################
    # Legend inside subplot A (upper left)
    #############################
    handles = list(custom_handles.values())
    labels = [h.get_label() for h in handles]
    #labels = [label.replace('0', 'A').replace('1', 'B') for label in labels]
    trans = str.maketrans({str(i): chr(ord('A') + i-1) for i in range(10)}) # remove -1 for non-zero padded version
    labels = [label.translate(trans) for label in labels]
    ax0 = axes[0, 0]
    ax0.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.95),
               fontsize=font_size-2, frameon=False)
    
    fig.tight_layout(rect=[0, 0, 1, 1])
    output_path = os.path.join(output_folder, f"{filename}.{fmt}")
    fig.savefig(output_path, format=fmt, bbox_inches="tight")
    plt.show()
    print(f"Combined merged figure saved to: {output_path}")
    
assemble_merged_knee_plots(interpData_wClusters_xr, output_folder,
                           filename="angle_jrl_figure",
                           fmt="jpg", corrected_alpha=corrected_alpha, font_size=14)
#%% END OF FULL CODE
#=========================================================================

if __name__ == '__main__':
    print("Full analysis and export complete.")