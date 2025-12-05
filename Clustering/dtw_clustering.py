# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 19:11:12 2025

@author: harald.penasso
"""
#%% --- SETUP ---
import os
wd  = r"C:\GitHub\c3dBox" # the local path to c3dBox
os.chdir(wd)
root_dir = r"C:\Data_Local\PRO_checked_mat"
g = 9.81
rms_fy_threshold_pc = 0.05
max_fy_threshold_pc = 0.1
#%% --- IMPORTS ---
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict  # for valid cycles stats
from pathlib import Path
from scipy.io import savemat
# local
from utils_py.mat2dict import loadmat_to_dict
from utils_py.access_mat import find_files, extract_knee_jrl_data
from Clustering import clustering as clustering_mod
from Clustering.clustering import cluster_timeseries#, optimize_gamma_for_k

# Simple logger that prints and stores lines
log_lines = []
def log(msg: str):
    print(msg)
    log_lines.append(msg)
# Use a timestamped base name so you can keep different runs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"knee_jrl_extraction_{timestamp}"
log_filename = os.path.normpath(root_dir + r"\\" + base_name + ".log")
npz_filename = os.path.normpath(root_dir + r"\\" + base_name + ".npz")
#%% --- ACCESS THE DATA ---
log(f"Scanning {root_dir} for files")
file_prefix = 'Rd'
file_suffix = '_splitCycles_osim_check'
file_type = 'mat'
file_dict = dict({k: dict() for k in find_files(root_dir,file_prefix,file_suffix,file_type)})
log(f" ... detected {len(file_dict.keys())} {file_prefix}***{file_suffix}.{file_type} files.")
#%% --- FILTER VALID CYCLES ---
log("Extracting valid cycles")
all_trials = []
meta_rows = []
debug_i = 0
valid_cycles_per_id = defaultdict(int)
valid_cycles_per_id_side = defaultdict(lambda: defaultdict(int))
for path in file_dict.keys():
    id_ = int(os.path.split(path)[-1].split('_')[0][2:6])
    trial = os.path.split(path)[-1].split('_')[4]
    #if debug_i != 8:
    #    debug_i += 1
    #    continue
    if True:#id_ < 6:
        log(f" ID {id_} {trial}:")
        d = loadmat_to_dict(path)
        bw = (d["meta"]["PROCESSING"]["Bodymass"] * g)
        for side in ("left_stride", "right_stride"):
            for cycle in d[side].keys():
                if "SO_forces" not in d[side][cycle]:
                    log(f" !!! missing 'SO_forces' in ID {id_}, {trial}, {side}, {cycle}")
                    continue  # skip this cycle
                man_sel = bool(d[side][cycle]["manually_selected"])
                kinetic = bool(d[side][cycle]["kinetic"])
                # Calculate SO error metric based on FY
                t = d[side][cycle]["SO_forces"]["time"]
                fy = d[side][cycle]["SO_forces"]["FY"]
                fy_normBW = fy / bw
                fz_keys = [k for k in d[side][cycle]["analog"].keys() if k.startswith("Force_Fz")]
                fz_max = np.max([np.abs(d[side][cycle]["analog"][k]).max() for k in fz_keys])
                if fz_max == 0:
                    continue # this is a kinematic cycle, avoid division by 0
                fy_normFz = fy / fz_max
                # limit to terminal stance at 10% to 30% stride cycle
                mid_stance_mask = (t-t.min() > 0.1*(t.max()-t.min())) & (t-t.min() < 0.3*(t.max()-t.min()))
                #rms_nBW = np.sqrt(np.mean(fy_normBW[mid_stance_mask]**2))
                #max_nBW = np.max(np.abs(fy_normBW[mid_stance_mask]))
                rms_nFz = np.sqrt(np.mean(fy_normFz[mid_stance_mask]**2))
                max_nFz = np.max(np.abs(fy_normFz[mid_stance_mask]))
                #rms_N = np.sqrt(np.mean(fy[mid_stance_mask]**2))
                #max_N = np.max(np.abs(fy[mid_stance_mask]))
                cnt = 0
                if (rms_nFz < rms_fy_threshold_pc) & (max_nFz < max_fy_threshold_pc) & man_sel & kinetic:
                    # get time- and bw normalized knee joint reaction loads
                    kjrl = extract_knee_jrl_data(d[side][cycle]["JRL"],side[0],bw)
                    kjrl = kjrl.rename(columns=lambda c: c.replace('_r_', '_').replace('_l_', '_'))
                    all_trials.append(kjrl.to_numpy(dtype="float64"))
                    meta_rows.append({"path": path, "side": side, "cycle": cycle})
                    # ID-level accumulation
                    valid_cycles_per_id[id_] += 1
                    valid_cycles_per_id_side[id_][side] += 1
                    cnt += 1
            log(f" ... {side} has {cnt} valid cycles")

data = np.stack(all_trials, axis=0)            
meta = pd.DataFrame(meta_rows)
feature_names = kjrl.columns.to_list()
# ---- Save data, meta, feature_names to a single compressed .npz ----
# Store meta as a structured array so we can reconstruct DataFrame later
meta_rec = meta.to_records(index=False)
np.savez_compressed(
    npz_filename,
    data=data,
    meta=meta_rec,
    feature_names=np.array(feature_names, dtype=object),
)
log(f"\nCompressed dataset written to {npz_filename}")
# Extraction summary
log("\n")
for id_, total in sorted(valid_cycles_per_id.items()):
    log(f"ID {id_}: {total} valid cycles total")
    for side, n in valid_cycles_per_id_side[id_].items():
        log(f"   {side}: {n} valid cycles")
log(f"-> extracted {data.shape[0]} valid cycles in total.")
# ---- Save log to .log file ----
with open(log_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"Log written to {log_filename}")
#%% --- PREPARE CLUSTERING ---
# Load the saved dataset (data, meta, feature_names) from compressed npz.
# Prefer the just-written file; otherwise grab the newest matching file in root_dir.
npz_candidates = sorted(Path(root_dir).glob("knee_jrl_extraction_*.npz"))
if os.path.exists(npz_filename):
    dataset_path = npz_filename
elif npz_candidates:
    dataset_path = str(npz_candidates[-1])
else:
    raise FileNotFoundError(f"No knee_jrl_extraction_*.npz file found in {root_dir}")

loaded = np.load(dataset_path, allow_pickle=True)
data = loaded["data"]
meta = pd.DataFrame(loaded["meta"])
feature_names = loaded["feature_names"].tolist()

log(f"Loaded dataset for clustering: {dataset_path}")
log(f" - data shape: {data.shape}")
log(f" - meta rows: {len(meta)}")
log(f" - features: {feature_names}")

# Prepare output locations
clustering_output_dir = os.path.join(root_dir, "clustering_results")
os.makedirs(clustering_output_dir, exist_ok=True)
excel_path = os.path.join(clustering_output_dir, f"{Path(dataset_path).stem}_clustering.xlsx")
# Debug log path (writeable)
debug_log_path = Path(wd) / "Clustering" / "bayes_debug.log"

def dbg(msg: str):
    with open(debug_log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

#%% --- CLUSTERING OPTIONS  ---
# Random seed used everywhere (also updates clustering.RANDOM_STATE_GLOBAL).
random_state = 42
clustering_mod.set_random_state_global(random_state)

# Method and distance metric
cluster_method = "agglo_ward"      # options: "agglo_ward", "ts_kmeans"
dm_method = "softdtw"              # keep soft-DTW for Ward

# Distance scaling:
# - use_scaler=True applies tslearn.TimeSeriesScalerMeanVariance (z-score per series).
use_scaler = True

# Optimization step (choose one mode)
optimization_mode = "grid"         # "grid" or "bayes"
opt_gammas = [0, 1e-2, 1]  # gamma values to scan (and to seed bayes)
opt_k_range = (2,3,4,5,6)            # dendrogram cuts to evaluate during optimization
gamma_space_bounds = (1e-8, 1e-4)   # bounds for Bayesian search
clustering_mod.set_gamma_space(*gamma_space_bounds)

#%% --- OPTIMIZE CLUSTERING ---
# Runs agglo_ward with grid_search="on" (or Bayes) to compute distances per gamma,
# produce dendrograms and per-k plots, and write metrics to Excel.
# Single call; cluster_timeseries performs grid or bayes internally based on optimization_mode.
dbg(f"[optimize] starting cluster_timeseries optimization_mode={optimization_mode}, opt_gammas={opt_gammas}, opt_k_range={opt_k_range}")
opt_run = cluster_timeseries(
    data=data,
    feature_names=feature_names,
    meta=meta,
    output_folder=clustering_output_dir,
    excel_path=excel_path,
    method=cluster_method,
    dm_method=dm_method,
    gamma=opt_gammas[0],          # placeholder; grid_search iterates over opt_gammas
    agglo_k_range=opt_k_range,
    gammas=opt_gammas,
    grid_search="on",
    use_scaler=use_scaler,
    plot=True,
    random_state=random_state,
    optimization_mode=optimization_mode,
)
dbg(f"[optimize] opt_run keys: {list(opt_run.keys()) if isinstance(opt_run, dict) else 'None'}")
opt_grid_results = opt_run.get("grid_search_results") if isinstance(opt_run, dict) else None
bayes_results_df = opt_run.get("bayes_results") if isinstance(opt_run, dict) else None
best_gamma_per_k = opt_run.get("bayes_best_gamma_per_k", {}) if isinstance(opt_run, dict) else {}
dbg(f"[optimize] grid_results rows: {len(opt_grid_results) if opt_grid_results is not None else 'None'}")
dbg(f"[optimize] bayes_best_gamma_per_k: {best_gamma_per_k}")
dbg(f"[optimize] bayes_results_df rows: {len(bayes_results_df) if bayes_results_df is not None else 'None'}")
if bayes_results_df is not None:
    dbg(f"[optimize] bayes_results_df content:\n{bayes_results_df}")
# Save bayes_results_df to Excel as a separate sheet if present
if bayes_results_df is not None and not bayes_results_df.empty:
    try:
        with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            bayes_results_df.to_excel(writer, sheet_name="bayes_results", index=False)
        dbg("[optimize] bayes_results_df written to sheet 'bayes_results'")
    except Exception as e:
        dbg(f"[optimize] Could not save bayes_results_df to Excel: {e}")
#%% --- USER INSPECTION AND DECISSION ---
# Inspect outputs in clustering_results:
# - Dendrograms per gamma: dendrogram_grid_softdtw_gamma*.pdf
# - Cluster plots per gamma/k: clustering_softdtw_grid_k*_gamma*.pdf
# - Metrics: Excel sheet 'grid_search_results' (silhouette, DB)
# Choose the best gamma/k based on separation (higher silhouette, lower DB) and visual inspection.
# Set the final choices below (single k).
final_k = 2
final_gamma = None
final_dendro_colors = None  # e.g., ["#1f77b4", "#ff7f0e"] to force dendrogram colors for final_k
# Prefer Bayesian result for this k, else best grid gamma at this k, else first opt gamma.
if final_k in best_gamma_per_k:
    final_gamma = best_gamma_per_k[final_k]
elif opt_grid_results is not None:
    subset = opt_grid_results[opt_grid_results["k"] == final_k]
    if not subset.empty:
        final_gamma = subset.sort_values("silhouette_dtw", ascending=False).iloc[0]["gamma"]
if final_gamma is None:
    final_gamma = opt_gammas[0]
log(f" ... Will be using k = {final_k} with final gamma = {final_gamma} in the final run.")
#%% --- FINAL CLUSTERING ---
final_run = cluster_timeseries(
    data=data,
    feature_names=feature_names,
    meta=meta,
    output_folder=clustering_output_dir,
    excel_path=excel_path,
    method=cluster_method,
    dm_method=dm_method,
    gamma=final_gamma,
    agglo_k_range=(final_k,),
    grid_search="off",
    use_scaler=use_scaler,
    plot=True,
    random_state=random_state,
    optimization_mode="grid",
    dendro_colors=final_dendro_colors,
)

labels_by_k = final_run.get("labels_by_k", {})
final_labels = labels_by_k.get(final_k, final_run.get("cluster_labels"))
if final_labels is None:
    raise ValueError(f"No cluster labels available for k={final_k}")

meta_final = meta.copy()
meta_final["cluster"] = final_labels

# Attach cluster labels back into each source .mat file without overwriting originals.
# Group meta by path, update all side/cycle entries per file, then save one _c copy per file.
clusters_per_file = {}
for path_val, grp in meta_final.groupby("path"):
    mat_path = Path(path_val)
    d = loadmat_to_dict(str(mat_path))
    updated = False
    for _, row in grp.iterrows():
        side = row["side"]
        cycle = row["cycle"]
        cluster_label = int(row["cluster"])
        clusters_per_file.setdefault(mat_path, {}).setdefault(cluster_label, 0)
        clusters_per_file[mat_path][cluster_label] += 1
        if side in d and cycle in d[side]:
            d[side][cycle]["cluster"] = cluster_label
            updated = True
        else:
            log(f"Missing side/cycle in {mat_path} -> {side}/{cycle}, skipping cluster write.")
    out_path = mat_path.with_name(f"{mat_path.stem}_c.mat")
    if updated:
        savemat(
            str(out_path),
            d,
            do_compression=True,
            long_field_names=True,
            oned_as="column",
        )
        log(f"Cluster labels written to {out_path}")
    else:
        log(f"No updates written for {mat_path} (no matching side/cycle).")

# Print summary: cycles per file per cluster, and ID/side distribution
log("\nCluster counts per file:")
for fpath, cl_map in clusters_per_file.items():
    log(f" {fpath.name}: " + ", ".join([f"cluster {c}: {n}" for c, n in cl_map.items()]))

# Build ID/side distributions
id_side_counts = meta_final.copy()
id_side_counts["id"] = id_side_counts["path"].apply(lambda p: Path(p).name.split("_")[0][2:6])
cluster_summary = id_side_counts.groupby(["cluster", "id", "side"]).size().reset_index(name="count")
log("\nID/side distribution per cluster:")
for _, row in cluster_summary.iterrows():
    log(f" cluster {row['cluster']}: ID {row['id']} {row['side']} -> {row['count']} cycles")
#%%
