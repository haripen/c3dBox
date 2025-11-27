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
rms_fy_threshold_nBW = 0.05
max_fy_threshold_nBW = 0.1
#%% --- IMPORTS ---
import numpy as np
# import matplotlib.pyplot as plt
# local
from utils_py.mat2dict import loadmat_to_dict
from utils_py.access_mat import find_files, extract_knee_jrl_data
#%% --- ACCESS THE DATA ---
file_dict = dict({k: dict() for k in find_files(root_dir,'Rd','_splitCycles_osim_check','mat')})

#%% --- FILTER VALID CYCLES ---
debug_i = 0
for path in file_dict.keys():
    if debug_i != 8:
        debug_i += 1
        continue
    d = loadmat_to_dict(path)
    bw = (d["meta"]["PROCESSING"]["Bodymass"] * g)
    for side in ("left_stride", "right_stride"):
        for cycle in d[side].keys():
            man_sel = bool(d[side][cycle]["manually_selected"])
            t = d[side][cycle]["SO_forces"]["time"]
            fy = d[side][cycle]["SO_forces"]["FY"]
            fy_norm = fy / bw
            # limit to terminal stance
            mid_stance_mask = (t-t.min() > 0.1*(t.max()-t.min())) & (t-t.min() < 0.3*(t.max()-t.min()))
            rms_nBW = np.sqrt(np.mean(fy_norm[mid_stance_mask]**2))
            max_nBW = np.max(np.abs(fy_norm[mid_stance_mask]))
            rms_N = np.sqrt(np.mean(fy[mid_stance_mask]**2))
            max_N = np.max(np.abs(fy[mid_stance_mask]))
            if (rms_nBW < rms_fy_threshold_nBW) & (max_nBW < max_fy_threshold_nBW) & man_sel:
                # get time- and bw normalized knee joint reaction loads
                kjrl = extract_knee_jrl_data(d[side][cycle]["JRL"],side[0],bw)
                # bring to format ID side cycle jrlPat_0 ... jrlPat_100, jrlPat_0 ... jrlPat_100, jrlMedTot_0 ... jrlMedTot_100
    break
            

#%%
d["left_stride"]["cycle1"]["manually_selected"]

d = loadmat_to_dict(file_list[0])


#%%
