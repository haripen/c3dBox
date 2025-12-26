# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:37:08 2025

@author: harald.penasso
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import Akima1DInterpolator

def find_files(
    root_dir: str,
    prefix_key: str = "",
    suffix_key: str = "",
    filetype: str = "mat",
) -> list[str]:
    """
    Return a list of full paths to all files in root_dir and its subdirectories
    whose filenames match:
        prefix_key + * + suffix_key + "." + filetype

    Examples:
    - prefix_key="PRO_", suffix_key="_checked", filetype="mat"
      matches: PRO_123_checked.mat

    - prefix_key="", suffix_key="", filetype="mat"
      matches: *.mat  (all .mat files)

    Parameters
    ----------
    root_dir : str
        Root directory to search (searched recursively).
    prefix_key : str, default ""
        String that the filename must start with (before the wildcard).
    suffix_key : str, default ""
        String that the filename must end with (before the extension).
    filetype : str, default "mat"
        File extension, with or without leading dot (e.g. "mat" or ".mat").

    Returns
    -------
    list[str]
        List of absolute (or joined) file paths matching the pattern.
    """
    # Normalize filetype: allow "mat" or ".mat"
    ext = filetype.lstrip(".")
    pattern = f"{prefix_key}*{suffix_key}.{ext}"
    search_pattern = os.path.join(root_dir, "**", pattern)
    return glob.glob(search_pattern, recursive=True)

def extract_knee_jrl_data(jrl_dict, side, bw):
    jrl_df = pd.DataFrame(jrl_dict)
    base_cols = (
        f"med_cond_weld_{side}_on_tibial_plat_{side}_in_tibial_plat_{side}",
        f"lat_cond_joint_{side}_on_lat_cond_{side}_in_lat_cond_{side}",
        f"fem_pat_{side}_on_patella_{side}_in_patella_{side}"
    )
    components = ("fx", "fy", "fz")
    kjrl_df = pd.DataFrame()
    for base in base_cols:
        kjrl_df[f"{base}_resultant"] = np.linalg.norm(jrl_df[[f"{base}_{comp}" for comp in components]].values, axis=1)/bw
    kjrl_df[f"mediototal_{side}_resultant_ratio_pc"] = (
        kjrl_df[f"{base_cols[0]}_resultant"] /
        (kjrl_df[f"{base_cols[0]}_resultant"]+kjrl_df[f"{base_cols[1]}_resultant"]) * 100
    )
    kjrl_df[f"total_{side}_resultant_tibfem_force"] = (
        kjrl_df[f"{base_cols[0]}_resultant"] +
        kjrl_df[f"{base_cols[1]}_resultant"]
    )
    kjrl_df.drop(columns=[f"{base_cols[0]}_resultant",f"{base_cols[1]}_resultant"],inplace=True)
    akima = Akima1DInterpolator(np.linspace(0, 100, kjrl_df.shape[0]), kjrl_df.values, axis=0)
    kjrl_df = pd.DataFrame(akima(np.linspace(0, 100, 101)), columns=kjrl_df.columns)
    return kjrl_df