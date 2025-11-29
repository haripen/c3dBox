#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Add OpenSim Outputs to Cycle-Split MAT Files (with short labels)
-------------------------------------------------------------------------

- Loads OpenSim output definitions from 'osim_outputs.json' (next to this script; works when frozen).
- Loads label translations from 'shortlabels_osim_outputs.json' to keep MATLAB struct field names ≤ 63 chars.
- Indexes OpenSim files in the provided OpenSim root using the JSON 'filenameID' regex patterns.
- Scans all '*_splitCycles.mat' under the cycle-split root.
- For each MAT file:
  * Extracts the trial core tokens from the filename.
  * Locates expected OpenSim outputs for that trial (exact core match; fallback by core prefix).
  * Loads each output via utils_py.osim_access.load_osimFile(file_path, cols2return).
  * Aligns the OpenSim time vector (DataFrame index) to each cycle's 'point["time"]' and injects the columns
    under safe (≤63 char) field names derived from the translation dictionary (or auto-shortened).
- Writes the updated MAT as '*_splitCycles_osim.mat' in the same folder.
- Logs missing/unmatched items to 'add_osim_missing.log' in the cycle-split root.

Author: Harald Penasso with ChatGPT assistance
Date: 2025-10-08
License: MIT
"""

import os
import re
import sys
import json
import gc
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import scipy.io

# repo-local imports
from utils_py.mat2dict import loadmat_to_dict
from utils_py.osim_access import load_osimFile  # universal loader → DataFrame (index='time')
#%%
# ------------------------
# Configuration & Constants
# ------------------------

DEFAULT_MAT_ROOT  = r"MAT_Root"
DEFAULT_OSIM_ROOT = r"OSIM_Root"

OSIM_JSON_NAME       = "osim_outputs.json"
SHORTLABELS_JSONNAME = "shortlabels_osim_outputs.json"
MISSING_LOG_NAME     = "add_osim_missing.log"

# Tolerances for time alignment
TIME_ATOL = 1e-6
TIME_RTOL = 1e-6

# Recognized filename "core" structure (from *_splitCycles.mat)
MAT_CORE_RE = re.compile(
    r"(?i)^(?P<pid1>[A-Za-z]+\d+)_(?P<pid2>[A-Za-z]+\d+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<orig>[^_]+)_"                    # original filename token (no underscores)
    r"(?P<cond>[A-Za-z][A-Za-z0-9]*\d+)"   # condition token with trailing number(s)
    r"(?P<steps>(?:_[A-Za-z0-9-]+)*)"      # optional processing steps
    r"_splitCycles$",
    re.IGNORECASE
)

# To extract the "core" from OpenSim filenames by key (case-insensitive)
EXTRACTORS: Dict[str, Tuple[str, str, bool]] = {
    "IK":            ("IK_", "", False),  # suffix is extension
    "IK_filt":       ("IK_filt_", "", False),
    "IK_markerErr":  ("ik_marker_errors_", "", True),  # drop ext when extracting core
    "ID":            ("ID_", ".sto", False),
    "SO_activation": ("SO_", "_StaticOptimization_activation.sto", False),
    "SO_forces":     ("SO_", "_StaticOptimization_force.sto", False),
    "JRL":           ("JRL_", "_JointReaction_ReactionLoads.sto", False),
}

# ------------------------
# Utility helpers
# ------------------------

def resolve_app_dir() -> Path:
    if getattr(sys, "frozen", False):  # PyInstaller
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def load_osim_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    norm = {}
    for key, sec in cfg.items():
        if not isinstance(sec, dict):
            continue
        norm[key] = {
            "time_colLabel": str(sec.get("time_colLabel", "time")),
            "filenameID": str(sec.get("filenameID", "")),
            "columns": list(sec.get("columns", []))
        }
    return norm

def compile_patterns(osim_cfg: Dict[str, Any]) -> Dict[str, re.Pattern]:
    compiled = {}
    for key, sec in osim_cfg.items():
        pat = sec.get("filenameID", "")
        try:
            compiled[key] = re.compile(pat)
        except re.error as e:
            raise ValueError(f"Invalid regex for key '{key}': {e}\nPattern: {pat}")
    return compiled

def load_shortlabels_json(app_dir: Path) -> Dict[str, str]:
    """Load short label mapping (original -> short) if available."""
    path = app_dir / SHORTLABELS_JSONNAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # coerce to str->str
    out = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out

def is_splitcycles_mat(filename: str) -> bool:
    return filename.lower().endswith("_splitcycles.mat")

def mat_trial_core_from_name(mat_stem: str) -> Optional[str]:
    m = MAT_CORE_RE.fullmatch(mat_stem)
    if not m:
        return None
    parts = [m.group("pid1"), m.group("pid2"), m.group("date"), m.group("time"),
             m.group("orig"), m.group("cond")]
    steps = m.group("steps") or ""
    return "_".join(parts) + steps

def mat_trial_base_from_core(core: str) -> str:
    m = re.match(
        r"(?i)^(?P<pid1>[A-Za-z]+\d+)_(?P<pid2>[A-Za-z]+\d+)_"
        r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
        r"(?P<orig>[^_]+)_(?P<cond>[A-Za-z][A-Za-z0-9]*\d+)"
        r"(?:_[A-Za-z0-9-]+)*$", core, flags=re.IGNORECASE
    )
    if not m:
        return core
    parts = [m.group("pid1"), m.group("pid2"), m.group("date"), m.group("time"),
             m.group("orig"), m.group("cond")]
    return "_".join(parts)

def casefold(s: str) -> str:
    return s.casefold() if hasattr(s, "casefold") else s.lower()

def to_numpy_1d(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype == object:
        flat = []
        for elem in arr.ravel():
            flat.extend(np.asarray(elem).ravel().tolist())
        arr = np.array(flat, dtype=float)
    return arr.astype(float).ravel()

def as_matlab_column(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=float).reshape(-1, 1)

def find_aligned_window(big: np.ndarray, sub: np.ndarray,
                        atol: float = TIME_ATOL, rtol: float = TIME_RTOL) -> Optional[Tuple[int, int]]:
    if sub.size == 0 or big.size == 0 or sub.size > big.size:
        return None
    if big.size == sub.size and np.allclose(big, sub, atol=atol, rtol=rtol):
        return (0, big.size)
    cand = np.where(np.isclose(big, sub[0], atol=atol, rtol=rtol))[0]
    candidates = np.unique(np.concatenate([cand, cand-1, cand+1]))
    for i0 in candidates:
        if i0 < 0 or i0 + sub.size > big.size:
            continue
        window = big[i0:i0 + sub.size]
        if np.allclose(window, sub, atol=atol, rtol=rtol):
            return (i0, i0 + sub.size)
    cand_start = np.where(np.isclose(big, sub[0], atol=atol, rtol=rtol))[0]
    cand_end   = np.where(np.isclose(big, sub[-1], atol=atol, rtol=rtol))[0]
    for i0 in cand_start:
        i1 = i0 + sub.size - 1
        if i1 in cand_end and i1 < big.size:
            window = big[i0:i1 + 1]
            if np.allclose(window, sub, atol=atol, rtol=rtol):
                return (i0, i1 + 1)
    return None

# ---------- Header probing to avoid KeyError when columns are missing ----------

def probe_osim_header_columns(file_path: Path) -> List[str]:
    """Return the column names (exact case) from the first data header line after 'endheader'."""
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip() == "endheader":
                break
        header_line = fh.readline()
    if not header_line:
        return []
    header_line = header_line.replace("\t", " ").strip()
    header_line = " ".join(header_line.split())
    return header_line.split(" ")

# ------------------------
# Indexing OpenSim outputs
# ------------------------

def extract_core_for_key(filename: str, key: str) -> Optional[str]:
    if key not in EXTRACTORS:
        return None
    prefix, suffix, drop_ext_for_suffix = EXTRACTORS[key]
    fl = filename
    fl_low = filename.lower()
    pref_low = prefix.lower()
    suff_low = suffix.lower()

    if not fl_low.startswith(pref_low):
        return None

    if drop_ext_for_suffix:
        stem = Path(fl).stem
        core = stem[len(prefix):]
        return core

    if suff_low:
        if not fl_low.endswith(suff_low):
            return None
        core = fl[len(prefix): len(fl) - len(suffix)]
        return core
    else:
        stem = Path(fl).stem
        core = stem[len(prefix):]
        return core

def build_osim_index(osim_root: Path,
                     patterns: Dict[str, re.Pattern]) -> Dict[str, Dict[str, Path]]:
    index: Dict[str, Dict[str, Path]] = {}
    for dirpath, _, files in os.walk(osim_root):
        for fname in files:
            for key, rx in patterns.items():
                if rx.fullmatch(fname):
                    core = extract_core_for_key(fname, key)
                    if core is None:
                        continue
                    core_l = casefold(core)
                    index.setdefault(core_l, {})
                    index[core_l][key] = Path(dirpath) / fname
    return index

# ------------------------
# Short label handling
# ------------------------

def auto_shorten(name: str, maxlen: int = 63) -> str:
    """
    Deterministic shortening for very long names, used only if:
      - not present in the translation dictionary, or
      - the mapped name is still > maxlen.
    Strategy:
      1) targeted replacements for common long phrases,
      2) if still too long: keep prefix and add a 6-char hash suffix.
    """
    # Targeted phrase compressions
    repl = (
        ("sagittal_articulation_frame", "sagFrame"),
        ("Lerner_knee", "LernerKnee"),
        ("femoral_cond", "femcond"),
        ("tibial_plat", "tibplat"),
        ("med_cond", "medcond"),
        ("lat_cond", "latcond"),
    )
    short = name
    for old, new in repl:
        short = short.replace(old, new)

    if len(short) <= maxlen:
        return short

    # Last resort: prefix + hash
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    keep = maxlen - 1 - len(h)  # room for '_' + hash
    return short[:keep] + "_" + h

AUTOSHORT_SEEN = set()

def make_fieldname(orig: str, translation: Dict[str, str], used: set, logf, key_for_log: str) -> str:
    """
    Map 'orig' to a safe MATLAB struct field name (≤63 chars), using:
      1) translation dict (if available),
      2) auto_shorten (if still too long),
      3) collision avoidance within 'used' via numbered suffix.
    """
    candidate = translation.get(orig, orig)
    if len(candidate) > 63:
        before = candidate
        candidate = auto_shorten(candidate, 63)
        key_tuple = (key_for_log, before, candidate)
        if key_tuple not in AUTOSHORT_SEEN:
            AUTOSHORT_SEEN.add(key_tuple)
            logf.write(f"[INFO] AUTOSHORT {key_for_log} :: '{before}' -> '{candidate}'\n")

    # Enforce MATLAB field name constraints (alnum or '_', start with letter)
    # Our tokens comply already, but guard against edge cases:
    if not candidate or not candidate[0].isalpha():
        candidate = "f_" + candidate
    candidate = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in candidate)

    # Avoid collisions within the same struct
    base = candidate
    i = 1
    while candidate in used or len(candidate) > 63:
        suffix = f"_{i}"
        # Trim base to fit
        trunc = 63 - len(suffix)
        candidate = (base[:trunc] if len(base) > trunc else base) + suffix
        i += 1

    used.add(candidate)
    return candidate

# ------------------------
# Main per-file processing
# ------------------------

def add_osim_to_mat(mat_path: Path,
                    osim_index: Dict[str, Dict[str, Path]],
                    osim_cfg: Dict[str, Any],
                    translation: Dict[str, str],
                    missing_logf,
                    ui_logger: Optional[Callable[[str], None]] = None) -> bool:
    """
    Load a '*_splitCycles.mat', locate & load OpenSim outputs via load_osimFile, inject per-cycle with safe labels,
    and save '*_splitCycles_osim.mat'.
    """
    try:
        d = loadmat_to_dict(str(mat_path))
    except Exception as e:
        print(f"[ERROR] Loading MAT failed: {mat_path}\n  -> {e}")
        missing_logf.write(f"[ERROR] LOAD_MAT {mat_path} :: {e}\n")
        return False

    mat_stem = mat_path.stem  # includes '_splitCycles'
    core = mat_trial_core_from_name(mat_stem)
    if not core:
        msg = f"[WARN] BAD_NAME {mat_path.name}"
        print(msg)
        missing_logf.write(msg + "\n")
        if ui_logger is not None:
            ui_logger(msg)
        return False
    core_l = casefold(core)
    core_base_l = casefold(mat_trial_base_from_core(core))

    found_for_core = osim_index.get(core_l, {})
    fallback_for_core: Dict[str, Path] = {}
    if not found_for_core:
        for k_core, mapping in osim_index.items():
            if k_core.startswith(core_base_l):
                for key, pth in mapping.items():
                    fallback_for_core.setdefault(key, pth)

    def log_missing(key: str, reason: str):
        print(f"[MISS] {key} -> {reason} :: {mat_path.name}")
        missing_logf.write(f"[MISS] {key} {mat_path.name} :: {reason}\n")

    keys_to_add = list(osim_cfg.keys())

    # Preload OpenSim datasets (load once per trial)
    loaded: Dict[str, Dict[str, Any]] = {}
    for key in keys_to_add:
        src_map = found_for_core if key in found_for_core else fallback_for_core
        if key not in src_map:
            log_missing(key, "no matching file found")
            continue
        fpath = src_map[key]
        cfg = osim_cfg[key]
        want_cols: List[str] = cfg["columns"]

        # Probe header to avoid KeyError in loader if some columns don't exist
        try:
            header_cols = probe_osim_header_columns(fpath)
        except Exception as e:
            log_missing(key, f"failed to read header '{fpath.name}': {e}")
            continue
        lut = {c.lower(): c for c in header_cols}

        present_realnames = [lut[c.lower()] for c in want_cols if c.lower() in lut]
        missing_cols_for_key = [c for c in want_cols if c.lower() not in lut]

        # Load only existing columns (loader ensures 'time' ⇢ index)
        try:
            df = load_osimFile(str(fpath), tuple(present_realnames))
        except Exception as e:
            log_missing(key, f"failed to load '{fpath.name}': {e}")
            continue

        loaded[key] = {
            "path": fpath,
            "df": df,  # pandas DataFrame with time index
            "present_map": {want: lut[want.lower()] for want in want_cols if want.lower() in lut},
            "missing_cols": missing_cols_for_key
        }

    # Identify stride containers present
    stride_roots = [k for k in ("left_stride", "right_stride", "exercise") if isinstance(d.get(k), dict)]
    if not stride_roots:
        print(f"[WARN] No stride roots found in MAT: {mat_path.name}")
        missing_logf.write(f"[WARN] NO_STRIDES {mat_path.name}\n")
        return False

    injected_any = False
    for stride_root in stride_roots:
        root_obj = d.get(stride_root, {})
        if not isinstance(root_obj, dict):
            continue
        for cycle_name, cycle_obj in root_obj.items():
            if not isinstance(cycle_obj, dict):
                continue
            pt = cycle_obj.get("point", {})
            if not isinstance(pt, dict) or "time" not in pt:
                missing_logf.write(f"[WARN] NO_POINT_TIME {mat_path.name} {stride_root}.{cycle_name}\n")
                continue
            cycle_time = to_numpy_1d(pt["time"])
            if cycle_time.size == 0:
                missing_logf.write(f"[WARN] EMPTY_POINT_TIME {mat_path.name} {stride_root}.{cycle_name}\n")
                continue

            for key in keys_to_add:
                if key not in loaded:
                    continue
                rec = loaded[key]
                df = rec["df"]

                # time from DataFrame index
                try:
                    sim_time = df.index.to_numpy(dtype=float)
                except Exception:
                    sim_time = to_numpy_1d(df.index.to_numpy())

                win = find_aligned_window(sim_time, cycle_time, atol=TIME_ATOL, rtol=TIME_RTOL)
                if win is None:
                    # For SO_* keys, emit an extra [timeE] diagnostic (also to UI if provided).
                    if key.startswith("SO_"):
                        pt_times = cycle_time
                        if pt_times.size > 1:
                            dt_point = float(np.median(np.diff(pt_times)))
                        else:
                            dt_point = 0.0

                        ex_n = int(min(3, pt_times.size, sim_time.size))
                        pt_samples = [float(t) for t in pt_times[:ex_n]]
                        os_samples = []
                        if ex_n > 0:
                            for tt in pt_times[:ex_n]:
                                j = int(np.argmin(np.abs(sim_time - tt)))
                                os_samples.append(float(sim_time[j]))
                            max_diff = float(np.max(np.abs(np.array(pt_samples) - np.array(os_samples))))
                        else:
                            max_diff = 0.0

                        msg = (
                            f"[timeE] {key} {mat_path.name} {stride_root}.{cycle_name} :: "
                            f"point_time={pt_samples} vs osim_time={os_samples}, "
                            f"dt_point={dt_point}, max_diff={max_diff}"
                        )
                        print(msg)
                        missing_logf.write(msg + "\n")
                        if ui_logger is not None:
                            ui_logger(msg)

                    log_missing(key, f"time alignment failed for {stride_root}.{cycle_name}")
                    continue
                i0, i1 = win

                # Build sub-struct for cycle with safe field names
                used_names = {"time"}
                block = {"time": as_matlab_column(cycle_time)}

                # Add columns under short/translated names
                for json_col, real_col in rec["present_map"].items():
                    try:
                        vec = df[real_col].to_numpy(dtype=float)
                    except Exception:
                        vec = to_numpy_1d(df[real_col].to_numpy())
                    seg = vec[i0:i1]
                    if seg.size != cycle_time.size:
                        log_missing(key, f"length mismatch for column '{json_col}' in {stride_root}.{cycle_name}")
                        continue

                    safe_name = make_fieldname(json_col, translation, used_names, missing_logf, key_for_log=key)
                    block[safe_name] = as_matlab_column(seg)

                if rec["missing_cols"]:
                    missing_logf.write(
                        f"[MISS] COLUMNS {key} {mat_path.name} {stride_root}.{cycle_name} :: "
                        f"missing={','.join(rec['missing_cols'])}\n"
                    )

                cycle_obj[key] = block
                injected_any = True

    if not injected_any:
        print(f"[WARN] Nothing injected for: {mat_path.name}")
        missing_logf.write(f"[WARN] NO_INJECT {mat_path.name}\n")
        return False

    out_path = mat_path.with_name(mat_path.stem + "_osim" + mat_path.suffix)
    try:
        scipy.io.savemat(str(out_path), d, oned_as="column", long_field_names=True, do_compression=True)
        print(f"[OK] Saved: {out_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Saving failed: {out_path}\n  -> {e}")
        missing_logf.write(f"[ERROR] SAVE {out_path} :: {e}\n")
        return False
    finally:
        del d
        gc.collect()

# ------------------------
# CLI / Orchestration
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Append OpenSim outputs into cycle-split MAT files (short labels)."
    )
    parser.add_argument("--mat-root", default=DEFAULT_MAT_ROOT,
                        help="Root folder containing *_splitCycles.mat files (recursively scanned).")
    parser.add_argument("--osim-root", default=DEFAULT_OSIM_ROOT,
                        help="Root folder containing OpenSim output files (recursively scanned).")
    parser.add_argument("--config", default=None,
                        help=f"Path to {OSIM_JSON_NAME} (defaults to file next to this script).")
    parser.add_argument("--show-stats", action="store_true",
                        help="Print simple stats after processing.")
    args = parser.parse_args()

    app_dir = resolve_app_dir()
    config_path = Path(args.config) if args.config else (app_dir / OSIM_JSON_NAME)
    if not config_path.exists():
        print(f"[FATAL] Config JSON not found: {config_path}")
        sys.exit(2)

    osim_cfg = load_osim_config(config_path)
    patterns = compile_patterns(osim_cfg)

    mat_root = Path(args.mat_root)
    osim_root = Path(args.osim_root)

    if not mat_root.exists():
        print(f"[FATAL] MAT root not found: {mat_root}")
        sys.exit(2)
    if not osim_root.exists():
        print(f"[FATAL] OpenSim root not found: {osim_root}")
        sys.exit(2)

    # Load translation dictionary for short labels
    translation = load_shortlabels_json(app_dir)
    if translation:
        print(f"[INFO] Loaded {len(translation)} short label mappings from {SHORTLABELS_JSONNAME}")
    else:
        print(f"[INFO] No '{SHORTLABELS_JSONNAME}' found. Will auto-shorten if needed.")

    print(f"[INFO] Indexing OpenSim outputs under: {osim_root}")
    osim_index = build_osim_index(osim_root, patterns)
    print(f"[INFO] Indexed {sum(len(v) for v in osim_index.values())} files across {len(osim_index)} trials.")

    missing_log_path = mat_root / MISSING_LOG_NAME
    with open(missing_log_path, "w", encoding="utf-8") as logf:
        all_mats: List[Path] = []
        for dirpath, _, files in os.walk(mat_root):
            for fname in files:
                if is_splitcycles_mat(fname):
                    all_mats.append(Path(dirpath) / fname)
        print(f"[INFO] Found {len(all_mats)} '*_splitCycles.mat' files under: {mat_root}")

        success = 0
        for mat_file in all_mats:
            ok = add_osim_to_mat(mat_file, osim_index, osim_cfg, translation, logf)
            success += int(ok)

    if args.show_stats:
        print("[INFO] Done.")
        print(f"  MAT root:  {mat_root}")
        print(f"  OSIM root: {osim_root}")
        print(f"  Missing log: {missing_log_path}")

if __name__ == "__main__":
    main()
