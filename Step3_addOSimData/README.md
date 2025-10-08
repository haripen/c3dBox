# [Step 3] Add OpenSim Data to Cycle‑Split MAT

This step follows **Step 2 — Split C3D‑MAT to Cycles** and appends selected OpenSim outputs to each cycle in the `*_splitCycles.mat` files. The structure and tone of this README intentionally mirror the Step 2 README for consistency.  

> Modeled after the Step 2 README style and sections.  

## Overview

- **Goal**: For every `*_splitCycles.mat` (per trial), find the corresponding OpenSim output files (IK, ID, SO, JRL, etc.), **align their time series to each cycle**, and **insert** the requested columns under each cycle (e.g., `left_stride/cycle1/IK/...`).  
- **Inputs**:  
  - Cycle‑split MAT files (output of Step 2).  
  - OpenSim outputs (e.g., `.mot`, `.sto`) in a separate OpenSim root folder.  
  - A configuration file `osim_outputs.json` describing file patterns and columns.  
  - An optional `shortlabels_osim_outputs.json` mapping for **long field names** (>63 characters) to short, readable names.
- **Outputs**: Saved as `*_splitCycles_osim.mat` next to each input MAT file (original preserved). MATLAB saves are **compressed**.

---

## Features

- **Auto‑discovery & matching**  
  - Recursively indexes OpenSim outputs by **regex** patterns from `osim_outputs.json`.  
  - Matches each cycle‑split file by a trial **core** (participant, date, time, original filename, condition, processing steps).  
  - Fallback matching: ignores processing steps if the exact core isn’t found.

- **Per‑cycle time alignment**  
  - Uses the cycle‑level `point["time"]` (from the MAT) and aligns to the **DataFrame index** from the OpenSim loader.  
  - Robust equality checks with tolerances; handles rounding/sampling differences.

- **Column selection & safety**  
  - Probes headers to request only columns that exist.  
  - Injects columns under **safe MATLAB field names (≤63 chars)** using `shortlabels_osim_outputs.json`; any that remain too long are auto‑shortened and logged.

- **Compact .mat files**  
  - Saves with `long_field_names=True` and `do_compression=True` to keep files small and compatible.

- **Clear logging**  
  - Writes `add_osim_missing.log` in the cycle‑split root: unmatched files, missing columns, time‑alignment failures, and any auto‑shortened labels.

---

## Repository Layout

```
root/
└─ c3dBox/
   ├─ Step3_addOSimData/
   │  ├─ step3_addosim.py          # CLI engine
   │  ├─ step3_addosim_gui.py      # PyQt5 GUI
   │  ├─ osim_outputs.json         # file patterns & columns
   │  └─ shortlabels_osim_outputs.json  # long→short field name mapping (optional but recommended)
   └─ utils_py/
      ├─ mat2dict.py
      └─ osim_access.py            # provides `load_osimFile(path, cols)` → pandas.DataFrame (index='time')
```

---

## Usage (GUI)

Start the graphical user interface from the repository root:

```bash
python .\Step3_addOSimData\step3_addosim_gui.py
```

**Steps**

1. **Select Cycle‑Split Root**: Folder containing the `*_splitCycles.mat` files (Step 2 output).  
2. **Select OpenSim Root**: Folder containing OpenSim outputs (recursively scanned).  
3. **Load Configs**:  
   - `osim_outputs.json` — which files & columns to pull.  
   - `shortlabels_osim_outputs.json` — mapping for long field names (optional; engine will auto‑shorten if missing).  
4. **Run**: Click **Add OpenSim Data**. Monitor progress & log; open the missing log if needed.  
5. **Result**: For each input `X_splitCycles.mat`, a `X_splitCycles_osim.mat` is saved **in the same folder**.

---

## Usage (CLI)

Run from the repository root (paths here are examples):

```bash
python .\Step3_addOSimData\step3_addosim.py ^
  --mat-root  "C:\Data_Local\opensim_WalkA_extracted_upd\Rd0001_Rd0001_1993_06_18\2022_10_25\matfiles\2022_10_25\openSim" ^
  --osim-root "C:\Data_Local\opensim_WalkA_extracted_upd\Rd0001_Rd0001_1993_06_18\2022_10_25\openSim" ^
  --show-stats
```

If `--config` or `--shortlabels` are omitted, the script loads `osim_outputs.json` and `shortlabels_osim_outputs.json` from the **same folder as the script**.

---

## Configuration Files

### `osim_outputs.json`

Defines, **per OpenSim key**, the filename regex and the list of columns to extract.

```json
{
  "IK": {
    "time_colLabel": "time",
    "filenameID": "(?i)^IK_(?P<pid>[A-Za-z]+\d+)_(?P=pid)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[^_]+_[A-Za-z][A-Za-z0-9]*\d+(?:_[A-Za-z0-9-]+)*\.mot$",
    "columns": ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]
  },
  "ID": {
    "time_colLabel": "time",
    "filenameID": "(?i)^ID_(?P<pid>[A-Za-z]+\d+)_(?P=pid)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[^_]+_[A-Za-z][A-Za-z0-9]*\d+(?:_[A-Za-z0-9-]+)*\.sto$",
    "columns": ["pelvis_tilt_moment", "pelvis_list_moment"]
  }
  // ... IK_filt, IK_markerErr, SO_activation, SO_forces, JRL
}
```

> The engine probes headers and only requests available columns; any missing columns are logged and skipped.

### `shortlabels_osim_outputs.json` (optional but recommended)

Maps **very long** column names to readable, predictable short names (≤63 chars). Example:

```json
{
  "Lerner_knee_r_on_sagittal_articulation_frame_r_in_sagittal_articulation_frame_r_fx": "LernerKnee_r_on_sagFrame_r_in_sagFrame_r_fx",
  "Lerner_knee_l_on_sagittal_articulation_frame_l_in_sagittal_articulation_frame_l_fy": "LernerKnee_l_on_sagFrame_l_in_sagFrame_l_fy"
}
```

If a column still exceeds 63 characters or isn’t listed, the engine **auto‑shortens** it and notes the mapping in `add_osim_missing.log`.

---

## What the Engine Does (in a bit more detail)

1. **Indexing**: Compile regex for each key (`IK`, `ID`, `SO_*`, `JRL`, …) and index matching files by a trial **core**.  
2. **Matching**: For each `*_splitCycles.mat`, extract its core and look up all required OpenSim files (exact, then relaxed by base core).  
3. **Loading**: Use `load_osimFile(file_path, cols)` to read only the requested columns. The loader returns a **pandas DataFrame** with `time` as **index**.  
4. **Alignment**: For each cycle, align the OpenSim time index to `point["time"]` using robust equality checks.  
5. **Injection**: Insert an OpenSim block (e.g., `IK`) under the cycle with:  
   - `time` (column vector)  
   - the requested columns as MATLAB fields (short labels applied).  
6. **Save**: Write `*_splitCycles_osim.mat` with **compression** and **long field names enabled**.  
7. **Log**: Record any misses (files, columns), alignment failures, and auto‑shortened labels.

---

## Example: Inspect an Updated File

```python
from utils_py.mat2dict import loadmat_to_dict

fn = r"C:\...\Rd0001_Rd0001_2022-10-25_13-55-20_Dynamic14_WalkA09_fp0_clean_cropped_splitCycles_osim.mat"
d = loadmat_to_dict(fn)

print(d.keys())  # expect: 'left_stride', 'right_stride' (or 'exercise'), 'meta', 'events'
print(d['right_stride']['cycle1'].keys())  # includes IK, ID, SO_forces, JRL, ...

block = d['right_stride']['cycle1']['JRL']
print(block.keys())  # 'time' + short labels
```

---

## Troubleshooting

- **No matching OpenSim files**: Check the `filenameID` regex and folder roots. See `add_osim_missing.log`.  
- **Time alignment failed**: The MAT and OpenSim times must cover the same cycle window. Confirm preprocessing and adjust tolerances if needed.  
- **Missing columns**: Ensure the column names in `osim_outputs.json` match the file headers (case‑insensitive).  
- **Field name too long**: Add an entry to `shortlabels_osim_outputs.json`; otherwise, the engine auto‑shortens and logs it.  
- **Large outputs**: Compression is enabled by default; consider pruning unused columns in `osim_outputs.json`.

---

## Credits & License

- Author: Harald Penasso with ChatGPT assistance  
- License: MIT License

Contributions via issues or pull requests are welcome!

---

## Build an Executable (optional)

**GUI (Windows):**
```bash
pyinstaller --onefile --windowed ^
  --add-data "Step3_addOSimData\osim_outputs.json;." ^
  --add-data "Step3_addOSimData\shortlabels_osim_outputs.json;." ^
  Step3_addOSimData\step3_addosim_gui.py
```

**GUI (macOS/Linux):**
```bash
pyinstaller --onefile --windowed   --add-data "Step3_addOSimData/osim_outputs.json:."   --add-data "Step3_addOSimData/shortlabels_osim_outputs.json:."   Step3_addOSimData/step3_addosim_gui.py
```

**CLI only:**
```bash
pyinstaller --onefile Step3_addOSimData/step3_addosim.py
```
