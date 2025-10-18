# Step 0 — Patch C3D Labels

This GUI patches **POINT** and **ANALOG** labels inside C3D files using alias groups built from all C3Ds under a chosen root. Each alias group is standardized to the **most frequently used spelling (case‑sensitive)** across your dataset. The tool can also:

- **Condense**: merge `POINT/LABELS*` → `POINT/LABELS` and `POINT/DESCRIPTIONS*` → `POINT/DESCRIPTIONS` when the *total* labels ≤ 255 (removes `*2`, `*3`, … blocks).
- **Reconcile**: if `sum(POINT/LABELS*) ≠ nPoints`, trim only the **excess** labels from the tail (mirrors trims to DESCRIPTIONS) until equality holds, and set `POINT.USED = nPoints`.

Writes are **in place** (unless you enable **Dry run**). No silent fallbacks: if packages/paths are missing, the app fails loudly.

---

## Environment (Step 0 → Step 3)

Use the **single Conda environment** shared across all steps.

```powershell
# from the repository root
conda env create -f environment.yml
conda activate c3d_to_matCycles
```

This installs Python and required packages (notably `ezc3d`, `PyQt5`/`pyqt`, `numpy`, `scipy`, `pyinstaller`, …).

> If you prefer `pip`, ensure `ezc3d` and a Qt binding (`PyQt5`) are installed; the UI uses PyQt5.

---

## Launch

From the repo root (or directly inside the Step 0 folder):

```powershell
python .\Step0_patchC3Dlabels\patchC3Dlabels_ui.py
```

---

## Typical workflow

1. **Build Inventory**  
   
   - Select a **Root** folder.  
   - Click **“1) Build Label Inventory”**.  
   - Inspect the tabs (**Point Labels**, **Analog Labels**).  
   - The **Most Frequent** column is editable—change the canonical spelling if needed.

2. **Save / Load mapping**  
   
   - **“2) Save Mapping JSON”** to persist your selections.  
   - **“3) Load Mapping JSON”** to reuse a previous policy (preview appears in the UI).

3. **Patch C3D files**  
   
   - Options:  
     - **Dry run**: log only, no writes.  
     - **Debug**: verbose before/after snapshots and per‑index edits.  
     - **Condense**: merge `LABELS*`/`DESCRIPTIONS*` to single blocks if ≤ 255 total.  
     - **Reconcile**: trim excess labels to match `nPoints` and set `POINT.USED`.  
   - Click **“4) Patch C3D Files”** to write changes **in place**.

> If `sum(POINT/LABELS*) < nPoints`, the app will **raise** (it never invents labels). Use **Debug** logs to diagnose the source file.

---

## `label_mapping.json`

A small policy file the UI reads/writes that pins **which spelling** to use for each alias group (separately for POINT and ANALOG) and which groups were **mutable** (i.e., had >1 variant during inventory).

**Example:**

```json
{
  "point": {
    "group_to_mostfrequent_raw": {
      "emg#1": "EMG.EMG_01_R M.tibialis anterior",
      "lasi": "LASI"
    },
    "mutable_groups": ["emg#1", "lasi"]
  },
  "analog": {
    "group_to_mostfrequent_raw": {
      "forcefx1": "Force.Fx1"
    },
    "mutable_groups": ["forcefx1"]
  }
}
```

- `group_to_mostfrequent_raw`: alias‑group → **chosen** label spelling used during patching.  
- `mutable_groups`: only these groups get rewritten (others remain untouched).  
- Version this file to keep labeling policy consistent across machines/runs.

---

## Notes

- **Most‑frequent is case‑sensitive**: outputs preserve the exact common spelling/casing found in your dataset.  
- **No fallbacks**: imports/paths must be valid; issues are raised immediately.  
- **Best run before Step 1 (C3D → MAT)** so downstream files inherit clean labels.

---

## Optional: build a single‑file app

```powershell
pyinstaller --onefile --windowed Step0_patchC3Dlabels\patchC3Dlabels_ui.py
```

This produces a standalone executable (handy for non‑Python users).
