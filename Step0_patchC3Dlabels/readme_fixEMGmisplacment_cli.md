# fixEMGmisplacement CLI

This CLI scans C3D files, computes EMG cycle envelopes, flags suspicious channels, and proposes label remapping per ID. It writes diagnostic plots and a JSON mapping, and can optionally apply the mapping to **copied** or **original** C3Ds.

---

## Requirements

- `ezc3d`, `numpy`, `scipy`
- `matplotlib` for plots (omit with `--no-plots`)

Use the repo conda env (`c3d_to_matCycles`) and add matplotlib if needed.

---

## Quick start (all IDs)

Run over **all IDs** and all C3Ds under the default data root:

```powershell
python .\Step0_patchC3Dlabels\fixEMGmisplacement_cli.py --task walk
```

- `--task walk` is a **case-insensitive substring** match on `.enf` `DESCRIPTION=...`
- This scans all C3Ds under `D:\Data_local\Pros_5er_hybrid_plain` by default

Outputs:
- `.\Step0_patchC3Dlabels\outputs\fixEMG_<task>_<timestamp>\`
  - `mapping\` plots
  - `processing\log.txt`
  - `mapping.json`

---

## Apply remapping (copy vs overwrite)

After reviewing plots/mapping, apply to **copied** C3Ds:

```powershell
python .\Step0_patchC3Dlabels\fixEMGmisplacement_cli.py --task walk --apply --apply-mode C3Dcopy
```

Overwrite originals (use with care):

```powershell
python .\Step0_patchC3Dlabels\fixEMGmisplacement_cli.py --task walk --apply --apply-mode C3Doverwrite
```

---

## Common options

```powershell
# Custom data root
--data-root D:\Data_local\Pros_5er_hybrid_plain

# Limit for quick tests (applied AFTER task filter)
--max-files 20

# Aggregation method for ID/global traces (default: pca)
--agg mean|median|pca

# Required correlation gain + MAD improvement for mapping suggestions (defaults: 0.10 / 0.20)
--corr-gain 0.10
--mad-gain 0.20

# Interactive review (shows suggested + borderline unsure swaps at end)
--interactive

# Optional: enforce CI gating for remaps
--use-ci

# Skip plots (faster)
--no-plots
```

---

## Examples

```powershell
# Example 1: run over all IDs (default root), task filter "walk"
python .\Step0_patchC3Dlabels\fixEMGmisplacement_cli.py --task walk

# Example 2: specify a custom root on the CLI
python .\Step0_patchC3Dlabels\fixEMGmisplacement_cli.py --data-root D:\Data_local\Pros_5er_hybrid_plain --task walk
```

Hard‑coded default root inside the script:

```
D:\Data_local\Pros_5er_hybrid_plain
```

---

## What counts as “all IDs and cycles”?

- **All IDs**: default scan includes every C3D under the data root, grouped by `SUBJECTS/NAMES`.
- **All cycles**: for each channel, all ipsilateral Foot‑Strike → Foot‑Strike cycles found in the C3D events are used.

---

## Notes

- NFU channels are ignored.
- Thresholds (SNR/correlation/lag) are derived empirically from the data distribution.
- Mapping suggestions use gains in corr + MAD by default; CI gating is optional via `--use-ci`.
- The JSON mapping format is `{id: {label_from: label_to}}`.

---

## Diagnostics scripts

### Export traces for review/training

Exports global traces and per‑ID processed traces (cycles + ID traces).

```powershell
python .\Step0_patchC3Dlabels\diagnostics\export_emg_traces.py --task walk --run-dir .\Step0_patchC3Dlabels\outputs\fixEMG_walk_YYYYMMDD_HHMMSS
```

Random IDs:
```powershell
python .\Step0_patchC3Dlabels\diagnostics\export_emg_traces.py --task walk --run-dir .\Step0_patchC3Dlabels\outputs\fixEMG_walk_YYYYMMDD_HHMMSS --random-ids 10 --seed 123
```

Outputs in:
`.\Step0_patchC3Dlabels\outputs\...\processing\exports\`

### Optimize detection thresholds (with feedback)

Uses expected labels (e.g., Rd0079=LR swap, Rd0012 OK) plus optional user feedback.

```powershell
python .\Step0_patchC3Dlabels\diagnostics\optimize_detection.py --export-dir .\Step0_patchC3Dlabels\outputs\fixEMG_walk_YYYYMMDD_HHMMSS\processing\exports --feedback user_feedback.jsonl --metric f1_combined
```

### Interactive user feedback (label_swap_feedback)

Shows plots and asks for user decision (swap / unsure / ok). Saves `user_feedback.jsonl`.

```powershell
python .\Step0_patchC3Dlabels\diagnostics\label_swap_feedback.py --export-dir .\Step0_patchC3Dlabels\outputs\fixEMG_walk_YYYYMMDD_HHMMSS\processing\exports --random-ids 10 --per-id 10 --n-swap 50 --n-random 50 --expected rd0079:lr,rd0012:ok
```

### Bootstrap global muscle stats (mapped, pooled L/R)

Applies mapping, pools left/right per muscle, bootstraps PCA dim1 stats, and plots pooled mean ± SD with side means.

```powershell
python .\Step0_patchC3Dlabels\diagnostics\bootstrapped_global_muscle_stats.py --task walk --run-dir .\Step0_patchC3Dlabels\outputs\fixEMG_walk_YYYYMMDD_HHMMSS --bootstrap 10000 --processes 20
```

Outputs:
`.\Step0_patchC3Dlabels\outputs\...\processing\bootstrap_stats\global_muscle_stats_walk.json`  
Plots (JPG + PDF): `.\Step0_patchC3Dlabels\outputs\...\processing\bootstrap_stats\plots\`

### Apply mapping.json to C3Ds (in place)

```powershell
python .\Step0_patchC3Dlabels\diagnostics\apply_mapping_json.py --task walk --mapping .\Step0_patchC3Dlabels\outputs\fixEMG_walk_YYYYMMDD_HHMMSS\mapping.json
```
