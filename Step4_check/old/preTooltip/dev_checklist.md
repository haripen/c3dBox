# Step4_check – Developer Test Checklist

A concise, repeatable checklist to verify the Step4_check UI end‑to‑end behavior.

---

## Preconditions

* [ ] You have a folder with test `.mat` files (e.g., `c3dBox/Step4_check/testing/`).
* [ ] No `_check.mat` exists yet for a clean run (or note existing ones for the “existing logs” path).
* [ ] App builds/launches with PySide6; matplotlib is used (no pyplot).

Keyboard cheatsheet

* Save: **Ctrl+S**
* Undo / Redo: **Ctrl+Z** / **Ctrl+Shift+Z**
* Rectangle mode: **s** = select, **d** = deselect
* Rescale Y to selected: **Ctrl+U** (same as clicking **New y‑lim**)

Artifacts to inspect

* Saved data: sibling `*_splitCycles[_osim]_check.mat`
* Audit logs: `Step4_check_logs/{PID}__{TrialType}.csv`

---

## 1) Pick root → dropdowns populate

1. [ ] Launch the app. Click **Set Root** and choose the testing folder.
2. [ ] Verify **Participant** dropdown populates with IDs parsed from filenames.
3. [ ] Select a participant → verify **Trial Type** dropdown populates with types for that PID.
4. [ ] Ensure file priority is respected when multiple versions exist: `_osim_check` > `_osim` > base `_splitCycles`.
5. [ ] When both PID and trial type are chosen, data loads and subplots render quickly.

**Pass if:** Both dropdowns populate as described, and plots appear without errors.

---

## 2) Toggle checkboxes → plots refresh

1. [ ] Uncheck **Left** → all left‑side lines vanish; recheck → lines return.
2. [ ] Uncheck **Right** → all right‑side lines vanish; recheck → lines return.
3. [ ] Toggle **Kinetic** (1) and **Kinematic** (0) filters → only matching cycles are drawn.
4. [ ] Hover lines to see tooltips with **original filename**, **trial‑type + stride‑side**, **cycle number**.

**Pass if:** Plot refresh is immediate and consistent with checkbox state.

---

## 3) Rectangle select/deselect → counts update

1. [ ] Press **d**, drag a rectangle across several lines → affected cycles become **deselected** (reduced alpha).
2. [ ] Observe the **status bar** counts update: Selected/Unselected for **KIN** and **KINEM** totals.
3. [ ] Press **s**, drag over some deselected lines → they revert to **selected** (full alpha), counts update again.

**Pass if:** Visual selection state and counts stay in sync across all subplots.

---

## 4) Undo / Redo

1. [ ] Make a selection change (e.g., deselect 3 lines).
2. [ ] Press **Ctrl+Z** → the change is undone; counts revert.
3. [ ] Press **Ctrl+Shift+Z** → the change is redone; counts match step 1.

**Pass if:** History behaves deterministically and updates both visuals and counts.

---

## 5) “New y‑lim” (or **Ctrl+U**) → rescale to selected visible lines + margin

1. [ ] With a mix of selected/deselected lines visible, click **New y‑lim** (or press **Ctrl+U**).
2. [ ] Each subplot rescales its Y‑axis using only **currently visible & selected** lines, applying the configured margin.

**Pass if:** No subplot uses deselected or hidden lines for autoscale; margins look correct.

---

## 6) Save → username popup → `_check.mat` written

1. [ ] With unsaved changes present, press **Ctrl+S** (or click **Save**).
2. [ ] If no username has been set this session, a **username popup** appears.
   • [ ] If prior logs exist under the chosen root, a **dropdown** suggests known usernames.
   • [ ] Otherwise, allow entering a new name.
3. [ ] Confirm save → a sibling `*_check.mat` is written next to the input file.
4. [ ] Inspect the `_check.mat`: per‑cycle fields **manually_selected**, **reconstruction_ok**, and QC flags (e.g., IK/SO) exist and reflect current state.
5. [ ] Make a second change and save again → only changed cycles register as changed in the audit log (see next section).

**Pass if:** File writes succeed, data round‑trips, and subsequent saves detect deltas.

---

## 7) Audit log entry created

1. [ ] After saving, open `Step4_check_logs/{PID}__{TrialType}.csv`.
2. [ ] Verify a new row with **timestamp**, **username**, a per‑**side×mode** breakdown of **changed final selection states**, and **final counts**.
3. [ ] Cross‑check that final counts equal the status bar values at the moment of save.

**Pass if:** Log row is appended (not overwritten) and values are correct.

---

## 8) Reopen app → “Set Username” shows discovered names

1. [ ] Close and relaunch the app.
2. [ ] Open the **Set Username** dialog/menu.
3. [ ] Verify the **dropdown** includes usernames discovered from existing log files under the current root.

**Pass if:** Known usernames are offered; selecting one suppresses future prompts until changed.

---

## Notes / Edge Cases

* [ ] Missing data keys: subplots remain empty without warnings; app stays stable.
* [ ] Navigation or close with unsaved changes: prompt to save; cancelling preserves state.
* [ ] QC dashing: any failed QC flag renders dashed lines; selection alpha still applies.
* [ ] EMG channels (keys containing “emg”, case‑insensitive) display processed envelopes.

---

### Sign‑off

* QA: ________  Date: ________
* Dev: ________  Date: ________
