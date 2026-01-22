# Step4_check

## Run the GUI

From the repository root, activate your environment and run the module:

```powershell
conda activate c3d_to_matCycles
python -m Step4_check.main
```

This is the recommended way to run Step4 because it uses package imports.

## Alternate (direct script)

If you prefer running the script directly, this also works now:

```powershell
conda activate c3d_to_matCycles
python .\Step4_check\main.py
```

Run the command from the repository root (`D:\Git\c3dBox`) so relative paths and layout JSON files resolve correctly.
