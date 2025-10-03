# [Step 2] Split C3D-MAT to Cycles

This GUI follows the step 1 [`C3D to MAT Converter`](https://github.com/haripen/c3dBox/tree/main/Step1_c3d_to_mat) and splits the gernerated `.mat` files based on C3D file event labels and [JSON](https://github.com/haripen/c3dBox/blob/main/Step2_split_by_event/cycle_config.json)-defined cycle-definitions to `.mat` files with time-discrete but segmented data.

## Prerequesits
Install Anaconda and create the `c3d_to_matCycles` environment from the [root folder](https://github.com/haripen/c3dBox/) of this repository.

## Features

- **Event-Based Splitting**: Split `.mat` files using configurable event labels.
- **JSON-Based Splitting Criteria**: Load cycle definition from a JSON configuration file automatically at startup or manually from the settings.
- **Data Organization:**
  - **Event-Driven Splitting:** Files are segmented based on specified event labels.
  - **Metadata Inclusion:** All relevant metadata preserved in split files.
- **Logging Support:** Optional verbose logging to debug or fine-tune splitting behavior.



## Usage

Start the graphical user interface from the root folder:

```bash
python .\Step2_split_by_event\split_by_event_pyqt.py
```

Steps:

1. **Select Input/Output Directories:**
   
   - Click "Browse" to select directories for input MAT files and output segmented files.

2. **Define Splitting Criteria:**
   
   - Choose event markers for splitting files (e.g., start/end events).

3. **Load JSON Configuration Manually (Optional):**
   
   - Load predefined splitting criteria from a JSON file from the settings menu.

4. **Run Splitting:**
   
   - Click "Run Split" to start. Monitor progress via the provided log window.

---

## JSON Configuration Format

Example JSON configuration for event-based splitting:

```json
{
  "split_events": ["EventStart", "Central Event", "EventEnd"]
}
```

Customize event labels and settings as needed.

---

## Credits & License

- Author: Harald Penasso with ChatGPT 4 assistance
- License: MIT License

Contributions via issues or pull requests are welcome!

---

## Loading the .mat in Python

Use the following snippet from the root level of the package:

```python
filepath_mat = r'.\example\mat\split_to_cycles\walk_splitCycles.mat'
from utils_py.mat2dict import loadmat_to_dict
imported_mat = loadmat_to_dict(filepath_mat)
print(imported_mat["events"]["event_labels"])
```
---

## If you like, create an executable

To create a standalone executable using PyInstaller with optional UPX compression:

```bash
pyinstaller --onefile --windowed --upx-dir "C:\upx-5.0.0-win64" split_by_event_pyqt.py
```
