# [Step 2] Split C3D-MAT to Cycles

This GUI or script follows the ``C3D to MAT Converter`` and splits its `.mat` files based on C3D file event labels. It features file selection, cycle definition by event labels via JSON, and structured export of segmented MAT files.

Use the following snippet after importing `.mat` files to convert event labels into MATLAB cell strings:

```matlab
events.event_labels = cellstr(events.event_labels);
```

In Python, use:

```python
from utils_py.mat2dict import loadmat_to_dict
```

---

## Features

- **GUI**: PyQt5-based interface with tooltips for ease of use.
- **Event-Based Splitting**: Split `.mat` files using configurable event labels.
- **JSON-Based Splitting Criteria**: Load custom cycle definition from a JSON configuration file.
- **Data Organization:**
  - **Event-Driven Splitting:** Files are segmented based on specified event labels.
  - **Metadata Inclusion:** All relevant metadata preserved in split files.
- **Logging Support:** Optional verbose logging to debug or fine-tune splitting behavior.

---

## Environment Setup

Use the provided YAML file to set up the Conda environment:

```bash
conda env create -f split_c3dmat_to_cycles.yml
conda activate split_c3dmat_to_cycles
```

YAML Example:

```yml
name: split_c3dmat_to_cycles
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pyqt
  - numpy
  - scipy
```

---

## Creating an Executable

To create a standalone executable using PyInstaller with optional UPX compression:

```bash
pyinstaller --onefile --windowed --upx-dir "C:\upx-5.0.0-win64" split_by_event_pyqt.py
```

---

## Usage

Run the executable or launch via:

```bash
python split_by_event_pyqt.py
```

Or run the script version:

```bash
python split_by_event.py
```

Steps:

1. **Select Input/Output Directories:**
   
   - Click "Browse" to select directories for input MAT files and output segmented files.

2. **Define Splitting Criteria:**
   
   - Choose event markers for splitting files (e.g., start/end events).

3. **Load JSON Configuration (Optional):**
   
   - Load predefined splitting criteria from a JSON file.

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

- Author: Harald Penasso with ChatGPT assistance
- License: MIT License

Dependencies:

- PyQt5
- numpy
- scipy

Contributions via issues or pull requests are welcome!
