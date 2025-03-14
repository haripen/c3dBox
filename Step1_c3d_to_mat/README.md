# [Step 1] C3D to MAT Converter

This GUI or script convert C3D files into MATLAB MAT files using ezC3D and PyQt5. It features an interface for selecting input and output folders, filtering data, and enriching MAT files with metadata, events, point, and analog data.

Use this line after importing a .mat file to convert event labels to a Matlab cell string.

```matlab
events.event_labels = cellstr(events.event_labels);
```

In Python use

```python
from utils_py.mat2dict import loadmat_to_dict
```

---

## Features

- **GUI**: PyQt5 interface with controls and helpful tooltips.
- **File Filtering:** Choose a filter mode (e.g., by file attributes or filename) and apply custom keywords for targeted file selection.
- **JSON-Based Data-Export Filters:** Automatically or manually load filter definitions from JSON files for precise data selection.
- **Meta Data Integration:** Includes detailed metadata such as sampling rates, frame ranges, and complete file headers.
- **Structured Data Export:**
  - **Events:** Exported in character arrays.
  - **Point and Analog Data:** Organized clearly with unique, MATLAB-friendly field names and filtered according to user criteria.
- **Logging Support:** Optional detailed logging of raw data labels for setting up the JSON or for debugging.

---

## Environment Setup

Use the provided YAML file to set up the environment with Conda:

```bash
conda env create -f c3d_to_mat.yml
conda activate c3d_to_mat
```

YAML Example:

```yml
name: c3d_to_mat
channels:
- conda-forge
dependencies:
- python=3.8
- pyqt
- pyqt5
- numpy
- scipy
- ezc3d
- pyinstaller
```

## Creating an Executable

To distribute a standalone executable using PyInstaller with UPX compression (just download and extract UPX to a folder):

```bash
pyinstaller --onefile --windowed --upx-dir "C:\upx-5.0.0-win64" c3d_to_mat_pyqt.py
```

---

## Usage

Run the executable directly or via:

```bash
python c3d_to_mat_pyqt.py
```

or run the script version

```bash
python c3d_to_mat
```

Steps:

1. Select Input/Output Folders:
   Use "Browse" to select the folders for source C3D files and the output location for MAT files.

2. Filtering Options:
   
   - Select the filter method (e.g., by filename or description).
   - Enter keywords to refine selection.

3. Load JSON Filter (optional):
   Use the settings menu to load a custom JSON filter file.

4. Conversion:
   Click "Run Conversion" to start processing. Monitor progress in the log.

---

## JSON Filter Format

Example JSON format for filtering:

```json
{
  "meta": [],
  "event": [],
  "analog": ["Label1", "Label2"],
  "point": ["Label3", "Label4"]
}
```

Customize the lists to match your specific filtering criteria.

---

## Credits & License

- Author: Harald Penasso with ChatGPT assistance  
- License: MIT License

Dependencies:

- PyQt5
- ezc3d
- numpy
- scipy

Contributions through issues or pull requests are welcome!
