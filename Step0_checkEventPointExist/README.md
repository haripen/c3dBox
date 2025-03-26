# C3D Consistency Checker

This script checks `.c3d` files within a specified directory structure for consistency in event markers and point data, based on user-defined sequences and keywords.

**Author:** Harald Penasso  
**Co-developed with:** GPT-4 (03-mini-high model)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Prerequisites

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate c3d_checker_env
```

## Building the Executable

Build a standalone executable using PyInstaller:

```bash
pyinstaller --onefile --windowed c3d_checker.py
```

### Adding UPX Compression (Optional)

If UPX is installed and extracted to `C:\upx`, you can compress the executable by adding the `--upx-dir` option:

```bash
pyinstaller --onefile --windowed --upx-dir="C:\upx" c3d_checker.py
```

## Usage

After launching the script or executable, you will encounter GUI prompts to guide your input:

1. **Select Source Folder**: Choose the root directory containing the `.c3d` files to be analyzed.
2. **Select Output Folder**: Specify a directory to store generated log files.
3. **Define Keyword Filter**: Enter comma-separated keywords. Only files with an ENF file description beginning with these keywords will be processed. Enter `__all` to analyze every file.
4. **Static Trial Keyword**: Specify a keyword identifying static trials (default is `stand`). These files will bypass event sequence validation.
5. **Extra Marker Data**: Enter additional marker names to check in the `.c3d` files. Leave blank if no extra markers need checking.
6. **Event Sequences**: Provide comma-separated lists of expected event sequences for both the left and right foot events.
7. **Additional Event Label**: Optionally enter an extra event label to check for its presence within the files.

After providing these inputs, the script will automatically process files accordingly.

## Outputs

The script generates two log files:

- **Files Checked Log**: Summarizes which files were processed, skipped, or caused errors.
- **Consistency Log**: Lists detailed inconsistencies found during the checks, such as missing markers or incorrect event sequences.

Both log files clearly document all user-defined inputs, chosen directories, and timestamps for straightforward verification and troubleshooting.