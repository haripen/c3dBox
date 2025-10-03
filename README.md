# c3dBox

A collection of tools useful when working with C3D files.

## Python installation and setup
Environment Setup

1. [Download ](https://www.anaconda.com/download/success)and install Python via anaconda
  
2. [Download ](https://github.com/haripen/c3dBox/archive/refs/heads/main.zip)the repository as ZIP File
  
3. Extract the ZIP file to a desired folder
  
4. In Anaconda Prompt Command Line with admin privileges, use ``cd My_Path`` or ``cd /d MyPathOnAnotherDrive`` to switch to the folder where you extraxted the ZIP content to.

To run all steps of this repository in one Conda environment after downloading the repository, create the Conda `c3d_to_matCycles` environment using `Anaconda Prompt` with admin privileges. Having navigated to the root folder of the downloaded repository, run the following command:
```powershell
conda env create -f environment.yml
```
This will install:
```yaml
name: c3d_to_matCycles
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - ezc3d
  - tk
  - pyinstaller
  - pyqt
```

Check out the folders for tools and infos on the steps.

**Thanks** to Sebastian Durstberger for showing me the hidden depths of C3D files!

---

License: [MIT License](https://github.com/haripen/c3dBox/blob/main/LICENSE)
