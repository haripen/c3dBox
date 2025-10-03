# c3dBox

A collection of tools useful when working with C3D files.

To run all steps in one environment after downloading the repository, create the Conda `c3d_to_matCycles` using `Anaconda Prompt` with admin privileges. Having navigated to the root folder of the downloaded repository, run the following command:
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
