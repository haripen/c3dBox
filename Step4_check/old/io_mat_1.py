# c3dBox/Step4_check/io_mat.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scipy.io import savemat
from ..utils_py.mat2dict import loadmat_to_dict

__all__ = ["load_dict", "derive_check_path", "save_dict_check"]


def load_dict(path: str | Path) -> dict[str, Any]:
    """
    Load a MATLAB .mat file into a Python dict using utils_py.mat2dict.loadmat_to_dict.

    Parameters
    ----------
    path : str | Path
        Path to the .mat file.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the .mat contents.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MAT file not found: {p}")
    return loadmat_to_dict(str(p))


def derive_check_path(original_path: str | Path) -> Path:
    """
    Derive the sibling *_check.mat path for a given original file.

    Examples
    --------
    /data/session1/file.mat       -> /data/session1/file_check.mat
    /data/session1/file           -> /data/session1/file_check.mat
    /data/session1/file.v1.mat    -> /data/session1/file.v1_check.mat

    Parameters
    ----------
    original_path : str | Path
        Original file path.

    Returns
    -------
    pathlib.Path
        Path to the sibling *_check.mat file.
    """
    p = Path(original_path)
    # If it ends with .mat, drop just that suffix; otherwise keep name as-is.
    stem = p.stem if p.suffix.lower() == ".mat" else p.name
    return p.with_name(f"{stem}_check.mat")


def save_dict_check(original_path: str | Path, data: Mapping[str, Any]) -> Path:
    """
    Save `data` as a sibling *_check.mat file using scipy.io.savemat with:
    - do_compression=True
    - long_field_names=True
    - format='5'

    Parameters
    ----------
    original_path : str | Path
        The source file path used to derive the *_check.mat destination.
    data : Mapping[str, Any]
        Dictionary-like object to save.

    Returns
    -------
    pathlib.Path
        The path to the saved *_check.mat file.
    """
    out_path = derive_check_path(original_path)
    # Ensure parent directory exists (should already, but harmless to enforce)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        str(out_path),
        data,  # type: ignore[arg-type]  # savemat accepts mapping-like
        do_compression=True,
        long_field_names=True,
        format="5",
    )
    return out_path
