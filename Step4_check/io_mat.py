# c3dBox/Step4_check/io_mat.py
"""
Tiny helpers to load/save MATLAB dicts for Step4_check.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Dict

from scipy.io import savemat, loadmat

__all__ = ["load_dict", "derive_check_path", "save_dict_check"]


def load_dict(path: str | Path) -> Dict[str, Any]:
    """
    Load a MATLAB .mat file into a Python dict.

    We first try a simple `scipy.io.loadmat(..., simplify_cells=True)`. If this
    SciPy feature is unavailable (older SciPy), we fall back to `squeeze_me=True`
    and basic post-processing.

    Parameters
    ----------
    path : str | Path
        Path to the .mat file.

    Returns
    -------
    dict
        A plain Python dictionary.
    """
    p = str(path)
    try:
        return loadmat(p, simplify_cells=True)  # type: ignore[call-arg]
    except TypeError:
        # SciPy < 1.8 does not support simplify_cells
        raw = loadmat(p, struct_as_record=False, squeeze_me=True)
        # remove matlab metadata keys that start with __
        return {k: v for (k, v) in raw.items() if not (k.startswith("__") and k.endswith("__"))}


def derive_check_path(original_path: str | Path) -> Path:
    p = Path(original_path)
    name = p.name
    if name.lower().endswith("_check.mat"):
        return p  # idempotent
    stem = p.stem if p.suffix.lower() == ".mat" else p.name
    return p.with_name(f"{stem}_check.mat")


def save_dict_check(original_path: str | Path, data: Mapping[str, Any]) -> Path:
    """
    Save `data` as a sibling *_check.mat file using scipy.io.savemat with:
    - do_compression=True
    - long_field_names=True
    - format='5'

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
