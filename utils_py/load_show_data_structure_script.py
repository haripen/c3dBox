# -*- coding: utf-8 -*-
"""
load file and show structure
"""
from .mat2dict import loadmat_to_dict
fn = r"C:\Data_Local\opensim_WalkA_extracted_upd\Rd0001_Rd0001_1993_06_18\2022_10_25\matfiles\Rd0001_Rd0001_2022-10-25_13-51-45_Dynamic01_WalkA01_fp0_clean_cropped_splitCycles_osim.mat"
fn = r"C:\Data_Local\PRO_checked_mat\Rd0001_Rd0001_2022-10-25_13-51-45_Dynamic01_WalkA01_fp0_clean_cropped_splitCycles_osim_check.mat"
d = loadmat_to_dict(fn)

from collections.abc import Mapping, Sequence

def maybe_save_summary(summary: str, default_filename: str = "dict_summary.txt") -> None:
    """
    Interactively ask the user whether to save `summary` to a text file.
    If yes, allow changing the filename and handle basic IO errors gracefully.
    """
    try:
        choice = input("Save summary to a text file? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return  # Non-interactive or cancelled

    if choice not in ("y", "yes"):
        return

    try:
        fname = input(f"Enter filename [{default_filename}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not fname:
        fname = default_filename

    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved to {fname}")
    except OSError as e:
        print(f"Could not save file: {e}")


def dict_tree_summary(d, max_list_items=8) -> str:
    """
    Build a hierarchical tree of a nested dict-like structure.
    For each endpoint value, show its type and size info.
    
    Parameters
    ----------
    d : dict
        The (possibly nested) dictionary to summarize.
    max_list_items : int
        Maximum number of sequence elements (list/tuple) to display per node.
    
    Returns
    -------
    str
        A pretty-printed tree.
    """
    lines = []
    seen_ids = set()

    # Optional imports (numpy, pandas, torch) are detected if present.
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None
    try:
        import pandas as _pd  # type: ignore
    except Exception:
        _pd = None
    try:
        import torch as _torch  # type: ignore
    except Exception:
        _torch = None

    def _is_mapping(x):
        return isinstance(x, Mapping)

    def _is_sequence(x):
        # Treat strings/bytes as scalars, not sequences
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

    def _endpoint_type_and_size(x):
        """
        Return (type_str, size_str) describing x.
        """
        tname = type(x).__name__

        # Numpy
        if _np is not None and isinstance(x, (_np.ndarray,)):
            dtype = str(x.dtype)
            shape = "x".join(map(str, x.shape))
            return (f"np.ndarray[{dtype}]", f"shape=({shape}), n={x.size}")

        # Pandas
        if _pd is not None:
            if isinstance(x, _pd.DataFrame):
                rows, cols = x.shape
                return ("pd.DataFrame", f"shape=({rows}x{cols}), columns={list(x.columns)}")
            if isinstance(x, _pd.Series):
                return ("pd.Series", f"len={x.shape[0]}, name={x.name}, dtype={x.dtype}")

        # Torch
        if _torch is not None and isinstance(x, (_torch.Tensor,)):
            shape = "x".join(map(str, list(x.shape)))
            dtype = str(x.dtype).replace("torch.", "")
            device = str(x.device)
            return (f"torch.Tensor[{dtype}]", f"shape=({shape}), n={x.numel()}, device={device}")

        # Bytes / string
        if isinstance(x, (str,)):
            return ("str", f"len={len(x)}")
        if isinstance(x, (bytes, bytearray)):
            return (tname, f"len={len(x)}")

        # Mapping / Sequence sizes (used when they are endpoints, e.g., empty)
        if _is_mapping(x):
            return (tname, f"keys={len(x)}")
        if _is_sequence(x):
            return (tname, f"len={len(x)}")

        # Generic objects
        if hasattr(x, "shape"):
            try:
                shape = tuple(getattr(x, "shape"))
                size = getattr(x, "size", None)
                size_str = f", n={size}" if isinstance(size, int) else ""
                return (tname, f"shape={shape}{size_str}")
            except Exception:
                pass

        if hasattr(x, "__len__"):
            try:
                return (tname, f"len={len(x)}")
            except Exception:
                pass

        # Scalar fallback
        return (tname, "scalar")

    def _is_endpoint(x):
        """
        Decide if we stop descent at x (True) or keep recursing (False).
        We recurse into dicts and sequences by default, except for special
        data containers (np/pd/torch) which are treated as endpoints.
        """
        # Treat special containers as endpoints
        if (_np is not None and isinstance(x, (_np.ndarray,))) or \
           (_pd is not None and isinstance(x, (_pd.DataFrame, _pd.Series))) or \
           (_torch is not None and isinstance(x, (_torch.Tensor,))):
            return True

        # Strings/bytes are scalar endpoints
        if isinstance(x, (str, bytes, bytearray)):
            return True

        # Recurse into dicts and list/tuple; treat sets as endpoints (unordered)
        if _is_mapping(x):
            return False
        if isinstance(x, (list, tuple)):
            return False

        # Other sequences (e.g., range) are endpoints
        if _is_sequence(x):
            return True

        # Everything else is an endpoint
        return True

    def _add_line(prefix, is_last, name, value, force_endpoint=False):
        branch = "└── " if is_last else "├── "
        connector = prefix + branch

        if force_endpoint or _is_endpoint(value):
            t, s = _endpoint_type_and_size(value)
            lines.append(f"{connector}{name}: <{t}> {s}")
            return

        # Non-endpoint containers (dict/list/tuple) get a heading line
        if _is_mapping(value):
            lines.append(f"{connector}{name}: dict (keys={len(value)})")
        elif isinstance(value, list):
            lines.append(f"{connector}{name}: list (len={len(value)})")
        elif isinstance(value, tuple):
            lines.append(f"{connector}{name}: tuple (len={len(value)})")
        else:
            # Fallback: treat as endpoint
            t, s = _endpoint_type_and_size(value)
            lines.append(f"{connector}{name}: <{t}> {s}")
            return

        # Prepare new prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Detect cycles
        obj_id = id(value)
        if obj_id in seen_ids:
            lines.append(f"{child_prefix}└── <cycle detected>")
            return
        seen_ids.add(obj_id)

        # Descend into children
        if _is_mapping(value):
            keys = list(value.keys())
            for idx, k in enumerate(keys):
                is_last_child = idx == len(keys) - 1
                v = value[k]
                _add_line(child_prefix, is_last_child, str(k), v)
        else:  # list/tuple
            n = len(value)
            limit = min(n, max_list_items)
            for i in range(limit):
                is_last_child = (i == limit - 1) and (n <= max_list_items)
                _add_line(child_prefix, is_last_child, f"[{i}]", value[i])
            if n > max_list_items:
                remaining = n - max_list_items
                lines.append(f"{child_prefix}└── … (+{remaining} more)")

    # Root handling
    if not _is_mapping(d):
        t, s = _endpoint_type_and_size(d)
        return f"<root>: <{t}> {s}"

    lines.append("root: dict (keys={})".format(len(d)))
    root_prefix = ""
    root_keys = list(d.keys())
    for idx, k in enumerate(root_keys):
        _add_line(root_prefix, idx == len(root_keys) - 1, str(k), d[k])

    return "\n".join(lines)


# ---------------------------
# Example:
if __name__ == "__main__":
    """
    example = {
        "meta": {"subject": "test", "tags": ["a", "b", "c"]},
        "signals": {
            "emg": [[1, 2, 3], [4, 5, 6]],
            "force": {"trial1": [100.0, 101.2], "trial2": [98.7, 99.1]},
        },
        "notes": "free text",
        "empty_list": [],
    }
    """
# --- Example integration ---
    report = dict_tree_summary(d)
    print(report)
    maybe_save_summary(report)
