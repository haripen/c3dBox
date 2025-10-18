# c3dBox/Step4_check/cycles.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

__all__ = ["CycleRef", "collect_cycles"]


@dataclass(frozen=True)
class CycleRef:
    """
    Lightweight pointer to a stride cycle.

    Attributes
    ----------
    side : str
        Either 'left_stride' or 'right_stride'.
    cycle : str
        The cycle identifier/name (stringified if originally non-string).
    groups : List[str]
        Group names available for this cycle (e.g., 'markers', 'events', ...).
    flags : Dict[str, Any]
        Flag values captured from the cycle (e.g., 'manually_selected', 'reconstruction_ok').
    """
    side: str
    cycle: str
    groups: List[str]
    flags: Dict[str, Any]


_FLAG_KEYS = {"manually_selected", "reconstruction_ok", "initial_manually_selected"}
_EXPECTED_SIDES = ("left_stride", "right_stride")


def _iter_sides(obj: Dict[str, Any]) -> Iterable[str]:
    for side in _EXPECTED_SIDES:
        if side in obj and isinstance(obj[side], dict):
            yield side


def _detect_groups(cycle_dict: Dict[str, Any], expected: Iterable[str] | None) -> List[str]:
    """
    Determine which groups are available for a cycle.

    If 'expected' is provided (e.g., from file_meta['groups']), use the
    intersection with keys present in the cycle. Otherwise, infer by heuristics.
    """
    if expected:
        return [g for g in expected if g in cycle_dict]

    # Heuristic: treat non-flag mapping/list-like entries as groups.
    groups: List[str] = []
    for k, v in cycle_dict.items():
        if k in _FLAG_KEYS:
            continue
        if isinstance(v, (dict, list, tuple)):
            groups.append(k)
    return groups


def _ensure_flags(cycle_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure per-cycle keys exist with required defaults and capture a stable
    snapshot for auditing.

    - manually_selected: default 1 (int)
    - reconstruction_ok: placeholder True (bool)
    - initial_manually_selected: first-seen snapshot (do not overwrite)
    """
    if "manually_selected" not in cycle_dict:
        cycle_dict["manually_selected"] = 1  # default
    if "reconstruction_ok" not in cycle_dict:
        cycle_dict["reconstruction_ok"] = True  # placeholder
    if "initial_manually_selected" not in cycle_dict:
        cycle_dict["initial_manually_selected"] = cycle_dict.get("manually_selected", 1)

    # Return a shallow flags view to embed in CycleRef
    return {
        "manually_selected": cycle_dict.get("manually_selected"),
        "reconstruction_ok": cycle_dict.get("reconstruction_ok"),
        "initial_manually_selected": cycle_dict.get("initial_manually_selected"),
    }


def collect_cycles(dict_obj: Dict[str, Any], file_meta: Dict[str, Any]) -> List[CycleRef]:
    """
    Collect CycleRef entries from a stride dictionary.

    Parameters
    ----------
    dict_obj : dict
        Container expected to optionally have 'left_stride' and/or 'right_stride'
        mapping to per-cycle dicts. This function is tolerant of missing sides
        and missing groups—those are simply skipped with no warnings.
        The function mutates per-cycle dictionaries to ensure required keys:
        - 'manually_selected' (default 1)
        - 'reconstruction_ok' (placeholder True)
        It also sets a one-time 'initial_manually_selected' snapshot for auditing.
    file_meta : dict
        Optional metadata. If file_meta['groups'] exists and is an iterable of
        group names, available groups are filtered to those present per cycle.

    Returns
    -------
    List[CycleRef]
        One entry per discovered cycle containing side, cycle name, available
        groups, and a flags dictionary.
    """
    expected_groups = None
    if isinstance(file_meta, dict):
        maybe_groups = file_meta.get("groups")
        if isinstance(maybe_groups, (list, tuple, set)):
            expected_groups = list(maybe_groups)

    out: List[CycleRef] = []

    for side in _iter_sides(dict_obj):
        side_block = dict_obj[side]  # type: ignore[index]
        # side_block should map cycle_name -> cycle_dict
        if not isinstance(side_block, dict):
            continue

        for cycle_name, cycle_dict in side_block.items():
            if not isinstance(cycle_dict, dict):
                # Skip non-dict entries silently to remain safe/tolerant
                continue

            # Ensure required flags and snapshot
            flags = _ensure_flags(cycle_dict)

            # Determine available groups (safe on missing: skip silently)
            groups = _detect_groups(cycle_dict, expected_groups)

            out.append(
                CycleRef(
                    side=side,
                    cycle=str(cycle_name),
                    groups=groups,
                    flags=flags,
                )
            )

    return out
