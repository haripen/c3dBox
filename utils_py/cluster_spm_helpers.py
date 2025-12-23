import os
import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


SIDES = ("left", "right")
SIDE_KEYS = {"left": "left_stride", "right": "right_stride"}


def get_bodymass(data_dict: Dict[str, Any]) -> Optional[float]:
    meta = data_dict.get("meta", {})
    for key in ("processing", "PROCESSING"):
        processing = meta.get(key, {})
        if isinstance(processing, dict) and "Bodymass" in processing:
            try:
                return float(processing["Bodymass"])
            except Exception:
                return None
    return None


def iter_cycles(data_dict: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for side in SIDES:
        side_key = SIDE_KEYS[side]
        side_dict = data_dict.get(side_key, {})
        if not isinstance(side_dict, dict):
            continue
        for cycle_id, cycle in side_dict.items():
            if isinstance(cycle, dict):
                yield side, str(cycle_id), cycle


def get_cluster_label(cycle_dict: Dict[str, Any]) -> Optional[int]:
    cluster = cycle_dict.get("cluster", None)
    if cluster is None:
        return None
    try:
        return int(np.asarray(cluster).item())
    except Exception:
        try:
            return int(cluster)
        except Exception:
            return None


def _select_point_key(point_dict: Dict[str, Any], side: str, base_name: str) -> Optional[str]:
    side_prefix = "L" if side == "left" else "R"
    candidates = [
        f"{side_prefix}{base_name}",
        base_name,
    ]
    for key in candidates:
        if key in point_dict:
            return key
    return None


def _interp_series(time: np.ndarray, values: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    if time is None or values is None:
        return None
    time = np.asarray(time).reshape(-1)
    values = np.asarray(values)
    if time.size < 2:
        return None
    t_min = time[0]
    t_max = time[-1]
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max == t_min:
        return None
    t = (time - t_min) / (t_max - t_min) * 100.0
    order = np.argsort(t)
    t = t[order]
    if values.ndim == 1:
        v = values[order]
        mask = np.isfinite(t) & np.isfinite(v)
        if mask.sum() < 2:
            return None
        return np.interp(np.linspace(0, 100, n_points), t[mask], v[mask])
    if values.ndim == 2:
        v = values[order, :]
        out = []
        for col in range(v.shape[1]):
            vc = v[:, col]
            mask = np.isfinite(t) & np.isfinite(vc)
            if mask.sum() < 2:
                out.append(None)
                continue
            out.append(np.interp(np.linspace(0, 100, n_points), t[mask], vc[mask]))
        if any(v is None for v in out):
            return None
        return np.stack(out, axis=1)
    return None


def extract_kinematic_cycles(
    data_dict: Dict[str, Any],
    variable_name: str,
    coord_index: int,
    n_points: int,
    forced_key_map: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[int], List[Dict[str, Any]], List[str]]:
    records: List[np.ndarray] = []
    labels: List[int] = []
    meta: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for side, cycle_id, cycle in iter_cycles(data_dict):
        cluster = get_cluster_label(cycle)
        if cluster is None:
            continue

        point_dict = cycle.get("point", {})
        if not isinstance(point_dict, dict):
            skipped.append(f"{cycle_id}:{variable_name}:no_point")
            continue
        forced_key = None
        if forced_key_map:
            forced_key = forced_key_map.get(variable_name)
        if isinstance(forced_key, dict):
            side_key = forced_key.get(side)
            key = side_key if side_key in point_dict else None
        elif isinstance(forced_key, str):
            key = forced_key if forced_key in point_dict else None
        else:
            key = _select_point_key(point_dict, side, variable_name)
        if key is None:
            skipped.append(f"{cycle_id}:{variable_name}:missing_key")
            continue
        values = point_dict.get(key, None)
        time = point_dict.get("time", None)
        if values is None or time is None:
            skipped.append(f"{cycle_id}:{variable_name}:missing_values")
            continue
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] <= coord_index:
            skipped.append(f"{cycle_id}:{variable_name}:bad_shape")
            continue

        series = values[:, coord_index]
        interp = _interp_series(time, series, n_points)
        if interp is None:
            skipped.append(f"{cycle_id}:{variable_name}:interp_failed")
            continue
        records.append(interp)
        labels.append(cluster)
        meta.append({"side": side, "cycle_id": cycle_id, "key": key})

    if records:
        return np.vstack(records), labels, meta, skipped
    return np.empty((0, n_points)), labels, meta, skipped


def _find_jrl_base(jrl_dict: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for base in candidates:
        fx = f"{base}_fx"
        fy = f"{base}_fy"
        fz = f"{base}_fz"
        if fx in jrl_dict and fy in jrl_dict and fz in jrl_dict:
            return base
    return None


def compute_knee_jrl_resultants(
    jrl_dict: Dict[str, Any],
    side: str,
    bw: float,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    if bw <= 0:
        return None, None, None, None
    side_letter = "l" if side == "left" else "r"
    med_base = _find_jrl_base(
        jrl_dict,
        [
            f"med_cond_joint_{side_letter}_on_med_cond_{side_letter}_in_med_cond_{side_letter}",
            f"med_cond_weld_{side_letter}_on_tibial_plat_{side_letter}_in_tibial_plat_{side_letter}",
        ],
    )
    lat_base = _find_jrl_base(
        jrl_dict,
        [
            f"lat_cond_joint_{side_letter}_on_lat_cond_{side_letter}_in_lat_cond_{side_letter}",
            f"lat_cond_weld_{side_letter}_on_tibial_plat_{side_letter}_in_tibial_plat_{side_letter}",
        ],
    )
    pat_base = _find_jrl_base(
        jrl_dict,
        [
            f"fem_pat_{side_letter}_on_patella_{side_letter}_in_patella_{side_letter}",
        ],
    )

    def resultant(base: Optional[str]) -> Optional[np.ndarray]:
        if base is None:
            return None
        fx = np.asarray(jrl_dict.get(f"{base}_fx", None))
        fy = np.asarray(jrl_dict.get(f"{base}_fy", None))
        fz = np.asarray(jrl_dict.get(f"{base}_fz", None))
        if fx is None or fy is None or fz is None:
            return None
        return np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) / bw

    med = resultant(med_base)
    lat = resultant(lat_base)
    pat = resultant(pat_base)
    total = None
    if med is not None and lat is not None:
        total = med + lat
    return med, lat, pat, total


def extract_knee_jrl_cycles(
    data_dict: Dict[str, Any],
    series_name: str,
    n_points: int,
    bw: float,
    preferred_side: str,
    single_side_vars: Iterable[str],
) -> Tuple[np.ndarray, List[int], List[Dict[str, Any]], List[str]]:
    records: List[np.ndarray] = []
    labels: List[int] = []
    meta: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for side, cycle_id, cycle in iter_cycles(data_dict):
        if series_name in single_side_vars and side != preferred_side:
            continue
        cluster = get_cluster_label(cycle)
        if cluster is None:
            continue
        jrl_dict = cycle.get("JRL", {})
        if not isinstance(jrl_dict, dict):
            skipped.append(f"{cycle_id}:JRL:missing")
            continue
        time = jrl_dict.get("time", None)
        if time is None:
            skipped.append(f"{cycle_id}:JRL:no_time")
            continue
        med, lat, pat, total = compute_knee_jrl_resultants(jrl_dict, side, bw)
        ratio = None
        if med is not None and lat is not None:
            denom = med + lat
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(denom == 0, np.nan, (med / denom) * 100.0)
        series_map = {
            "medial_resultant": med,
            "lateral_resultant": lat,
            "patellofemoral_resultant": pat,
            "total_tibfem_resultant": total,
            "mediototal_ratio_pc": ratio,
        }
        values = series_map.get(series_name, None)
        if values is None:
            skipped.append(f"{cycle_id}:JRL:{series_name}:missing")
            continue
        interp = _interp_series(time, values, n_points)
        if interp is None:
            skipped.append(f"{cycle_id}:JRL:{series_name}:interp_failed")
            continue
        records.append(interp)
        labels.append(cluster)
        meta.append({"side": side, "cycle_id": cycle_id})

    if records:
        return np.vstack(records), labels, meta, skipped
    return np.empty((0, n_points)), labels, meta, skipped


def filter_clusters(
    data: np.ndarray,
    labels: List[int],
    meta: List[Dict[str, Any]],
    min_cycles: int,
    include: Optional[Iterable[int]] = None,
    exclude: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, List[int], List[Dict[str, Any]]]:
    if data.size == 0:
        return data, labels, meta
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()

    indices = []
    for idx, lab in enumerate(labels):
        if include_set is not None and lab not in include_set:
            continue
        if lab in exclude_set:
            continue
        indices.append(idx)

    if not indices:
        return np.empty((0, data.shape[1])), [], []

    data = data[indices, :]
    labels = [labels[i] for i in indices]
    meta = [meta[i] for i in indices]

    counts = {lab: labels.count(lab) for lab in sorted(set(labels))}
    keep_labels = {lab for lab, cnt in counts.items() if cnt >= min_cycles}
    if not keep_labels:
        return np.empty((0, data.shape[1])), [], []
    filtered_indices = [i for i, lab in enumerate(labels) if lab in keep_labels]
    return data[filtered_indices, :], [labels[i] for i in filtered_indices], [meta[i] for i in filtered_indices]


def split_by_cluster(data: np.ndarray, labels: List[int]) -> Dict[int, np.ndarray]:
    out: Dict[int, List[np.ndarray]] = {}
    for row, lab in zip(data, labels):
        out.setdefault(lab, []).append(row)
    return {lab: np.vstack(rows) for lab, rows in out.items()}


def count_tests_for_clusters(n_clusters: int) -> int:
    if n_clusters <= 1:
        return 0
    if n_clusters == 2:
        return 1
    return 1 + len(list(combinations(range(n_clusters), 2)))
