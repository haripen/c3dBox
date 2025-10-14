# c3dBox/Step4_check/time_norm.py
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline


def resample_to_101(x: ArrayLike, t: Optional[ArrayLike] = None) -> np.ndarray:
    """
    Resample a 1D or 2D time series to 101 points along its first axis using cubic splines.

    Parameters
    ----------
    x : array-like
        Input data with shape (N,) or (N, D). For biomech arrays like joint angles the
        expected shape is (N, 3) with N samples and 3 coordinates in columns.
    t : array-like, optional
        Sample times of length N. If None, times are assumed to be uniformly spaced
        from 0 to 1.

    Returns
    -------
    y101 : np.ndarray
        Resampled array with shape (101,) for 1D input or (101, D) for 2D input.

    Notes
    -----
    - Uses cubic splines when there are at least 4 valid (finite) samples; otherwise
      falls back to linear interpolation for that column.
    - Handles NaNs/inf per column by fitting only on finite samples; if fewer than 2
      finite points are available, returns all-NaN for that column.
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr2d = arr[:, None]
        squeeze = True
    elif arr.ndim == 2:
        arr2d = arr
        squeeze = False
    else:
        raise ValueError(f"x must be 1D or 2D (got shape {arr.shape}).")

    n = arr2d.shape[0]
    if n < 2:
        # Nothing meaningful to resample
        return np.repeat(arr2d, 101, axis=0) if not (n == 1 and squeeze) else np.repeat(arr2d[:, 0], 101)

    # Prepare time vectors
    if t is None:
        t_vec = np.linspace(0.0, 1.0, n, dtype=float)
    else:
        t_vec = np.asarray(t, dtype=float)
        if t_vec.shape[0] != n:
            raise ValueError(f"t must have length {n} (got {t_vec.shape[0]}).")

        # Enforce increasing time order (and reorder data accordingly)
        sort_idx = np.argsort(t_vec)
        if not np.all(sort_idx == np.arange(n)):
            t_vec = t_vec[sort_idx]
            arr2d = arr2d[sort_idx, :]

        # Deduplicate times if necessary (averaging duplicates per column)
        unique_t, inverse, counts = np.unique(t_vec, return_inverse=True, return_counts=True)
        if unique_t.size != t_vec.size:
            # Average values at identical time stamps per column
            tmp = np.zeros((unique_t.size, arr2d.shape[1]), dtype=float)
            for j in range(arr2d.shape[1]):
                sums = np.bincount(inverse, weights=np.nan_to_num(arr2d[:, j], nan=0.0))
                valid_counts = np.bincount(inverse, weights=np.isfinite(arr2d[:, j]).astype(float))
                # Avoid divide-by-zero: where no finite data, set NaN
                with np.errstate(invalid="ignore", divide="ignore"):
                    tmp[:, j] = sums / np.where(valid_counts == 0, np.nan, valid_counts)
            arr2d = tmp
            t_vec = unique_t

    t_new = np.linspace(t_vec[0], t_vec[-1], 101, dtype=float)
    y_new = np.full((101, arr2d.shape[1]), np.nan, dtype=float)

    for j in range(arr2d.shape[1]):
        y = arr2d[:, j]
        finite_mask = np.isfinite(y) & np.isfinite(t_vec)
        tj = t_vec[finite_mask]
        yj = y[finite_mask]

        if tj.size < 2:
            # Not enough info -> all NaN
            continue

        # Choose interpolation method
        try:
            if tj.size >= 4:
                cs = CubicSpline(tj, yj, bc_type="natural")
                y_new[:, j] = cs(t_new)
            else:
                # Fallback to linear if too few points for cubic
                y_new[:, j] = np.interp(t_new, tj, yj)
        except Exception:
            # Robust fallback in case spline fails (e.g., still non-monotonic)
            y_new[:, j] = np.interp(t_new, tj, yj)

    return y_new[:, 0] if squeeze else y_new


# -------------------------- Unit tests -------------------------------------- #
def _max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmax(np.abs(a - b)))


if __name__ == "__main__":
    # 1) 1D sinusoid, explicit time
    N = 187
    t = np.linspace(0, 2 * np.pi, N)
    sig = np.sin(t)
    sig101 = resample_to_101(sig, t)
    assert sig101.shape == (101,), "1D output must have shape (101,)"
    t101 = np.linspace(0, 2 * np.pi, 101)
    ref = np.sin(t101)
    err = _max_abs_err(sig101, ref)
    print(f"Test 1 (1D sine) max abs err = {err:.3e}")
    assert err < 2e-3

    # 2) 2D with three columns (N,3): sin, cos, constant
    X = np.column_stack([np.sin(t), np.cos(t), np.ones_like(t)])
    X101 = resample_to_101(X, t)
    assert X101.shape == (101, 3), "2D output must preserve second dimension."
    ref2 = np.column_stack([np.sin(t101), np.cos(t101), np.ones_like(t101)])
    err2 = _max_abs_err(X101, ref2)
    print(f"Test 2 ((N,3) sine/cos/const) max abs err = {err2:.3e}")
    assert err2 < 2e-3

    # 3) 1D sinusoid without providing time (assumes uniform)
    N3 = 180
    sig3 = np.sin(np.linspace(0, 2 * np.pi, N3))
    sig3_101 = resample_to_101(sig3)  # t=None (uniform)
    assert sig3_101.shape == (101,), "Uniform-time 1D should output (101,)."
    ref3 = np.sin(np.linspace(0, 2 * np.pi, 101))
    err3 = _max_abs_err(sig3_101, ref3)
    print(f"Test 3 (1D sine, t=None) max abs err = {err3:.3e}")
    assert err3 < 2e-3

    # 4) Shape preservation on wider 2D input
    N4, D4 = 200, 10
    # Use smooth columns to avoid spline overshoot complaints
    t4 = np.linspace(0, 1, N4)
    X4 = np.column_stack([np.sin(2 * np.pi * (k + 1) * t4) for k in range(D4)])
    X4_101 = resample_to_101(X4, t4)
    assert X4_101.shape == (101, D4), "Output second dimension must be preserved."

    print("All tests passed ✅")
