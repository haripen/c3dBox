from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import spm1d
except Exception:  # pragma: no cover - environment dependent
    spm1d = None

try:
    import power1d
except Exception:  # pragma: no cover - environment dependent
    power1d = None


def interval_indices(start: float, end: float, n_points: int) -> np.ndarray:
    time = np.linspace(0, 100, n_points)
    start_idx = int(np.searchsorted(time, start, side="left"))
    end_idx = int(np.searchsorted(time, end, side="right") - 1)
    start_idx = max(0, min(start_idx, n_points - 1))
    end_idx = max(0, min(end_idx, n_points - 1))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    return np.arange(start_idx, end_idx + 1)


def interval_indices_min(start: float, end: float, n_points: int, min_points: int = 3) -> np.ndarray:
    idx = interval_indices(start, end, n_points)
    if idx.size >= min_points:
        return idx
    center = int(round((idx[0] + idx[-1]) / 2.0)) if idx.size else int(round((n_points - 1) / 2.0))
    half = min_points // 2
    start_idx = max(0, center - half)
    end_idx = start_idx + min_points - 1
    if end_idx >= n_points:
        end_idx = n_points - 1
        start_idx = max(0, end_idx - min_points + 1)
    return np.arange(start_idx, end_idx + 1)


def endpoints_to_percent(start: float, end: float, n_points: int) -> Tuple[float, float]:
    scale = 100.0 / float(n_points - 1)
    return float(start) * scale, float(end) * scale


def spm_ttest2_inference(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float,
    two_tailed: bool = True,
):
    if spm1d is None:
        raise ImportError("spm1d is required for SPM inference.")
    return spm1d.stats.ttest2(data_a, data_b).inference(alpha=alpha, two_tailed=two_tailed)


def spm_ttest_paired_inference(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float,
    two_tailed: bool = True,
):
    if spm1d is None:
        raise ImportError("spm1d is required for SPM inference.")
    if hasattr(spm1d.stats, "ttest_paired"):
        ttest = spm1d.stats.ttest_paired(data_a, data_b)
    else:
        ttest = spm1d.stats.ttest(data_a - data_b)
    return ttest.inference(alpha=alpha, two_tailed=two_tailed)


def extract_spm_intervals(
    inference,
    alpha_corr: float,
    n_points: int,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    for cluster in inference.clusters:
        if cluster.P > alpha_corr:
            continue
        start, end = endpoints_to_percent(cluster.endpoints[0], cluster.endpoints[1], n_points)
        intervals.append((start, end))
    return intervals


def compute_power1d_2sample(
    data_a: np.ndarray,
    data_b: np.ndarray,
    start: float,
    end: float,
    n_points: int,
    alpha_corr: float,
    iterations: int,
    progress_bar: bool,
    min_points: int = 3,
) -> Optional[Dict[str, float]]:
    if power1d is None:
        return None
    idx = interval_indices_min(start, end, n_points, min_points=min_points)
    if idx.size < 1:
        return None

    values_a = data_a[:, idx]
    values_b = data_b[:, idx]
    if values_a.shape[0] < 3 or values_b.shape[0] < 3:
        return None

    mean_a = values_a.mean(axis=0)
    mean_b = values_b.mean(axis=0)
    ja, jb = values_a.shape[0], values_b.shape[0]
    pooled_mean = (ja * mean_a + jb * mean_b) / float(ja + jb)

    baseline = power1d.geom.Continuum1D(pooled_mean)
    signal0 = power1d.geom.Null(Q=idx.size)
    signal_a = power1d.geom.Continuum1D(mean_a - pooled_mean)
    signal_b = power1d.geom.Continuum1D(mean_b - pooled_mean)

    residuals = np.vstack([values_a - mean_a, values_b - mean_b])
    noise = power1d.noise.from_residuals(residuals)

    model_a0 = power1d.models.DataSample(baseline, signal0, noise, J=ja)
    model_b0 = power1d.models.DataSample(baseline, signal0, noise, J=jb)
    model_a1 = power1d.models.DataSample(baseline, signal_a, noise, J=ja)
    model_b1 = power1d.models.DataSample(baseline, signal_b, noise, J=jb)

    teststat = power1d.stats.t_2sample_fn(ja, jb)
    exp0 = power1d.models.Experiment([model_a0, model_b0], teststat)
    exp1 = power1d.models.Experiment([model_a1, model_b1], teststat)
    sim = power1d.models.ExperimentSimulator(exp0, exp1)
    results = sim.simulate(iterations, progress_bar=progress_bar, two_tailed=True)
    results.set_alpha(float(alpha_corr))

    return {
        "power1d_power": float(results.p_reject1),
        "power1d_iterations": int(iterations),
    }


def compute_power1d_1sample(
    window: np.ndarray,
    start: float,
    end: float,
    n_points: int,
    alpha_corr: float,
    iterations: int,
    progress_bar: bool,
    min_points: int = 3,
) -> Optional[Dict[str, float]]:
    if power1d is None:
        return None
    idx = interval_indices_min(start, end, n_points, min_points=min_points)
    if idx.size < 1:
        return None
    if window.shape[0] < 3:
        return None

    window = window[:, idx]
    mean_wave = window.mean(axis=0)

    baseline = power1d.geom.Null(Q=idx.size)
    signal0 = power1d.geom.Null(Q=idx.size)
    signal1 = power1d.geom.Continuum1D(mean_wave)
    residuals = window - mean_wave
    noise = power1d.noise.from_residuals(residuals)

    model0 = power1d.models.DataSample(baseline, signal0, noise, J=window.shape[0])
    model1 = power1d.models.DataSample(baseline, signal1, noise, J=window.shape[0])
    teststat = power1d.stats.t_1sample_fn(window.shape[0])
    exp0 = power1d.models.Experiment([model0], teststat)
    exp1 = power1d.models.Experiment([model1], teststat)
    sim = power1d.models.ExperimentSimulator(exp0, exp1)
    results = sim.simulate(iterations, progress_bar=progress_bar, two_tailed=True)
    results.set_alpha(float(alpha_corr))

    return {
        "power1d_power": float(results.p_reject1),
        "power1d_iterations": int(iterations),
    }


def build_power_intervals(
    inference,
    alpha_corr: float,
    n_points: int,
    power_fn: Callable[[float, float], Optional[Dict[str, float]]],
) -> List[Tuple[float, float, Optional[float]]]:
    intervals: List[Tuple[float, float, Optional[float]]] = []
    for cluster in inference.clusters:
        if cluster.P > alpha_corr:
            continue
        start, end = endpoints_to_percent(cluster.endpoints[0], cluster.endpoints[1], n_points)
        power_stats = power_fn(start, end)
        power_value = power_stats["power1d_power"] if power_stats else None
        intervals.append((start, end, power_value))
    return intervals


def annotate_spm_power(
    ax,
    intervals: List[Tuple[float, float]],
    power_intervals: Optional[List[Tuple[float, float, Optional[float]]]] = None,
    p_color: str = "gray",
    p_alpha: float = 0.25,
    p_ymin: float = 0.02,
    p_ymax: float = 0.09,
    power_color: str = "black",
    power_alpha: float = 0.5,
    power_lw: float = 2.0,
    power_y_start: float = 0.04,
    power_y_step: float = 0.025,
    power_thresholds: Tuple[float, float] = (0.8, 0.95),
) -> None:
    for start, end in intervals:
        ax.axvspan(start, end, color=p_color, alpha=p_alpha, ymin=p_ymin, ymax=p_ymax)
    if power_intervals:
        for start, end, power in power_intervals:
            if power is None:
                continue
            y = power_y_start
            if power >= power_thresholds[0]:
                ax.hlines(
                    y,
                    start,
                    end,
                    color=power_color,
                    alpha=power_alpha,
                    lw=power_lw,
                    transform=ax.get_xaxis_transform(),
                )
                y += power_y_step
            if power >= power_thresholds[1]:
                ax.hlines(
                    y,
                    start,
                    end,
                    color=power_color,
                    alpha=power_alpha,
                    lw=power_lw,
                    transform=ax.get_xaxis_transform(),
                )


def add_power_p_legend(ax, text: str = "p (gray) | power >=0.8 / >=0.95 (black)") -> None:
    ax.text(
        0.99,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="black",
    )


def spm_power_selftest_two_sided(
    alpha: float = 0.05,
    n_samples: int = 50,
    n_points: int = 101,
    noise_sigma: float = 0.05,
    seed: int = 0,
    iterations: int = 2000,
    progress_bar: bool = False,
) -> Dict[str, object]:
    if spm1d is None:
        raise ImportError("spm1d is required for the SPM/power self-test.")
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, n_points)
    a = np.tile(base, (n_samples, 1)) + rng.normal(0.0, noise_sigma, size=(n_samples, n_points))
    b = np.tile(base[::-1], (n_samples, 1)) + rng.normal(0.0, noise_sigma, size=(n_samples, n_points))
    inference = spm_ttest2_inference(a, b, alpha=alpha, two_tailed=True)
    intervals = extract_spm_intervals(inference, alpha, n_points)
    has_low = any(end < 50.0 for _, end in intervals)
    has_high = any(start > 50.0 for start, _ in intervals)

    def power_fn(start: float, end: float) -> Optional[Dict[str, float]]:
        return compute_power1d_2sample(
            a,
            b,
            start,
            end,
            n_points,
            alpha,
            iterations,
            progress_bar,
        )

    power_intervals = build_power_intervals(inference, alpha, n_points, power_fn)
    return {
        "intervals": intervals,
        "power_intervals": power_intervals,
        "has_low": has_low,
        "has_high": has_high,
    }
