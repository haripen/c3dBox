# -*- coding: utf-8 -*-
"""
Time-series clustering helpers updated for the npz-based knee JRL dataset.

Expected inputs:
- data: np.ndarray shaped (n_samples, n_timesteps, n_features)
- feature_names: list[str] with length n_features
- meta: optional pandas.DataFrame with one row per sample
"""

from __future__ import annotations

import itertools
import os
import textwrap
import time
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from skopt import gp_minimize
from skopt.space import Real
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score as dtw_silhouette_score
from tslearn.metrics import cdist_dtw, cdist_soft_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# ---------------------------------------------------------------------------
# Global defaults used by the clustering/optimization helpers
# ---------------------------------------------------------------------------
GAMMA_SPACE: Optional[Real] = None
RANDOM_STATE_GLOBAL: Optional[int] = None


def set_random_state_global(value: int) -> None:
    """Override the module-level random state used during optimization helpers."""
    global RANDOM_STATE_GLOBAL
    RANDOM_STATE_GLOBAL = int(value)


def set_gamma_space(low: float, high: float, prior: str = "log-uniform") -> None:
    """Override the module-level gamma search space for Bayesian optimization."""
    global GAMMA_SPACE
    GAMMA_SPACE = Real(low, high, prior=prior, name="gamma")


def _resolve_random_state(random_state: Optional[int]) -> int:
    """Return a concrete random state int using local override, global, or fallback."""
    if random_state is not None:
        return int(random_state)
    if RANDOM_STATE_GLOBAL is not None:
        return int(RANDOM_STATE_GLOBAL)
    return 42  # fallback default

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_feature_names(feature_names: Optional[Iterable[str]], n_features: int) -> List[str]:
    """Normalize feature names to a list of strings with fallback defaults."""
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    names = list(feature_names)
    if len(names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, received {len(names)}")
    return [str(n) for n in names]


def _prepare_data_array(data: np.ndarray, use_scaler: bool) -> Tuple[np.ndarray, Optional[TimeSeriesScalerMeanVariance]]:
    """
    Validate and optionally scale the time-series array.

    Returns the prepared array and the scaler (if applied).
    """
    if data.ndim != 3:
        raise ValueError(f"`data` must be 3D (n_samples, n_timesteps, n_features). Got shape {data.shape}.")
    prepared = np.ascontiguousarray(data, dtype=np.float64)
    scaler = None
    if use_scaler:
        scaler = TimeSeriesScalerMeanVariance()
        prepared = scaler.fit_transform(prepared)
    return prepared, scaler


def _build_time_axis(n_timesteps: int) -> np.ndarray:
    """Return a 0-100 % normalized time axis matching the number of timesteps."""
    return np.linspace(0, 100, n_timesteps)


def _plot_cluster_traces(
    clustering_data: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    time_axis: np.ndarray,
    output_folder: str,
    dm_method: str,
    gamma: float,
    descriptor: str,
) -> Optional[str]:
    """Plot per-cluster traces and means for a given clustering result."""
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    num_clusters_found = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

    sns.set_style("whitegrid")
    plot_palette_size = max(1, num_clusters_found)
    palette = sns.color_palette("tab10", plot_palette_size)

    num_plot_vars = len(feature_names)
    num_cols = 3
    num_rows = (num_plot_vars + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, squeeze=False)
    axes = axes.flatten()

    cluster_items = sorted(zip(unique_clusters, cluster_counts), key=lambda item: item[1], reverse=True)
    cluster_to_palette_idx = {label: idx for idx, (label, _count) in enumerate(item for item in cluster_items if item[0] != -1)}

    for i, feat_name in enumerate(feature_names):
        ax = axes[i]
        plot_legend_handles = {}
        for cluster_label, count in cluster_items:
            color = "gray" if cluster_label == -1 else palette[cluster_to_palette_idx.get(cluster_label, 0)]
            alpha = 0.05 if cluster_label == -1 else 0.1
            cluster_indices = np.where(cluster_labels == cluster_label)[0]
            if len(cluster_indices) == 0:
                continue
            traces = clustering_data[cluster_indices, :, i]
            for row in traces:
                ax.plot(time_axis, row, color=color, alpha=alpha, linestyle="-", linewidth=0.8)

            mean_values = traces.mean(axis=0)
            line, = ax.plot(
                time_axis,
                mean_values,
                color=color,
                linewidth=3,
                linestyle="-",
                label=f"{'Noise' if cluster_label == -1 else f'Cluster {cluster_label}'} Mean ({count} samples)",
                zorder=1000,
                path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()],
            )
            if cluster_label not in plot_legend_handles:
                plot_legend_handles[cluster_label] = line

        ax.set_title(f"Clustering: {feat_name}")
        ax.set_xlabel("Time / Cycle (%)")
        ax.set_ylabel(feat_name)

    for j in range(num_plot_vars, len(axes)):
        fig.delaxes(axes[j])

    handles = list(plot_legend_handles.values()) if "plot_legend_handles" in locals() else []
    labels = [h.get_label() for h in handles]
    if handles:
        handles_labels_sorted = sorted(
            zip(handles, labels),
            key=lambda x: int(x[1].split(" ")[1]) if "Cluster" in x[1] else -1,
        )
        handles_sorted, labels_sorted = zip(*handles_labels_sorted)
        fig.legend(handles_sorted, labels_sorted, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=min(len(labels), 4))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    clustering_fig_path = os.path.join(
        output_folder,
        f"clustering_{dm_method}_{descriptor}_gamma{gamma if dm_method=='softdtw' else 'na'}.pdf",
    )
    try:
        fig.savefig(clustering_fig_path, format="pdf", bbox_inches="tight")
        print(f"Clustering figure saved to '{clustering_fig_path}'")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR saving clustering figure: {e}")
        clustering_fig_path = None
    plt.close(fig)
    return clustering_fig_path


# ---------------------------------------------------------------------------
# Distance matrix helpers
# ---------------------------------------------------------------------------
def precompute_distance_matrix(data: np.ndarray, method: str = "softdtw", gamma: float = 1.0, n_jobs: int = -1) -> Tuple[Optional[np.ndarray], float]:
    """
    Precomputes the pairwise distance matrix using either DTW or soft-DTW.
    Adjusts soft-DTW output to be non-negative with a zero diagonal for metric compatibility.
    """
    n_samples = data.shape[0]
    print(f"\n--- Precomputing Distance Matrix ({method.upper()}) ---")
    print(f"Calculating pairwise distances for {n_samples} samples...")
    if method == "softdtw":
        print(f"  Using soft-DTW with gamma={gamma}")
    elif method == "dtw":
        print(f"  Using classic DTW (n_jobs={n_jobs})")
    else:
        print(f"ERROR: Unknown distance matrix method '{method}'. Use 'dtw' or 'softdtw'.")
        return None, np.nan

    start_time = time.time()
    distance_matrix = None
    computation_time = np.nan

    try:
        data_prepared = np.ascontiguousarray(data, dtype=np.float64)

        if method == "softdtw":
            distance_matrix = cdist_soft_dtw(data_prepared, gamma=gamma)
            min_val = np.min(distance_matrix)
            print(f"  Adjusting soft-DTW matrix: subtracting min value ({min_val:.4f}) to ensure non-negativity.")
            distance_matrix = distance_matrix - min_val
            print("  Adjusting soft-DTW matrix: setting diagonal to 0.")
            np.fill_diagonal(distance_matrix, 0)
        elif method == "dtw":
            distance_matrix = cdist_dtw(data_prepared, n_jobs=n_jobs)

        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Pairwise distances calculated and adjusted (if softDTW) in {computation_time:.2f} seconds.")

    except MemoryError:
        print("\nERROR: MemoryError calculating the full distance matrix!")
        distance_matrix = None
        computation_time = np.nan
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: An unexpected error occurred during distance calculation: {e}")
        distance_matrix = None
        computation_time = np.nan

    return distance_matrix, computation_time


def calculate_dtw_clustering_metrics(distance_matrix: Optional[np.ndarray], computation_time: float, cluster_labels: np.ndarray) -> Dict[str, object]:
    """
    Calculates internal clustering evaluation metrics suitable for time series,
    using a precomputed DTW or soft-DTW distance matrix.
    """
    print("\n--- Calculating Clustering Metrics from Precomputed Distances ---")

    if distance_matrix is None:
        print("ERROR: Distance matrix is None. Cannot calculate metrics.")
        try:
            distribution = dict(zip(*np.unique(cluster_labels, return_counts=True)))
        except Exception:  # noqa: BLE001
            distribution = {}
        return {
            "silhouette_dtw": np.nan,
            "davies_bouldin": np.nan,
            "cluster_distribution": distribution,
            "dtw_within": {},
            "dtw_between": {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan},
            "dtw_computation_time": computation_time if not np.isnan(computation_time) else 0.0,
        }

    n_samples = distance_matrix.shape[0]
    unique_labels = np.unique(cluster_labels)
    n_clusters_found = len(unique_labels)

    distribution = dict(zip(*np.unique(cluster_labels, return_counts=True)))
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of clusters found: {n_clusters_found}")
    print(f"  Cluster Distribution: {distribution}")
    print(f"  Distance Matrix Computation Time: {computation_time:.2f} seconds")

    if n_samples <= 1 or n_clusters_found <= 1:
        print("WARN: Cannot calculate Silhouette or Davies-Bouldin. Need > 1 sample and > 1 cluster.")
        return {
            "silhouette_dtw": np.nan,
            "davies_bouldin": np.nan,
            "cluster_distribution": distribution,
            "dtw_within": {},
            "dtw_between": {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan},
            "dtw_computation_time": computation_time,
        }

    print("Calculating DTW Silhouette Score...")
    silhouette_dtw = np.nan
    try:
        silhouette_dtw = dtw_silhouette_score(
            X=distance_matrix,
            labels=cluster_labels,
            metric="precomputed",
            verbose=0,
        )
        print(f"  DTW Silhouette Score: {silhouette_dtw:.4f}")
    except ValueError as ve:  # noqa: BLE001
        print(f"\nWARN: Could not calculate DTW Silhouette Score: {ve}")
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: An unexpected error occurred during Silhouette Score calculation: {e}")

    print("Calculating Davies-Bouldin Index (DTW-based)...")
    davies_bouldin = np.nan
    medoids: Dict[int, int] = {}
    intra_cluster_distances: Dict[int, float] = {}
    calculation_possible = True

    for cluster in unique_labels:
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            calculation_possible = False
            break
        if len(indices) == 1:
            medoids[cluster] = indices[0]
            intra_cluster_distances[cluster] = 0.0
            continue

        submatrix = distance_matrix[np.ix_(indices, indices)]
        avg_distances = np.mean(submatrix, axis=1)
        medoid_index_in_cluster = np.argmin(avg_distances)
        medoid_global_index = indices[medoid_index_in_cluster]
        medoids[cluster] = medoid_global_index
        intra_cluster_distances[cluster] = np.mean(distance_matrix[indices, medoid_global_index])

    if calculation_possible and len(unique_labels) > 1:
        try:
            r_list = []
            for i in unique_labels:
                if i not in medoids:
                    continue
                max_r_i = 0.0
                for j in unique_labels:
                    if i == j or j not in medoids:
                        continue
                    m_ij = distance_matrix[medoids[i], medoids[j]]
                    if m_ij == 0:
                        continue
                    s_i = intra_cluster_distances.get(i, 0.0)
                    s_j = intra_cluster_distances.get(j, 0.0)
                    r_ij = (s_i + s_j) / m_ij
                    if r_ij > max_r_i:
                        max_r_i = r_ij
                if max_r_i > 0:
                    r_list.append(max_r_i)

            if r_list:
                davies_bouldin = np.mean(r_list)
                print(f"  Davies-Bouldin (DTW): {davies_bouldin:.4f}")
            else:
                print("  WARN: Could not compute Davies-Bouldin index (no valid inter-cluster comparisons possible).")
                davies_bouldin = np.nan

        except Exception as e:  # noqa: BLE001
            print(f"\nERROR: Could not calculate Davies-Bouldin (DTW): {e}")
            davies_bouldin = np.nan
    elif len(unique_labels) <= 1:
        print("  Skipping Davies-Bouldin: Only one cluster found.")
        davies_bouldin = np.nan

    print("Calculating DTW distance statistics...")
    dtw_within: Dict[int, Dict[str, float]] = {}
    for cluster in unique_labels:
        indices = np.where(cluster_labels == cluster)[0]
        values = np.array([])
        if len(indices) > 1:
            submatrix = distance_matrix[np.ix_(indices, indices)]
            triu_idx = np.triu_indices_from(submatrix, k=1)
            if triu_idx[0].size > 0:
                values = submatrix[triu_idx]
        dtw_within[cluster] = {
            "mean": np.mean(values) if values.size > 0 else np.nan,
            "std": np.std(values) if values.size > 0 else np.nan,
            "min": np.min(values) if values.size > 0 else np.nan,
            "max": np.max(values) if values.size > 0 else np.nan,
        }
    if dtw_within:
        df_within = pd.DataFrame(dtw_within).T
        df_within.index.name = "Cluster"
        print(textwrap.indent(df_within.to_string(), "    "))
    else:
        print("    No within-cluster distances to report.")

    between_values: List[float] = []
    if len(unique_labels) > 1:
        for i, cluster_i in enumerate(unique_labels):
            indices_i = np.where(cluster_labels == cluster_i)[0]
            if len(indices_i) == 0:
                continue
            for cluster_j in unique_labels[i + 1 :]:
                indices_j = np.where(cluster_labels == cluster_j)[0]
                if len(indices_j) == 0:
                    continue
                submatrix = distance_matrix[np.ix_(indices_i, indices_j)]
                if submatrix.size > 0:
                    between_values.extend(submatrix.flatten())

    between_values_arr = np.array(between_values)
    dtw_between = {
        "mean": np.mean(between_values_arr) if between_values_arr.size > 0 else np.nan,
        "std": np.std(between_values_arr) if between_values_arr.size > 0 else np.nan,
        "min": np.min(between_values_arr) if between_values_arr.size > 0 else np.nan,
        "max": np.max(between_values_arr) if between_values_arr.size > 0 else np.nan,
    }
    if between_values_arr.size > 0:
        df_between = pd.DataFrame([dtw_between])
        print(textwrap.indent(df_between.to_string(index=False), "    "))
    else:
        print("    No between-cluster distances to report (need > 1 non-empty cluster).")

    print("-------------------------------------------------")

    results = {
        "silhouette_dtw": silhouette_dtw,
        "davies_bouldin": davies_bouldin,
        "cluster_distribution": distribution,
        "dtw_within": dtw_within,
        "dtw_between": dtw_between,
        "dtw_computation_time": computation_time,
    }
    return results


# ---------------------------------------------------------------------------
# Core clustering routine
# ---------------------------------------------------------------------------
def cluster_timeseries(
    data: np.ndarray,
    feature_names: Optional[Iterable[str]],
    output_folder: str,
    excel_path: Optional[str] = None,
    meta: Optional[pd.DataFrame] = None,
    method: str = "ts_kmeans",
    use_scaler: bool = True,
    grid_search: str = "off",
    ks: Optional[List[int]] = None,
    gammas: Optional[List[float]] = None,
    dm_method: str = "softdtw",
    n_clusters: int = 2,
    gamma: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    plot: bool = True,
    agglo_k_range: Iterable[int] = (2, 3, 4),
    optimization_mode: str = "grid",
    dendro_colors: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Perform time series clustering using DTW/soft-DTW with optional grid search.

    Returns a dict containing prepared data, labels (if applicable), metrics,
    and grid search results.

    Key parameters for agglo_ward:
    - dm_method: distance metric (softdtw or dtw)
    - gamma: soft-DTW smoothing parameter
    - agglo_k_range: list/tuple of k values to evaluate when cutting the dendrogram
    """
    os.makedirs(output_folder, exist_ok=True)

    ks = ks or [n_clusters]
    gammas = gammas or [gamma]
    feature_names_list = _ensure_feature_names(feature_names, data.shape[-1])
    time_axis = _build_time_axis(data.shape[1])

    random_state_val = _resolve_random_state(random_state)

    clustering_data, _ = _prepare_data_array(data, use_scaler=use_scaler)
    print(f"Clustering data prepared with shape: {clustering_data.shape}")  # (n_samples, n_timesteps, n_features)

    grid_results_df = pd.DataFrame()
    bayes_results_df = pd.DataFrame()
    bayes_best_gamma_per_k: Dict[int, float] = {}

    if grid_search.lower() == "on":
        print("\n=== Starting Grid Search ===")
        results_list: List[Dict[str, object]] = []
        param_grid = [(None, g) for g in gammas] if method == "agglo_ward" else list(itertools.product(ks, gammas if dm_method == "softdtw" else [None]))
        precomputed_matrices: Dict[object, Tuple[np.ndarray, float]] = {}

        for k_val, gamma_val in param_grid:
            print(f"\n--- Grid Search: k={k_val}, gamma={gamma_val if gamma_val is not None else 'N/A'} ---")
            current_dm_method = dm_method
            current_gamma = gamma_val if current_dm_method == "softdtw" else None
            matrix_key = current_gamma if current_dm_method == "softdtw" else "dtw"

            if matrix_key not in precomputed_matrices:
                print(f"Calculating distance matrix for gamma={current_gamma}" if current_dm_method == "softdtw" else "Calculating DTW distance matrix")
                precomputed_matrix, precomputation_time = precompute_distance_matrix(
                    clustering_data,
                    method=current_dm_method,
                    gamma=current_gamma if current_gamma is not None else 1.0,
                    n_jobs=n_jobs,
                )
                if precomputed_matrix is None:
                    print(f"WARN: Failed to compute distance matrix for {matrix_key}. Skipping this grid point.")
                    continue
                precomputed_matrices[matrix_key] = (precomputed_matrix, precomputation_time)
            else:
                print(f"Using precomputed distance matrix for {matrix_key}")
                precomputed_matrix, precomputation_time = precomputed_matrices[matrix_key]

            if method == "agglo_ward":
                dist_mat = precomputed_matrix
                condensed = squareform(dist_mat, checks=False)
                z = linkage(condensed, method="ward")

                for cut_k in agglo_k_range:
                    labels = fcluster(z, t=cut_k, criterion="maxclust")
                    metrics = calculate_dtw_clustering_metrics(
                        distance_matrix=dist_mat,
                        computation_time=precomputation_time,
                        cluster_labels=labels,
                    )
                    if plot:
                        threshold = z[-(cut_k - 1), 2] if cut_k > 1 else 0
                        gamma_fmt = f"{gamma_val:.4g}"
                        fig, ax = plt.subplots(figsize=(9, 5))
                        dendrogram(
                            z,
                            ax=ax,
                            no_labels=True,
                            distance_sort="descending",
                            show_leaf_counts=True,
                            above_threshold_color="gray",
                            color_threshold=threshold,
                        )
                        # Legend mapping colors to clusters with counts
                        cluster_labels = labels
                        unique_clusters = np.unique(cluster_labels)
                        counts = {c: (cluster_labels == c).sum() for c in unique_clusters}
                        handles = []
                        palette = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
                        for idx, c in enumerate(unique_clusters):
                            color = palette[idx % len(palette)]
                            handles.append(mpl.lines.Line2D([0], [0], color=color, lw=2, label=f"Cluster {c} (n={counts[c]})"))
                        if handles:
                            ax.legend(handles=handles, loc="upper right")
                        ax.set_title(f"Ward Dendrogram (gamma={gamma_fmt}, k={cut_k})", pad=10)
                        ax.set_ylabel("Linkage distance (Ward / dSSE)")
                        ax.set_xlabel("Samples / merged clusters (ordered)")
                        for spine in ("top", "right"):
                            ax.spines[spine].set_visible(False)
                        ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
                        plt.tight_layout()
                        dendro_path = os.path.join(
                            output_folder,
                            f"dendrogram_grid_softdtw_gamma{gamma_fmt}_k{cut_k}.pdf",
                        )
                        try:
                            fig.savefig(dendro_path, format="pdf", bbox_inches="tight")
                            print(f"Dendrogram saved to '{dendro_path}'")
                        except Exception as e:  # noqa: BLE001
                            print(f"ERROR saving dendrogram: {e}")
                        plt.close(fig)

                    result_row = {
                        "k": cut_k,
                        "gamma": gamma_val,
                        "dm_method": dm_method,
                        "silhouette_dtw": metrics["silhouette_dtw"],
                        "davies_bouldin": metrics["davies_bouldin"],
                        "n_clusters_found": cut_k,
                        "dtw_comp_time_s": metrics["dtw_computation_time"],
                    }
                    results_list.append(result_row)
            else:  # ts_kmeans
                model_params = {
                    "n_clusters": k_val,
                    "random_state": random_state,
                    "n_init": 3,
                    "verbose": 0,
                }
                if current_dm_method == "softdtw":
                    model_params["metric"] = "softdtw"
                    model_params["metric_params"] = {"gamma": current_gamma}
                else:
                    model_params["metric"] = "dtw"

                try:
                    model = TimeSeriesKMeans(**model_params)
                    cluster_labels = model.fit_predict(clustering_data)
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR during TimeSeriesKMeans fitting: {e}")
                    continue

                metrics = calculate_dtw_clustering_metrics(
                    distance_matrix=precomputed_matrix,
                    computation_time=precomputation_time,
                    cluster_labels=cluster_labels,
                )
                result_row = {
                    "k": k_val,
                    "gamma": current_gamma if current_dm_method == "softdtw" else np.nan,
                    "dm_method": current_dm_method,
                    "silhouette_dtw": metrics["silhouette_dtw"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "n_clusters_found": len(metrics.get("cluster_distribution", {})),
                    "cluster_distribution": str(metrics.get("cluster_distribution", {})),
                    "dtw_comp_time_s": metrics["dtw_computation_time"],
                }
                results_list.append(result_row)

        grid_results_df = pd.DataFrame(results_list)
        if not grid_results_df.empty:
            print("\n=== Grid Search Complete ===")
            print(grid_results_df.to_string())
            if excel_path:
                try:
                    with pd.ExcelWriter(excel_path, mode="w", engine="openpyxl") as writer:
                        grid_results_df.to_excel(writer, sheet_name="grid_search_results", index=False)
                    print(f"Grid search results saved to '{excel_path}' (sheet: grid_search_results)")
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR: Could not save grid search results to Excel: {e}")
        else:
            print("\n=== Grid Search Complete (no valid results) ===")

        results_payload = {
            "prepared_data": clustering_data,
            "grid_search_results": grid_results_df,
            "feature_names": feature_names_list,
            "bayes_results": bayes_results_df,
            "bayes_best_gamma_per_k": bayes_best_gamma_per_k,
        }

        # Optional Bayesian refinement using grid results as seeds (no per-step plotting)
        if optimization_mode == "bayes" and method == "agglo_ward" and not grid_results_df.empty:
            bayes_records = []
            bayes_trials = []
            for k_val in agglo_k_range:
                # run bayes optimization; capture full trial history if available
                best_g, _ = optimize_gamma_for_k(
                    k_value=k_val,
                    data=clustering_data,
                    initial_results_df=grid_results_df,
                    n_calls=15,
                    random_state=random_state_val,
                    method="agglo_ward",
                    agglo_k_range=(k_val,),
                )
                if best_g is None:
                    continue
                bayes_best_gamma_per_k[k_val] = best_g

                # evaluate metrics at best_g (no plots here)
                dist_opt, dist_time = precompute_distance_matrix(
                    clustering_data,
                    method=dm_method,
                    gamma=best_g,
                    n_jobs=n_jobs,
                )
                if dist_opt is None:
                    continue
                z_tmp = linkage(squareform(dist_opt, checks=False), method="ward")
                labels_tmp = fcluster(z_tmp, t=k_val, criterion="maxclust")
                metrics_tmp = calculate_dtw_clustering_metrics(
                    distance_matrix=dist_opt,
                    computation_time=dist_time,
                    cluster_labels=labels_tmp,
                )
                bayes_records.append(
                    {
                        "k": k_val,
                        "gamma_opt": best_g,
                        "silhouette_dtw": metrics_tmp.get("silhouette_dtw"),
                        "davies_bouldin": metrics_tmp.get("davies_bouldin"),
                        "dtw_comp_time_s": metrics_tmp.get("dtw_computation_time"),
                    }
                )
                # record this gamma as a tried combo (for bayes_opt_tried sheet)
                bayes_trials.append(
                    {
                        "k": k_val,
                        "gamma": best_g,
                        "dm_method": dm_method,
                        "silhouette_dtw": metrics_tmp.get("silhouette_dtw"),
                        "davies_bouldin": metrics_tmp.get("davies_bouldin"),
                        "n_clusters_found": k_val,
                        "dtw_comp_time_s": metrics_tmp.get("dtw_computation_time"),
                    }
                )
            bayes_results_df = pd.DataFrame(bayes_records)
            results_payload["bayes_results"] = bayes_results_df
            results_payload["bayes_best_gamma_per_k"] = bayes_best_gamma_per_k
            if not bayes_results_df.empty:
                print("\n=== BAYES OPTIMIZATION COMPLETE ===")
                print(bayes_results_df.sort_values("k").to_string(index=False))
                if excel_path:
                    try:
                        with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                            bayes_results_df.sort_values("k").to_excel(writer, sheet_name="bayes_results", index=False)
                        print(f"Bayes results saved to sheet 'bayes_results' in '{excel_path}'")
                    except Exception as e:  # noqa: BLE001
                        print(f"Could not save bayes_results to Excel: {e}")
            bayes_trials_df = pd.DataFrame(bayes_trials)
            results_payload["bayes_trials"] = bayes_trials_df
            if excel_path and not bayes_trials_df.empty:
                try:
                    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                        bayes_trials_df.sort_values(["k", "gamma"]).to_excel(writer, sheet_name="bayes_opt_tried", index=False)
                    print(f"Bayes trial table saved to sheet 'bayes_opt_tried' in '{excel_path}'")
                except Exception as e:
                    print(f"Could not save bayes_opt_tried to Excel: {e}")

        return results_payload

    print(f"\n=== Running Single Clustering (method={method}, distance={dm_method}, gamma={gamma if dm_method=='softdtw' else 'N/A'}, random_state={random_state_val}) ===")
    labels_by_k: Dict[int, np.ndarray] = {}
    hier_df = None
    best_k = n_clusters
    cluster_plot_paths: List[Dict[str, object]] = []

    if method == "ts_kmeans":
        precomputed_matrix, precomputation_time = precompute_distance_matrix(
            clustering_data,
            method=dm_method,
            gamma=gamma,
            n_jobs=n_jobs,
        )
        if precomputed_matrix is None:
            raise RuntimeError("Failed to compute distance matrix. Aborting clustering.")
        print(f"Running TimeSeriesKMeans (k={n_clusters}, metric={dm_method}, gamma={gamma if dm_method=='softdtw' else 'N/A'})...")
        model_params = {
            "n_clusters": n_clusters,
            "random_state": random_state_val,
            "n_init": 5,
            "verbose": 0,
        }
        if dm_method == "softdtw":
            model_params["metric"] = "softdtw"
            model_params["metric_params"] = {"gamma": gamma}
        else:
            model_params["metric"] = "dtw"
        model = TimeSeriesKMeans(**model_params)
        cluster_labels = model.fit_predict(clustering_data)
    elif method == "agglo_ward":
        precomputed_matrix, precomputation_time = precompute_distance_matrix(
            clustering_data,
            method=dm_method,
            gamma=gamma,
            n_jobs=n_jobs,
        )
        if precomputed_matrix is None:
            raise RuntimeError("Failed to compute distance matrix. Aborting clustering.")
        condensed = squareform(precomputed_matrix, checks=False)
        z = linkage(condensed, method="ward")
        hier_results = []
        labels_by_k: Dict[int, np.ndarray] = {}
        for k_cut in agglo_k_range:
            labels = fcluster(z, t=k_cut, criterion="maxclust")
            labels_by_k[k_cut] = labels
            metrics_tmp = calculate_dtw_clustering_metrics(
                distance_matrix=precomputed_matrix,
                computation_time=precomputation_time,
                cluster_labels=labels,
            )
            hier_results.append(
                {
                    "k": k_cut,
                    "silhouette_dtw": metrics_tmp["silhouette_dtw"],
                    "davies_bouldin": metrics_tmp["davies_bouldin"],
                    "distribution": metrics_tmp["cluster_distribution"],
                }
            )
            if plot:
                path = _plot_cluster_traces(
                    clustering_data=clustering_data,
                    cluster_labels=labels,
                    feature_names=feature_names_list,
                    time_axis=time_axis,
                    output_folder=output_folder,
                    dm_method=dm_method,
                    gamma=gamma,
                    descriptor=f"k{k_cut}",
                )
                if path:
                    cluster_plot_paths.append({"k": k_cut, "path": path})
        hier_df = pd.DataFrame(hier_results)
        best = max(hier_results, key=lambda r: (r["silhouette_dtw"] or -999, -r["k"]))
        best_k = best["k"]
        print(f"-> Recommended cluster count (highest silhouette): k={best_k}")
        cluster_labels = labels_by_k[best_k]
        # Plot final dendrogram with cut line and filename including gamma and k
        if plot:
            threshold = z[-(best_k - 1), 2] if best_k > 1 else 0
            gamma_fmt = f"{gamma:.4g}" if dm_method == "softdtw" else "na"
            rc_ctx = mpl.rc_context({"axes.prop_cycle": mpl.cycler(color=dendro_colors)}) if dendro_colors else mpl.rc_context()
            with rc_ctx:
                fig, ax = plt.subplots(figsize=(9, 5))
                dendrogram(
                    z,
                    ax=ax,
                    no_labels=True,
                    distance_sort="descending",
                    show_leaf_counts=True,
                    above_threshold_color="gray",
                    color_threshold=threshold,
                )
                # Legend mapping colors to clusters with counts
                unique_clusters = np.unique(cluster_labels)
                counts = {c: (cluster_labels == c).sum() for c in unique_clusters}
                handles = []
                palette = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
                for idx, c in enumerate(unique_clusters):
                    color = palette[idx % len(palette)]
                    handles.append(mpl.lines.Line2D([0], [0], color=color, lw=2, label=f"Cluster {c} (n={counts[c]})"))
                if handles:
                    ax.legend(handles=handles, loc="upper right")
                ax.set_title(f"Ward Linkage Dendrogram (gamma={gamma_fmt}, k={best_k})", pad=10)
                ax.set_ylabel("Linkage distance (Ward / dSSE)")
                ax.set_xlabel("Samples / merged clusters (ordered)")
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
                plt.tight_layout()
                dendro_path = os.path.join(output_folder, f"dendrogram_{dm_method}_gamma{gamma_fmt}_k{best_k}.pdf")
                try:
                    fig.savefig(dendro_path, format="pdf", bbox_inches="tight")
                    print(f"Dendrogram saved to '{dendro_path}'")
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR saving dendrogram: {e}")
                plt.close(fig)
        if excel_path:
            with pd.ExcelWriter(excel_path, mode="w", engine="openpyxl") as writer:
                hier_df.to_excel(writer, sheet_name="hierarchy_metrics", index=False)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    metrics = calculate_dtw_clustering_metrics(
        distance_matrix=precomputed_matrix,
        computation_time=precomputation_time,
        cluster_labels=cluster_labels,
    )

    meta_with_clusters = None
    if meta is not None:
        meta_with_clusters = meta.copy()
        meta_with_clusters["cluster"] = cluster_labels

    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    num_clusters_found = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    print(f"Identified {num_clusters_found} clusters (excluding noise if any). Distribution: {metrics.get('cluster_distribution', {})}")

    if plot:
        descriptor_k = best_k if method == "agglo_ward" else n_clusters
        path = _plot_cluster_traces(
            clustering_data=clustering_data,
            cluster_labels=cluster_labels,
            feature_names=feature_names_list,
            time_axis=time_axis,
            output_folder=output_folder,
            dm_method=dm_method,
            gamma=gamma,
            descriptor=f"k{descriptor_k}",
        )
        if path:
            cluster_plot_paths.append({"k": descriptor_k, "path": path})

    if excel_path:
        print(f"Saving results summary to '{excel_path}'...")
        summary_df = pd.DataFrame(
            {
                "Method": [method],
                "Scaler": ["TimeSeriesScalerMeanVariance" if use_scaler else "None"],
                "Distance_Metric": [dm_method],
                "Gamma_SoftDTW": [gamma if dm_method == "softdtw" else np.nan],
                "Num_Clusters_Requested": [n_clusters],
                "Num_Clusters_Found": [num_clusters_found],
                "Silhouette_DTW": [metrics["silhouette_dtw"]],
                "Davies_Bouldin_DTW": [metrics["davies_bouldin"]],
                "DTW_Distance_Time_s": [metrics["dtw_computation_time"]],
                "Random_State": [random_state_val],
            }
        )

        dist_data = [{"Cluster": k, "Sample_Count": v} for k, v in metrics.get("cluster_distribution", {}).items()]
        distribution_df = pd.DataFrame(dist_data).sort_values(by="Cluster").reset_index(drop=True)

        within_data = []
        dtw_within = metrics.get("dtw_within", {})
        if isinstance(dtw_within, dict) and dtw_within:
            for cluster, stats in dtw_within.items():
                row = {"Cluster": cluster}
                row.update(stats)
                within_data.append(row)
            dtw_within_df = pd.DataFrame(within_data).sort_values(by="Cluster").reset_index(drop=True)
            dtw_within_df.columns = ["Cluster", "DTW_Within_Mean", "DTW_Within_Std", "DTW_Within_Min", "DTW_Within_Max"]
        else:
            dtw_within_df = pd.DataFrame(columns=["Cluster", "DTW_Within_Mean", "DTW_Within_Std", "DTW_Within_Min", "DTW_Within_Max"])

        dtw_between = metrics.get("dtw_between", {})
        if isinstance(dtw_between, dict) and not all(np.isnan(v) for v in dtw_between.values()):
            dtw_between_df = pd.DataFrame([dtw_between])
            dtw_between_df.columns = ["DTW_Between_Mean", "DTW_Between_Std", "DTW_Between_Min", "DTW_Between_Max"]
        else:
            dtw_between_df = pd.DataFrame(columns=["DTW_Between_Mean", "DTW_Between_Std", "DTW_Between_Min", "DTW_Between_Max"])

        try:
            with pd.ExcelWriter(excel_path, mode="w", engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="clustering_summary", index=False)
                distribution_df.to_excel(writer, sheet_name="cluster_distribution", index=False)
                dtw_within_df.to_excel(writer, sheet_name="dtw_within_distribution", index=False)
                dtw_between_df.to_excel(writer, sheet_name="dtw_between_distribution", index=False)
                if meta_with_clusters is not None:
                    meta_with_clusters.to_excel(writer, sheet_name="meta_with_clusters", index=False)
            print(f"Clustering information written to '{excel_path}'.")
        except Exception as e:  # noqa: BLE001
            print(f"\nERROR: Could not save results to Excel: {e}")

    return {
        "prepared_data": clustering_data,
        "cluster_labels": cluster_labels,
        "labels_by_k": labels_by_k if method == "agglo_ward" else {},
        "best_k": best_k if method == "agglo_ward" else n_clusters,
        "hierarchy_metrics": hier_df if method == "agglo_ward" else None,
        "metrics": metrics,
        "feature_names": feature_names_list,
        "meta": meta_with_clusters,
        "cluster_plots": cluster_plot_paths,
        "distance_matrix": precomputed_matrix,
    }


# ---------------------------------------------------------------------------
# Optimization helpers (Bayesian search)
# ---------------------------------------------------------------------------
def objective_function_silhouette(gamma: float, k_value: int, data: np.ndarray, random_state: Optional[int] = None) -> float:
    """
    Objective function for Bayesian Optimization.
    Calculates -Silhouette score for a given gamma and fixed k.
    NOTE: Assumes 'data' is already scaled and prepared!
    """
    print(f"  Optimizing: Evaluating gamma={gamma:.4f} for k={k_value}...")
    start_time = time.time()

    dm_method = "softdtw"
    precomputed_matrix, precomputation_time = precompute_distance_matrix(
        data,
        method=dm_method,
        gamma=gamma,
    )

    if precomputed_matrix is None:
        return 10.0

    model_params = {
        "n_clusters": k_value,
        "random_state": _resolve_random_state(random_state),
        "metric": dm_method,
        "metric_params": {"gamma": gamma},
        "n_init": 3,
        "verbose": 0,
    }
    model = TimeSeriesKMeans(**model_params)
    try:
        cluster_labels = model.fit_predict(data)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: TimeSeriesKMeans failed for gamma={gamma:.4f}, k={k_value}: {e}. Returning worst score.")
        return 10.0

    metrics = calculate_dtw_clustering_metrics(
        distance_matrix=precomputed_matrix,
        computation_time=precomputation_time,
        cluster_labels=cluster_labels,
    )
    silhouette = metrics.get("silhouette_dtw", np.nan)
    eval_time = time.time() - start_time
    print(f"  Optimizing: gamma={gamma:.4f}, k={k_value} -> Silhouette={silhouette:.4f} (Eval time: {eval_time:.2f}s)")

    if np.isnan(silhouette):
        return 10.0
    return -silhouette


def objective_function_hac(gamma: float, data: np.ndarray, agglo_k_range: Iterable[int] = (2, 3, 4)) -> float:
    """
    Bayesian objective for agglo_ward: returns -best silhouette over the provided k cuts.
    """
    dist_mat, _ = precompute_distance_matrix(
        data,
        method="softdtw",
        gamma=gamma,
        n_jobs=-1,
    )
    if dist_mat is None:
        return 10.0

    z = linkage(squareform(dist_mat, checks=False), method="ward")
    best_sil = -np.inf
    for k in agglo_k_range:
        labels = fcluster(z, t=k, criterion="maxclust")
        sil = dtw_silhouette_score(
            X=dist_mat,
            labels=labels,
            metric="precomputed",
        )
        if sil > best_sil:
            best_sil = sil
    return -best_sil if not np.isnan(best_sil) else 10.0


def optimize_gamma_for_k(
    k_value: int,
    data: np.ndarray,
    initial_results_df: pd.DataFrame,
    n_calls: int = 20,
    random_state: Optional[int] = None,
    method: str = "ts_kmeans",
    agglo_k_range: Iterable[int] = (2, 3, 4),
) -> Tuple[Optional[float], Optional[float]]:
    """
    Uses Bayesian Optimization to find the best gamma for a specific k.
    """
    global RANDOM_STATE_GLOBAL
    RANDOM_STATE_GLOBAL = _resolve_random_state(random_state)

    print(f"\n--- Optimizing Gamma for k = {k_value} ---")

    initial_k_df = initial_results_df[
        (initial_results_df["k"] == k_value)
        & (initial_results_df["dm_method"] == "softdtw")
        & (initial_results_df["silhouette_dtw"].notna())
    ].copy()

    if initial_k_df.empty:
        print(f"WARN: No valid initial grid search results found for k={k_value} with softDTW. Skipping optimization.")
        return None, None

    x0 = [[g] for g in initial_k_df["gamma"].tolist()]
    y0 = (-initial_k_df["silhouette_dtw"]).tolist()

    print(f"Starting Bayesian Optimization with {len(x0)} initial points and {n_calls} total calls.")
    print(f"Initial gammas (x0): {x0}")
    print(f"Initial objectives (-SIL): {y0}")

    if method == "ts_kmeans":
        objective_for_this_k = lambda gamma_list: objective_function_silhouette(
            gamma=gamma_list[0],
            k_value=k_value,
            data=data,
            random_state=_resolve_random_state(random_state),
        )
    else:
        objective_for_this_k = lambda gamma_list: objective_function_hac(
            gamma=gamma_list[0],
            data=data,
            agglo_k_range=agglo_k_range,
        )

    try:
        gamma_space = GAMMA_SPACE or Real(5e-5, 1e1, prior="log-uniform", name="gamma")
        result = gp_minimize(
            func=objective_for_this_k,
            dimensions=[gamma_space],
            x0=x0,
            y0=y0,
            n_calls=n_calls,
            random_state=_resolve_random_state(random_state),
            noise=1e-10,
        )

        best_gamma = result.x[0]
        best_neg_silhouette = result.fun
        best_silhouette = -best_neg_silhouette

        print(f"--- Optimization Complete for k = {k_value} ---")
        print(f"  Best Gamma found: {best_gamma:.4f}")
        print(f"  Best Silhouette score found: {best_silhouette:.4f}")

        return best_gamma, best_silhouette

    except Exception as e:  # noqa: BLE001
        print(f"ERROR during Bayesian Optimization for k={k_value}: {e}")
        return None, None
