def precompute_distance_matrix(data, method='softdtw', gamma=1.0, n_jobs=-1):
    """
    Precomputes the pairwise distance matrix using either DTW or soft-DTW.
    Adjusts soft-DTW output to be non-negative with a zero diagonal for metric compatibility.

    Args:
        data (np.ndarray): The time series dataset.
                           Shape: (n_samples, n_timesteps, n_features)
                           Should be C-contiguous and float64 for optimal performance.
        method (str): The distance metric to use ('dtw' or 'softdtw'). Default: 'softdtw'.
        gamma (float): The gamma parameter for soft-DTW. Ignored if method is 'dtw'. Default: 1.0.
        n_jobs (int): Number of CPU cores to use for parallel computation (for 'dtw'). Default: -1 (use all).

    Returns:
        tuple: A tuple containing:
            - distance_matrix (np.ndarray or None): The computed pairwise distance matrix
              (n_samples, n_samples), or None if an error occurred.
            - computation_time (float or np.nan): The time taken for computation in seconds,
              or np.nan if an error occurred.
    """
    n_samples = data.shape[0]
    print(f"\n--- Precomputing Distance Matrix ({method.upper()}) ---")
    print(f"Calculating pairwise distances for {n_samples} samples...")
    if method == 'softdtw':
        print(f"  Using soft-DTW with gamma={gamma}")
    elif method == 'dtw':
        print(f"  Using classic DTW (n_jobs={n_jobs})")
    else:
        print(f"ERROR: Unknown distance matrix method '{method}'. Use 'dtw' or 'softdtw'.")
        return None, np.nan

    start_time = time.time()
    distance_matrix = None
    computation_time = np.nan

    try:
        data_prepared = np.ascontiguousarray(data, dtype=np.float64)

        if method == 'softdtw':
            # Calculate raw soft-DTW similarity matrix
            distance_matrix = cdist_soft_dtw(data_prepared, gamma=gamma)
            # 1. Shift the matrix so the minimum value is 0.0
            min_val = np.min(distance_matrix)
            print(f"  Adjusting soft-DTW matrix: Subtracting min value ({min_val:.4f}) to ensure non-negativity.")
            distance_matrix = distance_matrix - min_val # Element-wise subtraction
            # 2. Fill the diagonal with zeros
            print("  Adjusting soft-DTW matrix: Setting diagonal to 0.")
            np.fill_diagonal(distance_matrix, 0)
        elif method == 'dtw':
            distance_matrix = cdist_dtw(data_prepared, n_jobs=n_jobs)
            # Classic DTW already has zeros on the diagonal and is non-negative

        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Pairwise distances calculated and adjusted (if softDTW) in {computation_time:.2f} seconds.")

    except MemoryError:
        # ... (error handling remains the same) ...
        print("\nERROR: MemoryError calculating the full distance matrix!")
        distance_matrix = None
        computation_time = np.nan
    except Exception as e:
        # ... (error handling remains the same) ...
        print(f"\nERROR: An unexpected error occurred during distance calculation: {e}")
        distance_matrix = None
        computation_time = np.nan

    return distance_matrix, computation_time

def calculate_dtw_clustering_metrics(distance_matrix, computation_time, cluster_labels):
    """
    Calculates internal clustering evaluation metrics suitable for time series,
    using a PRECOMPUTED DTW or soft-DTW distance matrix.

    Args:
        distance_matrix (np.ndarray): The precomputed pairwise distance matrix.
                                      Shape: (n_samples, n_samples). Assumes DTW or soft-DTW.
        computation_time (float): The time taken to compute the distance_matrix (in seconds).
        cluster_labels (np.ndarray): The cluster assignments for each sample.
                                      Shape: (n_samples,)

    Returns:
        dict: A dictionary containing the calculated metrics and summary info.
              Keys: 'silhouette_dtw', 'davies_bouldin', 'cluster_distribution',
                    'dtw_within', 'dtw_between', 'dtw_computation_time'
              Returns NaNs for metrics if calculation is not possible (e.g., < 2 clusters,
              distance matrix computation failed).
    """
    print("\n--- Calculating Clustering Metrics from Precomputed Distances ---")

    if distance_matrix is None:
        print("ERROR: Distance matrix is None. Cannot calculate metrics.")
        # Attempt to get distribution if possible
        try:
            distribution = dict(zip(*np.unique(cluster_labels, return_counts=True)))
        except Exception:
            distribution = {}
        return {
            'silhouette_dtw': np.nan,
            'davies_bouldin': np.nan,
            'cluster_distribution': distribution,
            'dtw_within': {},
            'dtw_between': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'dtw_computation_time': computation_time if not np.isnan(computation_time) else 0.0 # Use provided time if available
        }

    n_samples = distance_matrix.shape[0]
    unique_labels = np.unique(cluster_labels)
    n_clusters_found = len(unique_labels)

    # --- Basic Info ---
    distribution = dict(zip(*np.unique(cluster_labels, return_counts=True)))
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of clusters found: {n_clusters_found}")
    print(f"  Cluster Distribution: {distribution}")
    print(f"  Distance Matrix Computation Time: {computation_time:.2f} seconds")


    # Handle cases where silhouette score cannot be calculated
    if n_samples <= 1 or n_clusters_found <= 1:
        print(f"WARN: Cannot calculate Silhouette or Davies-Bouldin. Need > 1 sample and > 1 cluster.")
        print(f"      (Found {n_samples} samples and {n_clusters_found} unique cluster labels: {unique_labels})")
        return {
            'silhouette_dtw': np.nan,
            'davies_bouldin': np.nan,
            'cluster_distribution': distribution,
            'dtw_within': {}, # Still try to compute if possible, but likely empty
            'dtw_between': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'dtw_computation_time': computation_time
        }

    # --- Calculate DTW Silhouette Score ---
    print("Calculating DTW Silhouette Score...")
    silhouette_dtw = np.nan # Default to NaN
    try:
        silhouette_dtw = dtw_silhouette_score(
            X=distance_matrix, # Use the precomputed matrix
            labels=cluster_labels,
            metric='precomputed', # Specify that X is a distance matrix
            verbose=0
        )
        print(f"  DTW Silhouette Score: {silhouette_dtw:.4f}")
        # Interpretation Guidance
        print("  Silhouette Interpretation Guide:")
        if silhouette_dtw > 0.7:
            print("    - Strong structure detected (clusters are dense and well-separated).")
        elif silhouette_dtw > 0.5:
            print("    - Reasonable structure detected.")
        elif silhouette_dtw > 0.25:
            print("    - Weak structure detected (could be artificial, consider different k).")
        else:
            print("    - No substantial structure detected (clustering may not be meaningful).")
    except ValueError as ve:
         # Specifically catch ValueError which often occurs with invalid inputs (e.g. all points in one cluster)
        print(f"\nWARN: Could not calculate DTW Silhouette Score: {ve}")
        print(f"      This often happens if the number of labels is invalid (e.g., only 1 cluster found for {n_samples} samples).")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during Silhouette Score calculation: {e}")


    # --- Compute Davies-Bouldin index using DTW distances ---
    print("Calculating Davies-Bouldin Index (DTW-based)...")
    davies_bouldin = np.nan # Default to NaN
    medoids = {}
    intra_cluster_distances = {}
    calculation_possible = True

    for cluster in unique_labels:
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            print(f"  WARN: Cluster {cluster} has no members. Skipping for DB index.")
            calculation_possible = False
            break # Cannot compute if a cluster is empty
        if len(indices) == 1:
             print(f"  WARN: Cluster {cluster} has only one member. Intra-cluster distance is 0.")
             medoids[cluster] = indices[0]
             intra_cluster_distances[cluster] = 0.0
             continue # Continue to next cluster

        # Extract the submatrix for the cluster
        submatrix = distance_matrix[np.ix_(indices, indices)]
        # Average distance per candidate within the cluster
        avg_distances = np.mean(submatrix, axis=1)
        # Medoid: sample with the smallest average distance to others in the cluster
        medoid_index_in_cluster = np.argmin(avg_distances)
        medoid_global_index = indices[medoid_index_in_cluster]
        medoids[cluster] = medoid_global_index
        # S_i: average distance from all samples in the cluster to the medoid
        intra_cluster_distances[cluster] = np.mean(distance_matrix[indices, medoid_global_index])
        # Alternative S_i: average distance among all pairs in the cluster (more common but slower)
        # triu_idx = np.triu_indices_from(submatrix, k=1)
        # intra_cluster_distances[cluster] = np.mean(submatrix[triu_idx]) if triu_idx[0].size > 0 else 0.0


    if calculation_possible and len(unique_labels) > 1:
        try:
            R_list = []
            for i in unique_labels:
                if i not in medoids: continue # Skip if medoid wasn't found (e.g., empty cluster)
                max_R_i = 0.0 # Initialize max ratio for cluster i
                for j in unique_labels:
                    if i == j or j not in medoids: continue # Skip self-comparison or if medoid missing

                    # Distance between medoids (M_ij)
                    M_ij = distance_matrix[medoids[i], medoids[j]]

                    if M_ij == 0:
                         # Avoid division by zero. Happens if medoids are identical (e.g., duplicate time series)
                         # Or if distance is exactly zero for distinct series (possible with some DTW variants/data)
                         print(f"  WARN: Distance between medoids of cluster {i} and {j} is 0. Skipping this pair for DB index.")
                         # In this scenario, you might assign a large penalty or handle based on context.
                         # For simplicity here, we skip calculating this specific R_ij, which might slightly
                         # underestimate the DB index if other R_ij values are non-zero. If all M_ij are 0, DB will be 0.
                         continue # Skip this pair

                    S_i = intra_cluster_distances.get(i, 0.0) # Get intra-cluster distance, default 0
                    S_j = intra_cluster_distances.get(j, 0.0)
                    R_ij = (S_i + S_j) / M_ij
                    if R_ij > max_R_i:
                        max_R_i = R_ij

                if max_R_i > 0: # Only add if a valid comparison was made
                    R_list.append(max_R_i)

            if R_list: # Check if any ratios were calculated
                davies_bouldin = np.mean(R_list)
                print(f"  Davies-Bouldin (DTW): {davies_bouldin:.4f}")
            else:
                print("  WARN: Could not compute Davies-Bouldin index (no valid inter-cluster comparisons possible).")
                davies_bouldin = np.nan

        except Exception as e:
            print(f"\nERROR: Could not calculate Davies-Bouldin (DTW): {e}")
            davies_bouldin = np.nan
    elif len(unique_labels) <= 1:
         print("  Skipping Davies-Bouldin: Only one cluster found.")
         davies_bouldin = np.nan


    # --- Compute within-cluster DTW distance statistics ---
    print("Calculating DTW distance statistics...")
    dtw_within = {}
    for cluster in unique_labels:
        indices = np.where(cluster_labels == cluster)[0]
        values = np.array([]) # Default to empty array
        if len(indices) > 1:
            submatrix = distance_matrix[np.ix_(indices, indices)]
            # Extract only the upper triangle (excluding the diagonal)
            triu_idx = np.triu_indices_from(submatrix, k=1)
            if triu_idx[0].size > 0:
                values = submatrix[triu_idx]
        # Store stats even if values is empty (will result in NaNs)
        dtw_within[cluster] = {
            'mean': np.mean(values) if values.size > 0 else np.nan,
            'std': np.std(values) if values.size > 0 else np.nan,
            'min': np.min(values) if values.size > 0 else np.nan,
            'max': np.max(values) if values.size > 0 else np.nan,
        }
    print(' - within:')
    if dtw_within:
        # Convert dict of dicts to DataFrame for nice printing
        df_within = pd.DataFrame(dtw_within).T # Transpose to get clusters as rows
        df_within.index.name = 'Cluster'
        print(textwrap.indent(df_within.to_string(), '    '))
    else:
        print("    No within-cluster distances to report.")

    # --- Compute between-cluster DTW distance statistics ---
    between_values = []
    if len(unique_labels) > 1:
        for i, cluster_i in enumerate(unique_labels):
            indices_i = np.where(cluster_labels == cluster_i)[0]
            if len(indices_i) == 0: continue # Skip empty clusters

            for cluster_j in unique_labels[i+1:]: # Compare with subsequent clusters only
                 indices_j = np.where(cluster_labels == cluster_j)[0]
                 if len(indices_j) == 0: continue # Skip empty clusters

                 # Extract the rectangular submatrix of distances between clusters i and j
                 submatrix = distance_matrix[np.ix_(indices_i, indices_j)]
                 if submatrix.size > 0:
                     between_values.extend(submatrix.flatten()) # Add all pairwise distances

    between_values = np.array(between_values)
    dtw_between = {
        'mean': np.mean(between_values) if between_values.size > 0 else np.nan,
        'std': np.std(between_values) if between_values.size > 0 else np.nan,
        'min': np.min(between_values) if between_values.size > 0 else np.nan,
        'max': np.max(between_values) if between_values.size > 0 else np.nan,
    }
    print(' - between (overall):')
    if between_values.size > 0:
         # Create a DataFrame for consistent printing
         df_between = pd.DataFrame([dtw_between])
         print(textwrap.indent(df_between.to_string(index=False), '    '))
    else:
         print("    No between-cluster distances to report (need > 1 non-empty cluster).")

    print("-------------------------------------------------")

    results = {
        'silhouette_dtw': silhouette_dtw,
        'davies_bouldin': davies_bouldin,
        'cluster_distribution': distribution,
        'dtw_within': dtw_within,       # dict of stats per cluster
        'dtw_between': dtw_between,     # dict of overall stats
        'dtw_computation_time': computation_time
    }
    return results

def cluster_timeseries(dataset, output_folder, excel_path,
                       method="ts_kmeans", # Clustering method: "ts_kmeans" or "agglo_ward"
                       use_scaler='_scMnVar',
                       # --- Grid Search / Parameter Settings ---
                       grid_search="off", # "on" to run grid search, "off" to run with specified params
                       ks=[2],            # List of k values for grid search (or single value if grid_search="off")
                       gammas=[1.0],      # List of gamma values for grid search (or single value)
                       dm_method='softdtw',# 'dtw' or 'softdtw' (used if grid_search="off" or as default in grid search)
                       n_clusters=2,     # Default k if grid_search="off"
                       gamma=1.0,        # Default gamma if grid_search="off"
                       random_state=42,
                       # --- Other settings ---
                       n_jobs=-1         # For parallel DTW calculation
                       ):
    """
    Performs time series clustering on time series data, optionally running a grid search
    over k and gamma, precomputing the distance matrix, and saving results.

    Args:
        dataset (xr.Dataset): Input data containing time series features.
        output_folder (str): Path to save figures.
        excel_path (str): Path to save summary and results (.xlsx).
        method (str): Clustering method (currently only "ts_kmeans" and "agglo_ward").
        use_scaler (str): Suffix indicating scaling applied or to apply ('_scMnVar' or '').
        grid_search (str): "on" to perform grid search over ks and gammas, "off" to run once.
        ks (list): List of integers for the number of clusters (k) to try in grid search.
        gammas (list): List of floats for the gamma parameter (soft-DTW) to try in grid search.
        dm_method (str): Distance metric 'dtw' or 'softdtw' for the run (if grid_search="off")
                         or as the method during grid search.
        n_clusters (int): Number of clusters if grid_search="off".
        gamma (float): Gamma value for soft-DTW if grid_search="off".
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of jobs for parallel DTW calculation.

    Returns:
        xr.Dataset or None: The dataset with added 'cluster_labels' if grid_search="off",
                            otherwise None (as grid search just saves results).
    """

    # --- 1. Prepare Data ---
    feature_vars = [
        f"fem_pat_{'EXERC'}_on_patella_{'EXERC'}_in_patella_{'EXERC'}_resultant_pcNewtons",
        f"total_{'EXERC'}_resultant_force_pcNewtons",
        f"mediototal_{'EXERC'}_resultant_ratio_pc"
    ]
    # Add scaler suffix if needed (assuming data is already scaled if suffix present)
    feature_vars_scaled = [f + use_scaler if use_scaler and use_scaler != "_scMnVar" else f for f in feature_vars]

    # Select relevant data and reshape for tslearn (n_samples, n_timesteps, n_features)
    try:
        clustering_data_list = [dataset[var].values for var in feature_vars_scaled]
        # Stack along the last axis to create the feature dimension
        clustering_data = np.stack(clustering_data_list, axis=-1)
        # Ensure it's a 3D array (it should be, but double-check)
        if clustering_data.ndim == 2: # If only one feature was selected
             clustering_data = clustering_data[:, :, np.newaxis]

    except KeyError as e:
        print(f"ERROR: Feature variable not found in dataset: {e}")
        print(f"Available variables: {list(dataset.data_vars)}")
        return None if grid_search == "off" else False # Indicate failure

    # Apply scaling if specified
    if use_scaler == "_scMnVar":
        print("Applying TimeSeriesScalerMeanVariance...")
        scaler = TimeSeriesScalerMeanVariance()
        # Reshape for scaler (n_samples, n_timesteps * n_features) if needed, then back
        # Or apply feature-wise? tslearn scaler handles 3D data correctly.
        clustering_data = scaler.fit_transform(clustering_data) # Modifies data in place potentially
        print("Scaling complete.")

    # Ensure data is contiguous and float64 for C extensions efficiency
    clustering_data = np.ascontiguousarray(clustering_data, dtype=np.float64)
    print(f"Clustering Data prepared with shape: {clustering_data.shape}") # (n_samples, n_timesteps, n_features)

    # --- 2. Grid Search Logic ---
    if grid_search.lower() == "on":
        print("\n=== Starting Grid Search ===")
        results_list = []
        # For agglo_ward we only grid‐search gamma; k is determined internally
        if method == "agglo_ward":
            param_grid = [(None, g) for g in gammas]
        else:
            param_grid = list(itertools.product(ks, gammas if dm_method == 'softdtw' else [None]))

        # --- Precompute Distance Matrix ONCE per gamma (if softDTW) or just ONCE (if DTW) ---
        # Store precomputed matrices to avoid recalculation within the k-loop
        precomputed_matrices = {} # Key: gamma (or 'dtw'), Value: (matrix, time)

        for k_val, gamma_val in param_grid:
            print(f"\n--- Grid Search: k={k_val}, gamma={gamma_val if gamma_val is not None else 'N/A'} ---")

            current_dm_method = dm_method # Use the overall method specified
            current_gamma = gamma_val if current_dm_method == 'softdtw' else None

            # --- Get or Compute Distance Matrix ---
            matrix_key = current_gamma if current_dm_method == 'softdtw' else 'dtw'
            if matrix_key not in precomputed_matrices:
                print(f"Calculating distance matrix for gamma={current_gamma}" if current_dm_method == 'softdtw' else "Calculating DTW distance matrix")
                precomputed_matrix, precomputation_time = precompute_distance_matrix(
                    clustering_data,
                    method=current_dm_method,
                    gamma=current_gamma if current_gamma is not None else 1.0, # Pass default if None, although ignored by dtw
                    n_jobs=n_jobs
                )
                if precomputed_matrix is None:
                    print(f"WARN: Failed to compute distance matrix for {matrix_key}. Skipping this grid point.")
                    continue # Skip to next grid combination
                precomputed_matrices[matrix_key] = (precomputed_matrix, precomputation_time)
            else:
                print(f"Using precomputed distance matrix for {matrix_key}")
                precomputed_matrix, precomputation_time = precomputed_matrices[matrix_key]


            # --- Run Clustering ---
            if method == "ts_kmeans":
                print(f"Running TimeSeriesKMeans (k={k_val}, metric={current_dm_method}, gamma={current_gamma})...")
                model_params = {
                    "n_clusters": k_val,
                    "random_state": random_state,
                    "n_init": 3, # Add n_init for stability
                    "verbose": 0 # Set to 1 for more details
                }
                if current_dm_method == 'softdtw':
                    model_params["metric"] = "softdtw"
                    model_params["metric_params"] = {"gamma": current_gamma}
                else: # 'dtw'
                    model_params["metric"] = "dtw"
                    # No metric_params needed for standard dtw

                tskmeans_start_time = time.time()
                model = TimeSeriesKMeans(**model_params)
                tskmeans_end_time = time.time()
                eval_time = tskmeans_end_time - tskmeans_start_time
                print(f"  TimeSeriesKMeans time: {eval_time:.2f}s")

                try:
                    cluster_labels = model.fit_predict(clustering_data)
                except Exception as e:
                    print(f"ERROR during TimeSeriesKMeans fitting: {e}")
                    print("Skipping metric calculation for this grid point.")
                    continue # Skip to next grid point

            elif method == "agglo_ward":
                # We ignore k_val here, only gamma_val matters:
                print(f"Running Agglomerative (Ward) with Soft‑DTW γ={gamma_val}…")

                # 1) We already have precomputed_matrix for this gamma_val
                dist_mat = precomputed_matrix

                # 2) Build linkage tree
                condensed = squareform(dist_mat, checks=False)
                Z = linkage(condensed, method='ward')

                # 3) Evaluate cuts k=2…4, pick best silhouette
                best_sil, best_k = -1, None
                for cut_k in range(2, 5):
                    labels = fcluster(Z, t=cut_k, criterion='maxclust')
                    sil = dtw_silhouette_score(
                        X=dist_mat,
                        labels=labels,
                        metric='precomputed'
                    )
                    print(f"  k={cut_k} → silhouette={sil:.4f}")
                    if sil > best_sil:
                        best_sil, best_k = sil, cut_k

                print(f"  → best silhouette {best_sil:.4f} at k={best_k}")

                # 4) Final cut & metrics
                final_labels = fcluster(Z, t=best_k, criterion='maxclust')
                metrics = calculate_dtw_clustering_metrics(
                    distance_matrix=dist_mat,
                    computation_time=precomputation_time,
                    cluster_labels=final_labels
                )

                # 5) Store result for this gamma
                result_row = {
                    "k":        best_k,
                    "gamma":    gamma_val,
                    "dm_method": dm_method,
                    "silhouette_dtw": metrics['silhouette_dtw'],
                    "davies_bouldin": metrics['davies_bouldin'],
                    "n_clusters_found": best_k,
                    "dtw_comp_time_s": metrics['dtw_computation_time']
                }
                results_list.append(result_row)
                continue

            else:
                print(f"ERROR: Method {method} not implemented in this version.")
                continue

            # --- Calculate Metrics using Precomputed Matrix ---
            metrics_start_time = time.time()
            metrics = calculate_dtw_clustering_metrics(
                distance_matrix=precomputed_matrix,
                computation_time=precomputation_time,
                cluster_labels=cluster_labels
            )
            metrics_end_time = time.time()
            eval_time = metrics_end_time - metrics_start_time
            print(f"  calculate_dtw_clustering_metrics time: {eval_time:.2f}s")

            # --- Store Results ---
            result_row = {
                "k": k_val,
                "gamma": current_gamma if current_dm_method == 'softdtw' else np.nan,
                "dm_method": current_dm_method,
                "silhouette_dtw": metrics['silhouette_dtw'],
                "davies_bouldin": metrics['davies_bouldin'],
                "n_clusters_found": len(metrics.get('cluster_distribution', {})),
                "cluster_distribution": str(metrics.get('cluster_distribution', {})), # Store as string for Excel
                "dtw_comp_time_s": metrics['dtw_computation_time']
            }
            results_list.append(result_row)

        # --- Save Grid Search Results ---
        if results_list:
            grid_results_df = pd.DataFrame(results_list)
            print("\n=== Grid Search Complete ===")
            print(grid_results_df.to_string())
            try:
                # Use mode 'a' and if_sheet_exists='replace' to add/overwrite sheet
                with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                    grid_results_df.to_excel(writer, sheet_name='grid_search_results', index=False)
                print(f"\nGrid search results saved to sheet 'grid_search_results' in '{excel_path}'")
            except Exception as e:
                print(f"\nERROR: Could not save grid search results to Excel: {e}")
                print("Saving to CSV as fallback: grid_search_results.csv")
                grid_results_df.to_csv("grid_search_results.csv", index=False)

            print("\nGrid search finished. Set grid_search='off' and update parameters (k, gamma, dm_method) based on results.")
            # sys.exit() # Stop execution after grid search
            # return None # Alternative to sys.exit()

        else:
            print("\n=== Grid Search Complete (No valid results obtained) ===")
            sys.exit()
            # return None


    # --- 3. Single Run Logic (grid_search="off") ---
    else:
        print(f"\n=== Running Single Clustering (k={n_clusters}, method={dm_method}, gamma={gamma if dm_method=='softdtw' else 'N/A'}) ===")

        # --- Precompute Distance Matrix ---
        #precomputed_matrix, precomputation_time = precompute_distance_matrix(
        #    clustering_data,
        #    method=dm_method,
        #    gamma=gamma,
        #    n_jobs=n_jobs
        #)

        #if precomputed_matrix is None:
        #     print("ERROR: Failed to compute distance matrix. Aborting clustering.")
        #     return None # Indicate failure

        # --- Run Clustering ---
        if method == "ts_kmeans":
             print(f"Running TimeSeriesKMeans (k={n_clusters}, metric={dm_method}, gamma={gamma if dm_method=='softdtw' else 'N/A'})...")
             model_params = {
                 "n_clusters": n_clusters,
                 "random_state": random_state,
                 "n_init": 5, # Use more inits for final run
                 "verbose": 0
             }
             if dm_method == 'softdtw':
                 model_params["metric"] = "softdtw"
                 model_params["metric_params"] = {"gamma": gamma}
             else: # 'dtw'
                 model_params["metric"] = "dtw"
             tskmeans_start_time = time.time()
             model = TimeSeriesKMeans(**model_params)
             tskmeans_end_time = time.time()
             eval_time = tskmeans_end_time - tskmeans_start_time
             print(f"  TimeSeriesKMeans time: {eval_time:.2f}s")

             try:
                 cluster_labels = model.fit_predict(clustering_data)
             except Exception as e:
                 print(f"ERROR during TimeSeriesKMeans fitting: {e}")
                 return None

        elif method == "agglo_ward":
            # 1) Precompute DTW / Soft‑DTW distance matrix
            precomputed_matrix, dm_time = precompute_distance_matrix(
                clustering_data,
                method=dm_method,
                gamma=gamma,
                n_jobs=n_jobs
            )
            if precomputed_matrix is None:
                print("ERROR: could not compute distance matrix for agglo_ward. Aborting.")
                return None

            # 2) Perform Ward’s hierarchical clustering
            #    SciPy linkage expects a condensed distance vector:
            condensed = squareform(precomputed_matrix, checks=False)
            Z = linkage(condensed, method='ward')

            # 3) (Optional) plot dendrogram
            fig, ax = plt.subplots(figsize=(9, 5))
            dendrogram(
                Z,
                ax=ax,
                no_labels=True,           # set False if you want leaf labels
                distance_sort='descending',
                show_leaf_counts=True,    # small counts under leaves help readability
                above_threshold_color='gray',
                color_threshold=None,     # color every merge uniquely unless cut line used
                leaf_rotation=90,
                leaf_font_size=8,
                truncate_mode=None        # or 'lastp' to show only the last p merges
            )
            ax.set_title("Ward Linkage Dendrogram", pad=10)
            ax.set_ylabel("Linkage distance (Ward / ΔSSE)")
            ax.set_xlabel("Samples / merged clusters (ordered)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.grid(True, axis='y', linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_folder, "dendrogram_agglo_ward.pdf"))
            plt.close(fig)
            
            # 4) Evaluate cluster‐cuts from k=2…4
            hier_results = []
            for k_cut in range(2, 5):
                labels = fcluster(Z, t=k_cut, criterion='maxclust')
                metrics = calculate_dtw_clustering_metrics(
                    distance_matrix=precomputed_matrix,
                    computation_time=dm_time,
                    cluster_labels=labels
                )
                hier_results.append({
                    "k":        k_cut,
                    "silhouette_dtw": metrics['silhouette_dtw'],
                    "davies_bouldin": metrics['davies_bouldin'],
                    "distribution":    metrics['cluster_distribution']
                })

            # 5) Save the k=2…4 results to Excel
            hier_df = pd.DataFrame(hier_results)
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                hier_df.to_excel(writer, sheet_name='hierarchy_metrics', index=False)

            # 6) pick the best k by max silhouette (tie→smallest k)
            best = max(hier_results, key=lambda r: (r['silhouette_dtw'] or -999, -r['k']))
            best_k = best['k']
            print(f"→ Recommended cluster count (highest silhouette): k={best_k}")

            # 7) re‑cut tree at best_k and assign labels
            final_labels = fcluster(Z, t=best_k, criterion='maxclust')

            # 8) attach to dataset exactly like k‑means does
            dataset = dataset.assign(cluster_labels=(('sample',), final_labels))
            for var in feature_vars:
                dataset[f"{var}_cluster"] = ('sample', final_labels)

            # 9) reuse your existing plotting loop (it will pick up .cluster on each var)
            print("Generating hierarchical cluster overlay plots…")
            # (You already have the code below that generates & saves plots)
            # … so just let your normal plotting section run.
            return dataset

        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # --- Calculate Metrics ---
        metrics_start_time = time.time()
        metrics = calculate_dtw_clustering_metrics(
            distance_matrix=precomputed_matrix,
            computation_time=precomputation_time,
            cluster_labels=cluster_labels
        )
        metrics_end_time = time.time()
        eval_time = metrics_end_time - metrics_start_time
        print(f"  calculate_dtw_clustering_metrics time: {eval_time:.2f}s")

        # Extract metrics for summary
        silhouette_dtw = metrics['silhouette_dtw']
        davies_bouldin = metrics['davies_bouldin']
        cluster_distribution_dict = metrics['cluster_distribution']
        dtw_within = metrics['dtw_within']
        dtw_between = metrics['dtw_between']
        dtw_comp_time = metrics['dtw_computation_time'] # Use the time from metrics dict

        # --- Assign Labels and Plot ---
        print("\nAssigning cluster labels to dataset...")
        dataset = dataset.assign(cluster_labels=(('sample',), cluster_labels))

        # Add cluster labels as coordinates for easier selection/plotting if needed
        # Also helpful to associate unscaled data with clusters if scaling was applied
        for var in feature_vars: # Iterate through original base names
             scaled_var = var + use_scaler if use_scaler and use_scaler != "_scMnVar" else var
             unscaled_var = var # Original name without suffix
             # Add cluster coord based on scaled var name (if exists)
             if scaled_var in dataset:
                  dataset[f"{scaled_var}_cluster"] = ('sample', cluster_labels)
             # Always add cluster coord based on unscaled var name
             if unscaled_var in dataset:
                  dataset[f"{unscaled_var}_cluster"] = ('sample', cluster_labels)


        print("Generating cluster plots...")
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        num_clusters_found = len(unique_clusters) - (1 if -1 in unique_clusters else 0) # Exclude noise if present
        print(f"Identified {num_clusters_found} clusters (excluding noise if any). Distribution: {cluster_distribution_dict}")

        # Determine palette size based on actual clusters found (excluding -1)
        plot_palette_size = max(1, num_clusters_found) # Need at least 1 color
        palette = sns.color_palette("tab10", plot_palette_size)

        num_plot_vars = len(feature_vars) # Plot unscaled variables
        num_cols = 3
        num_rows = (num_plot_vars + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, squeeze=False) # Ensure axes is 2D
        axes = axes.flatten() # Flatten for easy iteration

        # Sort clusters by size for potentially clearer plotting (largest first)
        # Handle potential noise cluster (-1) separately or sort ignoring it
        cluster_items = sorted(cluster_distribution_dict.items(), key=lambda item: item[1], reverse=True)

        # Map cluster labels to palette indices (0, 1, 2...) handling noise (-1)
        cluster_to_palette_idx = {label: idx for idx, (label, count) in enumerate(
                                   item for item in cluster_items if item[0] != -1)}

        for i, var in enumerate(feature_vars): # Plot original variables
            ax = axes[i]
            # Plot individual traces first with low alpha
            for cluster_label, count in cluster_items:
                color = 'gray' if cluster_label == -1 else palette[cluster_to_palette_idx.get(cluster_label, 0)] # Default to first color if issue
                alpha = 0.05 if cluster_label == -1 else 0.1
                cluster_indices = np.where(cluster_labels == cluster_label)[0]

                if len(cluster_indices) > 0:
                    # Plot individual time series for this cluster
                    # Use .isel instead of direct numpy indexing for xarray DataArrays
                    cluster_data = dataset[var].isel(sample=cluster_indices)
                    # Plot each sample's time series
                    for sample_idx in range(len(cluster_indices)):
                         # Plotting the transpose (.T) assumes time is the second dimension (index 1)
                         ax.plot(cluster_data.coords[cluster_data.dims[1]], # Get time coordinates
                                 cluster_data.isel(sample=sample_idx).values, # Get data for one sample
                                 color=color, alpha=alpha, linestyle='-')


            # Plot means on top
            plot_legend_handles = {} # To store one handle per cluster mean
            for cluster_label, count in cluster_items:
                color = 'gray' if cluster_label == -1 else palette[cluster_to_palette_idx.get(cluster_label, 0)]
                label_str = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}" # Use original label
                cluster_indices = np.where(cluster_labels == cluster_label)[0]

                if len(cluster_indices) > 0:
                    # Calculate mean across samples for this cluster
                    mean_values = dataset[var].isel(sample=cluster_indices).mean(dim='sample')
                    time_coords = mean_values.coords[mean_values.dims[0]] # Get time coordinates for the mean plot

                    # Plot mean line
                    line, = ax.plot(time_coords, mean_values.values, color=color, linewidth=3, linestyle='-',
                                    label=f"{label_str} Mean ({count} samples)", # Add count to legend
                                    zorder=1000, # Ensure mean is plotted on top
                                    path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()]) # Outline effect
                    if cluster_label not in plot_legend_handles:
                        plot_legend_handles[cluster_label] = line


            ax.set_title(f"Clustering: {var}")
            ax.set_xlabel("Time / Cycle (%)") # Adjust label as needed
            ax.set_ylabel(var.replace("_pcNewtons", " (N)").replace("_pc", " (%)")) # Basic unit cleaning

        # Remove empty subplots
        for j in range(num_plot_vars, len(axes)):
            fig.delaxes(axes[j])

        # Create a single legend for the figure
        # Extract handles and labels from the last populated axis or build manually
        handles = list(plot_legend_handles.values())
        labels = [h.get_label() for h in handles]
        if handles:
             # Sort legend by cluster label (numerically, handling -1)
             handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: int(x[1].split(' ')[1]) if 'Cluster' in x[1] else -1)
             handles_sorted, labels_sorted = zip(*handles_labels_sorted)
             fig.legend(handles_sorted, labels_sorted, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(len(labels), 4)) # Place legend above plots

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for legend
        clustering_fig_path = os.path.join(output_folder, f'clustering_{dm_method}_k{n_clusters}_gamma{gamma if dm_method=="softdtw" else "na"}_figure.pdf')
        try:
            fig.savefig(clustering_fig_path, format='pdf', bbox_inches='tight')
            print(f"Clustering figure saved to '{clustering_fig_path}'")
        except Exception as e:
            print(f"ERROR saving clustering figure: {e}")
        plt.show() # Display the plot


        # --- Save Summary and Detailed Metrics to Excel ---
        print(f"Saving results summary to '{excel_path}'...")
        summary_df = pd.DataFrame({
            "Method": [method],
            "Scaler": [use_scaler if use_scaler else "None"],
            "Distance_Metric": [dm_method],
            "Gamma_SoftDTW": [gamma if dm_method == 'softdtw' else np.nan],
            "Num_Clusters_Requested": [n_clusters],
            "Num_Clusters_Found": [num_clusters_found], # Excludes noise
            "Silhouette_DTW": [silhouette_dtw],
            "Davies_Bouldin_DTW": [davies_bouldin],
            "DTW_Distance_Time_s": [dtw_comp_time],
            "Random_State": [random_state]
        })

        dist_data = [{"Cluster": k, "Sample_Count": v} for k, v in cluster_distribution_dict.items()]
        distribution_df = pd.DataFrame(dist_data).sort_values(by="Cluster").reset_index(drop=True)

        within_data = []
        # Check if dtw_within is populated correctly
        if isinstance(dtw_within, dict) and dtw_within:
             for cluster, stats in dtw_within.items():
                 # stats should be a dict like {'mean': val, 'std': val,...}
                 row = {"Cluster": cluster}
                 row.update(stats) # Add mean, std, min, max keys/values
                 within_data.append(row)
             dtw_within_df = pd.DataFrame(within_data).sort_values(by="Cluster").reset_index(drop=True)
             # Rename columns for clarity
             dtw_within_df.columns = ['Cluster', 'DTW_Within_Mean', 'DTW_Within_Std', 'DTW_Within_Min', 'DTW_Within_Max']
        else:
            dtw_within_df = pd.DataFrame(columns=['Cluster', 'DTW_Within_Mean', 'DTW_Within_Std', 'DTW_Within_Min', 'DTW_Within_Max'])


        # Check if dtw_between is populated
        if isinstance(dtw_between, dict) and not all(np.isnan(v) for v in dtw_between.values()):
             dtw_between_df = pd.DataFrame([dtw_between]) # Convert single dict to DataFrame
              # Rename columns for clarity
             dtw_between_df.columns = ['DTW_Between_Mean', 'DTW_Between_Std', 'DTW_Between_Min', 'DTW_Between_Max']
        else:
            dtw_between_df = pd.DataFrame(columns=['DTW_Between_Mean', 'DTW_Between_Std', 'DTW_Between_Min', 'DTW_Between_Max'])


        try:
            # Use mode 'a' and if_sheet_exists='replace' to add/overwrite sheets
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                summary_df.to_excel(writer, sheet_name='clustering_summary', index=False)
                distribution_df.to_excel(writer, sheet_name='cluster_distribution', index=False)
                dtw_within_df.to_excel(writer, sheet_name='dtw_within_distribution', index=False)
                dtw_between_df.to_excel(writer, sheet_name='dtw_between_distribution', index=False)
            print(f"Clustering information successfully appended/updated in '{excel_path}'.")
        except Exception as e:
             print(f"\nERROR: Could not save results to Excel: {e}")
             print("Attempting to save sheets as separate CSV files...")
             try:
                  summary_df.to_csv("clustering_summary.csv", index=False)
                  distribution_df.to_csv("cluster_distribution.csv", index=False)
                  dtw_within_df.to_csv("dtw_within_distribution.csv", index=False)
                  dtw_between_df.to_csv("dtw_between_distribution.csv", index=False)
                  print("Successfully saved results to CSV files.")
             except Exception as csv_e:
                  print(f"ERROR saving to CSV files: {csv_e}")


        return dataset # Return the dataset with labels added
        
def objective_function_silhouette(gamma, k_value, data, random_state):
    """
    Objective function for Bayesian Optimization.
    Calculates -Silhouette score for a given gamma and fixed k.
    NOTE: Assumes 'data' is already scaled and prepared!
    """
    print(f"  Optimizing: Evaluating gamma={gamma:.4f} for k={k_value}...")
    start_time = time.time()

    # --- 1. Precompute Distance Matrix ---
    # Using softDTW as gamma optimization is relevant for it
    dm_method = 'softdtw'
    print("line 1358 sets fixed softdtw")
    precomputed_matrix, precomputation_time = precompute_distance_matrix(
        data,
        method=dm_method,
        gamma=gamma,
        # n_jobs = -1 # Set as needed
    )

    if precomputed_matrix is None:
        print(f"  WARN: Failed distance matrix computation for gamma={gamma:.4f}. Returning worst score.")
        # Return a value indicating failure (worse than any expected score)
        # Since we minimize -SIL, return a large positive number
        return 10.0 # Or larger if needed

    # --- 2. Run Clustering ---
    model_params = {
        "n_clusters": k_value,
        "random_state": random_state,
        "metric": dm_method,
        "metric_params": {"gamma": gamma},
        "n_init": 3, # Fewer inits during optimization for speed
        "verbose": 0
    }
    tskmeans_start_time = time.time()
    model = TimeSeriesKMeans(**model_params)
    tskmeans_end_time = time.time()
    eval_time = tskmeans_end_time - tskmeans_start_time
    print(f"  TimeSeriesKMeans time: {eval_time:.2f}s")
    try:
        cluster_labels = model.fit_predict(data)
    except Exception as e:
        print(f"  WARN: TimeSeriesKMeans failed for gamma={gamma:.4f}, k={k_value}: {e}. Returning worst score.")
        return 10.0 # Indicate failure

    # --- 3. Calculate Metrics ---
    metrics_start_time = time.time()
    metrics = calculate_dtw_clustering_metrics(
        distance_matrix=precomputed_matrix,
        computation_time=precomputation_time,
        cluster_labels=cluster_labels
    )
    metrics_end_time = time.time()
    eval_time = metrics_end_time - metrics_start_time
    print(f"  calculate_dtw_clustering_metrics time: {eval_time:.2f}s")

    silhouette = metrics.get('silhouette_dtw', np.nan)
    end_time = time.time()
    eval_time = end_time - start_time
    print(f"  Optimizing: gamma={gamma:.4f}, k={k_value} -> Silhouette={silhouette:.4f} (Eval time: {eval_time:.2f}s)")


    # Handle NaN or calculation failures
    if np.isnan(silhouette):
        # If silhouette fails, return a very bad score to deter optimizer
        return 10.0
    else:
        # We want to MAXIMIZE silhouette, gp_minimize MINIMIZES, so return NEGATIVE silhouette
        return -silhouette

def objective_function_hac(gamma, data):
    """
    Bayesian objective for agglo_ward: returns -best silhouette over cuts k=2..10
    """
    # 1) Build the soft‑DTW distance matrix
    dist_mat, _ = precompute_distance_matrix(
        data,
        method="softdtw",
        gamma=gamma,
        n_jobs=-1
    )
    if dist_mat is None:
        return 10.0  # worst possible

    # 2) Build Ward linkage tree
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(dist_mat, checks=False), method="ward")

    # 3) Scan k=2..4, track best silhouette
    best_sil = -np.inf
    for k in range(2, 5):
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = dtw_silhouette_score(
            X=dist_mat,
            labels=labels,
            metric="precomputed"
        )
        if sil > best_sil:
            best_sil = sil

    # 4) Return negative so gp_minimize *maximizes* silhouette
    return -best_sil if not np.isnan(best_sil) else 10.0

def optimize_gamma_for_k(k_value, data, initial_results_df, n_calls=20, random_state=42, method="ts_kmeans"):
    """
    Uses Bayesian Optimization to find the best gamma for a specific k.

    Args:
        k_value (int): The fixed number of clusters k.
        data (np.ndarray): The prepared clustering data.
        initial_results_df (pd.DataFrame): DataFrame of initial grid search results.
        n_calls (int): Number of optimization iterations (evaluations of objective).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (best_gamma, best_score) found for this k. Score is the metric value (e.g. Silhouette).
               Returns (None, None) if optimization cannot proceed.
    """
    global RANDOM_STATE_CACHE
    RANDOM_STATE_CACHE = random_state # Store for objective function

    print(f"\n--- Optimizing Gamma for k = {k_value} ---")

    # Filter initial results for this k and softDTW
    initial_k_df = initial_results_df[
        (initial_results_df['k'] == k_value) &
        (initial_results_df['dm_method'] == 'softdtw') &
        (initial_results_df['silhouette_dtw'].notna()) # Only use valid starting points
    ].copy()

    if initial_k_df.empty:
        print(f"WARN: No valid initial grid search results found for k={k_value} with softDTW. Skipping optimization.")
        # Optionally, run gp_minimize without initial points, but it might take longer
        # For now, we skip if no starting point
        # Or, you could provide default initial points if none are found
        # x0 = [1.0] # Default gamma guess
        # y0 = [objective_function_silhouette(1.0, k_value, data, random_state)]
        return None, None # Indicate failure/skip


    # Prepare initial points for the optimizer
    # Ensure correct column names from your grid search excel/df
    x0 = [[g] for g in initial_k_df['gamma'].tolist()] # Create a list of lists for x0
    y0 = (-initial_k_df['silhouette_dtw']).tolist() # Objective minimizes -SIL
    
    print(f"Starting Bayesian Optimization with {len(x0)} initial points and {n_calls} total calls.")
    print(f"Initial gammas (x0): {x0}")
    print(f"Initial objectives (-SIL): {y0}")

    # Define the objective function specific to this k
    # Using a lambda captures the current k_value
    if method == "ts_kmeans":
        # optimize gamma for soft‑DTW k‑means with fixed k_value
        objective_for_this_k = lambda gamma_list: objective_function_silhouette(
            gamma=gamma_list[0],
            k_value=k_value,
            data=data,
            random_state=RANDOM_STATE_CACHE
        )
    else:  # method == "agglo_ward"
        # optimize gamma for agglomerative Ward (no fixed k_value used)
        objective_for_this_k = lambda gamma_list: objective_function_hac(
            gamma=gamma_list[0],
            data=data
        )

    try:
         # Run Bayesian Optimization
         result = gp_minimize(
             func=objective_for_this_k,
             dimensions=[GAMMA_SPACE], # Pass the defined gamma search space
             x0=x0,                 # Initial gamma values
             y0=y0,                 # Initial objective function values (-SIL)
             n_calls=n_calls,       # Total number of evaluations (including initial points)
             random_state=random_state,
             noise=1e-10 # Add slight noise assumption for numerical stability if needed
         )

         best_gamma = result.x[0]
         best_neg_silhouette = result.fun
         best_silhouette = -best_neg_silhouette # Convert back to actual Silhouette

         print(f"--- Optimization Complete for k = {k_value} ---")
         print(f"  Best Gamma found: {best_gamma:.4f}")
         print(f"  Best Silhouette score found: {best_silhouette:.4f}")

         return best_gamma, best_silhouette

    except Exception as e:
         print(f"ERROR during Bayesian Optimization for k={k_value}: {e}")
         return None, None # Indicate failure