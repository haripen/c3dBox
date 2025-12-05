## Orientation

I've saved the collection to `knee_jrl_extraction_20251201_122818.npz` using

```python
np.savez_compressed(
    npz_filename,
    data=data,
    meta=meta_rec,
    feature_names=np.array(feature_names, dtype=object),
)
```

where

```python
base_name = f"knee_jrl_extraction_{timestamp}"
log_filename = os.path.normpath(root_dir + r"\\" + base_name + ".log")
npz_filename = os.path.normpath(root_dir + r"\\" + base_name + ".npz")
```

We do not work on `jrl_v6_ISB.py` but we transfer its functionality. The aim is using the new data structure to feed it into the same, updated, or improved scaling, optimization, and agglomerative soft-dtw time-series clustering with ward linkage as in `jrl_v6_ISB.py`.

## First Task

Append dtw_clustering.py with loading the saved data from file. Load the data in cell #%% --- PREPARE CLUSTERING --- of `dtw_clustering.py`

## Second Task

Have a look at the `clustering.py` which contains all the functions taken from `jrl_v6_ISB.py`. Update the functions to fit the new data structure.

## Third Taks

Understand the clustering steps in `jrl_v6_ISB.py` and bring them to `dtw_clustering.py`

## Fourth Task

I ran

```
%runcell -n '--- CLUSTERING ---' C:/GitHub/c3dBox/Clustering/dtw_clustering.py
Loaded dataset for clustering: C:\Data_Local\PRO_checked_mat\knee_jrl_extraction_20251201_122818.npz
 - data shape: (200, 101, 3)
 - meta rows: 200
 - features: ['fem_pat_on_patella_in_patella_resultant', 'mediototal_resultant_ratio_pc', 'total_resultant_tibfem_force']
Clustering data prepared with shape: (200, 101, 3)

=== Running Single Clustering (k=3, method=softdtw, gamma=1.0) ===

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=1.0
  Adjusting soft-DTW matrix: subtracting min value (-151.7771) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 89.82 seconds.

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=1.0
  Adjusting soft-DTW matrix: subtracting min value (-151.7771) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 77.94 seconds.

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 45, 2: 155}
  Distance Matrix Computation Time: 77.94 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.2292
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.6852
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        76.156612  34.114263  11.404294  207.763048
    2        67.716385  24.481172   6.343437  192.404040
         mean       std       min        max
    91.955895 35.164688 22.206275 221.705714
-------------------------------------------------

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 3
  Cluster Distribution: {1: 45, 2: 56, 3: 99}
  Distance Matrix Computation Time: 77.94 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1098
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.6325
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        76.156612  34.114263  11.404294  207.763048
    2        50.589694  21.379491   6.343437  129.516157
    3        68.130946  26.105154   7.008645  192.404040
         mean       std       min        max
    83.167671 31.503845 22.206275 221.705714
-------------------------------------------------

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 4
  Cluster Distribution: {1: 14, 2: 31, 3: 56, 4: 99}
  Distance Matrix Computation Time: 77.94 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1303
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.0977
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        72.550904  26.783238  12.848334  129.426080
    2        58.111343  26.662348  11.404294  127.388379
    3        50.589694  21.379491   6.343437  129.516157
    4        68.130946  26.105154   7.008645  192.404040
         mean       std       min        max
    83.605899 31.588946 22.206275 221.705714
-------------------------------------------------
-> Recommended cluster count (highest silhouette): k=2

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 45, 2: 155}
  Distance Matrix Computation Time: 89.82 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.2292
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.6852
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        76.156612  34.114263  11.404294  207.763048
    2        67.716385  24.481172   6.343437  192.404040
         mean       std       min        max
    91.955895 35.164688 22.206275 221.705714
-------------------------------------------------
Identified 2 clusters (excluding noise if any). Distribution: {1: 45, 2: 155}
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_k3_gamma1.0_figure.pdf'
Saving results summary to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx'...
Clustering information written to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx'.
Clustered meta saved to C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_meta_with_clusters.csv
```

It looks like --- Precomputing Distance Matrix (SOFTDTW) --- ran twice which seems unnecessary, but i did not yet run the gridsearch with bayesian optimization.

After running the `agglo_ward` method, and for all methods, the user should inspect the plotted dendrogram (currently missing), and for each `k`, look at the separation metrics saved to the sub-folder in Excel, as well as the clustered feature plot for all preset `ks` at the optimal gamma. Based on their observations, the user sets the number of clusters to keep and runs a cell (to be appended) where the cluster labels are assigned to every `path`, `side `and `cycle `according to the meta-data. The final outputs are then generated and saved the according folder, with the folder name appended with a `_chosen`. Finally, the code reads the `.mat` file from the `path `specified in the meta file using `loadmat_to_dict` and adds the key `cluster` to `d[path][side][cycle]`, setting the value to the cluster label. The `.mat` file is saved with the new field next to the original using the command `savemat(str(root_dir), d_upd, one_disp=False, long_field_names=True, do_compression=True)` and appending `_c` to the file name ensuring no overwrite.

Clean up and explain the settings on options in the call of

```python
clustering_run = cluster_timeseries(
    data=data,
    feature_names=feature_names,
    meta=meta,
    output_folder=clustering_output_dir,
    excel_path=excel_path,
    method="agglo_ward",
    use_scaler=True,
    grid_search="off",
    ks=[2, 3, 4],
    gammas=[0.1, 1.0, 10.0],
    dm_method="softdtw",
    n_clusters=3,
    gamma=1.0,
    random_state=42,
    n_jobs=-1,
    plot=True,
)
```

Focussing on `method="agglo_ward`, thus removin all unnecessary options here.

Refactor the code to include the following cells

1. --- OPTIMIZE CLUSTERING --- 

2. --- USER INSPECTION AND DECISSION --- (as a comment, explain what the user should set, look for, and where it is found)

3. --- FINAL CLUSTERING --- (this is also where the cluster allocation is written to the `*_c.mat`file)

## Task 5

I moved the user-set final settings and choices to `#%% --- USER INSPECTION AND DECISSION ---`. Using `opt_gammas = [0.00005, 0.01, 1.0, 10]` to account for the Bayesian optimization space. I set `cluster_method = "agglo_ward"` and use `method=cluster_method,`. The clustering should print which method is used. I do not see the Bayesian optimization of gamma in action and also not where I could enable it. If grid search in `--- OPTIMIZE CLUSTERING ---` informs the Bayesian optimization in `--- FINAL CLUSTERING ---`  keep and use it. Otherwise I'd like to have an option to use either grid search or Bayesian optimization at the `--- OPTIMIZE CLUSTERING ---`step. If Bayesian optimization of gamma will be used at the optimization step already, we get only one optimal gamma for every k. Choose the method based on computational efficiency and getting the best possible seperation between clusters. Avoid redundant runs. Using `final_k_choices = (2, 3, 4)` is illogical because the user exactly defines one final `k`: ensure this functionality and logic.

Since 

```python
GAMMA_SPACE = Real(5e-5, 1e1, prior="log-uniform", name="gamma")
RANDOM_STATE_GLOBAL = 42 
```

are set in `dtw_clustering.py` with `gamma_space_bounds = (5e-5, 1e1)` and `random_state = 42` , resp., ensure to remove them from `clustering.py` but keep the code working

Here are the latest outputs of the run:

```
%runcell -n '--- OPTIMIZE CLUSTERING ---' C:/GitHub/c3dBox/Clustering/dtw_clustering.py
Clustering data prepared with shape: (200, 101, 3)

=== Starting Grid Search ===

--- Grid Search: k=None, gamma=0.1 ---
Calculating distance matrix for gamma=0.1

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=0.1
  Adjusting soft-DTW matrix: subtracting min value (-10.3867) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 69.58 seconds.
Running Agglomerative (Ward) with Soft-DTW gamma=0.1
Dendrogram saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\dendrogram_grid_softdtw_gamma0.1.pdf'
  k=2 -> silhouette=0.2167

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 35, 2: 165}
  Distance Matrix Computation Time: 69.58 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.2167
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 1.7334
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        43.800035  21.392458  5.463782   99.724833
    2        50.409750  20.978017  2.883941  168.142944
         mean      std       min        max
    64.445738 27.02154 12.293251 173.369791
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g0.1_k2_gamma0.1.pdf'
  k=3 -> silhouette=0.1706

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 3
  Cluster Distribution: {1: 35, 2: 25, 3: 140}
  Distance Matrix Computation Time: 69.58 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1706
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 1.6828
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        43.800035  21.392458  5.463782   99.724833
    2        44.995246  19.020537  5.060482   99.954393
    3        47.224638  18.892990  2.883941  147.721517
         mean       std       min        max
    62.665634 25.909616 12.293251 173.369791
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g0.1_k3_gamma0.1.pdf'
  k=4 -> silhouette=0.1560

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 4
  Cluster Distribution: {1: 35, 2: 25, 3: 50, 4: 90}
  Distance Matrix Computation Time: 69.58 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1560
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 1.9848
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        43.800035  21.392458  5.463782   99.724833
    2        44.995246  19.020537  5.060482   99.954393
    3        35.807253  18.411836  2.883941   99.181782
    4        45.569899  18.660437  3.982926  147.721517
         mean       std       min        max
    59.117833 24.071711 12.293251 173.369791
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g0.1_k4_gamma0.1.pdf'

--- Grid Search: k=None, gamma=1.0 ---
Calculating distance matrix for gamma=1.0

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=1.0
  Adjusting soft-DTW matrix: subtracting min value (-151.7771) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 72.07 seconds.
Running Agglomerative (Ward) with Soft-DTW gamma=1.0
Dendrogram saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\dendrogram_grid_softdtw_gamma1.0.pdf'
  k=2 -> silhouette=0.2292

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 45, 2: 155}
  Distance Matrix Computation Time: 72.07 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.2292
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.6852
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        76.156612  34.114263  11.404294  207.763048
    2        67.716385  24.481172   6.343437  192.404040
         mean       std       min        max
    91.955895 35.164688 22.206275 221.705714
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g1.0_k2_gamma1.0.pdf'
  k=3 -> silhouette=0.1098

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 3
  Cluster Distribution: {1: 45, 2: 56, 3: 99}
  Distance Matrix Computation Time: 72.07 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1098
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.6325
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        76.156612  34.114263  11.404294  207.763048
    2        50.589694  21.379491   6.343437  129.516157
    3        68.130946  26.105154   7.008645  192.404040
         mean       std       min        max
    83.167671 31.503845 22.206275 221.705714
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g1.0_k3_gamma1.0.pdf'
  k=4 -> silhouette=0.1303

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 4
  Cluster Distribution: {1: 14, 2: 31, 3: 56, 4: 99}
  Distance Matrix Computation Time: 72.07 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1303
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.0977
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        72.550904  26.783238  12.848334  129.426080
    2        58.111343  26.662348  11.404294  127.388379
    3        50.589694  21.379491   6.343437  129.516157
    4        68.130946  26.105154   7.008645  192.404040
         mean       std       min        max
    83.605899 31.588946 22.206275 221.705714
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g1.0_k4_gamma1.0.pdf'

--- Grid Search: k=None, gamma=10.0 ---
Calculating distance matrix for gamma=10.0

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=10.0
  Adjusting soft-DTW matrix: subtracting min value (-1676.5244) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 72.69 seconds.
Running Agglomerative (Ward) with Soft-DTW gamma=10.0
Dendrogram saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\dendrogram_grid_softdtw_gamma10.0.pdf'
  k=2 -> silhouette=0.1555

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 113, 2: 87}
  Distance Matrix Computation Time: 72.69 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1555
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 3.3794
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        97.683448  38.193116  11.644873  278.179560
    2        93.586044  40.789965  10.031116  258.126712
          mean       std       min       max
    114.920425 44.668125 33.294366 314.16773
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g10.0_k2_gamma10.0.pdf'
  k=3 -> silhouette=0.1162

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 3
  Cluster Distribution: {1: 113, 2: 35, 3: 52}
  Distance Matrix Computation Time: 72.69 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1162
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.6508
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        97.683448  38.193116  11.644873  278.179560
    2        93.690570  36.333571  25.922595  202.283631
    3        71.407630  31.815720  10.031116  179.602005
          mean       std       min       max
    114.106573 44.069955 33.294366 314.16773
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g10.0_k3_gamma10.0.pdf'
  k=4 -> silhouette=0.1310

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 4
  Cluster Distribution: {1: 31, 2: 82, 3: 35, 4: 52}
  Distance Matrix Computation Time: 72.69 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1310
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.3929
Calculating DTW distance statistics...
                  mean        std        min         max
    Cluster                                             
    1        94.798697  35.203419  13.982951  202.782458
    2        87.658980  34.894849  11.644873  216.552163
    3        93.690570  36.333571  25.922595  202.283631
    4        71.407630  31.815720  10.031116  179.602005
          mean       std       min       max
    113.605275 43.165852 33.294366 314.16773
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_grid_g10.0_k4_gamma10.0.pdf'

=== Grid Search Complete ===
   k  gamma dm_method  silhouette_dtw  davies_bouldin  n_clusters_found  dtw_comp_time_s
0  2    0.1   softdtw        0.216714        1.733414                 2        69.579064
1  3    0.1   softdtw        0.170574        1.682799                 3        69.579064
2  4    0.1   softdtw        0.156040        1.984822                 4        69.579064
3  2    1.0   softdtw        0.229165        2.685169                 2        72.068980
4  3    1.0   softdtw        0.109796        2.632519                 3        72.068980
5  4    1.0   softdtw        0.130281        2.097679                 4        72.068980
6  2   10.0   softdtw        0.155485        3.379404                 2        72.689677
7  3   10.0   softdtw        0.116170        2.650779                 3        72.689677
8  4   10.0   softdtw        0.131015        2.392865                 4        72.689677
Grid search results saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx' (sheet: grid_search_results)
```

## Task 6

This is the latest output using optimization_mode = "bayes":

```
%runcell -n '--- OPTIMIZE CLUSTERING ---' C:/GitHub/c3dBox/Clustering/dtw_clustering.py
Clustering data prepared with shape: (200, 101, 3)

=== Starting Grid Search ===

=== Running Single Clustering (method=agglo_ward, distance=softdtw, gamma=5e-05, random_state=42) ===

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=5e-05
  Adjusting soft-DTW matrix: subtracting min value (-0.0000) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 73.76 seconds.

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 16, 2: 184}
  Distance Matrix Computation Time: 73.76 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.4095
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 0.7794
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        43.661208  18.419361  1.906568  149.621151
         mean       std       min        max
    77.788413 25.029681 22.348534 165.302425
-------------------------------------------------

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 3
  Cluster Distribution: {1: 16, 2: 66, 3: 118}
  Distance Matrix Computation Time: 73.76 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1727
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.2320
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        31.546880  15.512902  2.255871   97.986977
    3        42.987959  17.770196  1.906568  134.605282
         mean       std      min        max
    55.877306 24.362218 8.557905 165.302425
-------------------------------------------------

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 4
  Cluster Distribution: {1: 16, 2: 66, 3: 73, 4: 45}
  Distance Matrix Computation Time: 73.76 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1932
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 1.7646
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        31.546880  15.512902  2.255871   97.986977
    3        36.689373  13.276757  2.637360   88.669767
    4        38.722006  20.227006  1.906568  115.331982
         mean       std      min        max
    54.338779 23.186757 8.557905 165.302425
-------------------------------------------------
-> Recommended cluster count (highest silhouette): k=2

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 16, 2: 184}
  Distance Matrix Computation Time: 73.76 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.4095
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 0.7794
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        43.661208  18.419361  1.906568  149.621151
         mean       std       min        max
    77.788413 25.029681 22.348534 165.302425
-------------------------------------------------
Identified 2 clusters (excluding noise if any). Distribution: {1: 16, 2: 184}
Saving results summary to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx'...
Clustering information written to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx'.
```

I expected to see a `=== BAYES OPTIMIZATION COMPLETE ===` table similar to `=== GRID SEARCH COMPLETE ===` with `k`, `gamma_opt` and seperation stats but this was not shown. No figures were saved but the user needs to inspect them for the next step. I'd also expect to see Bayes-results printed during or after the Bayes optimization steps. Ensure to run Bayes optimization if it is selected, print the final table, and save the plots. The user should then select the desired k, which in this case comes with a known optimal gamma.



## Output Task 8

```
%runcell -n '--- OPTIMIZE CLUSTERING ---' C:/GitHub/c3dBox/Clustering/dtw_clustering.py
[optimize] starting cluster_timeseries optimization_mode=bayes, opt_gammas=[5e-05, 0.01, 1.0, 10], opt_k_range=(2, 3, 4)
Clustering data prepared with shape: (200, 101, 3)

=== Starting Grid Search ===

=== Running Single Clustering (method=agglo_ward, distance=softdtw, gamma=5e-05, random_state=42) ===

--- Precomputing Distance Matrix (SOFTDTW) ---
Calculating pairwise distances for 200 samples...
  Using soft-DTW with gamma=5e-05
  Adjusting soft-DTW matrix: subtracting min value (-0.0000) to ensure non-negativity.
  Adjusting soft-DTW matrix: setting diagonal to 0.
Pairwise distances calculated and adjusted (if softDTW) in 88.15 seconds.
Dendrogram saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\dendrogram_softdtw_gamma5e-05.pdf'

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 16, 2: 184}
  Distance Matrix Computation Time: 88.15 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.4095
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 0.7794
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        43.661208  18.419361  1.906568  149.621151
         mean       std       min        max
    77.788413 25.029681 22.348534 165.302425
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_k2_gamma5e-05.pdf'

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 3
  Cluster Distribution: {1: 16, 2: 66, 3: 118}
  Distance Matrix Computation Time: 88.15 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1727
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 2.2320
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        31.546880  15.512902  2.255871   97.986977
    3        42.987959  17.770196  1.906568  134.605282
         mean       std      min        max
    55.877306 24.362218 8.557905 165.302425
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_k3_gamma5e-05.pdf'

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 4
  Cluster Distribution: {1: 16, 2: 66, 3: 73, 4: 45}
  Distance Matrix Computation Time: 88.15 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.1932
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 1.7646
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        31.546880  15.512902  2.255871   97.986977
    3        36.689373  13.276757  2.637360   88.669767
    4        38.722006  20.227006  1.906568  115.331982
         mean       std      min        max
    54.338779 23.186757 8.557905 165.302425
-------------------------------------------------
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_k4_gamma5e-05.pdf'
-> Recommended cluster count (highest silhouette): k=2

--- Calculating Clustering Metrics from Precomputed Distances ---
  Number of samples: 200
  Number of clusters found: 2
  Cluster Distribution: {1: 16, 2: 184}
  Distance Matrix Computation Time: 88.15 seconds
Calculating DTW Silhouette Score...
  DTW Silhouette Score: 0.4095
Calculating Davies-Bouldin Index (DTW-based)...
  Davies-Bouldin (DTW): 0.7794
Calculating DTW distance statistics...
                  mean        std       min         max
    Cluster                                            
    1        51.300765  22.128465  4.681827  107.793364
    2        43.661208  18.419361  1.906568  149.621151
         mean       std       min        max
    77.788413 25.029681 22.348534 165.302425
-------------------------------------------------
Identified 2 clusters (excluding noise if any). Distribution: {1: 16, 2: 184}
Clustering figure saved to 'C:\Data_Local\PRO_checked_mat\clustering_results\clustering_softdtw_k2_gamma5e-05.pdf'
Saving results summary to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx'...
Clustering information written to 'C:\Data_Local\PRO_checked_mat\clustering_results\knee_jrl_extraction_20251201_122818_clustering.xlsx'.
[optimize] opt_run keys: ['prepared_data', 'cluster_labels', 'labels_by_k', 'best_k', 'hierarchy_metrics', 'metrics', 'feature_names', 'meta', 'cluster_plots', 'distance_matrix']
[optimize] grid_results rows: None
[optimize] bayes_best_gamma_per_k: {}
[optimize] bayes_results_df rows: None
```
