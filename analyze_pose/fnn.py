import os
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import zscore
from scipy.spatial import KDTree

# This function rearranges the time series data into a format needed for analysis.
# It creates "embedded" versions of the data, which help reveal hidden patterns.
def embed_time_series(data, embedding_dim, lag):
    data_length = len(data) - (embedding_dim - 1) * lag
    embedded = np.zeros((data_length, embedding_dim))
    for c in range(embedding_dim):
        embedded[:, c] = data[c * lag : c * lag + data_length]
    return embedded

# This function calculates the percentage of "False Nearest Neighbours" (FNN) for different dimensions.
# FNN helps decide how many variables are needed to describe the system's behavior.
def fnn(timeseries, tlag, min_dimension, max_dimension):
    # Make sure the input data is in the right format
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        timeseries = timeseries.values.flatten()  # Convert to a simple array
    elif not isinstance(timeseries, np.ndarray):
        raise ValueError("Input timeseries must be a NumPy array or Pandas Series/DataFrame")
    
    time_series = np.array(timeseries)

    percent = np.ones(max_dimension - min_dimension + 1)

    # Calculate the average value and spread of the data
    mean_x = np.mean(time_series)
    Ra = np.sqrt(np.mean((time_series - mean_x) ** 2))

    # List of dimensions to test
    de = np.arange(min_dimension, max_dimension + 1)
    
    # For each dimension, check how many points have "false" neighbors
    for c in tqdm(de, desc="Processing FNN"):
        number_false = 0
        max_l = len(time_series) - c * tlag

        # Rearrange the data for this dimension
        curr_embedding = embed_time_series(time_series, c, tlag)

        # Build a KDTree to quickly find nearest neighbors
        tree = KDTree(curr_embedding)

        for c2 in range(max_l):
            # Find the nearest neighbor (excluding itself)
            dist, NN = tree.query(curr_embedding[c2], k=2)
            nearest_d = dist[1]  # Distance to the nearest neighbor
            NN = NN[1]           # Index of the nearest neighbor

            if NN >= max_l:
                # If the neighbor is out of bounds, skip
                test_stat1 = 0
                test_stat2 = 0
            else:
                # Calculate how different the neighbor is in the next dimension
                if nearest_d == 0:
                    test_stat1 = 1
                else:
                    test_stat1 = np.abs(time_series[c2 + c * tlag] - time_series[NN + c * tlag]) / nearest_d
                
                test_stat2 = np.abs(time_series[c2 + c * tlag] - time_series[NN + c * tlag]) / Ra

            # If either test is true, count as a false neighbor
            number_false += ((test_stat1 >= 15) | (test_stat2 >= 2))

        percent[c - min_dimension] = number_false / max_l

    # Return the dimensions tested and the percentage of false neighbors for each
    return de, percent * 100

# ── Parameters ─────────────────────────────────────────────────────
ROOT_DIR       = "data/preprocessed_pose/experimental_pose"             # Folder with data files (set this to your data location), change to data/preprocessed_pose/baseline_pose if needed
N_FILES_TO_USE = 100            # Only look at up to 100 files
N_SERIES       = 100            # Analyze 100 random data series
MIN_DIM, MAX_DIM = 1, 10        # Range of dimensions to test
TAU            = 20             # Time lag to use
SEED           = 42             # Random seed for reproducibility
FIG_DIR        = "figs/ami_fnn"      # Where to save the plots
os.makedirs(FIG_DIR, exist_ok=True)

# ── BUILD POOL OF VALID (file, col) PAIRS ──────────────────────────
# Find all CSV files in the folder
csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))[:N_FILES_TO_USE]
pool = []      # List of (file path, column name) pairs

for fname in tqdm(csv_files, desc="Scanning files"):
    fpath = os.path.join(ROOT_DIR, fname)
    head  = pd.read_csv(fpath, nrows=5)  # Read just the first few rows to get column info
    # Only use columns with numbers, and skip probability columns
    cols  = [c for c in head.select_dtypes(include=[np.number]).columns
             if not c.endswith("_prob")]
    for col in cols:
        # Only use columns with enough data points
        npts = pd.read_csv(fpath, usecols=[col])[col].dropna().shape[0]
        if npts >= (MAX_DIM + 1) * TAU:
            pool.append((fpath, col))

# If not enough usable series are found, stop the program
if len(pool) < N_SERIES:
    raise RuntimeError(f"Only {len(pool)} usable series; need {N_SERIES}")

random.seed(SEED)
sampled = random.sample(pool, N_SERIES)  # Pick 100 random (file, column) pairs

# ── RUN FNN ON THE SAMPLED SERIES ──────────────────────────────────
fnn_curves = []      # Store the FNN results for each series

for fpath, col in tqdm(sampled, desc="FNN series"):
    s = pd.read_csv(fpath, usecols=[col])[col].dropna().values
    dims, vals = fnn(s, tlag=TAU,
                     min_dimension=MIN_DIM,
                     max_dimension=MAX_DIM)
    fnn_curves.append(vals)

fnn_mat   = np.vstack(fnn_curves)        # Combine all results into one big table
mean_fnn  = fnn_mat.mean(axis=0)         # Average FNN percentage for each dimension
sem_fnn   = fnn_mat.std(axis=0, ddof=1) / math.sqrt(fnn_mat.shape[0])  # Standard error 

# ── PLOT MEAN ± 1 SEM ──────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(dims, mean_fnn, marker="o", label="Mean %FNN")

ax.set_xlabel("Embedding Dimension")
ax.set_ylabel("% False Nearest Neighbours")
ax.set_title(f"FNN (100 random series across {len(csv_files)} files)")
ax.grid(True)
ax.legend()

# Save the plot as PNG and SVG images
png = os.path.join(FIG_DIR, "fnn_avg.png")
svg = os.path.join(FIG_DIR, "fnn_avg.svg")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(svg,           bbox_inches="tight")
