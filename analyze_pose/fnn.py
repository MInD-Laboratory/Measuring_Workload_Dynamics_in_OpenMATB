import os
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import zscore
from scipy.spatial import KDTree


# Embedding the time series based on dimension and lag
def embed_time_series(data, embedding_dim, lag):
    data_length = len(data) - (embedding_dim - 1) * lag
    embedded = np.zeros((data_length, embedding_dim))
    for c in range(embedding_dim):
        embedded[:, c] = data[c * lag : c * lag + data_length]
    return embedded

def fnn(timeseries, tlag, min_dimension, max_dimension):
    # Ensure the input is a NumPy array
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        timeseries = timeseries.values.flatten()  # Convert to 1D NumPy array if it's a Series or DataFrame
    elif not isinstance(timeseries, np.ndarray):
        raise ValueError("Input timeseries must be a NumPy array or Pandas Series/DataFrame")
    
    # Preprocess the time series
    time_series = np.array(timeseries)

    percent = np.ones(max_dimension - min_dimension + 1)

    # Mean and radius of the attractor
    mean_x = np.mean(time_series)
    Ra = np.sqrt(np.mean((time_series - mean_x) ** 2))

    # Embedding dimensions to check
    de = np.arange(min_dimension, max_dimension + 1)
    
    for c in tqdm(de, desc="Processing FNN"):
        number_false = 0
        max_l = len(time_series) - c * tlag

        # Embed the time series
        curr_embedding = embed_time_series(time_series, c, tlag)

        # Build KDTree for nearest neighbor search
        tree = KDTree(curr_embedding)

        for c2 in range(max_l):
            # Search for the nearest neighbor
            dist, NN = tree.query(curr_embedding[c2], k=2)  # Nearest neighbor excluding itself
            nearest_d = dist[1]  # Exclude self, take the second closest
            NN = NN[1]

            if NN >= max_l:
                # If NN is beyond available data, assume not false
                test_stat1 = 0
                test_stat2 = 0
            else:
                if nearest_d == 0:
                    test_stat1 = 1
                else:
                    test_stat1 = np.abs(time_series[c2 + c * tlag] - time_series[NN + c * tlag]) / nearest_d
                
                test_stat2 = np.abs(time_series[c2 + c * tlag] - time_series[NN + c * tlag]) / Ra

            # Use Abarbanel's criteria: test_stat1 >= 15 or test_stat2 >= 2
            number_false += ((test_stat1 >= 15) | (test_stat2 >= 2))

        percent[c - min_dimension] = number_false / max_l

    return de, percent * 100

# ── Parameters ─────────────────────────────────────────────────────
ROOT_DIR       = ""
N_FILES_TO_USE = 100          # cap files to scan
N_SERIES       = 100          # how many random (file,col) pairs
MIN_DIM, MAX_DIM = 1, 10
TAU            = 20
SEED           = 42
FIG_DIR        = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
# ── BUILD POOL OF VALID (file, col) PAIRS ──────────────────────────
csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))[:N_FILES_TO_USE]
pool = []      # (path, col)

for fname in tqdm(csv_files, desc="Scanning files"):
    fpath = os.path.join(ROOT_DIR, fname)
    head  = pd.read_csv(fpath, nrows=5)
    cols  = [c for c in head.select_dtypes(include=[np.number]).columns
             if not c.endswith("_prob")]
    for col in cols:
        npts = pd.read_csv(fpath, usecols=[col])[col].dropna().shape[0]
        if npts >= (MAX_DIM + 1) * TAU:
            pool.append((fpath, col))

if len(pool) < N_SERIES:
    raise RuntimeError(f"Only {len(pool)} usable series; need {N_SERIES}")

random.seed(SEED)
sampled = random.sample(pool, N_SERIES)

# ── RUN FNN ON THE SAMPLED SERIES ──────────────────────────────────
fnn_curves = []      # list of 1-D arrays (len = MAX_DIM-MIN_DIM+1)

for fpath, col in tqdm(sampled, desc="FNN series"):
    s = pd.read_csv(fpath, usecols=[col])[col].dropna().values
    dims, vals = fnn(s, tlag=TAU,
                     min_dimension=MIN_DIM,
                     max_dimension=MAX_DIM)
    fnn_curves.append(vals)

fnn_mat   = np.vstack(fnn_curves)        # shape → (100, n_dims)
mean_fnn  = fnn_mat.mean(axis=0)
# sem_fnn   = fnn_mat.std(axis=0, ddof=1) / math.sqrt(fnn_mat.shape[0])

# ── PLOT MEAN ± 1 SEM ──────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(dims, mean_fnn, marker="o", label="Mean %FNN")
# ax.fill_between(dims, mean_fnn - sem_fnn, mean_fnn + sem_fnn,
#                 alpha=0.3, label="± 1 SEM")
ax.set_xlabel("Embedding Dimension")
ax.set_ylabel("% False Nearest Neighbours")
ax.set_title(f"FNN (100 random series across {len(csv_files)} files)")
ax.grid(True)
ax.legend()

png = os.path.join(FIG_DIR, "fnn_avg.png")
svg = os.path.join(FIG_DIR, "fnn_avg.svg")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(svg,           bbox_inches="tight")
print(f"Figure saved → {png} & {svg}")
