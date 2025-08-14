import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------- Cross-AMI -----------------------------
def _safe_minmax_scale(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=float)
    lo, hi = np.nanmin(a[finite]), np.nanmax(a[finite])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=float)
    out = (a - lo) / (hi - lo)
    out[~finite] = np.nan
    return out

def cross_ami(timeseries1, timeseries2, min_lag, max_lag):
    """
    Cross Average Mutual Information (X-AMI) between x and y.

    Convention here: for lag = L, we compare x[0 : N-L] with y[L : N].
    If the X-AMI peaks at lag L>0, it suggests x leads y by ~L samples.
    """
    # Flatten to 1D numpy
    if isinstance(timeseries1, (pd.Series, pd.DataFrame)):
        timeseries1 = timeseries1.values.flatten()
    if isinstance(timeseries2, (pd.Series, pd.DataFrame)):
        timeseries2 = timeseries2.values.flatten()
    if not isinstance(timeseries1, np.ndarray) or not isinstance(timeseries2, np.ndarray):
        raise ValueError("Both inputs must be NumPy arrays or Pandas Series/DataFrame.")

    x = np.asarray(timeseries1).astype(float)
    y = np.asarray(timeseries2).astype(float)

    # drop NaNs consistently (keep overlapping finite region)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    length = min(len(x), len(y))
    if length == 0:
        return np.column_stack((np.arange(0), np.zeros(0)))

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(min_lag, max_lag + 1) if min_lag < max_lag else np.arange(0, 51)
    else:
        print("warning: max_lag exceeds n/2-1; setting to n/2-1")
        lag = np.arange(0, length // 2)

    # normalize each to [0,1] (robust to constant signals)
    x = _safe_minmax_scale(x)
    y = _safe_minmax_scale(y)

    ami_values = np.zeros(len(lag), dtype=float)

    for i in tqdm(range(len(lag)), desc="Processing Cross-AMI", leave=False):
        L = int(lag[i])
        # number of aligned samples for this lag
        N = length - L
        if N <= 2:
            ami_values[i] = 0.0
            continue

        # number of equiprobable bins (Scott-esque, like your AMI)
        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.nanvar(x[:N], ddof=1) == 0 or np.nanvar(y[L:], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        xw = x[:N]
        yw = y[L:]

        # bin edges in [0,1]
        ami_sum = 0.0
        for k1 in range(1, k + 1):
            x_lo, x_hi = (k1 - 1) / k, k1 / k
            mask_x = (xw > x_lo) & (xw <= x_hi)
            if not mask_x.any():
                continue
            px1 = mask_x.sum() / N

            for k2 in range(1, k + 1):
                y_lo, y_hi = (k2 - 1) / k, k2 / k
                mask_y = (yw > y_lo) & (yw <= y_hi)
                if not mask_y.any():
                    continue
                py2 = mask_y.sum() / N

                # joint (aligned)
                ppp = (mask_x & mask_y).sum() / N
                if ppp > 0 and px1 > 0 and py2 > 0:
                    ami_sum += ppp * np.log2(ppp / (px1 * py2))

        ami_values[i] = ami_sum

    return np.column_stack((lag, ami_values))

# ----------------------------- Parameters -----------------------------
MODE = "experimental"  # "experimental" or "baseline"
ROOT_DIR = f"data/preprocessed_pose/{MODE}_pose"

# Signal pairs for Cross-AMI (x leads y if peak at positive lag)
TARGET_PAIRS = [
    ("center_face_magnitude", "avg_pupil_magnitude"),
]

N_FILES_TO_USE = 100
MIN_LAG = 1
MAX_LAG = 100

# -------------------- Run Cross-AMI across files ---------------------
pair_results = {pair: [] for pair in TARGET_PAIRS}
csv_files = sorted([f for f in os.listdir(ROOT_DIR) if f.endswith(".csv")])[:N_FILES_TO_USE]

for file in tqdm(csv_files, desc="Files"):
    df = pd.read_csv(os.path.join(ROOT_DIR, file))

    for (xcol, ycol) in TARGET_PAIRS:
        if xcol not in df.columns or ycol not in df.columns:
            continue

        # drop NaNs per column, then re-align (inner join on index)
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        xy = pd.concat([x, y], axis=1).dropna()
        if len(xy) < 2 * MAX_LAG:
            continue

        res = cross_ami(xy.iloc[:, 0].values, xy.iloc[:, 1].values, MIN_LAG, MAX_LAG)
        if res.shape[0] == 0:
            continue
        pair_results[(xcol, ycol)].append(res[:, 1])  # only AMI values

# -------------------- Average Cross-AMI per pair ---------------------
averaged = {}
lags = None
for pair, arrs in pair_results.items():
    if not arrs:
        continue
    stacked = np.vstack(arrs)  # shape: n_files x n_lags
    averaged[pair] = np.nanmean(stacked, axis=0)
    if lags is None:
        # use lags from last computed res; they’re consistent by construction
        lags = cross_ami(np.arange(2*MAX_LAG), np.arange(2*MAX_LAG), MIN_LAG, MAX_LAG)[:, 0]

# ----------------------------- Plot ---------------------------------
os.makedirs("figures", exist_ok=True)

for (xcol, ycol), ami_curve in averaged.items():
    plt.figure()
    plt.plot(lags, ami_curve, marker="o", linewidth=1)
    plt.title(f"Cross-AMI (Averaged)\n{xcol} → {ycol}")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Cross Average Mutual Information")
    plt.grid(True)
    plt.tight_layout()
    out = f"figures/cross_ami_{xcol}_vs_{ycol}.svg".replace(" ", "_")
    plt.savefig(out)
    plt.show()
    print(f"[OK] Saved {out}")
