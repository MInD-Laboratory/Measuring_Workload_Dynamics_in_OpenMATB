import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- AMI ----------------
def ami(timeseries, min_lag, max_lag):
    # 1D numpy
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        x = timeseries.values.flatten()
    elif isinstance(timeseries, np.ndarray):
        x = timeseries.flatten()
    else:
        raise ValueError("timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = x[np.isfinite(x)]
    length = len(x)
    if length < 2*max_lag:
        return None  # too short for requested lags

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(min_lag, max_lag + 1) if min_lag < max_lag else np.arange(0, 51)
    else:
        print("warning: max_lag exceeds n/2-1; setting to n/2-1")
        lag = np.arange(0, length // 2)

    # scale to [0,1] (guard against constant series)
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.column_stack((lag, np.zeros_like(lag, dtype=float)))
    x = (x - lo) / (hi - lo)

    ami_values = np.zeros(len(lag), dtype=float)
    for i in tqdm(range(len(lag)), desc='Processing AMI', leave=False):
        N = length - lag[i]
        if N <= 2:
            ami_values[i] = 0.0
            continue
        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.var(x[:N], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        x1 = x[:N]
        x2 = x[lag[i]:]
        s = 0.0
        for k1 in range(1, k + 1):
            x1_lo, x1_hi = (k1 - 1) / k, k1 / k
            mask1 = (x1 > x1_lo) & (x1 <= x1_hi)
            if not mask1.any():
                continue
            px1 = mask1.sum() / N
            for k2 in range(1, k + 1):
                x2_lo, x2_hi = (k2 - 1) / k, k2 / k
                mask2 = (x2 > x2_lo) & (x2 <= x2_hi)
                if not mask2.any():
                    continue
                px2 = mask2.sum() / N
                pxy = (mask1 & mask2).sum() / N
                if pxy > 0:
                    s += pxy * np.log2(pxy / (px1 * px2))
        ami_values[i] = s

    return np.column_stack((lag, ami_values))

# ---------------- Params ----------------
MODE = "experimental"  # "experimental" or "baseline"
ROOT_DIR = f"data/preprocessed_pose/{MODE}_pose"
N_FILES_TO_USE = 100
MIN_LAG = 1
MAX_LAG = 100

# Explicit signals (we'll drop *_prob below)
TARGET_COLUMNS = [
    "center_face_x","center_face_y","center_face_prob",
    "left_eye_x","left_eye_y","left_eye_prob",
    "right_eye_x","right_eye_y","right_eye_prob",
    "left_pupil_x","left_pupil_y","left_pupil_prob",
    "right_pupil_x","right_pupil_y","right_pupil_prob",
    "center_face_magnitude","left_eye_magnitude","right_eye_magnitude",
    "left_pupil_magnitude","right_pupil_magnitude",
    "avg_pupil_x","avg_pupil_y","avg_pupil_magnitude",
    "blink_dist","head_rotation_angle","mouth_dist",
]

# drop prob columns
TARGET_COLUMNS = [c for c in TARGET_COLUMNS if not c.endswith("_prob")]

# ---------------- Run across files/columns → one curve ----------------
csv_files = sorted([f for f in os.listdir(ROOT_DIR) if f.endswith(".csv")])[:N_FILES_TO_USE]

all_curves = []     # list of AMI value arrays
lags_ref = None
used = 0

for file in tqdm(csv_files, desc="Files"):
    df = pd.read_csv(os.path.join(ROOT_DIR, file))
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(series) < 2*MAX_LAG:
            continue
        res = ami(series, MIN_LAG, MAX_LAG)
        if res is None or res.shape[0] == 0:
            continue
        if lags_ref is None:
            lags_ref = res[:, 0]
        # ensure same lag vector length
        if res[:, 0].shape != lags_ref.shape or not np.allclose(res[:, 0], lags_ref):
            # skip mismatched lag vectors (shouldn't happen, but be safe)
            continue
        all_curves.append(res[:, 1])
        used += 1

if not all_curves:
    raise RuntimeError("No usable series found (check columns, NaNs, or MAX_LAG vs series length).")

# Average across all selected columns/files → one curve
stacked = np.vstack(all_curves)
ami_avg = np.nanmean(stacked, axis=0)

# ---------------- Plot ----------------
plt.figure()
plt.plot(lags_ref, ami_avg, marker='o', linewidth=1)
plt.title(f"AMI Curve (Averaged across columns & files)\nMODE={MODE}")
plt.xlabel("Time Lag (samples)")
plt.ylabel("Average Mutual Information")
plt.grid(True)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
out_path = f"figures/ami_curve_{MODE}_ALL_SIGNALS.svg"
plt.savefig(out_path)
plt.show()
print(f"[OK] Averaged over {used} series | saved -> {out_path}")
