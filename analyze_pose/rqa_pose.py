import os
import sys
# Add a folder to the system path so we can use custom analysis tools
sys.path.append(os.path.abspath("/Users/cartersale/Library/CloudStorage/OneDrive-MacquarieUniversity/Research/Projects/2025_MATBExp4/03_Analysis"))  
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from scipy.signal import detrend as scipy_detrend
import math
from rqa.utils import rqa_utils_cpp, norm_utils
import seaborn as sns

# ── SETTINGS ───────────────────────────────────────────────────────
ROOT_DIR        = "data/pose/baseline_pose"      # Where the data files are
OUT_CSV         = "data/rqa/baseline_pose_rqa.csv"  # Where to save results

SAMPLE_RATE_HZ  = 60                      # How many data points per second
WIN_SECONDS     = 60                      # Length of each analysis window (seconds)
OVERLAP_FRAC    = 0.5                     # Windows overlap by 50%
TARGET_COLUMNS  = [                       # Which measurements to analyze
    "avg_pupil_magnitude",
    "center_face_magnitude",
    "head_rotation_angle",
    "mouth_dist",
    "center_face_x",
    "center_face_y",
    "avg_pupil_x",
    "avg_pupil_y",
    "blink_dist"
]

# Parameters for the RQA analysis
PARAMS = {                                
    "norm": 1,           # How to normalize the data
    "eDim": 4,           # Embedding dimension
    "tLag": 20,          # Time lag
    "rescaleNorm": 1,    # Whether to rescale distances
    "radius": 0.15,      # Threshold for recurrence
    "tw": 2,             # Ignore lines close to the diagonal
    "minl": 4,           # Minimum line length
}

def compute_rqa(series, p):
    """
    Run Recurrence Quantification Analysis (RQA) on a single measurement series.
    Returns results and an error code.
    """
    data_norm = norm_utils.normalize_data(series, p["norm"])
    ds        = rqa_utils_cpp.rqa_dist(data_norm, data_norm,
                                       dim=p["eDim"], lag=p["tLag"])
    _, rs, _, err = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=p["rescaleNorm"],
        rad=p["radius"],
        diag_ignore=p["tw"],
        minl=p["minl"],
        rqa_mode="auto",
    )
    return rs, err

def sliding_windows(total_len, win_len, step):
    """
    Divide the data into overlapping windows for analysis.
    Yields start and end indices for each window.
    """
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step

def main():
    results = []  # Store results for all files and windows
    win_len  = WIN_SECONDS * SAMPLE_RATE_HZ  # How many data points in each window
    step     = int(win_len * (1 - OVERLAP_FRAC))  # How far to move for each new window

    # List all CSV files in the data folder
    csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))

    for csv_file in tqdm(csv_files, desc="Files"):
        # Get participant ID and condition from the filename
        pid, cond = os.path.splitext(csv_file)[0].split("_", 1)
        df = pd.read_csv(os.path.join(ROOT_DIR, csv_file))

        # Only keep columns that are both requested and present in the file
        cols_present = [c for c in TARGET_COLUMNS if c in df.columns]
        if not cols_present:
            tqdm.write(f"⚠️  {csv_file}: none of TARGET_COLUMNS found, skipping.")
            continue

        for col in cols_present:
            series = df[col].values
            total_n = len(series)

            # Divide the data into overlapping windows and analyze each window
            for w_idx, (s, e) in enumerate(
                    tqdm(sliding_windows(total_n, win_len, step),
                         leave=False, desc=f"{pid}-{cond}:{col}")):
                rs, err = compute_rqa(series[s:e], PARAMS)
                if err != 0:
                    tqdm.write(f"‼️  {pid}_{cond} window {w_idx} (col {col}) error code {err}")
                    continue

                # Save the results for this window
                results.append({
                    "participant":        pid,
                    "condition":          cond,
                    "column":             col,
                    "window_index":         w_idx,
                    "window_start":       s,
                    "window_end":         e,
                    "perc_recur":         float(rs["perc_recur"]),       # % recurrence
                    "perc_determ":        float(rs["perc_determ"]),      # % determinism
                    "maxl_found":         float(rs["maxl_found"]),       # Longest line
                    "mean_line_length":   float(rs["mean_line_length"]), # Average line length
                    "std_line_length":    float(rs["std_line_length"]),  # Std dev of line length
                    "entropy":            float(rs["entropy"]),          # Entropy
                    "laminarity":         float(rs["laminarity"]),       # Laminarity
                    "trapping_time":      float(rs["trapping_time"]),    # Trapping time
                    "vmax":               float(rs["vmax"]),             # Vertical max
                    "divergence":         float(rs["divergence"]),       # Divergence
                    "trend_lower_diag":   float(rs["trend_lower_diag"]), # Trend lower diagonal
                    "trend_upper_diag":   float(rs["trend_upper_diag"]), # Trend upper diagonal
                })

    # ── SAVE RESULTS ────────────────────────────────────────────────
    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        print(f"\n✅  RQA results written to {OUT_CSV}  ({len(results)} rows)")
    else:
        print("\n😐  No results generated.")

if __name__ == "__main__":
    # Run the main function if this file