import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rqa.utils import norm_utils, rqa_utils_cpp

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODE = "baseline"  # "experimental" or "baseline"
ROOT_DIR = f"data/preprocessed_pose/{MODE}_pose"
OUT_CSV  = f"data/rqa/{MODE}_pose_crqa_head_eye.csv"

SAMPLE_RATE_HZ = 60        # How many data points per second (frames per second)
WIN_SECONDS    = 60        # How long each analysis window is (in seconds)
OVERLAP_FRAC   = 0.5       # How much windows overlap (50%)

# Parameters for the analysis method
PARAMS = {
    "norm":        1,    # How to normalize the data
    "eDim":        4,    # Embedding dimension (for RQA)
    "tLag":        40,   # Time lag (for RQA)
    "rescaleNorm": 1,    # Whether to rescale distances
    "radius":      0.2,  # Threshold for recurrence
    "minl":        2,    # Minimum line length for analysis
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sliding_windows(total_len, win_len, step):
    # This function divides the data into overlapping windows for analysis
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step

def compute_cross_rqa(x, y, p):
    """Run cross-recurrence quantification analysis (CRQA) on two data series."""
    try:
        # Normalize both data series
        x_n = norm_utils.normalize_data(x, p["norm"])
        y_n = norm_utils.normalize_data(y, p["norm"])
        # Calculate distances between points in the two series
        ds  = rqa_utils_cpp.rqa_dist(x_n, y_n, dim=p["eDim"], lag=p["tLag"])

        # Calculate RQA statistics from the distance matrix
        _, rs, _, err = rqa_utils_cpp.rqa_stats(
            ds["d"],
            rescale=p["rescaleNorm"],
            rad=p["radius"],
            diag_ignore=0,
            minl=p["minl"],
            rqa_mode="cross",
        )
        return rs, err

    except RuntimeError as e:
        tqdm.write(f"⛔️ RuntimeError in cross-RQA: {e}")
        return None, -1

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    results = []  # Store results for all files and windows
    win_len = WIN_SECONDS * SAMPLE_RATE_HZ  # How many data points in each window
    step    = int(win_len * (1 - OVERLAP_FRAC))  # How far to move for each new window

    # List all CSV files in the data folder
    csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))
    for csv_file in tqdm(csv_files, desc="Files"):
        # Get participant ID and condition from the filename
        pid, cond = os.path.splitext(csv_file)[0].split("_", 1)
        # Read the data from the file
        df = pd.read_csv(os.path.join(ROOT_DIR, csv_file))

        # Make sure both required columns are present
        if not {"center_face_magnitude", "avg_pupil_magnitude"} <= set(df.columns):
            tqdm.write(f"⚠️  {csv_file}: missing pupil columns, skipping.")
            continue

        # Get the two data series to compare
        series1 = df["center_face_magnitude"].values
        series2 = df["avg_pupil_magnitude"].values
        total_n = len(series1)

        # Divide the data into overlapping windows and analyze each window
        for w_idx, (s, e) in enumerate(
                sliding_windows(total_n, win_len, step)):
            xw = series1[s:e]
            yw = series2[s:e]
            if not (np.isfinite(xw).all() and np.isfinite(yw).all()):
                continue
            rs, err = compute_cross_rqa(series1[s:e], series2[s:e], PARAMS)
            if err != 0:
                tqdm.write(f"‼️  {pid}_{cond} window {w_idx} error {err}")
                continue

            # Save the results for this window
            results.append({
                "participant":      pid,
                "condition":        cond,
                "window_index":     w_idx,
                "start_sample":     s,
                "end_sample":       e,
                "perc_recur":       float(rs["perc_recur"]),       # % recurrence
                "perc_determ":      float(rs["perc_determ"]),      # % determinism
                "maxl_found":       float(rs["maxl_found"]),       # Longest line
                "mean_line_length": float(rs["mean_line_length"]), # Average line length
                "std_line_length":  float(rs["std_line_length"]),  # Std dev of line length
                "entropy":          float(rs["entropy"]),          # Entropy
                "laminarity":       float(rs["laminarity"]),       # Laminarity
                "trapping_time":    float(rs["trapping_time"]),    # Trapping time
                "vmax":             float(rs["vmax"]),             # Vertical max
                "divergence":       float(rs["divergence"]),       # Divergence
                "trend_lower_diag": float(rs["trend_lower_diag"]), # Trend lower diagonal
                "trend_upper_diag": float(rs["trend_upper_diag"]), # Trend upper diagonal
            })

    # Save all results to a CSV file if any were found
    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        print(f"\n✅  Cross-RQA results written to {OUT_CSV} ({len(results)} rows)")
    else:
        print("\n😐  No cross-RQA results generated.")

# Run the main function if this file is executed directly
if __name__ == "__main__":
    main()