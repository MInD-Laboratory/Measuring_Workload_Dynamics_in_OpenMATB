import os
import sys
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

ROOT_DIR        = "data/pose/baseline_pose"
OUT_CSV         = "data/rqa/baseline_pose_rqa.csv"

SAMPLE_RATE_HZ  = 60                      # fps
WIN_SECONDS     = 60                      # window length
OVERLAP_FRAC    = 0.5                     # 50 % overlap
TARGET_COLUMNS  = [                       # edit freely
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

PARAMS = {                                
    "norm": 1, 
    "eDim": 4,
    "tLag": 20,
    "rescaleNorm": 1,
    "radius": 0.15,
    "tw": 2,
    "minl": 4,
}

def compute_rqa(series, p):
    """Return rs-dict and error code for one univariate series."""
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
    """Yield (start, end) indices for sliding windows."""
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step

def main():
    results = []
    win_len  = WIN_SECONDS * SAMPLE_RATE_HZ
    step     = int(win_len * (1 - OVERLAP_FRAC))

    csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))

    for csv_file in tqdm(csv_files, desc="Files"):
        pid, cond = os.path.splitext(csv_file)[0].split("_", 1)
        df = pd.read_csv(os.path.join(ROOT_DIR, csv_file))

        # sanity: only keep requested columns that exist
        cols_present = [c for c in TARGET_COLUMNS if c in df.columns]
        if not cols_present:
            tqdm.write(f"⚠️  {csv_file}: none of TARGET_COLUMNS found, skipping.")
            continue

        for col in cols_present:
            series = df[col].values
            total_n = len(series)

            for w_idx, (s, e) in enumerate(
                    tqdm(sliding_windows(total_n, win_len, step),
                         leave=False, desc=f"{pid}-{cond}:{col}")):
                rs, err = compute_rqa(series[s:e], PARAMS)
                if err != 0:
                    tqdm.write(f"‼️  {pid}_{cond} window {w_idx} (col {col}) error code {err}")
                    continue

                results.append({
                    "participant":        pid,
                    "condition":          cond,
                    "column":             col,
                    "window_index":         w_idx,
                    "window_start":       s,
                    "window_end":         e,
                    "perc_recur":         float(rs["perc_recur"]),
                    "perc_determ":        float(rs["perc_determ"]),
                    "maxl_found":         float(rs["maxl_found"]),
                    "mean_line_length":   float(rs["mean_line_length"]),
                    "std_line_length":    float(rs["std_line_length"]),
                    "entropy":            float(rs["entropy"]),
                    "laminarity":         float(rs["laminarity"]),
                    "trapping_time":      float(rs["trapping_time"]),
                    "vmax":               float(rs["vmax"]),
                    "divergence":         float(rs["divergence"]),
                    "trend_lower_diag":   float(rs["trend_lower_diag"]),
                    "trend_upper_diag":   float(rs["trend_upper_diag"]),
                })

    # ── SAVE ───────────────────────────────────────────────────────────────────
    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        print(f"\n✅  RQA results written to {OUT_CSV}  ({len(results)} rows)")
    else:
        print("\n😐  No results generated.")

if __name__ == "__main__":
    main()