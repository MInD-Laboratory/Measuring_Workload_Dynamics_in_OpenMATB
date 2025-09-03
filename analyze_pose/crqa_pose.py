import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rqa.utils import norm_utils, rqa_utils_cpp

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODE = "baseline"  # "experimental" or "baseline"
ROOT_DIR = f"data/preprocessed_pose/{MODE}_pose"
OUT_CSV  = f"data/rqa/{MODE}_pose_crqa_head_eye_xy.csv"  # single file for both axes

SAMPLE_RATE_HZ = 60        # frames per second
WIN_SECONDS    = 60        # window length (seconds)
OVERLAP_FRAC   = 0.5       # 50% overlap

# Parameters for the analysis method
PARAMS = {
    "norm":        1,    # normalization method
    "eDim":        4,    # embedding dimension
    "tLag":        40,   # time lag
    "rescaleNorm": 1,    # rescale distances
    "radius":      0.2,  # recurrence threshold
    "minl":        2,    # minimum line length
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sliding_windows(total_len, win_len, step):
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step

def compute_cross_rqa(x, y, p):
    """Run cross-recurrence quantification analysis (CRQA) on two data series."""
    try:
        x_n = norm_utils.normalize_data(x, p["norm"])
        y_n = norm_utils.normalize_data(y, p["norm"])
        ds  = rqa_utils_cpp.rqa_dist(x_n, y_n, dim=p["eDim"], lag=p["tLag"])

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
    results = []
    win_len = WIN_SECONDS * SAMPLE_RATE_HZ
    step    = int(win_len * (1 - OVERLAP_FRAC))

    csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))
    for csv_file in tqdm(csv_files, desc="Files"):
        pid, cond = os.path.splitext(csv_file)[0].split("_", 1)
        df = pd.read_csv(os.path.join(ROOT_DIR, csv_file))

        # Require both X and Y columns for head and eye
        required = {"center_face_x", "center_face_y", "avg_pupil_x", "avg_pupil_y"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            tqdm.write(f"⚠️  {csv_file}: missing columns {sorted(missing)}, skipping.")
            continue

        axes = {
            "x": ("center_face_x", "avg_pupil_x"),
            "y": ("center_face_y", "avg_pupil_y"),
        }

        for axis, (head_col, eye_col) in axes.items():
            label = f"crqa_head_eye_{axis}"
            series1 = df[head_col].values
            series2 = df[eye_col].values
            total_n = len(series1)

            for w_idx, (s, e) in enumerate(sliding_windows(total_n, win_len, step)):
                xw = series1[s:e]
                yw = series2[s:e]
                if not (np.isfinite(xw).all() and np.isfinite(yw).all()):
                    continue

                rs, err = compute_cross_rqa(xw, yw, PARAMS)
                if err != 0:
                    tqdm.write(f"‼️  {pid}_{cond} {label} window {w_idx} error {err}")
                    continue

                results.append({
                    "participant":      pid,
                    "condition":        cond,
                    "column":           label,                       # ← requested field
                    "window_index":     w_idx,
                    "start_sample":     s,
                    "end_sample":       e,
                    "perc_recur":       float(rs["perc_recur"]),
                    "perc_determ":      float(rs["perc_determ"]),
                    "maxl_found":       float(rs["maxl_found"]),
                    "mean_line_length": float(rs["mean_line_length"]),
                    "std_line_length":  float(rs["std_line_length"]),
                    "entropy":          float(rs["entropy"]),
                    "laminarity":       float(rs["laminarity"]),
                    "trapping_time":    float(rs["trapping_time"]),
                    "vmax":             float(rs["vmax"]),
                    "divergence":       float(rs["divergence"]),
                    "trend_lower_diag": float(rs["trend_lower_diag"]),
                    "trend_upper_diag": float(rs["trend_upper_diag"]),
                })

    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        print(f"\n✅  Cross-RQA results written to {OUT_CSV} ({len(results)} rows)")
    else:
        print("\n😐  No cross-RQA results generated.")

if __name__ == "__main__":
    main()
