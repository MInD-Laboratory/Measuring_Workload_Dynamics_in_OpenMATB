import os
import sys
sys.path.append(os.path.abspath("/Users/cartersale/Library/CloudStorage/OneDrive-MacquarieUniversity/Research/Projects/2025_MATBExp4/03_Analysis"))  
import pandas as pd
from tqdm import tqdm
from rqa.utils import norm_utils, rqa_utils_cpp

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR        = "/Users/cartersale/Library/CloudStorage/OneDrive-MacquarieUniversity/Research/Projects/2025_MATBExp4/02_Data/pose/pre_filtered_keypoints_nozscore"
OUT_CSV        = "/Users/cartersale/Documents/nature_matb_submission/data/rqa/crqa_head_eye_bsl.csv"
SAMPLE_RATE_HZ = 60        # fps
WIN_SECONDS    = 60        # window length
OVERLAP_FRAC   = 0.5       # 50% overlap

PARAMS = {
    "norm":        1,
    "eDim":        4,
    "tLag":        40,
    "rescaleNorm": 1,
    "radius":      0.2,
    "minl":        2,
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sliding_windows(total_len, win_len, step):
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step

def compute_cross_rqa(x, y, p):
    """Return (rs‐dict, err_code) for two series."""
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

        # need both columns
        if not {"center_face_magnitude", "avg_pupil_magnitude"} <= set(df.columns):
            tqdm.write(f"⚠️  {csv_file}: missing pupil columns, skipping.")
            continue

        series1 = df["center_face_magnitude"].values
        series2 = df["avg_pupil_magnitude"].values
        total_n = len(series1)

        for w_idx, (s, e) in enumerate(
                sliding_windows(total_n, win_len, step)):
            rs, err = compute_cross_rqa(series1[s:e], series2[s:e], PARAMS)
            if err != 0:
                tqdm.write(f"‼️  {pid}_{cond} window {w_idx} error {err}")
                continue

            results.append({
                "participant":      pid,
                "condition":        cond,
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