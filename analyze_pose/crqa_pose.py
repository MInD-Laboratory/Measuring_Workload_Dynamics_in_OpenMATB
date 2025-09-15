import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rqa.utils import norm_utils, rqa_utils_cpp

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODE = "baseline"  # "experimental" or "baseline"
ROOT_DIR = f"data/preprocessed_pose/{MODE}_pose"
OUT_CSV  = f"data/rqa/{MODE}_pose_crqa_head_eye_xy.csv"   # detailed x/y CRQA results
POSE_RQA_CSV = f"data/rqa/{MODE}_pose_rqa.csv"            # existing per-signal RQA to append to if present

SAMPLE_RATE_HZ = 60
WIN_SECONDS    = 60
OVERLAP_FRAC   = 0.5

PARAMS = {
    "norm":        1,
    "eDim":        4,
    "tLag":        40,
    "rescaleNorm": 1,
    "radius":      0.25,
    "minl":        2,
}

def sliding_windows(total_len, win_len, step):
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step

def compute_cross_rqa(x, y, p):
    try:
        x_n = norm_utils.normalize_data(x, p["norm"])
        y_n = norm_utils.normalize_data(y, p["norm"])
        ds  = rqa_utils_cpp.rqa_dist(x_n, y_n, dim=p["eDim"], lag=p["tLag"])
        _, rs, _, err = rqa_utils_cpp.rqa_stats(
            ds["d"], rescale=p["rescaleNorm"], rad=p["radius"],
            diag_ignore=0, minl=p["minl"], rqa_mode="cross",
        )
        return rs, err
    except RuntimeError as e:
        tqdm.write(f"⛔️ RuntimeError in cross-RQA: {e}")
        return None, -1

METRIC_COLS = [
    "perc_recur", "perc_determ", "maxl_found", "mean_line_length", "std_line_length",
    "entropy", "laminarity", "trapping_time", "vmax", "divergence",
    "trend_lower_diag", "trend_upper_diag"
]

def main():
    results = []
    win_len = WIN_SECONDS * SAMPLE_RATE_HZ
    step    = int(win_len * (1 - OVERLAP_FRAC))

    csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))
    for csv_file in tqdm(csv_files, desc="Files"):
        pid, cond = os.path.splitext(csv_file)[0].split("_", 1)
        df = pd.read_csv(os.path.join(ROOT_DIR, csv_file))

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
            # keep both results separate, and use the requested base name
            label = f"crqa_eye_head_{axis}"
            s1 = df[head_col].values
            s2 = df[eye_col].values
            total_n = len(s1)

            for w_idx, (s, e) in enumerate(sliding_windows(total_n, win_len, step)):
                xw = s1[s:e]
                yw = s2[s:e]
                if not (np.isfinite(xw).all() and np.isfinite(yw).all()):
                    continue

                rs, err = compute_cross_rqa(xw, yw, PARAMS)
                if err != 0:
                    tqdm.write(f"‼️  {pid}_{cond} {label} window {w_idx} error {err}")
                    continue

                row = {
                    "participant":   str(pid),
                    "condition":     cond,
                    "column":        label,             # <- separate x and y labels
                    "window_index":  w_idx,
                    "window_start":  s,                 # align with RQA schema
                    "window_end":    e,                 # align with RQA schema
                }
                row.update({k: float(rs[k]) for k in METRIC_COLS})
                results.append(row)

    if not results:
        print("\n😐  No cross-RQA results generated.")
        return

    # 1) Write detailed X/Y results
    crqa_xy_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    crqa_xy_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅  Cross-RQA results written to {OUT_CSV} ({len(crqa_xy_df)} rows)")

    # 2) Append both X and Y rows to existing pose RQA, if present
    if os.path.exists(POSE_RQA_CSV):
        base_df = pd.read_csv(POSE_RQA_CSV)

        # ensure schema compatibility
        for c in ["window_start", "window_end"]:
            if c not in base_df.columns:
                base_df[c] = np.nan
        for c in METRIC_COLS:
            if c not in base_df.columns:
                base_df[c] = np.nan

        # column order to match RQA file (participant, condition, column, window_index, window_start, window_end, metrics…)
        cols = ["participant", "condition", "column", "window_index", "window_start", "window_end"] + METRIC_COLS
        # if base has extra columns, keep them; otherwise, order as above
        base_cols = [c for c in cols if c in base_df.columns] + [c for c in base_df.columns if c not in cols]
        crqa_cols = [c for c in base_cols if c in crqa_xy_df.columns] + [c for c in base_cols if c not in crqa_xy_df.columns]
        # reindex to union of columns
        base_u = base_df.reindex(columns=base_cols)
        crqa_u = crqa_xy_df.reindex(columns=base_cols)

        merged = pd.concat([base_u, crqa_u], ignore_index=True)
        merged.to_csv(POSE_RQA_CSV, index=False)
        print(f"🧩  Appended CRQA X/Y rows to {POSE_RQA_CSV} as 'crqa_eye_head_x' and 'crqa_eye_head_y'.")
    else:
        print(f"ℹ️  {POSE_RQA_CSV} not found. Skipped appending. You still have {OUT_CSV}.")

if __name__ == "__main__":
    main()
