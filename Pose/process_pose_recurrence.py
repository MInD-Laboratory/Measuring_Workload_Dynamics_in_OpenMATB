import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rqa.utils import rqa_utils_cpp, norm_utils

# ── SETTINGS ───────────────────────────────────────────────────────
SESSIONS = ["baseline", "experimental"]
NORMALIZATIONS = ["original", "procrustes_global"]
ROOT_DIR_PATTERN = "data/processed_data/{session}/features/per_frame/{normalization}"
OUT_DIR = "data/rqa"

SAMPLE_RATE_HZ = 60
WIN_SECONDS = 60
OVERLAP_FRAC = 0.5

# RQA Parameters
RQA_PARAMS = {
    "norm": 1,
    "eDim": 4,
    "tLag": 20,
    "rescaleNorm": 1,
    "radius": 0.15,
    "tw": 2,
    "minl": 4,
}

# CRQA Parameters
CRQA_PARAMS = {
    "norm": 1,
    "eDim": 4,
    "tLag": 40,
    "rescaleNorm": 1,
    "radius": 0.3,
    "minl": 2,
}

# Define columns for each normalization type
RQA_COLUMNS = {
    "original": [
        "interocular",
        "head_rotation_rad",
        "blink_aperture",
        "mouth_aperture",
        "pupil_dx",
        "pupil_dy",
        "pupil_metric",
        "center_face_magnitude",
        "center_face_x",
        "center_face_y",
    ],
    "procrustes_global": [
        "interocular",
        "head_rotation_rad",
        "head_tx",
        "head_ty",
        "head_scale",
        "head_motion_mag",
        "blink_aperture",
        "mouth_aperture",
        "pupil_dx",
        "pupil_dy",
        "pupil_metric",
    ],
}

# Define CRQA pairs: (head_col, pupil_col, label)
CRQA_PAIRS = {
    "original": [
        ("center_face_magnitude", "pupil_metric", "crqa_head_pupil_mag"),
        ("center_face_x", "pupil_dx", "crqa_head_pupil_x"),
        ("center_face_y", "pupil_dy", "crqa_head_pupil_y"),
    ],
    "procrustes_global": [
        ("head_motion_mag", "pupil_metric", "crqa_head_pupil_mag"),
        ("head_tx", "pupil_dx", "crqa_head_pupil_x"),
        ("head_ty", "pupil_dy", "crqa_head_pupil_y"),
    ],
}

METRIC_COLS = [
    "perc_recur", "perc_determ", "maxl_found", "mean_line_length", "std_line_length",
    "entropy", "laminarity", "trapping_time", "vmax", "divergence",
    "trend_lower_diag", "trend_upper_diag"
]


def sliding_windows(total_len, win_len, step):
    """Generate overlapping windows."""
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step


def compute_rqa(series, p):
    """Run RQA on a single measurement series."""
    data_norm = norm_utils.normalize_data(series, p["norm"])
    ds = rqa_utils_cpp.rqa_dist(data_norm, data_norm, dim=p["eDim"], lag=p["tLag"])
    _, rs, _, err = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=p["rescaleNorm"],
        rad=p["radius"],
        diag_ignore=p["tw"],
        minl=p["minl"],
        rqa_mode="auto",
    )
    return rs, err


def compute_cross_rqa(x, y, p):
    """Run cross-RQA between two measurement series."""
    try:
        x_n = norm_utils.normalize_data(x, p["norm"])
        y_n = norm_utils.normalize_data(y, p["norm"])
        ds = rqa_utils_cpp.rqa_dist(x_n, y_n, dim=p["eDim"], lag=p["tLag"])
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


def process_session_normalization(session, normalization):
    """Process RQA and CRQA for one session/normalization combination."""
    print(f"\n{'='*70}")
    print(f"Processing: {session} / {normalization}")
    print(f"{'='*70}")
    
    root_dir = ROOT_DIR_PATTERN.format(session=session, normalization=normalization)
    
    if not os.path.exists(root_dir):
        print(f"⚠️  Directory not found: {root_dir}, skipping.")
        return
    
    results = []
    win_len = WIN_SECONDS * SAMPLE_RATE_HZ
    step = int(win_len * (1 - OVERLAP_FRAC))
    
    csv_files = sorted(f for f in os.listdir(root_dir) if f.endswith("_perframe.csv"))
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {root_dir}")
        return
    
    # Get RQA columns and CRQA pairs for this normalization
    rqa_cols = RQA_COLUMNS[normalization]
    crqa_pairs = CRQA_PAIRS[normalization]
    
    for csv_file in tqdm(csv_files, desc="Files"):
        # Parse filename: {participant}_{condition}_perframe.csv
        base_name = os.path.splitext(csv_file)[0]
        parts = base_name.replace("_perframe", "").split("_")
        if len(parts) >= 2:
            pid = parts[0]
            cond = "_".join(parts[1:])
        else:
            tqdm.write(f"⚠️  Cannot parse filename: {csv_file}, skipping.")
            continue
        
        df = pd.read_csv(os.path.join(root_dir, csv_file))
        
        # ── RQA Analysis ──
        cols_present = [c for c in rqa_cols if c in df.columns]
        if not cols_present:
            tqdm.write(f"⚠️  {csv_file}: none of RQA columns found, skipping RQA.")
        else:
            for col in cols_present:
                series = df[col].values
                total_n = len(series)
                
                for w_idx, (s, e) in enumerate(
                    sliding_windows(total_n, win_len, step)
                ):
                    win = series[s:e]
                    if not np.isfinite(win).all():
                        continue
                    
                    rs, err = compute_rqa(win, RQA_PARAMS)
                    if err != 0:
                        tqdm.write(f"‼️  {pid}_{cond} RQA window {w_idx} (col {col}) error {err}")
                        continue
                    
                    row = {
                        "participant": str(pid),
                        "condition": cond,
                        "column": col,
                        "window_index": w_idx,
                        "window_start": s,
                        "window_end": e,
                    }
                    row.update({k: float(rs[k]) for k in METRIC_COLS})
                    results.append(row)
        
        # ── CRQA Analysis ──
        for head_col, pupil_col, label in crqa_pairs:
            if head_col not in df.columns or pupil_col not in df.columns:
                tqdm.write(f"⚠️  {csv_file}: missing {head_col} or {pupil_col}, skipping CRQA {label}.")
                continue
            
            s1 = df[head_col].values
            s2 = df[pupil_col].values
            total_n = len(s1)
            
            for w_idx, (s, e) in enumerate(
                sliding_windows(total_n, win_len, step)
            ):
                xw = s1[s:e]
                yw = s2[s:e]
                if not (np.isfinite(xw).all() and np.isfinite(yw).all()):
                    continue
                
                rs, err = compute_cross_rqa(xw, yw, CRQA_PARAMS)
                if err != 0:
                    tqdm.write(f"‼️  {pid}_{cond} CRQA {label} window {w_idx} error {err}")
                    continue
                
                row = {
                    "participant": str(pid),
                    "condition": cond,
                    "column": label,
                    "window_index": w_idx,
                    "window_start": s,
                    "window_end": e,
                }
                row.update({k: float(rs[k]) for k in METRIC_COLS})
                results.append(row)
    
    # ── SAVE RESULTS ──
    if not results:
        print(f"😐  No results generated for {session}/{normalization}.")
        return
    
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"{session}_{normalization}_rqa_crqa.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv, index=False)
    print(f"✅  Results written to {out_csv} ({len(df_results)} rows)")


def main():
    """Process all session/normalization combinations."""
    print("Starting batch RQA and CRQA analysis...")
    print(f"Sessions: {SESSIONS}")
    print(f"Normalizations: {NORMALIZATIONS}")
    
    for session in SESSIONS:
        for normalization in NORMALIZATIONS:
            process_session_normalization(session, normalization)
    
    print(f"\n{'='*70}")
    print("All processing complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()