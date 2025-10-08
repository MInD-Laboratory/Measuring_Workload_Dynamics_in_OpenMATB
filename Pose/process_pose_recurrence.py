#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA and CRQA Analysis with Multiple Normalization Methods

Computes RQA (Recurrence Quantification Analysis) and CRQA (Cross-RQA) on pose data
with different alignment and normalization approaches:
  - Alignments: original, procrustes_global
  - Normalizations: none (original), minmax, zscore

Output files are named: {session}_{alignment}_{normalization}_rqa_crqa.csv
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rqa.utils import rqa_utils_cpp, norm_utils

# ============================================================================
# SETTINGS
# ============================================================================

SESSIONS = ["experimental", "baseline"]
ALIGNMENTS = ["original"]
NORMALIZATIONS = ["original", "minmax", "zscore"]  # Add all three normalization types

# Path pattern - uses double underscore between alignment and normalization
ROOT_DIR_PATTERN = "data/processed_data/{session}/features/per_frame/{alignment}__{normalization}"

OUT_DIR = "data/rqa"

SAMPLE_RATE_HZ = 60
WIN_SECONDS = 60
OVERLAP_FRAC = 0.5

# ============================================================================
# RQA PARAMETERS
# ============================================================================

RQA_PARAMS = {
    "norm": 1,          # Normalization method for RQA algorithm
    "eDim": 4,          # Embedding dimension
    "tLag": 20,         # Time lag
    "rescaleNorm": 1,   # Rescale normalization
    "radius": 0.15,     # Recurrence threshold
    "tw": 2,            # Theiler window
    "minl": 4,          # Minimum line length
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

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

# Define columns for each alignment type (regardless of normalization)
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sliding_windows(total_len, win_len, step):
    """Generate overlapping windows."""
    i = 0
    while i + win_len <= total_len:
        yield i, i + win_len
        i += step


def compute_rqa(series, p):
    """
    Run RQA on a single measurement series.
    
    Args:
        series: 1D numpy array of measurements
        p: RQA parameter dictionary
        
    Returns:
        tuple: (results_dict, error_code)
    """
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
    """
    Run cross-RQA between two measurement series.
    
    Args:
        x: First time series
        y: Second time series
        p: CRQA parameter dictionary
        
    Returns:
        tuple: (results_dict, error_code)
    """
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


def load_normalized_data(file_path, normalization_method):
    """
    Load data from CSV file.
    
    Since your data is already normalized in separate directories with the
    naming convention {alignment}__{normalization}, we just load it directly.
    
    Args:
        file_path: Path to CSV file
        normalization_method: 'original', 'minmax', or 'zscore' (for metadata only)
        
    Returns:
        DataFrame (already normalized based on which directory it came from)
    """
    df = pd.read_csv(file_path)
    return df


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_combination(session, alignment, normalization):
    """
    Process RQA and CRQA for one session/alignment/normalization combination.
    
    Args:
        session: 'baseline' or 'experimental'
        alignment: 'original' or 'procrustes_global'
        normalization: 'original', 'minmax', or 'zscore'
    """
    print(f"\n{'='*70}")
    print(f"Processing: {session} / {alignment}__{normalization}")
    print(f"{'='*70}")
    
    # Build directory path with double underscore
    root_dir = ROOT_DIR_PATTERN.format(
        session=session, 
        alignment=alignment, 
        normalization=normalization
    )
    
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
    
    # Get RQA columns and CRQA pairs for this alignment
    rqa_cols = RQA_COLUMNS[alignment]
    crqa_pairs = CRQA_PAIRS[alignment]
    
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
        
        # Load data with normalization applied
        df = load_normalized_data(
            os.path.join(root_dir, csv_file), 
            normalization
        )
        
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
                        "alignment": alignment,
                        "normalization": normalization,
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
                    "alignment": alignment,
                    "normalization": normalization,
                    "column": label,
                    "window_index": w_idx,
                    "window_start": s,
                    "window_end": e,
                }
                row.update({k: float(rs[k]) for k in METRIC_COLS})
                results.append(row)
    
    # ── SAVE RESULTS ──
    if not results:
        print(f"😐  No results generated for {session}/{alignment}/{normalization}.")
        return
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Output filename includes all three identifiers with double underscore format
    out_csv = os.path.join(
        OUT_DIR, 
        f"{session}_{alignment}__{normalization}_rqa_crqa.csv"
    )
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv, index=False)
    print(f"✅  Results written to {out_csv} ({len(df_results)} rows)")


def main():
    """Process all session/alignment/normalization combinations."""
    print("="*70)
    print("STARTING BATCH RQA AND CRQA ANALYSIS")
    print("="*70)
    print(f"Sessions: {SESSIONS}")
    print(f"Alignments: {ALIGNMENTS}")
    print(f"Normalizations: {NORMALIZATIONS}")
    print(f"Total combinations: {len(SESSIONS) * len(ALIGNMENTS) * len(NORMALIZATIONS)}")
    
    # Track completion
    completed = []
    failed = []
    
    for session in SESSIONS:
        for alignment in ALIGNMENTS:
            for normalization in NORMALIZATIONS:
                try:
                    process_combination(session, alignment, normalization)
                    completed.append(f"{session}_{alignment}_{normalization}")
                except Exception as e:
                    error_msg = f"{session}_{alignment}_{normalization}"
                    failed.append(error_msg)
                    print(f"❌ Error processing {error_msg}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successful: {len(completed)}/{len(SESSIONS)*len(ALIGNMENTS)*len(NORMALIZATIONS)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for f in failed:
            print(f"   - {f}")
    
    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Output files: {session}_{alignment}_{normalization}_rqa_crqa.csv")


if __name__ == "__main__":
    main()