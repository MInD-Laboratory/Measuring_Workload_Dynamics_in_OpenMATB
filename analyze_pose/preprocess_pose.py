"""
Preprocess OpenPose CSVs, collapse keypoints into regions, and MASK low-quality windows.

Plain-English flow:
1) Load which time windows are low quality from a QC CSV you already generated.
2) For those windows, set ONLY the affected metrics to missing (NaN). Everything else stays.
3) Save the cleaned CSVs and a simple report of how much got masked.

If anything important is missing or mismatched, the script stops with a clear error.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

    # input_directory = 'data/pose/experimental_pose'
    # output_directory = 'data/preprocessed_pose/experimental_pose'
# =========================== USER SETTINGS ===========================
# 1) Where your raw OpenPose CSVs are
RAW_INPUT_DIR = 'data/raw_pose/baseline_pose'  # Change to 'experimental_pose' if needed

# 2) Where to save cleaned, preprocessed CSVs
OUTPUT_DIR = 'data/preprocessed_pose/baseline_pose'  # Change to /experimental_pose if needed

# 3) Where the QC window indices CSV lives (produced by your QC step)
QC_DIR = 'data/qc_outputs_bsl'
QC_BAD_IDX_FILE = os.path.join(QC_DIR, 'metric_bad_window_indices.csv')

# 4) Confidence threshold for averaging:
#    if a region's average confidence is below this, we treat positions as missing (NaN) before interpolation
CONFIDENCE_THRESHOLD = 0.3

# 5) QC window spec — MUST match the QC step (60 s * 30 fps = 1800; 60 s * 60 fps = 3600, etc.)
QC_WINDOW_FRAMES = 1800
QC_OVERLAP = 0.0
QC_STEP_FRAMES = int(round(QC_WINDOW_FRAMES * (1.0 - QC_OVERLAP)))

# Map QC metric labels → columns to blank (set NaN) in the processed table
# Keys here are LOWERCASE and must match the QC CSV "metric" values (case-insensitive).
QC_TO_COLUMNS = {
    "eyes": [
        "blink_dist",
        "left_eye_x", "left_eye_y", "left_eye_prob", "left_eye_magnitude",
        "right_eye_x", "right_eye_y", "right_eye_prob", "right_eye_magnitude",
    ],
    "head_rotation": [
        "head_rotation_angle",
    ],
    "mouth_dist": [
        "mouth_dist",
    ],
    "pupils_combined": [
        "avg_pupil_x", "avg_pupil_y", "avg_pupil_magnitude",
        "left_pupil_x", "left_pupil_y", "left_pupil_prob", "left_pupil_magnitude",
        "right_pupil_x", "right_pupil_y", "right_pupil_prob", "right_pupil_magnitude",
    ],
    "center_face": [
        "center_face_x", "center_face_y", "center_face_prob", "center_face_magnitude",
    ],
}

# ====================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------ Helpers ------------------------------

def load_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV file into a table."""
    return pd.read_csv(file_path)

def compute_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw keypoints into mean positions + mean confidence for regions.
    If confidence is low, mark positions as NaN. Then interpolate gaps.
    """
    averaged_df = pd.DataFrame({
        # Center-face region (27–35)
        'center_face_x': df[[f'x{i}' for i in range(27, 36)]].mean(axis=1),
        'center_face_y': df[[f'y{i}' for i in range(27, 36)]].mean(axis=1),
        'center_face_prob': df[[f'prob{i}' for i in range(27, 36)]].mean(axis=1),

        # Eyes (36–41 left, 42–47 right)
        'left_eye_x': df[[f'x{i}' for i in range(36, 42)]].mean(axis=1),
        'left_eye_y': df[[f'y{i}' for i in range(36, 42)]].mean(axis=1),
        'left_eye_prob': df[[f'prob{i}' for i in range(36, 42)]].mean(axis=1),

        'right_eye_x': df[[f'x{i}' for i in range(42, 48)]].mean(axis=1),
        'right_eye_y': df[[f'y{i}' for i in range(42, 48)]].mean(axis=1),
        'right_eye_prob': df[[f'prob{i}' for i in range(42, 48)]].mean(axis=1),

        # Pupils (68, 69)
        'left_pupil_x': df['x68'],
        'left_pupil_y': df['y68'],
        'left_pupil_prob': df['prob68'],

        'right_pupil_x': df['x69'],
        'right_pupil_y': df['y69'],
        'right_pupil_prob': df['prob69'],
    })

    # If confidence below threshold, treat positions as missing
    for part in ['center_face', 'left_eye', 'right_eye', 'left_pupil', 'right_pupil']:
        prob_col = f'{part}_prob'
        for axis in ['x', 'y']:
            val_col = f'{part}_{axis}'
            averaged_df.loc[averaged_df[prob_col] < CONFIDENCE_THRESHOLD, val_col] = np.nan

    # Fill short gaps by interpolation
    averaged_df.interpolate(method='linear', limit_direction='both', inplace=True)
    return averaged_df

def compute_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """For each *_x with matching *_y, add *_magnitude = sqrt(x^2 + y^2)."""
    for col in df.columns:
        if col.endswith('_x'):
            y_col = col.replace('_x', '_y')
            if y_col in df.columns:
                mag_col = col.replace('_x', '_magnitude')
                df[mag_col] = np.sqrt(df[col]**2 + df[y_col]**2)
    return df

def compute_combined_eyes(df: pd.DataFrame) -> pd.DataFrame:
    """Average pupil positions and magnitude."""
    df['avg_pupil_x'] = (df['left_pupil_x'] + df['right_pupil_x']) / 2
    df['avg_pupil_y'] = (df['left_pupil_y'] + df['right_pupil_y']) / 2
    df['avg_pupil_magnitude'] = np.sqrt(df['avg_pupil_x']**2 + df['avg_pupil_y']**2)
    return df

def compute_blink(df_raw: pd.DataFrame) -> pd.Series:
    """Blink proxy: average vertical eyelid distances (both eyes)."""
    top_right_x = df_raw[['x37', 'x38']].mean(axis=1)
    top_right_y = df_raw[['y37', 'y38']].mean(axis=1)
    bottom_right_x = df_raw[['x40', 'x41']].mean(axis=1)
    bottom_right_y = df_raw[['y40', 'y41']].mean(axis=1)
    right_eye_dist = np.sqrt((top_right_x - bottom_right_x)**2 + (top_right_y - bottom_right_y)**2)

    top_left_x = df_raw[['x43', 'x44']].mean(axis=1)
    top_left_y = df_raw[['y43', 'y44']].mean(axis=1)
    bottom_left_x = df_raw[['x46', 'x47']].mean(axis=1)
    bottom_left_y = df_raw[['y46', 'y47']].mean(axis=1)
    left_eye_dist = np.sqrt((top_left_x - bottom_left_x)**2 + (top_left_y - bottom_left_y)**2)

    return (right_eye_dist + left_eye_dist) / 2.0

def compute_head_rotation(df_raw: pd.DataFrame) -> pd.Series:
    """Head rotation angle (radians) from keypoints 36→45."""
    dx = df_raw['x45'] - df_raw['x36']
    dy = df_raw['y45'] - df_raw['y36']
    return np.arctan2(dy, dx)

def compute_mouth_distance(df_raw: pd.DataFrame) -> pd.Series:
    """Mouth opening proxy (distance between keypoints 62 and 66)."""
    return np.sqrt((df_raw['x62'] - df_raw['x66'])**2 + (df_raw['y62'] - df_raw['y66'])**2)

# ------------------------- Load QC bad-window indices --------------------------

def load_bad_windows(bad_idx_csv: str) -> dict:
    """
    Build a dictionary:
        bad_map[file_basename][metric_lower] = list of (start_frame, end_frame_exclusive)

    Accepts either explicit spans OR window_index (rebuilds spans from QC window spec).
    Normalizes filenames (basenames) and metric names (lowercase).
    """
    if not os.path.isfile(bad_idx_csv):
        sys.exit(f"[FATAL] QC bad-window index CSV not found: {bad_idx_csv}")

    df = pd.read_csv(bad_idx_csv)

    if 'file' not in df.columns or 'metric' not in df.columns:
        sys.exit("[FATAL] QC CSV must include 'file' and 'metric' columns.")

    # Normalize: basenames + lowercase metric labels
    df['file'] = df['file'].astype(str).map(lambda s: os.path.basename(s).strip())
    df['metric'] = df['metric'].astype(str).str.strip().str.lower()

    has_spans = {'start_frame', 'end_frame_exclusive'}.issubset(df.columns)
    has_index = 'window_index' in df.columns

    if not has_spans and not has_index:
        sys.exit("[FATAL] QC CSV must have either span columns (start_frame, end_frame_exclusive) or window_index.")

    # Rebuild spans if needed
    if not has_spans and has_index:
        df = df.copy()
        widx = df['window_index'].astype(int)
        df['start_frame'] = (widx * QC_STEP_FRAMES).astype(int)
        df['end_frame_exclusive'] = (df['start_frame'] + QC_WINDOW_FRAMES).astype(int)

    # Sanity note: spans not equal to QC window length are okay for the final truncated window
    span_len = (df['end_frame_exclusive'] - df['start_frame']).to_numpy()
    n_off = int((span_len != QC_WINDOW_FRAMES).sum())
    if n_off:
        print(f"[QC WARNING] {n_off} span(s) not equal to QC_WINDOW_FRAMES={QC_WINDOW_FRAMES}. "
              "Likely final truncated windows or QC mismatch.")

    bad_map: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for (fname, metric), grp in df.groupby(['file', 'metric']):
        spans = list(zip(grp['start_frame'].astype(int), grp['end_frame_exclusive'].astype(int)))
        bad_map.setdefault(fname, {})[metric] = spans

    total = sum(len(v) for d in bad_map.values() for v in d.values())
    print(f"[QC] Loaded {total} bad window span(s) from {bad_idx_csv} (window={QC_WINDOW_FRAMES}, step={QC_STEP_FRAMES}).")
    return bad_map

# ------------------------------ Apply masks ------------------------------------

def apply_bad_masks(file_basename: str, df_proc: pd.DataFrame, bad_map: dict):
    """
    Turn only the QC-flagged metric windows into NaN.

    Returns:
        df_proc (DataFrame)        : modified with NaNs where appropriate
        stats (dict)               : per-metric counts of frames/windows masked
        total_masked_frames (int)  : total frames set to NaN across all metrics
    """
    n_frames = len(df_proc)
    stats = {}
    total_masked_frames = 0

    present = bad_map.get(file_basename, {})  # dict: metric -> list of (start, end)
    if not present:
        # No QC spans for this file → nothing to mask
        for qc_metric in QC_TO_COLUMNS.keys():
            stats[qc_metric] = {
                "frames_total": n_frames,
                "frames_masked": 0,
                "pct_masked": 0.0,
                "windows_masked": 0
            }
        return df_proc, stats, total_masked_frames

    # Build a boolean mask per QC metric (True = we will blank it to NaN)
    metric_masks = {m: np.zeros(n_frames, dtype=bool) for m in QC_TO_COLUMNS.keys()}

    # Fill those masks from spans
    for qc_metric_raw, windows in present.items():
        qc_metric = qc_metric_raw.lower()
        if qc_metric not in metric_masks:
            continue  # QC includes a metric we don't track here
        for (s, e) in windows:
            s = max(0, int(s))
            e = min(n_frames, int(e))
            if s < e:
                metric_masks[qc_metric][s:e] = True

    # Apply the masks (write NaN) and collect stats
    for qc_metric, mask in metric_masks.items():
        frames_masked = int(mask.sum())
        if frames_masked > 0:
            for col in QC_TO_COLUMNS[qc_metric]:
                if col in df_proc.columns:
                    df_proc.loc[mask, col] = np.nan
        stats[qc_metric] = {
            "frames_total": n_frames,
            "frames_masked": frames_masked,
            "pct_masked": (frames_masked / n_frames) if n_frames > 0 else np.nan,
            "windows_masked": int(len(present.get(qc_metric, [])))
        }
        total_masked_frames += frames_masked

    return df_proc, stats, total_masked_frames

# ------------------------------- Main runner -----------------------------------

def process_data(input_dir: str = RAW_INPUT_DIR, output_dir: str = OUTPUT_DIR):
    """
    1) Load QC map and verify filenames overlap.
    2) For each CSV: compute features → apply masks → save → append to drop report.
    3) Save the drop report at the end.
    """
    # Load QC once and fail fast if missing
    bad_map = load_bad_windows(QC_BAD_IDX_FILE)

    # Verify at least one filename matches between QC and input dir
    proc_files = {os.path.basename(f) for f in os.listdir(input_dir) if f.endswith('.csv')}
    qc_files = set(bad_map.keys())
    intersection = proc_files & qc_files
    if not intersection:
        sys.exit(
            "[FATAL] No filename overlap between QC and RAW_INPUT_DIR.\n"
            f"QC files ({len(qc_files)}): {sorted(list(qc_files))[:5]}...\n"
            f"RAW files ({len(proc_files)}): {sorted(list(proc_files))[:5]}..."
        )
    print(f"[OK] QC CSV found and {len(intersection)} matching file(s).")

    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    report_rows = []

    for csv_file in tqdm(csv_files, desc="Preprocessing CSV files w/ QC masking"):
        file_path = os.path.join(input_dir, csv_file)
        df_raw = load_csv(file_path)

        # Build the preprocessed feature table from raw keypoints
        averaged_df = compute_averages(df_raw)
        averaged_df = compute_magnitude(averaged_df)
        averaged_df = compute_combined_eyes(averaged_df)
        averaged_df['blink_dist'] = compute_blink(df_raw)
        averaged_df['head_rotation_angle'] = compute_head_rotation(df_raw)
        averaged_df['mouth_dist'] = compute_mouth_distance(df_raw)

        # Apply the QC masks (set only flagged metric windows to NaN)
        averaged_df, stats, total_masked_frames = apply_bad_masks(
            file_basename=os.path.basename(csv_file),
            df_proc=averaged_df,
            bad_map=bad_map
        )

        # If QC says this file has bad windows but we masked nothing, stop and show examples
        if (csv_file in bad_map) and sum(len(v) for v in bad_map[csv_file].values()) > 0 and total_masked_frames == 0:
            sample = []
            for m, spans in bad_map[csv_file].items():
                if spans:
                    s0, e0 = spans[0]
                    sample.append(f"{m}[{s0}:{e0})")
                if len(sample) >= 3:
                    break
            sys.exit(f"[FATAL] Masking produced 0 NaNs for {csv_file} despite QC spans: {', '.join(sample)}. "
                     "Check filename/metric normalization and column names.")

        # Save the cleaned CSV
        out_path = os.path.join(output_dir, os.path.splitext(csv_file)[0] + ".csv")
        averaged_df.to_csv(out_path, index=False)

        # Append per-metric masking stats for the drop report
        for qc_metric, st in stats.items():
            report_rows.append({
                "file": csv_file,
                "qc_metric": qc_metric,
                "frames_total": st["frames_total"],
                "frames_masked": st["frames_masked"],
                "pct_masked": st["pct_masked"],
                "windows_masked": st["windows_masked"]
            })

    # Write the drop report (one row per file × metric)
    report_path = os.path.join(output_dir, "preprocess_drop_report.csv")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"Wrote preprocessed files to: {output_dir}")
    print(f"Wrote drop report: {report_path}")

# --------------------------------- Entrypoint ----------------------------------

if __name__ == "__main__":
    process_data()
