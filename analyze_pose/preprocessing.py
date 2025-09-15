"""
=============================================================================
FACIAL POSE PROCESSING UTILITIES
=============================================================================

This module provides functions for processing facial landmark data from OpenPose
to create clean, standardized features for analysis. It performs:

1. QUALITY CONTROL (QC): Identifies time windows where landmark data is poor
2. PREPROCESSING: Converts raw landmarks into meaningful features 
3. FILTERING: Applies temporal smoothing to reduce noise

The pipeline supports two coordinate normalization methods:
- Original: Uses eye corners to stabilize head pose
- Procrustes: Aligns all landmarks to a reference shape (more robust)
"""

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt

# =============================================================================
# CONFIGURATION: LANDMARK DEFINITIONS
# =============================================================================

# Mapping of metrics to the keypoint indices they depend on (for QC stage)
METRIC_KPS = {
    "eyes": list(range(37, 42)) + list(range(43, 48)),  # Eye contour landmarks
    "head_rotation": [37, 46],                           # Outer eye corners  
    "mouth_dist": [62, 66],                             # Top/bottom lip centers
    "pupils_combined": [69, 70],                        # Left/right pupil centers
    "center_face": list(range(28, 36)),                 # Nose bridge region
}

# All landmark indices that we need for quality control
RELEVANT_KPS = sorted({kp for kps in METRIC_KPS.values() for kp in kps})

# Eye landmark definitions for blink detection
EYES = {
    "L": [37, 38, 39, 40, 41, 42],  # Left eye contour (6 points)
    "R": [43, 44, 45, 46, 47, 48],  # Right eye contour (6 points)
}

# Reference points for head stabilization (outer eye corners)
ANCHOR_L = 37  # Left outer eye corner
ANCHOR_R = 46  # Right outer eye corner

INDEX_OFFSET = 0  # No offset: landmark numbers match CSV column numbers exactly

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _col(name_idx: int, axis: str) -> str:
    """Convert landmark index and axis to column name (e.g., 'x37', 'y37', 'prob37')."""
    idx = name_idx + INDEX_OFFSET
    return f"{axis}{idx}"

def _have_cols(df: pd.DataFrame, idxs: list[int]) -> bool:
    """Check if DataFrame has x and y columns for all specified landmark indices."""
    cols = [f"x{(i + INDEX_OFFSET)}" for i in idxs] + [f"y{(i + INDEX_OFFSET)}" for i in idxs]
    return all(c in df.columns for c in cols)

# =============================================================================
# QUALITY CONTROL (QC) FUNCTIONS
# =============================================================================

def window_ranges(n_rows: int, window_size: int, overlap: float):
    """
    Calculate sliding window positions for quality control analysis.
    Divides data into overlapping segments to check quality in each segment separately.
    """
    if n_rows < window_size:
        return []  # File too short for even one window
    
    step = max(1, int(round(window_size * (1 - overlap))))
    return [(s, s + window_size) for s in range(0, n_rows - window_size + 1, step)]

def _max_true_run_length(b: pd.Series) -> int:
    """Find the longest consecutive sequence of True values in a boolean series."""
    run = max_run = 0
    for v in b.to_numpy():
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def kp_missing_series(df_w: pd.DataFrame, i: int, conf_thresh: float) -> pd.Series:
    """
    Determine which frames have missing or low-confidence data for a landmark.
    A landmark is considered "missing" if coordinates are NaN or confidence is below threshold.
    """
    xi = i 
    xcol, ycol, pcol = f"x{xi}", f"y{xi}", f"prob{xi}"
    
    # If columns don't exist, consider all frames as missing
    if (xcol not in df_w.columns) or (ycol not in df_w.columns) or (pcol not in df_w.columns):
        return pd.Series(True, index=df_w.index)
    
    # A landmark is "good" if it has high confidence AND valid coordinates
    good = (df_w[pcol] >= conf_thresh) & df_w[xcol].notna() & df_w[ycol].notna()
    return ~good  # Return True for missing/bad landmarks

def analyze_file_qc(fp: str, window_size: int, overlap: float, conf_thresh: float, max_interp: int):
    """
    Perform quality control analysis on a single CSV file.
    
    This function:
    1. Divides the data into overlapping windows
    2. Checks each landmark in each window for excessive missing data
    3. Marks windows as "bad" if key landmarks are missing too often
    4. Groups landmarks by facial region (eyes, mouth, etc.)
    
    Returns:
    Three DataFrames: keypoint_summary, metric_summary, bad_window_indices
    """
    df = pd.read_csv(fp)
    base = os.path.basename(fp)
    wranges = window_ranges(len(df), window_size, overlap)
    total_windows = len(wranges)

    # Handle files too short for analysis
    if total_windows == 0:
        kp_rows = [{"file": base, "keypoint": i, "bad_windows": 0, "total_windows": 0, "pct_bad": np.nan} for i in RELEVANT_KPS]
        met_rows = [{"file": base, "metric": m, "bad_windows": 0, "total_windows": 0, "pct_bad": np.nan} for m in METRIC_KPS.keys()]
        return pd.DataFrame(kp_rows), pd.DataFrame(met_rows), pd.DataFrame([])

    # Initialize counters
    kp_bad_counts = {i: 0 for i in RELEVANT_KPS}
    metric_bad_counts = {m: 0 for m in METRIC_KPS}
    metric_bad_indices = {m: [] for m in METRIC_KPS}

    # Analyze each window
    for win_idx, (s, e) in enumerate(wranges):
        df_w = df.iloc[s:e].reset_index(drop=True)

        # Check each individual landmark
        kp_bad = {}
        for i in RELEVANT_KPS:
            missing = kp_missing_series(df_w, i, conf_thresh)
            longest_gap = _max_true_run_length(missing)
            is_bad = (longest_gap > max_interp)
            kp_bad[i] = is_bad
            if is_bad:
                kp_bad_counts[i] += 1

        # Check each facial region (metric) - a region is bad if ANY of its landmarks are bad
        for m, kps in METRIC_KPS.items():
            if any(kp_bad.get(i, True) for i in kps):
                metric_bad_counts[m] += 1
                metric_bad_indices[m].append((win_idx, s, e))

    # Build summary statistics
    kp_rows = []
    for i in RELEVANT_KPS:
        bw = kp_bad_counts[i]
        kp_rows.append({
            "file": base, "keypoint": i, "bad_windows": bw, 
            "total_windows": total_windows, "pct_bad": bw / total_windows
        })

    met_rows = []
    for m in METRIC_KPS:
        bw = metric_bad_counts[m]
        met_rows.append({
            "file": base, "metric": m, "bad_windows": bw, 
            "total_windows": total_windows, "pct_bad": bw / total_windows
        })

    # Record specific bad windows for later masking
    met_idx_rows = []
    for m, entries in metric_bad_indices.items():
        for (win_idx, s, e) in entries:
            met_idx_rows.append({
                "file": base, "metric": m, "window_index": win_idx, 
                "start_frame": s, "end_frame_exclusive": e
            })

    return pd.DataFrame(kp_rows), pd.DataFrame(met_rows), pd.DataFrame(met_idx_rows)

def run_qc(input_dir: str, output_dir: str, window_size: int, overlap: float, conf_thresh: float, max_interp: int):
    """Run quality control analysis on all CSV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    all_kp, all_met, all_met_idx = [], [], []

    for f in tqdm(files, desc="QC scanning"):
        fp = os.path.join(input_dir, f)
        try:
            kp_df, met_df, met_idx_df = analyze_file_qc(fp, window_size, overlap, conf_thresh, max_interp)
        except Exception as e:
            # Record error information if analysis fails
            kp_df = pd.DataFrame([{"file": f, "keypoint": None, "bad_windows": None, "total_windows": None, "pct_bad": None, "error": str(e)}])
            met_df = pd.DataFrame([{"file": f, "metric": None, "bad_windows": None, "total_windows": None, "pct_bad": None, "error": str(e)}])
            met_idx_df = pd.DataFrame([{"file": f, "metric": None, "window_index": None, "start_frame": None, "end_frame_exclusive": None, "error": str(e)}])

        all_kp.append(kp_df)
        all_met.append(met_df)
        all_met_idx.append(met_idx_df)

    # Combine results and save
    out_kp = pd.concat(all_kp, ignore_index=True)
    out_met = pd.concat(all_met, ignore_index=True)
    out_met_idx = pd.concat(all_met_idx, ignore_index=True)

    out_kp_path = os.path.join(output_dir, "keypoint_bad_windows.csv")
    out_met_path = os.path.join(output_dir, "metric_bad_windows.csv")
    out_met_idx_path = os.path.join(output_dir, "metric_bad_window_indices.csv")

    out_kp.to_csv(out_kp_path, index=False)
    out_met.to_csv(out_met_path, index=False)
    out_met_idx.to_csv(out_met_idx_path, index=False)

    print("Wrote QC results:")
    print(" ", out_kp_path)
    print(" ", out_met_path)
    print(" ", out_met_idx_path)

    return out_kp_path, out_met_path, out_met_idx_path

def summarize_bad_windows(path: str):
    """Quick summary statistics from QC results."""
    df = pd.read_csv(path)
    total_bad = df['bad_windows'].sum()
    total_windows = df['total_windows'].sum()
    pct = total_bad / total_windows * 100 if total_windows > 0 else float('nan')
    return total_bad, total_windows, pct

# =============================================================================
# COORDINATE NORMALIZATION: ORIGINAL METHOD
# =============================================================================

def _stabilize_points(df_raw: pd.DataFrame, landmark_indices: list[int]) -> dict[int, tuple[pd.Series, pd.Series]]:
    """
    Apply head stabilization using eye corners as reference points.
    
    This normalizes for head position, rotation, and size by:
    1. Centering coordinates at the midpoint between eye corners
    2. Rotating to make the eye line horizontal  
    3. Scaling by the distance between eye corners
    """
    # Get eye corner coordinates
    xL, yL = df_raw[_col(ANCHOR_L, "x")], df_raw[_col(ANCHOR_L, "y")]
    xR, yR = df_raw[_col(ANCHOR_R, "x")], df_raw[_col(ANCHOR_R, "y")]

    # Center at midpoint between eyes
    cx = (xL + xR) / 2.0
    cy = (yL + yR) / 2.0

    # Calculate rotation angle to make eye line horizontal
    dx = xR - xL
    dy = yR - yL
    theta = np.arctan2(dy, dx)

    cosT = np.cos(theta)
    sinT = np.sin(theta)

    # Calculate scaling factor (inter-ocular distance)
    d_io = np.sqrt(dx*dx + dy*dy).where(lambda s: s > 1e-6, np.nan)

    # Apply transformation to all requested landmarks
    stabilized = {}
    for k in landmark_indices:
        X = df_raw[_col(k, "x")] - cx
        Y = df_raw[_col(k, "y")] - cy
        Xr =  cosT * X + sinT * Y   # Rotate by -theta
        Yr = -sinT * X + cosT * Y
        stabilized[k] = (Xr / d_io, Yr / d_io)
    
    return stabilized

# =============================================================================
# COORDINATE NORMALIZATION: PROCRUSTES METHOD  
# =============================================================================

def procrustes_analysis(X, Y):
    """
    Perform Procrustes analysis to align two sets of landmarks.
    
    Finds the optimal similarity transformation (translation + rotation + scaling) 
    to align shape Y to shape X. More robust than eye-corner method because it uses
    all available landmarks instead of just two points.
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    
    # Remove translation by centering both shapes at origin
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Calculate optimal scaling factor
    norm_X = np.sqrt(np.sum(X_centered**2))
    norm_Y = np.sqrt(np.sum(Y_centered**2))
    
    if norm_Y < 1e-10:  # Avoid division by zero
        return Y, 1.0, np.eye(2), np.zeros(2)
    
    scale = norm_X / norm_Y
    Y_scaled = Y_centered * scale
    
    # Find optimal rotation using Singular Value Decomposition (SVD)
    H = Y_scaled.T @ X_centered  # Cross-covariance matrix
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure we have a proper rotation (not a reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and final translation
    Y_rotated = Y_scaled @ R.T
    translation = X.mean(axis=0) - Y_rotated.mean(axis=0)
    Y_aligned = Y_rotated + translation
    
    return Y_aligned, scale, R, translation

def compute_procrustes_alignment(df: pd.DataFrame, reference_landmarks: list = None) -> pd.DataFrame:
    """
    Apply Procrustes alignment to all landmarks in a DataFrame.
    
    1. Selects stable landmarks as reference points
    2. Calculates the average shape across all valid frames
    3. Aligns each frame to this average shape
    4. Adds new columns with "_proc" suffix for aligned coordinates
    """
    if reference_landmarks is None:
        reference_landmarks = [29,30,36,45] # the temples and 2 points on the nose
    
    # Check if we have enough data for alignment
    x_cols = [f"x{i}" for i in reference_landmarks]
    y_cols = [f"y{i}" for i in reference_landmarks]
    required_cols = x_cols + y_cols
    
    available_cols = [col for col in required_cols if col in df.columns]
    if len(available_cols) < 6:  # Need at least 3 points (6 coordinates)
        print(f"Warning: Not enough landmark columns for Procrustes alignment. Found {len(available_cols)//2} points.")
        return df
    
    # Find which landmarks we actually have data for
    available_landmarks = []
    for i in reference_landmarks:
        if f"x{i}" in df.columns and f"y{i}" in df.columns:
            available_landmarks.append(i)
    
    if len(available_landmarks) < 3:
        print("Warning: Need at least 3 complete landmarks for Procrustes alignment.")
        return df
    
    # Extract landmark coordinates into 3D array: (frames, landmarks, coordinates)
    n_frames = len(df)
    n_landmarks = len(available_landmarks)
    
    coords = np.zeros((n_frames, n_landmarks, 2))
    valid_frames = np.ones(n_frames, dtype=bool)
    
    # Fill coordinate array and track which frames have complete data
    for frame_idx in range(n_frames):
        for lm_idx, landmark in enumerate(available_landmarks):
            x_val = df.loc[frame_idx, f"x{landmark}"]
            y_val = df.loc[frame_idx, f"y{landmark}"]
            
            if pd.isna(x_val) or pd.isna(y_val):
                valid_frames[frame_idx] = False
                break
            
            coords[frame_idx, lm_idx, 0] = x_val
            coords[frame_idx, lm_idx, 1] = y_val
    
    if valid_frames.sum() == 0:
        print("Warning: No frames with complete landmark data for Procrustes alignment.")
        return df
    
    # Calculate reference shape (average of all valid frames)
    reference_shape = coords[valid_frames].mean(axis=0)
    
    # Apply Procrustes alignment to each frame
    df_aligned = df.copy()
    
    print(f"Applying Procrustes alignment to {n_frames} frames using {n_landmarks} landmarks...")
    
    for frame_idx in range(n_frames):
        if not valid_frames[frame_idx]:
            # Set aligned coordinates to NaN for frames with missing data
            for landmark in available_landmarks:
                df_aligned.loc[frame_idx, f"x{landmark}_proc"] = np.nan
                df_aligned.loc[frame_idx, f"y{landmark}_proc"] = np.nan
            continue
        
        current_shape = coords[frame_idx]
        
        try:
            # Apply Procrustes transformation
            aligned_shape, scale, rotation, translation = procrustes_analysis(reference_shape, current_shape)
            
            # Store aligned coordinates in new columns
            for lm_idx, landmark in enumerate(available_landmarks):
                df_aligned.loc[frame_idx, f"x{landmark}_proc"] = aligned_shape[lm_idx, 0]
                df_aligned.loc[frame_idx, f"y{landmark}_proc"] = aligned_shape[lm_idx, 1]
                
        except Exception as e:
            print(f"Warning: Procrustes alignment failed for frame {frame_idx}: {e}")
            # Set to NaN on failure
            for landmark in available_landmarks:
                df_aligned.loc[frame_idx, f"x{landmark}_proc"] = np.nan
                df_aligned.loc[frame_idx, f"y{landmark}_proc"] = np.nan
    
    print(f"Procrustes alignment completed. Added _proc columns for {len(available_landmarks)} landmarks.")
    return df_aligned

# =============================================================================
# FEATURE COMPUTATION: ORIGINAL METHOD
# =============================================================================

def compute_blink(df_raw: pd.DataFrame) -> pd.Series:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection using original coordinates.
    EAR measures how "open" the eyes are. Lower values = more closed eyes.
    """
    needed = EYES["L"] + EYES["R"] + [ANCHOR_L, ANCHOR_R]
    if not _have_cols(df_raw, needed):
        missing = [c for i in needed for c in (f"x{i}", f"y{i}") if c not in df_raw.columns]
        raise KeyError(f"compute_blink(): missing columns: {missing[:8]}{'...' if len(missing)>8 else ''}")

    # Apply head stabilization
    S = _stabilize_points(df_raw, needed)

    def dist(i, j):
        """Calculate distance between two stabilized landmarks"""
        xi, yi = S[i]
        xj, yj = S[j]
        return np.hypot(xi - xj, yi - yj)

    # Left eye EAR: (||38-42|| + ||39-41||) / (2 * ||37-40||)
    L = EYES["L"]
    left_ear = (dist(L[1], L[5]) + dist(L[2], L[4])) / (2.0 * dist(L[0], L[3]))
    
    # Right eye EAR: (||44-48|| + ||45-47||) / (2 * ||43-46||)
    R = EYES["R"]
    right_ear = (dist(R[1], R[5]) + dist(R[2], R[4])) / (2.0 * dist(R[0], R[3]))

    return (left_ear + right_ear) / 2.0

def compute_mouth_distance(df_raw: pd.DataFrame) -> pd.Series:
    """Calculate mouth opening distance using original coordinates."""
    needed = [ANCHOR_L, ANCHOR_R, 63, 67]
    if not _have_cols(df_raw, needed):
        missing = [c for i in needed for c in (f"x{i}", f"y{i}") if c not in df_raw.columns]
        raise KeyError(f"compute_mouth_distance(): missing columns: {missing[:8]}{'...' if len(missing)>8 else ''}")
    
    S = _stabilize_points(df_raw, needed)
    x63, y63 = S[63]
    x67, y67 = S[67]
    return np.hypot(x63 - x67, y63 - y67)

def compute_head_rotation(df_raw: pd.DataFrame) -> pd.Series:
    """Calculate head rotation angle using original coordinates."""
    dx = df_raw[_col(46, "x")] - df_raw[_col(37, "x")]
    dy = df_raw[_col(46, "y")] - df_raw[_col(37, "y")]
    return np.arctan2(dy, dx)

# =============================================================================
# FEATURE COMPUTATION: PROCRUSTES METHOD
# =============================================================================

def compute_blink_procrustes(df: pd.DataFrame) -> pd.Series:
    """Calculate EAR using Procrustes-aligned coordinates when available."""
    def get_coord(landmark, axis):
        """Get coordinate, preferring Procrustes-aligned version"""
        proc_col = f"{axis}{landmark}_proc"
        orig_col = f"{axis}{landmark}"
        if proc_col in df.columns:
            return df[proc_col].fillna(df.get(orig_col, np.nan))
        return df.get(orig_col, pd.Series(np.nan, index=df.index))
    
    def compute_distance(lm1, lm2):
        """Calculate Euclidean distance between two landmarks"""
        x1, y1 = get_coord(lm1, 'x'), get_coord(lm1, 'y')
        x2, y2 = get_coord(lm2, 'x'), get_coord(lm2, 'y')
        return np.hypot(x1 - x2, y1 - y2)
    
    # Left EAR: (||38-42|| + ||39-41||) / (2 * ||37-40||)
    left_ear = (compute_distance(38, 42) + compute_distance(39, 41)) / (2.0 * compute_distance(37, 40))
    
    # Right EAR: (||44-48|| + ||45-47||) / (2 * ||43-46||)
    right_ear = (compute_distance(44, 48) + compute_distance(45, 47)) / (2.0 * compute_distance(43, 46))
    
    return (left_ear + right_ear) / 2.0

def compute_mouth_distance_procrustes(df: pd.DataFrame) -> pd.Series:
    """Calculate mouth opening using Procrustes-aligned coordinates."""
    def get_coord(landmark, axis):
        """Get coordinate, preferring Procrustes-aligned version"""
        proc_col = f"{axis}{landmark}_proc"
        orig_col = f"{axis}{landmark}"
        if proc_col in df.columns:
            return df[proc_col].fillna(df.get(orig_col, np.nan))
        return df.get(orig_col, pd.Series(np.nan, index=df.index))
    
    x63, y63 = get_coord(63, 'x'), get_coord(63, 'y')
    x67, y67 = get_coord(67, 'x'), get_coord(67, 'y')
    
    return np.hypot(x63 - x67, y63 - y67)

def compute_head_rotation_procrustes(df: pd.DataFrame) -> pd.Series:
    """Calculate head rotation using Procrustes-aligned coordinates."""
    def get_coord(landmark, axis):
        """Get coordinate, preferring Procrustes-aligned version"""
        proc_col = f"{axis}{landmark}_proc"
        orig_col = f"{axis}{landmark}"
        if proc_col in df.columns:
            return df[proc_col].fillna(df.get(orig_col, np.nan))
        return df.get(orig_col, pd.Series(np.nan, index=df.index))
    
    x37, y37 = get_coord(37, 'x'), get_coord(37, 'y')
    x46, y46 = get_coord(46, 'x'), get_coord(46, 'y')
    
    dx = x46 - x37
    dy = y46 - y37
    return np.arctan2(dy, dx)

# =============================================================================
# REGIONAL AVERAGING AND MOTION FEATURES
# =============================================================================

def compute_averages(df: pd.DataFrame, conf_thresh: float) -> pd.DataFrame:
    """
    Convert individual landmarks into regional averages using original coordinates.
    Creates meaningful regional features by averaging landmarks within each facial region.
    Low-confidence landmarks are masked out before averaging.
    """
    center_idxs = list(range(28, 37))  # Nose bridge landmarks
    left_eye_idxs = EYES["L"]          # Left eye contour
    right_eye_idxs = EYES["R"]         # Right eye contour

    def mean_cols(prefix: str, idxs: list[int]):
        """Calculate mean across specified columns"""
        return df[[f"{prefix}{i}" for i in idxs]].mean(axis=1)

    # Create regional averages
    averaged_df = pd.DataFrame({
        "center_face_x":    mean_cols("x", center_idxs),
        "center_face_y":    mean_cols("y", center_idxs),
        "center_face_prob": mean_cols("prob", center_idxs),

        "left_eye_x":       mean_cols("x", left_eye_idxs),
        "left_eye_y":       mean_cols("y", left_eye_idxs),
        "left_eye_prob":    mean_cols("prob", left_eye_idxs),

        "right_eye_x":      mean_cols("x", right_eye_idxs),
        "right_eye_y":      mean_cols("y", right_eye_idxs),
        "right_eye_prob":   mean_cols("prob", right_eye_idxs),

        # Individual pupil points (if available)
        "left_pupil_x":     df.get("x69", pd.Series(np.nan, index=df.index)),
        "left_pupil_y":     df.get("y69", pd.Series(np.nan, index=df.index)),
        "left_pupil_prob":  df.get("prob69", pd.Series(np.nan, index=df.index)),

        "right_pupil_x":    df.get("x70", pd.Series(np.nan, index=df.index)),
        "right_pupil_y":    df.get("y70", pd.Series(np.nan, index=df.index)),
        "right_pupil_prob": df.get("prob70", pd.Series(np.nan, index=df.index)),
    })

    # Apply confidence masking: set coordinates to NaN where confidence is too low
    for part in ["center_face", "left_eye", "right_eye", "left_pupil", "right_pupil"]:
        prob_col = f"{part}_prob"
        for axis in ["x", "y"]:
            val_col = f"{part}_{axis}"
            if prob_col in averaged_df.columns:
                averaged_df.loc[averaged_df[prob_col] < conf_thresh, val_col] = np.nan

    # Fill gaps using linear interpolation
    averaged_df.interpolate(method="linear", limit_direction="both", inplace=True)
    return averaged_df

def compute_procrustes_averages(df: pd.DataFrame, conf_thresh: float) -> pd.DataFrame:
    """
    Convert landmarks into regional averages using Procrustes-aligned coordinates when available.
    Prioritizes "_proc" suffixed columns, falling back to original coordinates.
    """
    center_idxs = list(range(28, 37))  
    left_eye_idxs = [37, 38, 39, 40, 41, 42]
    right_eye_idxs = [43, 44, 45, 46, 47, 48]

    def mean_cols_proc(prefix: str, idxs: list[int]):
        """Calculate mean, preferring Procrustes-aligned columns"""
        proc_cols = [f"{prefix}{i}_proc" for i in idxs]
        orig_cols = [f"{prefix}{i}" for i in idxs]
        
        available_proc = [col for col in proc_cols if col in df.columns]
        available_orig = [col for col in orig_cols if col in df.columns]
        
        if available_proc:
            return df[available_proc].mean(axis=1)
        elif available_orig:
            return df[available_orig].mean(axis=1)
        else:
            return pd.Series(np.nan, index=df.index)

    # Build regional features using best available coordinates
    averaged_df = pd.DataFrame({
        "center_face_x":    mean_cols_proc("x", center_idxs),
        "center_face_y":    mean_cols_proc("y", center_idxs),
        "center_face_prob": df[[f"prob{i}" for i in center_idxs if f"prob{i}" in df.columns]].mean(axis=1) if any(f"prob{i}" in df.columns for i in center_idxs) else pd.Series(np.nan, index=df.index),

        "left_eye_x":       mean_cols_proc("x", left_eye_idxs),
        "left_eye_y":       mean_cols_proc("y", left_eye_idxs),
        "left_eye_prob":    df[[f"prob{i}" for i in left_eye_idxs if f"prob{i}" in df.columns]].mean(axis=1) if any(f"prob{i}" in df.columns for i in left_eye_idxs) else pd.Series(np.nan, index=df.index),

        "right_eye_x":      mean_cols_proc("x", right_eye_idxs),
        "right_eye_y":      mean_cols_proc("y", right_eye_idxs),
        "right_eye_prob":   df[[f"prob{i}" for i in right_eye_idxs if f"prob{i}" in df.columns]].mean(axis=1) if any(f"prob{i}" in df.columns for i in right_eye_idxs) else pd.Series(np.nan, index=df.index),

        # Pupils: try Procrustes first, fall back to original
        "left_pupil_x":     df.get("x69_proc", df.get("x69", pd.Series(np.nan, index=df.index))),
        "left_pupil_y":     df.get("y69_proc", df.get("y69", pd.Series(np.nan, index=df.index))),
        "left_pupil_prob":  df.get("prob69", pd.Series(np.nan, index=df.index)),

        "right_pupil_x":    df.get("x70_proc", df.get("x70", pd.Series(np.nan, index=df.index))),
        "right_pupil_y":    df.get("y70_proc", df.get("y70", pd.Series(np.nan, index=df.index))),
        "right_pupil_prob": df.get("prob70", pd.Series(np.nan, index=df.index)),
    })

    # Apply confidence masking
    for part in ["center_face", "left_eye", "right_eye", "left_pupil", "right_pupil"]:
        prob_col = f"{part}_prob"
        for axis in ["x", "y"]:
            val_col = f"{part}_{axis}"
            if prob_col in averaged_df.columns:
                averaged_df.loc[averaged_df[prob_col] < conf_thresh, val_col] = np.nan

    # Fill gaps with interpolation
    averaged_df.interpolate(method="linear", limit_direction="both", inplace=True)
    return averaged_df

def compute_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate movement magnitude (speed) for each facial region.
    
    For every x/y coordinate pair, calculates frame-to-frame movement magnitude:
    magnitude = sqrt((x[t] - x[t-1])^2 + (y[t] - y[t-1])^2)
    
    This captures how much each facial region is moving between frames.
    """
    for col in list(df.columns):
        if col.endswith("_x"):
            y_col = col.replace("_x", "_y")
            if y_col in df.columns:
                mag_col = col.replace("_x", "_magnitude")
                
                # Calculate frame-to-frame differences
                dx = df[col].diff()
                dy = df[y_col].diff()
                
                # Calculate Euclidean distance (magnitude of movement)
                df[mag_col] = np.hypot(dx, dy)
                
                # Set first frame to 0 (no previous frame to compare to)
                df.loc[df.index[0], mag_col] = 0.0
    
    return df

def compute_combined_eyes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined eye features by averaging left and right eye measurements.
    Provides a single "average eye" measure that's more stable than individual eyes.
    """
    df["avg_pupil_x"] = (df["left_pupil_x"] + df["right_pupil_x"]) / 2
    df["avg_pupil_y"] = (df["left_pupil_y"] + df["right_pupil_y"]) / 2
    df["avg_pupil_magnitude"] = np.hypot(df["avg_pupil_x"], df["avg_pupil_y"])
    return df

# =============================================================================
# QUALITY CONTROL MASKING
# =============================================================================

def load_bad_windows(bad_idx_csv: str, qc_window_frames: int, qc_overlap: float):
    """
    Load the list of "bad" time windows identified during quality control.
    Converts QC results into format for masking unreliable data during preprocessing.
    
    Returns:
    bad_map: Dictionary mapping filename -> metric -> list of (start, end) frame ranges
    step_frames: Number of frames between window starts
    """
    if not os.path.isfile(bad_idx_csv):
        sys.exit(f"[FATAL] QC bad-window index CSV not found: {bad_idx_csv}")

    df = pd.read_csv(bad_idx_csv)
    if 'file' not in df.columns or 'metric' not in df.columns:
        sys.exit("[FATAL] QC CSV must include 'file' and 'metric' columns.")

    # Normalize filenames and metric names for consistent matching
    df['file'] = df['file'].astype(str).map(lambda s: os.path.basename(s).strip())
    df['metric'] = df['metric'].astype(str).str.strip().str.lower()

    # Check if we have explicit frame ranges or need to calculate them
    has_spans = {'start_frame', 'end_frame_exclusive'}.issubset(df.columns)
    has_index = 'window_index' in df.columns

    step_frames = int(round(qc_window_frames * (1.0 - qc_overlap)))

    # Calculate frame ranges from window indices if needed
    if not has_spans and has_index:
        df = df.copy()
        widx = df['window_index'].astype(int)
        df['start_frame'] = (widx * step_frames).astype(int)
        df['end_frame_exclusive'] = (df['start_frame'] + qc_window_frames).astype(int)

    # Build nested dictionary: filename -> metric -> list of bad frame ranges
    bad_map = {}
    for (fname, metric), grp in df.groupby(['file', 'metric']):
        spans = list(zip(grp['start_frame'].astype(int), grp['end_frame_exclusive'].astype(int)))
        bad_map.setdefault(fname, {})[metric] = spans

    total = sum(len(v) for d in bad_map.values() for v in d.values())
    print(f"[QC] Loaded {total} bad window span(s) from {bad_idx_csv} (window={qc_window_frames}, step={step_frames}).")
    return bad_map, step_frames

def apply_bad_masks(file_basename: str, df_proc: pd.DataFrame, bad_map: dict, qc_to_columns: dict):
    """
    Apply quality control masks by setting unreliable data to NaN.
    
    Takes bad windows identified during QC and masks out corresponding feature columns
    during those time periods. Different facial regions can have different bad windows.
    
    Returns:
    df_proc: Modified DataFrame with NaN masks applied
    stats: Dictionary with masking statistics per metric
    total_masked_frames: Total number of frames masked across all metrics
    """
    n_frames = len(df_proc)
    stats = {}
    total_masked_frames = 0

    # Get bad windows for this specific file
    present = bad_map.get(file_basename, {})
    
    # Create boolean masks for each metric
    metric_masks = {m: np.zeros(n_frames, dtype=bool) for m in qc_to_columns.keys()}

    # Build per-metric masks from bad window ranges
    for qc_metric_raw, windows in present.items():
        qc_metric = qc_metric_raw.lower()
        if qc_metric not in metric_masks:
            continue
            
        # Mark all frames in bad windows as True
        for (s, e) in windows:
            s = max(0, int(s))  # Ensure we don't go below 0
            e = min(n_frames, int(e))  # Ensure we don't exceed data length
            if s < e:
                metric_masks[qc_metric][s:e] = True

    # Apply masks and collect statistics
    for qc_metric, mask in metric_masks.items():
        frames_masked = int(mask.sum())
        
        # Set corresponding feature columns to NaN where mask is True
        if frames_masked > 0:
            for col in qc_to_columns[qc_metric]:
                if col in df_proc.columns:
                    df_proc.loc[mask, col] = np.nan
        
        # Record statistics for reporting
        stats[qc_metric] = {
            "frames_total": n_frames,
            "frames_masked": frames_masked,
            "pct_masked": (frames_masked / n_frames) if n_frames > 0 else np.nan,
            "windows_masked": int(len(present.get(qc_metric, [])))
        }
        total_masked_frames += frames_masked

    return df_proc, stats, total_masked_frames

# =============================================================================
# TEMPORAL FILTERING
# =============================================================================

def filter_data_safe_preserve_nans(
    df: pd.DataFrame,
    fs: float = 60.0,
    cutoff: float = 10.0,
    order: int = 4,
    audit: bool = True,
) -> pd.DataFrame:
    """
    Apply Butterworth low-pass filter to smooth temporal data while preserving NaN patterns.
    
    This filter removes high-frequency noise while preserving underlying signal trends.
    Key features:
    - Zero-phase filtering (no temporal shift)
    - Preserves original NaN locations exactly
    - Handles short sequences gracefully
    - Internal gap-filling for filter stability only
    
    Parameters:
    df: DataFrame with numeric columns to filter
    fs: Sampling rate in Hz (frames per second)
    cutoff: Cutoff frequency in Hz (frequencies above this are attenuated)
    order: Filter order (higher = steeper rolloff)
    audit: Whether to print summary statistics
    
    Returns:
    DataFrame with filtered data, original NaN patterns preserved
    """
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not num_cols:
        return out

    # Design Butterworth filter
    wn = cutoff / (fs / 2.0)  # Normalize to Nyquist frequency
    wn = min(max(wn, 1e-6), 0.999999)  # Clamp to valid range
    sos = butter(order, wn, btype='low', output='sos')

    # Track filtering results for audit
    all_nan_before, too_short, pad_failed = [], [], []

    for c in num_cols:
        x = out[c].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(x)

        # Skip columns with no valid data
        if not finite.any():
            all_nan_before.append(c)
            continue

        # Internal gap filling for filter stability only
        # (These filled values will be restored to NaN after filtering)
        idx = np.arange(x.size)
        x_filled = x.copy()
        if not finite.all():
            # Linear interpolation across gaps
            x_filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
            
            # Constant extrapolation at edges
            first, last = np.where(finite)[0][[0, -1]]
            x_filled[:first] = x_filled[first]
            x_filled[last+1:] = x_filled[last]

        # Check minimum length requirement for stable filtering
        min_len = max(25, 4 * order + 5)
        if x_filled.size < min_len:
            too_short.append(c)
            out[c] = x  # Keep original data
            continue

        # Apply zero-phase filtering
        try:
            y = sosfiltfilt(sos, x_filled, padtype='odd')
        except Exception:
            pad_failed.append(c)
            out[c] = x  # Keep original data
            continue

        # Restore original NaN locations
        y[~finite] = np.nan
        out[c] = y

    # Print audit summary if requested
    if audit:
        if all_nan_before:
            print(f"[Butterworth Filter] All‑NaN (unchanged): {len(all_nan_before)} cols")
        if too_short:
            print(f"[Butterworth Filter] Unfiltered (too short): {len(too_short)} cols")
        if pad_failed:
            print(f"[Butterworth Filter] Unfiltered (pad failure): {len(pad_failed)} cols")

    return out

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_data_consistent(
    input_dir: str, 
    output_dir: str, 
    qc_bad_idx_file: str, 
    qc_to_columns: dict,
    qc_window_frames: int, 
    qc_overlap: float, 
    conf_thresh: float,
    coordinate_system: str = "procrustes",
    apply_butterworth: bool = True,
    fs: float = 60.0, 
    cutoff: float = 10.0, 
    filter_order: int = 4
):
    """
    Main processing pipeline that coordinates all steps.
    
    Orchestrates the complete preprocessing workflow:
    1. Load quality control results
    2. Process each CSV file:
       a. Apply coordinate normalization (Procrustes or original)
       b. Compute facial features and regional averages
       c. Apply quality control masks
       d. Apply temporal smoothing filter
    3. Save processed files and generate report
    
    Parameters:
    input_dir: Directory containing raw CSV files
    output_dir: Directory to save processed files
    qc_bad_idx_file: Path to QC results CSV
    qc_to_columns: Mapping of QC metrics to feature columns
    qc_window_frames: QC window size (must match QC stage)
    qc_overlap: QC window overlap (must match QC stage)
    conf_thresh: Minimum confidence threshold for valid landmarks
    coordinate_system: "procrustes" or "original" normalization method
    apply_butterworth: Whether to apply temporal smoothing
    fs: Sampling rate for filtering
    cutoff: Filter cutoff frequency
    filter_order: Filter order
    
    Returns:
    Path to processing report CSV file
    """
    print("="*80)
    print("FACIAL POSE PROCESSING PIPELINE")
    print("="*80)
    
    # Load quality control results
    print("Loading quality control results...")
    bad_map, step_frames = load_bad_windows(qc_bad_idx_file, qc_window_frames, qc_overlap)
    
    # Verify file overlap between QC and input data
    proc_files = {os.path.basename(f) for f in os.listdir(input_dir) if f.endswith('.csv')}
    qc_files = set(bad_map.keys())
    intersection = proc_files & qc_files
    
    if not intersection:
        print("[WARNING] No filename overlap between QC and RAW_INPUT_DIR.")
        print(f"QC has {len(qc_files)} files, input has {len(proc_files)} files")
    else:
        print(f"[OK] Found {len(intersection)} files with QC data out of {len(proc_files)} input files")

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    print(f"Processing {len(csv_files)} files using {coordinate_system} coordinate system")
    if apply_butterworth:
        print(f"Butterworth filter: {cutoff}Hz cutoff, {fs}Hz sampling rate")
    
    report_rows = []

    # Process each file
    for csv_file in tqdm(csv_files, desc=f"Processing with {coordinate_system} coordinates"):
        file_path = os.path.join(input_dir, csv_file)
        
        try:
            # Load raw data
            df_raw = pd.read_csv(file_path)
            
            # Apply coordinate normalization
            if coordinate_system == "procrustes":
                print(f"  [{csv_file}] Applying Procrustes alignment...")
                df_raw = compute_procrustes_alignment(df_raw)
                
                # Use Procrustes-aware feature computation
                averaged_df = compute_procrustes_averages(df_raw, conf_thresh)
                averaged_df = compute_magnitude(averaged_df)
                averaged_df = compute_combined_eyes(averaged_df)
                
                # Compute derived features using Procrustes coordinates
                averaged_df['blink_dist'] = compute_blink_procrustes(df_raw)
                averaged_df['head_rotation_angle'] = compute_head_rotation_procrustes(df_raw)
                averaged_df['mouth_dist'] = compute_mouth_distance_procrustes(df_raw)
                
            else:  # original coordinate system
                # Use original stabilization pipeline
                averaged_df = compute_averages(df_raw, conf_thresh)
                averaged_df = compute_magnitude(averaged_df)
                averaged_df = compute_combined_eyes(averaged_df)
                
                # Compute derived features using original stabilization
                averaged_df['blink_dist'] = compute_blink(df_raw)
                averaged_df['head_rotation_angle'] = compute_head_rotation(df_raw)
                averaged_df['mouth_dist'] = compute_mouth_distance(df_raw)

            # Apply quality control masks
            averaged_df, stats, total_masked_frames = apply_bad_masks(
                file_basename=os.path.basename(csv_file),
                df_proc=averaged_df,
                bad_map=bad_map,
                qc_to_columns=qc_to_columns
            )

            # Apply temporal smoothing filter
            if apply_butterworth:
                averaged_df = filter_data_safe_preserve_nans(
                    averaged_df, 
                    fs=fs, 
                    cutoff=cutoff, 
                    order=filter_order, 
                    audit=False
                )

            # Save processed file
            out_path = os.path.join(output_dir, os.path.splitext(csv_file)[0] + ".csv")
            averaged_df.to_csv(out_path, index=False)

            # Record processing statistics
            for qc_metric, st in stats.items():
                report_rows.append({
                    "file": csv_file,
                    "qc_metric": qc_metric,
                    "frames_total": st["frames_total"],
                    "frames_masked": st["frames_masked"],
                    "pct_masked": st["pct_masked"],
                    "windows_masked": st["windows_masked"],
                    "coordinate_system": coordinate_system,
                    "butterworth_applied": apply_butterworth,
                    "butterworth_fs": fs if apply_butterworth else None,
                    "butterworth_cutoff": cutoff if apply_butterworth else None
                })
                
        except Exception as e:
            print(f"ERROR processing {csv_file}: {e}")
            # Record error in report
            report_rows.append({
                "file": csv_file,
                "qc_metric": "ERROR",
                "frames_total": 0,
                "frames_masked": 0,
                "pct_masked": np.nan,
                "windows_masked": 0,
                "coordinate_system": coordinate_system,
                "butterworth_applied": apply_butterworth,
                "error": str(e)
            })

    # Save processing report
    report_path = os.path.join(output_dir, "preprocess_drop_report_consistent.csv")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    
    print("="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Processed files saved to: {output_dir}")
    print(f"Processing report saved to: {report_path}")
    
    return report_path