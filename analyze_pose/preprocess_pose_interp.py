"""
PURPOSE
-------
Process raw face/eye keypoint CSVs (one row per frame). Mask low-confidence points,
fill only short gaps, compute summary signals, and write a lean CSV per input file.
Also log windows where a whole region goes missing for too long to interpolate safely.

HOW TO USE
----------
1) Set 'input_directory' to your raw CSV folder.
2) Set 'output_directory' for processed CSVs.
3) Run. A 'logs/dropped_windows.csv' will record "all-dead" regions over long runs.

KEY IDEAS
---------
- Each keypoint has (x, y, prob) per frame. If prob < CONFIDENCE_THRESHOLD, (x,y) is treated as NaN.
- Only fill short NaN runs (<= MAX_INTERP) by interpolation. Longer runs remain NaN.
- Region averages are over surviving keypoints only. A region is "present" if ANY keypoint survives that frame.
- Blink distance = vertical eyelid opening (avg of L/R). Head rotation = angle between eye corners.
- Mouth distance = distance between upper/lower lip landmarks.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------ Config (edit as needed) ------------
CONFIDENCE_THRESHOLD = 0.3   # if a keypoint's prob < threshold, its x/y is missing
WINDOW_SIZE = 1800           # frames per window for logging
OVERLAP = 0.0                # 0 = no overlap, 0.5 = 50% overlap
MAX_INTERP = 30              # max consecutive missing frames to fill by interpolation
LOG_DIRNAME = "logs"         # subfolder (inside output) for logs
# -------------------------------------------------

# ---- I/O ----
def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# ---- utilities ----
def window_ranges(n_rows: int, window_size: int, overlap: float):
    step = max(1, int(round(window_size * (1 - overlap))))
    for start in range(0, n_rows - window_size + 1, step):
        yield start, start + window_size

def _max_true_run_length(bools: pd.Series) -> int:
    run = max_run = 0
    for v in bools.to_numpy():
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run

# ---- logging logic: region is "present" if ANY keypoint survives in a frame ----
def _log_region_all_missing(df_w: pd.DataFrame, base: str, widx: int, s: int, e: int,
                            max_interp: int, log_rows: list):
    """
    For each frame, a keypoint survives if prob >= THRESHOLD and x,y are non-NaN.
    Region is missing if NONE of its keypoints survive at that frame.
    Log the window if the longest consecutive region-missing run > max_interp.
    """
    regions = {
        "center_face": list(range(27, 36)),
        "left_eye":    list(range(36, 42)),
        "right_eye":   list(range(42, 48)),
        # Pupils combined: present if either 68 or 69 survives
        "pupils_combined": [68, 69],
    }

    for region, idxs in regions.items():
        survivors = []
        for i in idxs:
            xcol, ycol, pcol = f"x{i}", f"y{i}", f"prob{i}"
            if (xcol not in df_w.columns) or (ycol not in df_w.columns) or (pcol not in df_w.columns):
                survivors.append(pd.Series(False, index=df_w.index))
                continue
            kp_survive = (df_w[pcol] >= CONFIDENCE_THRESHOLD) & df_w[xcol].notna() & df_w[ycol].notna()
            survivors.append(kp_survive)

        region_present = survivors[0]
        for s_i in survivors[1:]:
            region_present = region_present | s_i

        region_missing = ~region_present
        L = _max_true_run_length(region_missing)
        if L > max_interp and log_rows is not None:
            log_rows.append({
                "file": base,
                "window_index": widx,
                "start": s,
                "end": e,
                "reason": f"{region} all keypoints missing > {max_interp} frames",
                "region": region,
                "columns": ";".join([f"x{i},y{i},prob{i}" for i in idxs]),
                "details": json.dumps({"missing_run_all_dead": int(L), "keypoints": idxs})
            })

# ---- averaging after per-keypoint masking ----
def compute_averages(df: pd.DataFrame, max_interp: int = MAX_INTERP, interpolate: bool = True) -> pd.DataFrame:
    """
    Mask each keypoint by its own prob, then average survivors within each region.
    Optionally interpolate short gaps on the resulting region columns.
    """
    def masked_group_mean(idxs):
        x_cols = [f"x{i}" for i in idxs]
        y_cols = [f"y{i}" for i in idxs]
        p_cols = [f"prob{i}" for i in idxs]

        X = df[x_cols].copy()
        Y = df[y_cols].copy()
        P = df[p_cols].copy()

        bad = P < CONFIDENCE_THRESHOLD
        X = X.mask(bad)
        Y = Y.mask(bad)

        x_mean = X.mean(axis=1)
        y_mean = Y.mean(axis=1)
        valid_ratio = (~bad).sum(axis=1) / len(idxs)  # fraction of kps that survived per frame
        return x_mean, y_mean, valid_ratio

    # Regions
    cf_x, cf_y, cf_valid = masked_group_mean(range(27, 36))   # center face
    le_x, le_y, le_valid = masked_group_mean(range(36, 42))   # left eye
    re_x, re_y, re_valid = masked_group_mean(range(42, 48))   # right eye

    # Pupils (single landmarks, but we keep them separate here; we'll combine later)
    lp_x = df["x68"].where(df["prob68"] >= CONFIDENCE_THRESHOLD, np.nan)
    lp_y = df["y68"].where(df["prob68"] >= CONFIDENCE_THRESHOLD, np.nan)
    rp_x = df["x69"].where(df["prob69"] >= CONFIDENCE_THRESHOLD, np.nan)
    rp_y = df["y69"].where(df["prob69"] >= CONFIDENCE_THRESHOLD, np.nan)

    out = pd.DataFrame({
        "center_face_x": cf_x,
        "center_face_y": cf_y,
        "center_face_valid_ratio": cf_valid,

        "left_eye_x": le_x,
        "left_eye_y": le_y,
        "left_eye_valid_ratio": le_valid,

        "right_eye_x": re_x,
        "right_eye_y": re_y,
        "right_eye_valid_ratio": re_valid,

        "left_pupil_x": lp_x,
        "left_pupil_y": lp_y,
        "right_pupil_x": rp_x,
        "right_pupil_y": rp_y,
    })

    if interpolate:
        pos_cols = [c for c in out.columns if c.endswith("_x") or c.endswith("_y")]
        out[pos_cols] = out[pos_cols].interpolate(method="linear",
                                                  limit=max_interp,
                                                  limit_direction="both")
    return out

def compute_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.endswith("_x"):
            y_col = col.replace("_x", "_y")
            if y_col in df.columns:
                mag_col = col.replace("_x", "_magnitude")
                df[mag_col] = np.sqrt(df[col]**2 + df[y_col]**2)
    return df

def compute_combined_eyes(df: pd.DataFrame) -> pd.DataFrame:
    # Use whichever pupil survives; average when both exist; NaN when both missing.
    left_x, right_x = df["left_pupil_x"].to_numpy(), df["right_pupil_x"].to_numpy()
    left_y, right_y = df["left_pupil_y"].to_numpy(), df["right_pupil_y"].to_numpy()
    df["avg_pupil_x"] = np.nanmean(np.vstack([left_x, right_x]), axis=0)
    df["avg_pupil_y"] = np.nanmean(np.vstack([left_y, right_y]), axis=0)
    df["avg_pupil_magnitude"] = np.sqrt(df["avg_pupil_x"]**2 + df["avg_pupil_y"]**2)
    return df

def _mask_xy(df, i):
    xcol, ycol, pcol = f"x{i}", f"y{i}", f"prob{i}"
    x = df[xcol].where(df[pcol] >= CONFIDENCE_THRESHOLD, np.nan)
    y = df[ycol].where(df[pcol] >= CONFIDENCE_THRESHOLD, np.nan)
    return x, y

def compute_blink(df_raw: pd.DataFrame, max_interp: int = MAX_INTERP) -> pd.Series:
    # Right eye: top (37,38), bottom (40,41)
    r_top_x = pd.concat([_mask_xy(df_raw, 37)[0], _mask_xy(df_raw, 38)[0]], axis=1).mean(axis=1)
    r_top_y = pd.concat([_mask_xy(df_raw, 37)[1], _mask_xy(df_raw, 38)[1]], axis=1).mean(axis=1)
    r_bot_x = pd.concat([_mask_xy(df_raw, 40)[0], _mask_xy(df_raw, 41)[0]], axis=1).mean(axis=1)
    r_bot_y = pd.concat([_mask_xy(df_raw, 40)[1], _mask_xy(df_raw, 41)[1]], axis=1).mean(axis=1)
    r_dist = np.hypot(r_top_x - r_bot_x, r_top_y - r_bot_y)

    # Left eye: top (43,44), bottom (46,47)
    l_top_x = pd.concat([_mask_xy(df_raw, 43)[0], _mask_xy(df_raw, 44)[0]], axis=1).mean(axis=1)
    l_top_y = pd.concat([_mask_xy(df_raw, 43)[1], _mask_xy(df_raw, 44)[1]], axis=1).mean(axis=1)
    l_bot_x = pd.concat([_mask_xy(df_raw, 46)[0], _mask_xy(df_raw, 47)[0]], axis=1).mean(axis=1)
    l_bot_y = pd.concat([_mask_xy(df_raw, 46)[1], _mask_xy(df_raw, 47)[1]], axis=1).mean(axis=1)
    l_dist = np.hypot(l_top_x - l_bot_x, l_top_y - l_bot_y)

    blink = (r_dist + l_dist) / 2.0
    blink = blink.interpolate(method="linear", limit=max_interp, limit_direction="both")
    return blink

def compute_head_rotation(df_raw: pd.DataFrame, max_interp: int = MAX_INTERP) -> pd.Series:
    # Eye corners: 36 (left), 45 (right)
    x36, y36 = _mask_xy(df_raw, 36)
    x45, y45 = _mask_xy(df_raw, 45)
    dx, dy = (x45 - x36), (y45 - y36)
    angle = np.arctan2(dy, dx)
    angle = angle.interpolate(method="linear", limit=max_interp, limit_direction="both")
    return angle

def compute_mouth_distance(df_raw: pd.DataFrame, max_interp: int = MAX_INTERP) -> pd.Series:
    # Upper/lower lip: 62, 66
    x62, y62 = _mask_xy(df_raw, 62)
    x66, y66 = _mask_xy(df_raw, 66)
    mouth = np.hypot(x62 - x66, y62 - y66)
    mouth = mouth.interpolate(method="linear", limit=max_interp, limit_direction="both")
    return mouth

# ---- file-level processing ----
def process_one_file(file_path: str, output_dir: str,
                     window_size: int = WINDOW_SIZE, overlap: float = OVERLAP,
                     max_interp: int = MAX_INTERP, log_rows: list | None = None):
    df_raw = load_csv(file_path)
    n = len(df_raw)
    base = os.path.basename(file_path)

    # 1) Windowed logging: only when a region is "all-dead" for too long
    for widx, (s, e) in enumerate(window_ranges(n, window_size, overlap)):
        df_w = df_raw.iloc[s:e].reset_index(drop=True)
        _log_region_all_missing(df_w, base, widx, s, e, max_interp, log_rows)

    # 2) Full-sequence processing + derived metrics
    avg_full = compute_averages(df_raw, max_interp=max_interp, interpolate=True)
    avg_full = compute_magnitude(avg_full)
    avg_full = compute_combined_eyes(avg_full)
    avg_full["blink_dist"] = compute_blink(df_raw, max_interp=max_interp)
    avg_full["head_rotation_angle"] = compute_head_rotation(df_raw, max_interp=max_interp)
    avg_full["mouth_dist"] = compute_mouth_distance(df_raw, max_interp=max_interp)

    # 3) Save
    out_file = os.path.splitext(base)[0] + ".csv"
    out_path = os.path.join(output_dir, out_file)
    avg_full.to_csv(out_path, index=False)

def process_data(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, LOG_DIRNAME)
    os.makedirs(log_dir, exist_ok=True)
    log_rows = []

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        file_path = os.path.join(input_dir, csv_file)
        try:
            process_one_file(file_path, output_dir, WINDOW_SIZE, OVERLAP, MAX_INTERP, log_rows)
        except Exception as e:
            log_rows.append({
                "file": csv_file, "window_index": -1, "start": None, "end": None,
                "reason": f"FAILED: {e}", "region": None, "columns": None, "details": None
            })

    log_df = pd.DataFrame(log_rows, columns=["file","window_index","start","end","reason","region","columns","details"])
    log_path = os.path.join(log_dir, "dropped_windows.csv")
    log_df.to_csv(log_path, index=False)

if __name__ == "__main__":
    # CHANGE THESE PATHS AS NEEDED:
    input_directory = "data/pose/experimental_pose" 
    output_directory = "data/preprocessed_pose/experimental_pose"
    process_data(input_directory, output_directory)
