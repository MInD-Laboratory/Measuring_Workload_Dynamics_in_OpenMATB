import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from scipy.signal import detrend as scipy_detrend

DATA_DIR = "data/pose/baseline_pose"
FPS = 60
WINDOW_SECONDS = 60
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STEP_FRAMES = WINDOW_FRAMES // 2
METRICS = [
    "center_face_magnitude", "avg_pupil_magnitude", "avg_pupil_x", "avg_pupil_y","blink_dist","head_rotation_angle", "mouth_dist", "center_face_x", "center_face_y",
]

do_scale = True       # Scale each series to [0, 1]
do_detrend = False    # Apply linear detrending

def scale_series(series: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Scale to [0,1] using provided min and max.
    """
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(series)

def preprocess_window(raw: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Scale raw window data to [0,1] using global min/max, then optionally detrend.
    """
    s = raw.astype(float)
    s = scale_series(s, min_val, max_val)
    if do_detrend:
        s = scipy_detrend(s, type='linear')
    return s

def compute_rms(series: np.ndarray) -> float:
    """
    Compute root mean square of a series, ignoring NaNs.
    """
    if series.size == 0:
        return np.nan
    return np.sqrt(np.nanmean(np.square(series)))

def compute_windowed_summary(data_dir=DATA_DIR) -> pd.DataFrame:
    """
    For each CSV: normalize full-series metrics to [0,1],
    then slice into overlapping windows to compute
    mean-abs velocity, mean-abs acceleration, and RMS.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    rows = []

    for file_path in tqdm(csv_files, desc="Windowed summaries with global scaling"):
        fname = os.path.basename(file_path)
        pid, cond = fname.split("_")[0], fname.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(file_path)
        n = len(df)

        # Precompute global min/max per metric
        global_min = {m: np.nanmin(df[m].to_numpy()) for m in METRICS}
        global_max = {m: np.nanmax(df[m].to_numpy()) for m in METRICS}

        window_idx = 0
        start = 0
        while start + WINDOW_FRAMES <= n:
            end = start + WINDOW_FRAMES
            row = {
                "participant": pid,
                "condition": cond,
                "window_index": window_idx,
                "window_start": start / FPS,
                "window_end": (end - 1) / FPS
            }

            for metric in METRICS:
                raw_window = df[metric].to_numpy()[start:end]
                proc = preprocess_window(raw_window,
                                         global_min[metric],
                                         global_max[metric])
                vel = np.diff(proc, prepend=np.nan) * FPS
                acc = np.diff(vel, prepend=np.nan) * FPS

                row[f"{metric}_mean_abs_vel"] = np.nanmean(np.abs(vel))
                row[f"{metric}_mean_abs_acc"] = np.nanmean(np.abs(acc))
                row[f"{metric}_rms"] = compute_rms(proc)

            rows.append(row)
            window_idx += 1
            start += STEP_FRAMES

    return pd.DataFrame(rows)

windowed_summary = compute_windowed_summary()
windowed_summary.to_csv("data/linear_metrics/baseline_pose_linear.csv", index=False)
