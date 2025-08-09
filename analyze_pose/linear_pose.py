import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from scipy.signal import detrend as scipy_detrend

# Set up where the data is and how to process it
DATA_DIR = "data/preprocessed_pose/experimental_pose"   # Folder containing the data files, change to /baseline_pose if needed
FPS = 60                               # Frames per second (how often data is recorded)
WINDOW_SECONDS = 60                    # Length of each analysis window in seconds
WINDOW_FRAMES = WINDOW_SECONDS * FPS   # Number of frames in each window
STEP_FRAMES = WINDOW_FRAMES // 2       # How much to move the window each time (50% overlap)
METRICS = [                            # List of measurements to analyze
    "center_face_magnitude", "avg_pupil_magnitude", "avg_pupil_x", "avg_pupil_y",
    "blink_dist", "head_rotation_angle", "mouth_dist", "center_face_x", "center_face_y",
]

do_scale = True       # Should we scale each series to [0, 1]?
do_detrend = False    # Should we remove linear trends from the data?

def scale_series(series: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Scale the data to be between 0 and 1, using the minimum and maximum values.
    If all values are the same, return zeros.
    """
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(series)

def preprocess_window(raw: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Prepare a window of data: scale it to [0,1], and optionally remove trends.
    """
    s = raw.astype(float)
    s = scale_series(s, min_val, max_val)
    if do_detrend:
        s = scipy_detrend(s, type='linear')
    return s

def compute_rms(series: np.ndarray) -> float:
    """
    Calculate the root mean square (RMS) of the data, ignoring missing values.
    RMS is a measure of how much the values vary.
    """
    if series.size == 0:
        return np.nan
    return np.sqrt(np.nanmean(np.square(series)))

def compute_windowed_summary(data_dir=DATA_DIR) -> pd.DataFrame:
    """
    For each data file:
      - Scale all measurements to [0,1] using the whole file's min/max.
      - Divide the data into overlapping windows.
      - For each window, calculate:
          - Mean absolute velocity (how fast values change)
          - Mean absolute acceleration (how quickly velocity changes)
          - RMS (overall variation)
      - Save results for each window.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))  # Find all CSV files
    rows = []  # Store results for each window

    for file_path in tqdm(csv_files, desc="Windowed summaries with global scaling"):
        fname = os.path.basename(file_path)
        # Get participant ID and condition from the filename
        pid, cond = fname.split("_")[0], fname.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(file_path)
        n = len(df)

        # Find the min and max for each measurement in the whole file
        global_min = {m: np.nanmin(df[m].to_numpy()) for m in METRICS}
        global_max = {m: np.nanmax(df[m].to_numpy()) for m in METRICS}

        window_idx = 0
        start = 0
        # Slide the window through the data
        while start + WINDOW_FRAMES <= n:
            end = start + WINDOW_FRAMES
            row = {
                "participant": pid,
                "condition": cond,
                "window_index": window_idx,
                "window_start": start / FPS,      # Start time in seconds
                "window_end": (end - 1) / FPS     # End time in seconds
            }

            # For each measurement, process the window and calculate metrics
            for metric in METRICS:
                raw_window = df[metric].to_numpy()[start:end]
                proc = preprocess_window(raw_window,
                                         global_min[metric],
                                         global_max[metric])
                vel = np.diff(proc, prepend=np.nan) * FPS      # Velocity: change per second
                acc = np.diff(vel, prepend=np.nan) * FPS       # Acceleration: change in velocity per second

                row[f"{metric}_mean_abs_vel"] = np.nanmean(np.abs(vel))   # Average speed of change
                row[f"{metric}_mean_abs_acc"] = np.nanmean(np.abs(acc))   # Average acceleration
                row[f"{metric}_rms"] = compute_rms(proc)                 # Overall variation

            rows.append(row)
            window_idx += 1
            start += STEP_FRAMES   # Move window forward (with overlap)

    return pd.DataFrame(rows)

# Run the analysis and save the results to a CSV file
windowed_summary = compute_windowed_summary()