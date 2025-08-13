import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from scipy.signal import detrend as scipy_detrend

# Set the folder where the cleaned pose data files are stored
DATA_DIR = "data/preprocessed_pose/baseline_pose"  # Change to /baseline_pose if needed

# Settings for how to break up the data into chunks ("windows")
FPS = 60                 # How many data points per second (frames per second)
WINDOW_SECONDS = 60      # How long each window is (in seconds)
WINDOW_FRAMES = WINDOW_SECONDS * FPS   # How many data points in each window
STEP_FRAMES = WINDOW_FRAMES // 2       # How much windows overlap (50%)

# List of measurements to analyze (these columns must be in the data files)
METRICS = [
    "center_face_magnitude", "avg_pupil_magnitude", "avg_pupil_x", "avg_pupil_y",
    "blink_dist", "head_rotation_angle", "mouth_dist", "center_face_x", "center_face_y",
]

# Should we scale each measurement to [0, 1]? Should we remove trends?
do_scale = True       # Scale each series to [0, 1] using min/max for each file
do_detrend = False    # Remove linear trend after scaling (usually not needed)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def scale_series(series: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Scale the data to be between 0 and 1, using the minimum and maximum values.
    If all values are the same, return zeros.
    """
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return np.zeros_like(series)

def preprocess_window(raw: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Prepare a window of data: scale it to [0,1], and optionally remove trends.
    """
    s = raw.astype(float)
    s = scale_series(s, min_val, max_val) if do_scale else s
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

# ------------------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------------------

def compute_windowed_summary(data_dir=DATA_DIR) -> pd.DataFrame:
    """
    For each data file:
      1) Read the file, treating empty cells as missing data.
      2) Find the minimum and maximum value for each measurement in the whole file.
      3) Divide the data into overlapping windows (chunks of 60 seconds).
      4) If ANY measurement has missing data in a window, skip that window.
      5) Otherwise, calculate velocity, acceleration, and RMS for each measurement in the window.
    Returns a table with one row for each "clean" window.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    rows = []

    # Keep track of how many windows were kept or skipped for each file
    per_file_counts = []  # (filename, kept, skipped)

    for file_path in tqdm(csv_files, desc="Windowed summaries with global scaling"):
        fname = os.path.basename(file_path)

        # Read the CSV file and treat empty cells as missing data (NaN)
        df = pd.read_csv(
            file_path,
            na_values=["", " ", "NA", "NaN", "nan"],  # interpret empties as NaN
            keep_default_na=True
        )
        # Also convert cells with only spaces to missing data
        df = df.replace(r"^\s*$", np.nan, regex=True)

        n = len(df)
        if n < WINDOW_FRAMES:
            per_file_counts.append((fname, 0, 0))  # File too short, skip
            continue

        # Make sure all required measurements are present
        missing_cols = [m for m in METRICS if m not in df.columns]
        if missing_cols:
            raise ValueError(f"{fname}: missing expected columns: {missing_cols}")

        # Find min and max for each measurement in the whole file
        global_min = {m: np.nanmin(df[m].to_numpy()) for m in METRICS}
        global_max = {m: np.nanmax(df[m].to_numpy()) for m in METRICS}

        kept = 0
        skipped = 0

        window_idx = 0
        start = 0
        while start + WINDOW_FRAMES <= n:
            end = start + WINDOW_FRAMES

            # If ANY measurement has missing data in this window, skip the window
            has_nan_any_metric = False
            for metric in METRICS:
                window_vals = df[metric].to_numpy()[start:end]
                if np.isnan(window_vals).any():
                    has_nan_any_metric = True
                    break

            if has_nan_any_metric:
                skipped += 1
                start += STEP_FRAMES
                window_idx += 1
                continue

            # If all measurements are present, calculate features for this window
            row = {
                "file": fname,
                "participant": fname.split("_")[0],
                "condition": fname.split("_")[-1].replace(".csv", ""),
                "window_index": window_idx,
                "window_start_s": start / FPS,
                "window_end_s": (end - 1) / FPS
            }

            for metric in METRICS:
                raw_window = df[metric].to_numpy()[start:end]
                proc = preprocess_window(raw_window, global_min[metric], global_max[metric])

                # Calculate velocity (how fast values change) and acceleration (how quickly velocity changes)
                vel = np.diff(proc, prepend=np.nan) * FPS
                acc = np.diff(vel, prepend=np.nan) * FPS

                row[f"{metric}_mean_abs_vel"] = np.nanmean(np.abs(vel))   # Average speed of change
                row[f"{metric}_mean_abs_acc"] = np.nanmean(np.abs(acc))   # Average acceleration
                row[f"{metric}_rms"] = compute_rms(proc)                 # Overall variation

            rows.append(row)
            kept += 1
            start += STEP_FRAMES
            window_idx += 1

        per_file_counts.append((fname, kept, skipped))

    # Print a summary for each file: how many windows were kept/skipped
    for fname, kept, skipped in per_file_counts:
        total = kept + skipped
        if total > 0:
            print(f"[{fname}] kept {kept} / {total} windows ({kept/total:.1%}), skipped {skipped} ({skipped/total:.1%})")
        else:
            print(f"[{fname}] no full windows (file too short for {WINDOW_SECONDS}s).")

    return pd.DataFrame(rows)

# Run the analysis and keep only “clean” windows (no missing data)
windowed_summary = compute_windowed_summary()
# Save the results to a CSV file
output_path = "data/linear_pose_metrics/baseline_pose_metrics.csv"
windowed_summary.to_csv(output_path, index=False)
print(f"[OK] Saved {len(windowed_summary)} clean windows to {output_path}")