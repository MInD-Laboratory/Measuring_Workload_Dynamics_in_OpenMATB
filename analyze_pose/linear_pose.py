import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import detrend as scipy_detrend

# ------------------------------ SETTINGS ------------------------------
MODE = "baseline"  # "baseline" or "experimental"
DATA_DIR = f"data/preprocessed_pose/{MODE}_pose" # Folder containing the data files
OUT_PATH = f"data/linear_pose_metrics/{MODE}_pose_metrics.csv" # Output path for the metrics

FPS = 60 # Frames per second (how often data is recorded)
WINDOW_SECONDS = 60 # Length of each analysis window in seconds
WINDOW_FRAMES = WINDOW_SECONDS * FPS # Number of frames in each window
STEP_FRAMES = WINDOW_FRAMES // 2  # 50% overlap

METRICS = [
    "center_face_magnitude", "avg_pupil_magnitude", "avg_pupil_x", "avg_pupil_y",
    "blink_dist", "head_rotation_angle", "mouth_dist", "center_face_x", "center_face_y",
]

ANGLE_COLUMNS = ["head_rotation_angle"]      # unwrap before scaling
ROBUST_MINMAX = False                        # if True: use percentiles (1,99) instead of min/max, kept false
DO_DETREND = False                           # optional: detrend AFTER scaling for vel/acc, kept false
# ----------------------------------------------------------------------

def unwrap_if_angle(colname: str, values: np.ndarray):
    return np.unwrap(values) if colname in ANGLE_COLUMNS else values

def series_minmax(arr: np.ndarray, robust=False):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    if robust:
        lo, hi = np.percentile(a, [1, 99])
        if hi <= lo:  # degenerate
            hi = lo + 1e-9
        return float(lo), float(hi)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi

def scale_01(arr: np.ndarray, lo: float, hi: float):
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

def zero_center(arr: np.ndarray):
    return arr - np.nanmean(arr) if arr.size else arr

def rms_zero_centered(arr: np.ndarray):
    if arr.size == 0:
        return np.nan
    c = zero_center(arr)
    return np.sqrt(np.nanmean(c**2))

def mean_abs_diff_per_sec(series: np.ndarray, fps: int):
    if series.size == 0:
        return np.nan
    diffs = np.diff(series, prepend=np.nan)
    return np.nanmean(np.abs(diffs)) * fps

def mean_abs_acc_per_sec(series: np.ndarray, fps: int):
    if series.size < 2:
        return np.nan
    v = np.diff(series, prepend=np.nan)
    a = np.diff(v, prepend=np.nan)
    return np.nanmean(np.abs(a)) * (fps**2)

def compute_windowed_summary(data_dir=DATA_DIR, out_path=OUT_PATH) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    rows, per_file_counts = [], []

    for file_path in tqdm(csv_files, desc="Metrics from scaled [0,1] pose"):
        fname = os.path.basename(file_path)
        if "report" in fname.lower():
            continue  # ignore non-signal CSVs

        df = pd.read_csv(file_path, na_values=["", " ", "NA", "NaN", "nan"], keep_default_na=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Ensure required columns exist
        missing = [m for m in METRICS if m not in df.columns]
        if missing:
            print(f"[SKIP] {fname}: missing columns {missing}")
            continue

        n = len(df)
        if n < WINDOW_FRAMES:
            per_file_counts.append((fname, 0, 0))
            continue

        # --- Scale whole file to [0,1] per metric (after angle unwrap) ---
        scaled_df = pd.DataFrame(index=df.index)
        minmax = {}
        for m in METRICS:
            raw = unwrap_if_angle(m, df[m].to_numpy().astype(float))
            lo, hi = series_minmax(raw, robust=ROBUST_MINMAX)
            minmax[m] = (lo, hi)
            scaled_df[m] = scale_01(raw, lo, hi)

        kept = skipped = window_idx = 0
        start = 0

        while start + WINDOW_FRAMES <= n:
            end = start + WINDOW_FRAMES
            block = scaled_df.iloc[start:end]

            # Strict: skip any window with NaNs
            if block.isna().any().any():
                skipped += 1
                start += STEP_FRAMES; window_idx += 1
                continue

            row = {
                "file": fname,
                "participant": fname.split("_")[0],
                "condition": fname.split("_")[-1].replace(".csv", ""),
                "window_index": window_idx,
                "window_start_s": start / FPS,
                "window_end_s": (end - 1) / FPS,
            }

            for m in METRICS:
                s = block[m].to_numpy().astype(float)

                # optional detrend on the already-scaled series (doesn't change RMS if mean-centered)
                s_proc = scipy_detrend(s, type="linear") if DO_DETREND else s

                # velocity/acc on scaled domain
                row[f"{m}_mean_abs_vel"] = mean_abs_diff_per_sec(s_proc, FPS)
                row[f"{m}_mean_abs_acc"] = mean_abs_acc_per_sec(s_proc, FPS)

                # RMS on scaled, zero-centered (std of the segment)
                row[f"{m}_rms"] = rms_zero_centered(s)

            rows.append(row)
            kept += 1
            start += STEP_FRAMES; window_idx += 1

        per_file_counts.append((fname, kept, skipped))

    for fname, kept, skipped in per_file_counts:
        tot = kept + skipped
        msg = f"[{fname}] kept {kept} / {tot} ({kept/tot:.1%})" if tot else f"[{fname}] no full windows"
        print(msg)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(out)} clean windows to {out_path}")
    return out

if __name__ == "__main__":
    compute_windowed_summary()
