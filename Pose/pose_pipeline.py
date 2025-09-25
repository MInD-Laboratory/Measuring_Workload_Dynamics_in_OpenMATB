#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_pipeline.py

Standalone facial pose pipeline — clear, stepwise, extensively commented.

What this script does (in order), controlled by flags at the top:

  1) Load raw OpenPose CSVs (x1,y1,prob1,...,x70,y70,prob70) from RAW_DIR
  2) Filter to relevant keypoints (your sets)                [RUN_FILTER]
  3) Mask low-confidence (conf < CONF_THRESH) to NaN         [RUN_MASK]
  4) Interpolate short gaps (≤ MAX_INTERP_RUN) + Butterworth [RUN_INTERP_FILTER]
  5) Normalize to screen size (2560×1440)                   [RUN_NORM]
  6) Build templates (global + per-participant)             [RUN_TEMPLATES]
  7) Features:
       A) Procrustes vs global template (windowed 60s, 50% overlap)
       A) Procrustes vs participant template (same)
       B) Original (no Procrustes), same windowing           [RUN_FEATURES_*]
     - Per-metric: drop windows containing any NaNs
     - Save three CSVs: procrustes_global, procrustes_participant, original
  8) Interocular scaling + linear metrics (vel, acc, RMS)    [RUN_LINEAR]
     - Save three CSVs for linear metrics corresponding to step 7 outputs.

A JSON summary is saved with:
  - config & flags,
  - per-file masking stats,
  - windows dropped (total & per metric) per route,
  - template info,
  - any errors encountered.

Assumptions:
  - Filenames are exactly "<participantID>_<condition>.csv" e.g., "472_H.csv".
  - Conditions are L/M/H (you can add more; parser is tolerant).
  - Image dimensions are 2560×1440.
  - Sampling is 60 Hz.

"""

from __future__ import annotations

# ======= TOP-LEVEL PARAMETERS (EDIT HERE) ====================================
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import math
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd
from tqdm import tqdm

# Filtering/Signal processing
try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---------------------- FLAGS: choose what to run -----------------------------
RUN_FILTER          = False   # Step 2
RUN_MASK            = False   # Step 3
RUN_INTERP_FILTER   = False   # Step 4
RUN_NORM            = False   # Step 5
RUN_TEMPLATES       = False   # Step 6
RUN_FEATURES_PROCRUSTES_GLOBAL      = False   # Step 7A (global template)
RUN_FEATURES_PROCRUSTES_PARTICIPANT = False   # Step 7A (participant template)
RUN_FEATURES_ORIGINAL               = False   # Step 7B (no template)
RUN_LINEAR          = True   # Step 8
SCALE_BY_INTEROCULAR = True  # Step 8: scale distance-like metrics by interocular; set False to disable

# ---------------------- OUTPUT toggles ---------------------------------------
SAVE_REDUCED            = True  # save reduced (relevant triplets)
SAVE_MASKED             = True  # save masked (after conf threshold)
SAVE_INTERP_FILTERED    = True  # save post interpolation+filtering coords
SAVE_NORM               = True  # save screen-normalized coords

# Per-frame save toggles (Step 7)
SAVE_PER_FRAME_PROCRUSTES_GLOBAL      = True   # write per-frame metrics aligned to GLOBAL template
SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT = True   # write per-frame metrics aligned to PARTICIPANT template
SAVE_PER_FRAME_ORIGINAL               = True   # write per-frame metrics without Procrustes


OVERWRITE               = True   # overwrite step outputs
OVERWRITE_TEMPLATES     = False  # if False and templates exist, reuse them

# ---------------------- Core Config ------------------------------------------
@dataclass
class Config:
    # IO
    RAW_DIR: str = "data/raw/experimental"
    OUT_BASE: str = "data/processed/experimental"

    # Sampling
    FPS: int = 60

    # Image/screen normalization
    IMG_WIDTH: int = 2560
    IMG_HEIGHT: int = 1440

    # Confidence masking
    CONF_THRESH: float = 0.30

    # Interpolation
    MAX_INTERP_RUN: int = 60  # frames

    # Butterworth filter
    FILTER_ORDER: int = 4
    CUTOFF_HZ: float = 10.0

    # Windowing
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5  # 50%

    # Landmark sets (1-based indices)
    PROCRUSTES_REF: Tuple[int, ...] = (30, 31, 37, 46)
    BLINK_L_TOP: Tuple[int, ...] = (38, 39)
    BLINK_L_BOT: Tuple[int, ...] = (41, 42)
    BLINK_R_TOP: Tuple[int, ...] = (44, 45)
    BLINK_R_BOT: Tuple[int, ...] = (47, 48)
    HEAD_ROT: Tuple[int, int] = (37, 46)
    MOUTH: Tuple[int, int] = (63, 67)
    CENTER_FACE: Tuple[int, ...] = tuple(range(28, 37))  # 28..36 inclusive

CFG = Config()

# =============================================================================
# ============================= UTILITIES =====================================
# =============================================================================

def list_csvs(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".csv"])

def parse_participant_condition(filename: str) -> Tuple[str, str]:
    """
    Expect "<participantID>_<condition>.csv"
    Returns (participant_id, condition). Raises if not matched.
    """
    base = Path(filename).name
    m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match '<participant>_<cond>.csv': {base}")
    return m.group(1), m.group(2)

def detect_conf_prefix_case_insensitive(columns: List[str]) -> str:
    cols_low = [c.lower() for c in columns]
    for prefix in ("prob", "c", "confidence"):
        if any(col.startswith(prefix) for col in cols_low):
            return prefix
    raise ValueError("Confidence prefix not found (expected 'prob*', 'c*', or 'confidence*').")

def find_real_colname(prefix: str, i: int, columns: List[str]) -> Optional[str]:
    target = f"{prefix}{i}".lower()
    for col in columns:
        if col.lower() == target:
            return col
    # allow weird separators (unlikely but safe)
    for col in columns:
        if col.lower().startswith(target):
            return col
    return None

def lm_triplet_colnames(i: int, conf_prefix: str, columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        find_real_colname("x", i, columns),
        find_real_colname("y", i, columns),
        find_real_colname(conf_prefix, i, columns),
    )

def relevant_indices() -> List[int]:
    s = set()
    s.update(CFG.PROCRUSTES_REF)
    s.update(CFG.HEAD_ROT)
    s.update(CFG.MOUTH)
    s.update(CFG.CENTER_FACE)
    s.update(CFG.BLINK_L_TOP); s.update(CFG.BLINK_L_BOT)
    s.update(CFG.BLINK_R_TOP); s.update(CFG.BLINK_R_BOT)
    s.update([69, 70]) # pupils
    return sorted(s)

def filter_df_to_relevant(df: pd.DataFrame, conf_prefix: str, indices: List[int]) -> pd.DataFrame:
    """Keep only x/y/conf for the given 1-based indices. Case-insensitive mapping."""
    kept: List[str] = []
    missing: List[str] = []
    cols = list(df.columns)
    for i in indices:
        x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
        if x and y and c:
            kept.extend([x, y, c])
        else:
            if not x: missing.append(f"x{i}")
            if not y: missing.append(f"y{i}")
            if not c: missing.append(f"{conf_prefix}{i}")
    if not kept:
        raise ValueError(f"No relevant triplets found. Missing examples: {missing[:8]}")
    return df.loc[:, kept].copy()

def confidence_mask(df_reduced: pd.DataFrame, conf_prefix: str, indices: List[int], thr: float) -> Tuple[pd.DataFrame, Dict]:
    """Set x,y,conf to NaN where conf<thr for each relevant landmark. Return masked copy + stats."""
    dfm = df_reduced.copy()
    cols = list(dfm.columns)
    n_frames = len(dfm)
    per_lm = {}
    total_considered = 0
    total_masked = 0
    for i in indices:
        x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
        if not (x and y and c):
            continue
        conf = pd.to_numeric(dfm[c], errors="coerce")
        low = conf < thr
        low = low.fillna(False)

        pre_x = dfm[x].notna()
        pre_y = dfm[y].notna()
        considered = int((pre_x | pre_y).sum()) * 2
        masked = int(((pre_x | pre_y) & low).sum()) * 2

        if low.any():
            dfm.loc[low, [x, y, c]] = np.nan

        per_lm[i] = {
            "frames_total": int(n_frames),
            "frames_low_conf": int(low.sum()),
            "coords_considered": considered,
            "coords_masked": masked,
            "pct_frames_low_conf": (int(low.sum()) / n_frames * 100.0) if n_frames else 0.0
        }
        total_considered += considered
        total_masked += masked

    overall = {
        "frames": int(n_frames),
        "n_landmarks_considered": len(per_lm),
        "total_coord_values": int(total_considered),
        "total_coords_masked": int(total_masked),
        "pct_coords_masked": (total_masked / total_considered * 100.0) if total_considered else 0.0
    }
    return dfm, {"per_landmark": per_lm, "overall": overall}

# ---------- Interpolation & Filtering ----------------------------------------

def find_nan_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of [start, end) index pairs where mask==True (NaN runs)."""
    runs = []
    if mask.size == 0:
        return runs
    in_run = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run = True
            start = i
        elif not m and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs

def interpolate_run_limited(series: pd.Series, max_run: int) -> pd.Series:
    """
    Linear interpolation but ONLY for NaN runs with length <= max_run.
    Longer runs remain NaN.
    """
    x = series.astype(float).copy()
    nan_mask = x.isna().values
    runs = find_nan_runs(nan_mask)
    # Create a mask of samples that are allowed to be interpolated
    allow = np.zeros_like(nan_mask, dtype=bool)
    for s, e in runs:
        if (e - s) <= max_run:
            allow[s:e] = True
    # Build an array where disallowed NaNs stay NaN during interpolation.
    y = x.copy()
    # Temporarily fill disallowed NaNs with a sentinel so they don't get interpolated
    # Simpler: perform interpolation on a copy where allowed NaNs are left, others remain NaN.
    y_interp = y.copy()
    # For pandas interpolate to work only on allowed NaNs, we can mask disallowed NaNs by not touching them.
    # Interpolate linearly on index:
    if allow.any():
        # We replace disallowed NaNs with a stopper by forward/backfill to prevent bridging large gaps
        # then revert them to NaN after interpolation.
        disallowed = nan_mask & (~allow)
        fillable = ~disallowed
        temp = y_interp.copy()
        # Interpolate only across fillable gaps
        temp[~fillable] = np.nan  # ensure disallowed NaNs remain NaN
        temp = temp.interpolate(method="linear", limit=None, limit_direction="both")
        # Combine: use interpolated where allowed, else original NaNs
        y_interp[allow] = temp[allow]
        y_interp[~allow & nan_mask] = np.nan
    return y_interp

def butterworth_segment_filter(series: pd.Series, order: int, cutoff_hz: float, fs: float) -> pd.Series:
    """
    Zero-phase Butterworth filtering applied per contiguous non-NaN segment.
    Segments shorter than padlen are skipped to avoid filtfilt errors.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for Butterworth filtering. Install scipy or disable RUN_INTERP_FILTER.")
    x = series.astype(float).values.copy()
    isnan = np.isnan(x)
    # Design filter
    nyq = fs / 2.0
    wn = min(0.999, cutoff_hz / nyq)
    b, a = butter(order, wn, btype='low', analog=False)
    # filtfilt padlen rule of thumb
    padlen = 3 * (max(len(a), len(b)) - 1)
    # Process contiguous valid segments
    start = 0
    while start < len(x):
        # skip NaNs
        while start < len(x) and np.isnan(x[start]):
            start += 1
        if start >= len(x):
            break
        end = start
        while end < len(x) and not np.isnan(x[end]):
            end += 1
        seg_len = end - start
        if seg_len > padlen + 1:
            x[start:end] = filtfilt(b, a, x[start:end])
        # else: leave as-is (unfiltered)
        start = end
    return pd.Series(x, index=series.index, dtype=float)

# ---------- Screen normalization ---------------------------------------------

def normalize_to_screen(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """
    Divide all x columns by width and all y columns by height.
    Assumes df contains only x*, y*, prob* (or c*/confidence*).
    """
    out = df.copy()
    x_cols = [c for c in out.columns if c.lower().startswith('x')]
    y_cols = [c for c in out.columns if c.lower().startswith('y')]
    out[x_cols] = out[x_cols] / float(width)
    out[y_cols] = out[y_cols] / float(height)
    return out

def interocular_series(df: pd.DataFrame, conf_prefix: Optional[str] = None) -> pd.Series:
    """
    Return per-frame interocular distance (sqrt((x46-x37)^2 + (y46-y37)^2)).
    Accepts an optional conf_prefix for backward compatibility with existing calls.
    If any of the required columns are missing, returns a Series of NaNs with same index.
    """
    cols = list(df.columns)
    x37_col = find_real_colname("x", 37, cols)
    y37_col = find_real_colname("y", 37, cols)
    x46_col = find_real_colname("x", 46, cols)
    y46_col = find_real_colname("y", 46, cols)

    # If any required column is missing, return NaN series (keeps callers safe)
    if not (x37_col and y37_col and x46_col and y46_col):
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Cast to float (coerce bad strings to NaN), compute Euclidean distance
    x37 = pd.to_numeric(df[x37_col], errors="coerce").astype(float)
    y37 = pd.to_numeric(df[y37_col], errors="coerce").astype(float)
    x46 = pd.to_numeric(df[x46_col], errors="coerce").astype(float)
    y46 = pd.to_numeric(df[y46_col], errors="coerce").astype(float)

    return np.sqrt((x46 - x37) ** 2 + (y46 - y37) ** 2)

# ---------- Procrustes (2D similarity transform) -----------------------------

def procrustes_frame_to_template(frame_xy: np.ndarray, templ_xy: np.ndarray, available_mask: np.ndarray) -> Tuple[bool, float, float, float, np.ndarray]:
    """
    Compute similarity transform that maps frame_xy -> templ_xy using points where available_mask==True.
    frame_xy, templ_xy: arrays shape (L, 2) for L landmarks (only relevant set).
    available_mask: boolean mask (L,) True where both frame and template points are finite.

    Returns: (ok, scale s, tx, ty, R2x2, transformed_points)
      - Rotation angle can be recovered via atan2(R[1,0], R[0,0]) if needed;
        but we directly report eye-based angle per spec.
    """
    idx = np.where(available_mask)[0]
    if idx.size < 3:
        return False, np.nan, np.nan, np.nan, np.full((2,2), np.nan), np.full_like(frame_xy, np.nan)

    X = frame_xy[idx, :]  # (n,2)
    Y = templ_xy[idx, :]  # (n,2)

    # Center
    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    Xc = X - muX
    Yc = Y - muY

    # Solve for R,s via SVD of Yc^T Xc
    # We want sR minimizing ||sR Xc - Yc||_F
    C = Xc.T @ Yc  # (2,2)
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    # Special reflection handling
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    varX = (Xc**2).sum()
    s = (S.sum()) / varX if varX > 0 else 1.0
    t = (muY.T - s * R @ muX.T).reshape(2)  # (2,)

    # Apply transform to all points (not only idx)
    Xall = frame_xy.copy()
    Xall_centered = Xall - muX  # careful: subtract muX of used points; reasonable approx
    Xtrans = (s * (R @ Xall_centered.T)).T + muY  # (L,2)
    return True, float(s), float(t[0]), float(t[1]), R, Xtrans

# ---------- Feature helpers --------------------------------------------------

def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle (radians) of vector p1->p2 vs +x axis using arctan2(dy, dx), wrapped to (-pi, pi]."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.atan2(dy, dx)

def blink_aperture_from_points(eye_top: np.ndarray, eye_bot: np.ndarray) -> float:
    """
    Aperture as vertical distance between average of top-two points and average of bottom-two points.
    Inputs: eye_top shape (2,2), eye_bot shape (2,2).
    """
    top_mean = eye_top.mean(axis=0)  # (x,y)
    bot_mean = eye_bot.mean(axis=0)
    # Absolute vertical separation; coordinates normalized later, so sign isn't critical
    return float(abs(top_mean[1] - bot_mean[1]))

def mouth_aperture(p63: np.ndarray, p67: np.ndarray) -> float:
    return float(np.linalg.norm(p67 - p63))

def pupil_proxy(eye_ring: np.ndarray) -> np.ndarray:
    """
    Proxy for pupil center: average of eye ring points.
    eye_ring shape (N,2). Returns (2,) center.
    """
    return eye_ring.mean(axis=0)

def center_face_magnitude(nose_points: np.ndarray, baseline: Optional[np.ndarray] = None) -> float:
    """
    RMS magnitude of nose points around a baseline (default = per-file mean across time).
    nose_points shape (K,2).
    """
    if baseline is None:
        base = nose_points.mean(axis=0, keepdims=True)
    else:
        base = baseline.reshape(1,2)
    diffs = nose_points - base
    dists = np.sqrt((diffs**2).sum(axis=1))
    return float(np.sqrt((dists**2).mean()))

# ---------- Windowing --------------------------------------------------------

def windows_indices(n: int, win: int, hop: int) -> List[Tuple[int,int,int]]:
    """
    Return list of windows as (start, end, index), 0-based, end exclusive.
    """
    out = []
    w = 0
    start = 0
    while start + win <= n:
        out.append((start, start + win, w))
        start += hop
        w += 1
    return out

# ---------- Linear metrics ---------------------------------------------------

def is_distance_like_metric(name: str) -> bool:
    # Angle and unitless scale are NOT distance-like
    if name in ("head_rotation_rad", "head_scale"):
        return False
    # Everything else we produce is distance-like (tx, ty, motion_mag, blink, mouth, pupil, center_face)
    return True

def linear_metrics(x: np.ndarray, fps: float) -> Tuple[float, float, float]:
    """
    Given a window time series (no NaNs), compute:
      - mean absolute velocity (per second)
      - mean absolute acceleration (per second^2)
      - RMS variability within window
    """
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    dt = 1.0 / fps
    vel = np.diff(x) / dt
    acc = np.diff(vel) / dt
    mean_abs_vel = float(np.mean(np.abs(vel)))
    mean_abs_acc = float(np.mean(np.abs(acc)))
    rms = float(np.sqrt(np.mean((x - np.mean(x))**2)))
    return mean_abs_vel, mean_abs_acc, rms

# =============================================================================
# ============================= PIPELINE ======================================
# =============================================================================

def ensure_dirs():
    base = Path(CFG.OUT_BASE)
    (base / "reduced").mkdir(parents=True, exist_ok=True)
    (base / "masked").mkdir(parents=True, exist_ok=True)
    (base / "interp_filtered").mkdir(parents=True, exist_ok=True)
    (base / "norm_screen").mkdir(parents=True, exist_ok=True)
    (base / "templates").mkdir(parents=True, exist_ok=True)
    (base / "features").mkdir(parents=True, exist_ok=True)
    (base / "linear_metrics").mkdir(parents=True, exist_ok=True)

def load_raw_files() -> List[Path]:
    files = list_csvs(CFG.RAW_DIR)
    if not files:
        print(f"No CSV files found in RAW_DIR: {CFG.RAW_DIR}")
        sys.exit(1)
    return files

def write_per_frame_metrics(
    out_root: Path,
    source: str,
    participant: str,
    condition: str,
    perframe: Dict[str, np.ndarray],
    interocular: np.ndarray,
    n_frames: int
) -> None:
    """
    Save per-frame feature series to CSV: one file per participant-condition per source.
    """
    out_dir = out_root / "per_frame" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{participant}_{condition}_perframe.csv"

    df_pf = pd.DataFrame({
        "participant": participant,
        "condition": condition,
        "frame": np.arange(n_frames, dtype=int),
        "interocular": interocular
    })
    # Attach each metric array (same length n_frames)
    for k, arr in perframe.items():
        df_pf[k] = arr
    df_pf.to_csv(out_path, index=False)

def run_pipeline():
    def want_any_steps_1_to_7() -> bool:
        return any([
            RUN_FILTER,
            RUN_MASK,
            RUN_INTERP_FILTER,
            RUN_NORM,
            RUN_TEMPLATES,
            RUN_FEATURES_PROCRUSTES_GLOBAL,
            RUN_FEATURES_PROCRUSTES_PARTICIPANT,
            RUN_FEATURES_ORIGINAL,
        ])

    ensure_dirs()

    # ========================================================================
    # LINEAR-ONLY MODE (rehydrate from disk)
    # ========================================================================
    if RUN_LINEAR and not want_any_steps_1_to_7():
        print("\n=== Linear-only mode: reading per-frame CSVs from disk ===")

        lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"
        lm_dir.mkdir(parents=True, exist_ok=True)

        def compute_linear_for_csv(out_name: str) -> Dict[str, int]:
            # Infer source from output name
            if "procrustes_global" in out_name:
                source = "procrustes_global"
            elif "procrustes_participant" in out_name:
                source = "procrustes_participant"
            elif "original" in out_name:
                source = "original"
            else:
                raise ValueError(f"Unknown source for {out_name}")

            per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / source
            if not per_frame_dir.exists():
                print(f"[skip] No per-frame dir for source '{source}': {per_frame_dir}")
                return {}

            rows = []
            drops_agg: Dict[str, int] = {}

            files = sorted(per_frame_dir.glob("*.csv"))
            if not files:
                print(f"[skip] No per-frame CSVs for '{source}' in {per_frame_dir}")
                return {}

            # Iterate saved per-frame CSVs (one per participant_condition)
            for pf in tqdm(files, desc=f"Linear ({source})", unit="file"):
                df = pd.read_csv(pf)  # columns: participant, condition, frame, interocular, metrics...
                pid = str(df["participant"].iloc[0]) if "participant" in df.columns and len(df) else "NA"
                cond = str(df["condition"].iloc[0]) if "condition" in df.columns and len(df) else "NA"

                metric_cols = [c for c in df.columns if c not in ("participant","condition","frame","interocular")]
                if "interocular" not in df.columns:
                    print(f"[warn] 'interocular' missing in {pf.name}; scaling disabled for this file.")
                    io = np.full(len(df), np.nan, dtype=float)
                else:
                    io = df["interocular"].to_numpy(dtype=float)

                # Scale each distance-like metric by interocular (per-frame) where appropriate
                scaled = {}
                for k in metric_cols:
                    arr = df[k].to_numpy(dtype=float)
                    if SCALE_BY_INTEROCULAR and is_distance_like_metric(k) and np.isfinite(io).any():
                        scaled[k] = arr / io
                    else:
                        scaled[k] = arr

                # Window and compute linear metrics
                win = CFG.WINDOW_SECONDS * CFG.FPS
                hop = int(win * (1.0 - CFG.WINDOW_OVERLAP))
                if hop <= 0:
                    hop = max(1, win // 2)

                n = len(df)
                for (s, e, widx) in windows_indices(n, win, hop):
                    base = {
                        "source": source,
                        "participant": pid,
                        "condition": cond,
                        "window_index": widx,
                        "t_start_frame": s,
                        "t_end_frame": e
                    }
                    for k, arr in scaled.items():
                        seg = arr[s:e]
                        if np.any(~np.isfinite(seg)) or len(seg) < 3:
                            drops_agg[k] = drops_agg.get(k, 0) + 1
                            base[f"{k}_mean_abs_vel"] = np.nan
                            base[f"{k}_mean_abs_acc"] = np.nan
                            base[f"{k}_rms"] = np.nan
                        else:
                            v, a, r = linear_metrics(seg.astype(float), CFG.FPS)
                            base[f"{k}_mean_abs_vel"] = v
                            base[f"{k}_mean_abs_acc"] = a
                            base[f"{k}_rms"] = r
                    rows.append(base)

            df_out = pd.DataFrame(rows)
            out_path = Path(CFG.OUT_BASE) / "linear_metrics" / out_name
            df_out.to_csv(out_path, index=False)
            print(f"[OK] Wrote {out_path}")
            return drops_agg

        # Choose sources to compute: either flags are on OR per_frame dirs exist with files
        srcs = []
        for src, flag in [
            ("procrustes_global", RUN_FEATURES_PROCRUSTES_GLOBAL),
            ("procrustes_participant", RUN_FEATURES_PROCRUSTES_PARTICIPANT),
            ("original", RUN_FEATURES_ORIGINAL),
        ]:
            per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / src
            has_files = per_frame_dir.exists() and any(per_frame_dir.glob("*.csv"))
            if flag or has_files:
                srcs.append(src)

        if not srcs:
            print("No per-frame feature CSVs found under features/per_frame/*.")
            print("Run Step 7 once to generate them, or enable a RUN_FEATURES_* flag.")
            # Still write a summary for transparency
            summary = {
                "config": asdict(CFG),
                "flags": {
                    "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK, "RUN_INTERP_FILTER": RUN_INTERP_FILTER,
                    "RUN_NORM": RUN_NORM, "RUN_TEMPLATES": RUN_TEMPLATES,
                    "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
                    "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
                    "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL,
                    "RUN_LINEAR": RUN_LINEAR,
                    "SAVE_REDUCED": SAVE_REDUCED, "SAVE_MASKED": SAVE_MASKED,
                    "SAVE_INTERP_FILTERED": SAVE_INTERP_FILTERED, "SAVE_NORM": SAVE_NORM,
                    "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
                    "SCALE_BY_INTEROCULAR": SCALE_BY_INTEROCULAR
                },
                "masking_overall": {},
                "window_drops": {
                    "procrustes_global": {},
                    "procrustes_participant": {},
                    "original": {},
                    "linear_metrics": {}
                }
            }
            summ_path = Path(CFG.OUT_BASE) / "pipeline_summary.json"
            with open(summ_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary written to: {summ_path}")
            print("Done.")
            return

        linear_drop_totals = {}
        if "procrustes_global" in srcs:
            linear_drop_totals["procrustes_global"] = compute_linear_for_csv("procrustes_global_linear.csv")
        if "procrustes_participant" in srcs:
            linear_drop_totals["procrustes_participant"] = compute_linear_for_csv("procrustes_participant_linear.csv")
        if "original" in srcs:
            linear_drop_totals["original"] = compute_linear_for_csv("original_linear.csv")

        # Summary
        summary = {
            "config": asdict(CFG),
            "flags": {
                "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK, "RUN_INTERP_FILTER": RUN_INTERP_FILTER,
                "RUN_NORM": RUN_NORM, "RUN_TEMPLATES": RUN_TEMPLATES,
                "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
                "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
                "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL,
                "RUN_LINEAR": RUN_LINEAR,
                "SAVE_REDUCED": SAVE_REDUCED, "SAVE_MASKED": SAVE_MASKED,
                "SAVE_INTERP_FILTERED": SAVE_INTERP_FILTERED, "SAVE_NORM": SAVE_NORM,
                "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
                "SCALE_BY_INTEROCULAR": SCALE_BY_INTEROCULAR
            },
            "masking_overall": {},
            "window_drops": {
                "procrustes_global": {},
                "procrustes_participant": {},
                "original": {},
                "linear_metrics": linear_drop_totals
            }
        }
        summ_path = Path(CFG.OUT_BASE) / "pipeline_summary.json"
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {summ_path}")
        print("Done.")
        return

    # ========================================================================
    # FULL/REGULAR MODE (Steps 1–7 as originally written)
    # ========================================================================
    files = load_raw_files()

    # State containers across steps (kept in-memory to avoid re-IO):
    perfile_data = {}     # file -> dict with keys at each step (reduced, masked, interp_filt, norm)
    perfile_meta = {}     # file -> metadata: participant, condition, conf_prefix
    perfile_mask_stats = {}

    # -------------------- STEP 1-3: LOAD → FILTER → MASK ---------------------
    print("\n=== Steps 1–3: Load → Filter → Mask ===")
    for fp in tqdm(files, desc="Load/Filter/Mask", unit="file"):
        df_raw = pd.read_csv(fp)
        conf_prefix = detect_conf_prefix_case_insensitive(list(df_raw.columns))
        indices = relevant_indices()

        # Step 2: Filter
        if RUN_FILTER:
            df_reduced = filter_df_to_relevant(df_raw, conf_prefix, indices)
            if SAVE_REDUCED:
                out = Path(CFG.OUT_BASE) / "reduced" / (fp.stem + "_reduced.csv")
                if OVERWRITE or not out.exists():
                    df_reduced.to_csv(out, index=False)
        else:
            # Previously sys.exit(1) — now just stop the pipeline if a required step is disabled
            print("Requested filtering step not enabled; set RUN_FILTER=True.")
            return

        # Step 3: Mask
        if RUN_MASK:
            df_masked, stats = confidence_mask(df_reduced, conf_prefix, indices, CFG.CONF_THRESH)
            perfile_mask_stats[fp.name] = stats["overall"]
            if SAVE_MASKED:
                out = Path(CFG.OUT_BASE) / "masked" / (fp.stem + "_masked.csv")
                if OVERWRITE or not out.exists():
                    df_masked.to_csv(out, index=False)
        else:
            print("Requested masking step not enabled; set RUN_MASK=True.")
            return

        perfile_data[fp.name] = {"reduced": df_reduced, "masked": df_masked}
        pid, cond = parse_participant_condition(fp.name)
        perfile_meta[fp.name] = {"participant": pid, "condition": cond, "conf_prefix": conf_prefix}

    # -------------------- STEP 4: INTERPOLATE + FILTER -----------------------
    print("\n=== Step 4: Interpolate (run-limited) + Butterworth filter ===")
    if RUN_INTERP_FILTER:
        if not SCIPY_AVAILABLE:
            print("scipy is required for RUN_INTERP_FILTER. Install scipy or disable this step.")
            return
        for fp in tqdm(files, desc="Interp/Filter", unit="file"):
            name = fp.name
            dfm = perfile_data[name]["masked"].copy()
            for col in dfm.columns:
                if col.lower().startswith("x") or col.lower().startswith("y"):
                    dfm[col] = interpolate_run_limited(dfm[col], CFG.MAX_INTERP_RUN)
                    dfm[col] = butterworth_segment_filter(dfm[col], CFG.FILTER_ORDER, CFG.CUTOFF_HZ, CFG.FPS)
            if SAVE_INTERP_FILTERED:
                out = Path(CFG.OUT_BASE) / "interp_filtered" / (fp.stem + "_interp_filt.csv")
                if OVERWRITE or not out.exists():
                    dfm.to_csv(out, index=False)
            perfile_data[name]["interp_filt"] = dfm
    else:
        print("Requested RUN_INTERP_FILTER=False. Downstream steps may fail without cleaned coordinates.")
        return

    # -------------------- STEP 5: SCREEN NORMALIZATION -----------------------
    print("\n=== Step 5: Normalize to screen size (2560×1440) ===")
    if RUN_NORM:
        for fp in tqdm(files, desc="Normalize", unit="file"):
            name = fp.name
            dfc = perfile_data[name]["interp_filt"]
            df_norm = normalize_to_screen(dfc, CFG.IMG_WIDTH, CFG.IMG_HEIGHT)
            if SAVE_NORM:
                out = Path(CFG.OUT_BASE) / "norm_screen" / (fp.stem + "_norm.csv")
                if OVERWRITE or not out.exists():
                    df_norm.to_csv(out, index=False)
            perfile_data[name]["norm"] = df_norm
    else:
        print("Requested RUN_NORM=False. Templates and features require normalized coords.")
        return

    # -------------------- STEP 6: TEMPLATES ----------------------------------
    print("\n=== Step 6: Templates (global + per-participant) ===")
    templ_dir = Path(CFG.OUT_BASE) / "templates"
    global_templ_path = templ_dir / "global_template.csv"

    part_to_files: Dict[str, List[str]] = {}
    for fp in files:
        pid = perfile_meta[fp.name]["participant"]
        part_to_files.setdefault(pid, []).append(fp.name)

    def compute_template_across_files(file_names: List[str]) -> pd.DataFrame:
        if not file_names:
            return pd.DataFrame()
        cols = perfile_data[file_names[0]]["norm"].columns
        accum = []
        for name in file_names:
            accum.append(perfile_data[name]["norm"][cols].astype(float))
        big = pd.concat(accum, axis=0, ignore_index=True)
        x_cols = [c for c in cols if c.lower().startswith("x")]
        y_cols = [c for c in cols if c.lower().startswith("y")]
        templ = pd.DataFrame(index=[0], columns=x_cols + y_cols, dtype=float)
        templ[x_cols] = big[x_cols].mean(axis=0, skipna=True).values
        templ[y_cols] = big[y_cols].mean(axis=0, skipna=True).values
        return templ

    if RUN_TEMPLATES:
        if global_templ_path.exists() and not OVERWRITE_TEMPLATES:
            global_template = pd.read_csv(global_templ_path)
        else:
            all_names = [fp.name for fp in files]
            global_template = compute_template_across_files(all_names)
            global_template.to_csv(global_templ_path, index=False)

        participant_templates: Dict[str, pd.DataFrame] = {}
        for pid, names in tqdm(part_to_files.items(), desc="Participant templates", unit="participant"):
            part_path = templ_dir / f"participant_{pid}_template.csv"
            if part_path.exists() and not OVERWRITE_TEMPLATES:
                participant_templates[pid] = pd.read_csv(part_path)
            else:
                templ = compute_template_across_files(names)
                templ.to_csv(part_path, index=False)
                participant_templates[pid] = templ
    else:
        print("RUN_TEMPLATES=False requested. Procrustes features require templates.")
        return

    # ========================= STEP 7: FEATURES ===============================
    print("\n=== Step 7: Features (windowed 60s, 50% overlap) ===")
    win = CFG.WINDOW_SECONDS * CFG.FPS
    hop = int(win * (1.0 - CFG.WINDOW_OVERLAP))
    if hop <= 0:
        hop = max(1, win // 2)

    rel_idxs = relevant_indices()
    feat_dir = Path(CFG.OUT_BASE) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    def colmaps_for(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cols = list(df.columns)
        conf_prefix = detect_conf_prefix_case_insensitive(cols)
        xs, ys = [], []
        for i in rel_idxs:
            x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
            xs.append(x)
            ys.append(y)
        return xs, ys
    

    # ---------- Helper to compute per-frame Procrustes metrics ---------------
    def procrustes_features_for_file(name: str, template_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute per-frame metrics for file 'name' by aligning its normalized coords to template_df.
        Returns dict of metric_name -> np.ndarray (length = n_frames).
        """
        df = perfile_data[name]["norm"]
        cols = list(df.columns)
        conf_prefix = detect_conf_prefix_case_insensitive(cols)
        xs, ys = colmaps_for(df)
        # Build frame landmark matrix (L,2) per frame on the fly
        # Template landmark matrix (L,2)
        templ_xy = np.column_stack([
            template_df[xs].values[0],  # shape: (L,)
            template_df[ys].values[0]
        ])
        n = len(df)
        L = len(rel_idxs)
        # Prepare outputs
        head_rot = np.full(n, np.nan, float)      # radians
        head_tx  = np.full(n, np.nan, float)
        head_ty  = np.full(n, np.nan, float)
        head_s   = np.full(n, np.nan, float)
        head_motion_mag = np.full(n, np.nan, float)
        blink_ap = np.full(n, np.nan, float)
        mouth_ap = np.full(n, np.nan, float)
        pupil_av = np.full(n, np.nan, float)

        # Precompute landmark groups indices inside rel_idxs
        def idx_of(lmk: int) -> int:
            return rel_idxs.index(lmk) if lmk in rel_idxs else -1

        # Indices for eyes, mouth, nose rings
        L_top_idxs = [idx_of(38), idx_of(39)]
        L_bot_idxs = [idx_of(41), idx_of(42)]
        R_top_idxs = [idx_of(44), idx_of(45)]
        R_bot_idxs = [idx_of(47), idx_of(48)]
        left_eye_ring  = [idx_of(i) for i in [37,38,39,40,41,42] if idx_of(i) >= 0]
        right_eye_ring = [idx_of(i) for i in [43,44,45,46,47,48] if idx_of(i) >= 0]
        mouth_pair = (idx_of(63), idx_of(67))
        eye_pair = (idx_of(37), idx_of(46))

        for t in range(n):
            # Build frame XY for relevant indices
            fx = [df.iloc[t, df.columns.get_loc(xc)] if xc is not None else np.nan for xc in xs]
            fy = [df.iloc[t, df.columns.get_loc(yc)] if yc is not None else np.nan for yc in ys]
            frame_xy = np.column_stack([np.asarray(fx, float), np.asarray(fy, float)])  # (L,2)
            # Available mask (finite in both frame and template)
            available = np.isfinite(frame_xy).all(axis=1) & np.isfinite(templ_xy).all(axis=1)

            ok, s, tx, ty, R, Xtrans = procrustes_frame_to_template(frame_xy, templ_xy, available)
            if not ok:
                continue

            # Head transform parameters
            head_s[t] = s
            head_tx[t] = tx
            head_ty[t] = ty
            head_motion_mag[t] = math.sqrt(tx*tx + ty*ty + (s - 1.0)**2)

            # Head rotation from transformed eye-corner vector (37->46)
            i37, i46 = eye_pair
            if i37 >= 0 and i46 >= 0 and np.isfinite(Xtrans[i37]).all() and np.isfinite(Xtrans[i46]).all():
                head_rot[t] = angle_between_points(Xtrans[i37], Xtrans[i46])

            # Blink aperture (average of both eyes)
            def safe_points(idxs: List[int]) -> Optional[np.ndarray]:
                pts = [Xtrans[i] for i in idxs if i >= 0 and np.isfinite(Xtrans[i]).all()]
                return np.vstack(pts) if len(pts) == 2 else None

            Ltop = safe_points(L_top_idxs); Lbot = safe_points(L_bot_idxs)
            Rtop = safe_points(R_top_idxs); Rbot = safe_points(R_bot_idxs)
            vals = []
            if Ltop is not None and Lbot is not None:
                vals.append(blink_aperture_from_points(Ltop, Lbot))
            if Rtop is not None and Rbot is not None:
                vals.append(blink_aperture_from_points(Rtop, Rbot))
            if vals:
                blink_ap[t] = float(np.mean(vals))

            # Mouth aperture
            m63, m67 = mouth_pair
            if m63 >= 0 and m67 >= 0 and np.isfinite(Xtrans[m63]).all() and np.isfinite(Xtrans[m67]).all():
                mouth_ap[t] = mouth_aperture(Xtrans[m63], Xtrans[m67])

            # Eye ring centers (already defined left_eye_ring, right_eye_ring)
            def eye_center(idxs: List[int]) -> Optional[np.ndarray]:
                pts = [Xtrans[i] for i in idxs if i >= 0 and np.isfinite(Xtrans[i]).all()]
                if len(pts) >= 3:
                    return np.vstack(pts).mean(axis=0)
                return None

            cL = eye_center(left_eye_ring)
            cR = eye_center(right_eye_ring)

            # Pupils: 69 (left), 70 (right)
            i69 = rel_idxs.index(69) if 69 in rel_idxs else -1
            i70 = rel_idxs.index(70) if 70 in rel_idxs else -1

            vals = []
            if i69 >= 0 and cL is not None and np.isfinite(Xtrans[i69]).all():
                vals.append(float(np.linalg.norm(Xtrans[i69] - cL)))
            if i70 >= 0 and cR is not None and np.isfinite(Xtrans[i70]).all():
                vals.append(float(np.linalg.norm(Xtrans[i70] - cR)))
            if vals:
                pupil_av[t] = float(np.mean(vals))

        return {
            "head_rotation_rad": head_rot,
            "head_tx": head_tx,
            "head_ty": head_ty,
            "head_scale": head_s,
            "head_motion_mag": head_motion_mag,
            "blink_aperture": blink_ap,
            "mouth_aperture": mouth_ap,
            "pupil_metric": pupil_av
        }

    # ---------- Helper to compute per-frame Original metrics -----------------
    def original_features_for_file(name: str) -> Dict[str, np.ndarray]:
        df = perfile_data[name]["norm"]
        cols = list(df.columns)
        conf_prefix = detect_conf_prefix_case_insensitive(cols)

        def col(i, axis: str) -> pd.Series:
            c = find_real_colname(axis, i, cols)
            return df[c].astype(float) if c else pd.Series([np.nan]*len(df))

        n = len(df)
        # Head rotation from eye-corner vector (37->46)
        x37, y37 = col(37, "x"), col(37, "y")
        x46, y46 = col(46, "x"), col(46, "y")
        head_rot = np.full(n, np.nan, float)
        vdx = (x46 - x37).values
        vdy = (y46 - y37).values
        valid = np.isfinite(vdx) & np.isfinite(vdy)
        head_rot[valid] = np.arctan2(vdy[valid], vdx[valid])

        # Blink aperture (same definition as Procrustes but in original coords)
        def avg2(s1, s2): return (s1 + s2) / 2.0
        # Left eye
        Ltop = avg2(col(38,"y"), col(39,"y"))
        Lbot = avg2(col(41,"y"), col(42,"y"))
        # Right eye
        Rtop = avg2(col(44,"y"), col(45,"y"))
        Rbot = avg2(col(47,"y"), col(48,"y"))
        blink = np.full(n, np.nan, float)
        L_ok = Ltop.notna() & Lbot.notna()
        R_ok = Rtop.notna() & Rbot.notna()
        if L_ok.any():
            blink[L_ok] = np.abs(Ltop[L_ok].values - Lbot[L_ok].values)
        if R_ok.any():
            tmp = np.abs(Rtop[R_ok].values - Rbot[R_ok].values)
            # average across eyes where both available, else use whichever available
            both = L_ok & R_ok
            blink[both] = (blink[both] + tmp[both]) / 2.0
            onlyR = R_ok & (~L_ok)
            blink[onlyR] = tmp[onlyR]

        # Mouth
        x63, y63 = col(63,"x"), col(63,"y")
        x67, y67 = col(67,"x"), col(67,"y")
        mouth = np.sqrt((x67 - x63)**2 + (y67 - y63)**2).values

        # Eye ring centers in original (normalized) coords
        def eye_center_xy(ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
            xs = [col(i,"x").values for i in ids if find_real_colname("x", i, cols)]
            ys = [col(i,"y").values for i in ids if find_real_colname("y", i, cols)]
            if not xs or not ys:
                return np.full(n, np.nan), np.full(n, np.nan)
            x_mat = np.vstack(xs); y_mat = np.vstack(ys)
            return np.nanmean(x_mat, axis=0), np.nanmean(y_mat, axis=0)

        cLx, cLy = eye_center_xy([37,38,39,40,41,42])
        cRx, cRy = eye_center_xy([43,44,45,46,47,48])

        # Pupils 69 (left), 70 (right)
        pLx, pLy = col(69, "x").values if find_real_colname("x", 69, cols) else np.full(n, np.nan), \
                col(69, "y").values if find_real_colname("y", 69, cols) else np.full(n, np.nan)
        pRx, pRy = col(70, "x").values if find_real_colname("x", 70, cols) else np.full(n, np.nan), \
                col(70, "y").values if find_real_colname("y", 70, cols) else np.full(n, np.nan)

        dL = np.sqrt((pLx - cLx)**2 + (pLy - cLy)**2)
        dR = np.sqrt((pRx - cRx)**2 + (pRy - cRy)**2)

        # Average over available eyes
        pupil = np.where(np.isfinite(dL) & np.isfinite(dR),
                        (dL + dR) / 2.0,
                        np.where(np.isfinite(dL), dL,
                                np.where(np.isfinite(dR), dR, np.nan)))

        # Center-face magnitude: RMS of nose points (28..36) around per-file mean
        nose_x = [col(i, "x").values for i in CFG.CENTER_FACE]
        nose_y = [col(i, "y").values for i in CFG.CENTER_FACE]
        nose_x = np.vstack(nose_x) if len(nose_x) else np.empty((0,n))
        nose_y = np.vstack(nose_y) if len(nose_y) else np.empty((0,n))
        cfm = np.full(n, np.nan, float)
        if nose_x.size and nose_y.size:
            # Per-file mean per landmark over time (ignore NaNs)
            mean_x = np.nanmean(nose_x, axis=1, keepdims=True)
            mean_y = np.nanmean(nose_y, axis=1, keepdims=True)
            dx = nose_x - mean_x
            dy = nose_y - mean_y
            dists = np.sqrt(dx**2 + dy**2)  # shape (K, n)
            cfm = np.sqrt(np.nanmean(dists**2, axis=0))  # (n,)

        return {
            "head_rotation_rad": head_rot,
            "blink_aperture": blink,
            "mouth_aperture": mouth,
            "pupil_metric": pupil,
            "center_face_magnitude": cfm
        }

    # ---------- Windowing + per-window summaries & drops ---------------------
    def window_features(metric_map: Dict[str, np.ndarray],
                        interocular: np.ndarray,
                        fps: int,
                        win: int,
                        hop: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Convert per-frame metric arrays to per-window summaries.
        Rule: if ANY NaN inside a window for a metric → that metric's value is NaN for that window.
        We return a DataFrame with columns per metric containing window means,
        plus metadata (window index, t_start, t_end), and a dict of "windows_dropped" per metric.
        """
        n = len(next(iter(metric_map.values()))) if metric_map else 0
        windows = windows_indices(n, win, hop)
        rows = []
        drops = {k: 0 for k in metric_map.keys()}
        for (s, e, widx) in windows:
            row = {"window_index": widx, "t_start_frame": s, "t_end_frame": e}
            for key, series in metric_map.items():
                seg = series[s:e]
                if np.any(~np.isfinite(seg)) or len(seg) == 0:
                    row[key] = np.nan
                    drops[key] += 1
                else:
                    row[key] = float(np.mean(seg))
            # interocular mean for this window (used later for scaling)
            seg_io = interocular[s:e]
            row["interocular_mean"] = float(np.mean(seg_io)) if np.all(np.isfinite(seg_io)) and len(seg_io) else np.nan
            rows.append(row)
        dfw = pd.DataFrame(rows)
        return dfw, drops

    # ---------- Compute Procrustes features (global + participant) -----------
    procrustes_global_rows = []
    procrustes_part_rows = []
    procrustes_global_drops_agg: Dict[str, int] = {}
    procrustes_part_drops_agg: Dict[str, int] = {}

    if RUN_FEATURES_PROCRUSTES_GLOBAL or RUN_FEATURES_PROCRUSTES_PARTICIPANT:
        print("Computing Procrustes features...")
        for fp in tqdm(files, desc="Procrustes features", unit="file"):
            name = fp.name
            pid = perfile_meta[name]["participant"]
            cond = perfile_meta[name]["condition"]

            if RUN_FEATURES_PROCRUSTES_GLOBAL:
                feats = procrustes_features_for_file(name, global_template)
                io = interocular_series(perfile_data[name]["norm"], perfile_meta[name]["conf_prefix"]).values
                n_frames = len(io)

                if SAVE_PER_FRAME_PROCRUSTES_GLOBAL:
                    write_per_frame_metrics(
                        feat_dir, "procrustes_global", pid, cond,
                        perframe=feats, interocular=io, n_frames=n_frames
                    )
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "condition", cond)
                dfw.insert(0, "participant", pid)
                dfw.insert(0, "source", "procrustes_global")
                procrustes_global_rows.append(dfw)
                # aggregate drops
                for k, v in drops.items():
                    procrustes_global_drops_agg[k] = procrustes_global_drops_agg.get(k, 0) + v

            if RUN_FEATURES_PROCRUSTES_PARTICIPANT:
                templ = participant_templates[pid]
                feats = procrustes_features_for_file(name, templ)
                io = interocular_series(perfile_data[name]["norm"], perfile_meta[name]["conf_prefix"]).values
                n_frames = len(io)
                if SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT:
                    write_per_frame_metrics(
                        feat_dir, "procrustes_participant", pid, cond,
                        perframe=feats, interocular=io, n_frames=n_frames
                    )
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "condition", cond)
                dfw.insert(0, "participant", pid)
                dfw.insert(0, "source", "procrustes_participant")
                procrustes_part_rows.append(dfw)
                for k, v in drops.items():
                    procrustes_part_drops_agg[k] = procrustes_part_drops_agg.get(k, 0) + v

    # ---------- Compute Original features -----------------------------------
    original_rows = []
    original_drops_agg: Dict[str, int] = {}
    if RUN_FEATURES_ORIGINAL:
        print("Computing Original (no Procrustes) features...")
        for fp in tqdm(files, desc="Original features", unit="file"):
            name = fp.name
            pid = perfile_meta[name]["participant"]
            cond = perfile_meta[name]["condition"]

            feats = original_features_for_file(name)
            io = interocular_series(perfile_data[name]["norm"], perfile_meta[name]["conf_prefix"]).values
            n_frames = len(io)
            if SAVE_PER_FRAME_ORIGINAL:
                write_per_frame_metrics(
                    feat_dir, "original", pid, cond,
                    perframe=feats, interocular=io, n_frames=n_frames
                )
            dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
            dfw.insert(0, "condition", cond)
            dfw.insert(0, "participant", pid)
            dfw.insert(0, "source", "original")
            original_rows.append(dfw)
            for k, v in drops.items():
                original_drops_agg[k] = original_drops_agg.get(k, 0) + v

    # Save Step 7 CSVs
    feat_dir = Path(CFG.OUT_BASE) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    procrustes_global_path = feat_dir / "procrustes_global_features.csv"
    procrustes_part_path   = feat_dir / "procrustes_participant_features.csv"
    original_path          = feat_dir / "original_features.csv"



    if RUN_FEATURES_PROCRUSTES_GLOBAL and procrustes_global_rows:
        pd.concat(procrustes_global_rows, ignore_index=True).to_csv(procrustes_global_path, index=False)
    if RUN_FEATURES_PROCRUSTES_PARTICIPANT and procrustes_part_rows:
        pd.concat(procrustes_part_rows, ignore_index=True).to_csv(procrustes_part_path, index=False)
    if RUN_FEATURES_ORIGINAL and original_rows:
        pd.concat(original_rows, ignore_index=True).to_csv(original_path, index=False)

    # ========================= STEP 8: LINEAR METRICS ========================
    print("\n=== Step 8: Interocular scaling + linear metrics (per window) ===")
    lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"
    lm_dir.mkdir(parents=True, exist_ok=True)

    def compute_linear_for_csv_from_perframe(source: str, out_name: str) -> Dict[str, int]:
        per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / source
        rows = []
        drops_agg: Dict[str, int] = {}
        for pf in sorted(per_frame_dir.glob("*.csv")):
            df = pd.read_csv(pf)
            pid = str(df["participant"].iloc[0])
            cond = str(df["condition"].iloc[0])
            metric_cols = [c for c in df.columns if c not in ("participant","condition","frame","interocular")]
            io = df["interocular"].to_numpy(float) if "interocular" in df.columns else np.full(len(df), np.nan)
            scaled = {}
            for k in metric_cols:
                arr = df[k].to_numpy(float)
                if SCALE_BY_INTEROCULAR and is_distance_like_metric(k) and np.isfinite(io).any():
                    scaled[k] = arr / io
                else:
                    scaled[k] = arr
            n = len(df)
            for (s, e, widx) in windows_indices(n, CFG.WINDOW_SECONDS*CFG.FPS, int(CFG.WINDOW_SECONDS*CFG.FPS*(1-CFG.WINDOW_OVERLAP))):
                base = {"source": source, "participant": pid, "condition": cond,
                        "window_index": widx, "t_start_frame": s, "t_end_frame": e}
                for k, arr in scaled.items():
                    seg = arr[s:e]
                    if np.any(~np.isfinite(seg)) or len(seg) < 3:
                        drops_agg[k] = drops_agg.get(k, 0) + 1
                        base[f"{k}_mean_abs_vel"] = np.nan
                        base[f"{k}_mean_abs_acc"] = np.nan
                        base[f"{k}_rms"] = np.nan
                    else:
                        v, a, r = linear_metrics(seg.astype(float), CFG.FPS)
                        base[f"{k}_mean_abs_vel"] = v
                        base[f"{k}_mean_abs_acc"] = a
                        base[f"{k}_rms"] = r
                rows.append(base)
        df_out = pd.DataFrame(rows)
        out_path = lm_dir / out_name
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Wrote {out_path}")
        return drops_agg

    linear_drop_totals = {}
    if RUN_LINEAR:
        if RUN_FEATURES_PROCRUSTES_GLOBAL:
            linear_drop_totals["procrustes_global"] = compute_linear_for_csv_from_perframe("procrustes_global", "procrustes_global_linear.csv")
        if RUN_FEATURES_PROCRUSTES_PARTICIPANT:
            linear_drop_totals["procrustes_participant"] = compute_linear_for_csv_from_perframe("procrustes_participant", "procrustes_participant_linear.csv")
        if RUN_FEATURES_ORIGINAL:
            linear_drop_totals["original"] = compute_linear_for_csv_from_perframe("original", "original_linear.csv")

    # ========================= SUMMARY JSON ==================================
    summary = {
        "config": asdict(CFG),
        "flags": {
            "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK, "RUN_INTERP_FILTER": RUN_INTERP_FILTER,
            "RUN_NORM": RUN_NORM, "RUN_TEMPLATES": RUN_TEMPLATES,
            "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
            "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
            "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL,
            "RUN_LINEAR": RUN_LINEAR,
            "SAVE_REDUCED": SAVE_REDUCED, "SAVE_MASKED": SAVE_MASKED,
            "SAVE_INTERP_FILTERED": SAVE_INTERP_FILTERED, "SAVE_NORM": SAVE_NORM,
            "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
            "SCALE_BY_INTEROCULAR": SCALE_BY_INTEROCULAR
        },
        "masking_overall": perfile_mask_stats,
        "window_drops": {
            "procrustes_global": procrustes_global_drops_agg if RUN_FEATURES_PROCRUSTES_GLOBAL else {},
            "procrustes_participant": procrustes_part_drops_agg if RUN_FEATURES_PROCRUSTES_PARTICIPANT else {},
            "original": original_drops_agg if RUN_FEATURES_ORIGINAL else {},
            "linear_metrics": linear_drop_totals if RUN_LINEAR else {}
        }
    }
    summ_path = Path(CFG.OUT_BASE) / "pipeline_summary.json"
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summ_path}")
    print("Done.")


# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

if __name__ == "__main__":
    print("POSE PIPELINE — standalone mode")
    print("Config:")
    for k,v in asdict(CFG).items():
        print(f"  {k}: {v}")
    print("\nFlags:")
    print(textwrap.indent(
        "\n".join([f"{k}: {globals()[k]}" for k in [
            "RUN_FILTER","RUN_MASK","RUN_INTERP_FILTER","RUN_NORM",
            "RUN_TEMPLATES","RUN_FEATURES_PROCRUSTES_GLOBAL","RUN_FEATURES_PROCRUSTES_PARTICIPANT",
            "RUN_FEATURES_ORIGINAL","RUN_LINEAR",
            "SAVE_REDUCED","SAVE_MASKED","SAVE_INTERP_FILTERED","SAVE_NORM",
            "OVERWRITE","OVERWRITE_TEMPLATES"
        ]]),
        "  "
    ))
    if not SCIPY_AVAILABLE and RUN_INTERP_FILTER:
        print("\nERROR: scipy is required for RUN_INTERP_FILTER. Install scipy or set RUN_INTERP_FILTER=False.")
        sys.exit(1)
    run_pipeline()
