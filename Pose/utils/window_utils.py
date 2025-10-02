"""Windowing and time-series analysis utilities for pose data.

Provides functions for sliding window analysis, time-domain feature extraction,
and metric classification for pose analysis pipelines.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def windows_indices(n: int, win: int, hop: int) -> List[Tuple[int,int,int]]:
    """Generate sliding window indices for time series analysis.

    Creates overlapping or non-overlapping windows across a sequence.
    Each window is defined by start and end frame indices.

    Args:
        n: Total length of the sequence
        win: Window size in frames
        hop: Step size between windows (hop < win creates overlap)

    Returns:
        List of tuples: (start_frame, end_frame, window_index)

    Example:
        >>> windows_indices(10, 4, 2)
        [(0, 4, 0), (2, 6, 1), (4, 8, 2), (6, 10, 3)]
    """
    out = []  # List to store window definitions
    w = 0     # Window index counter
    start = 0 # Current starting position

    # Generate windows until we can't fit another complete window
    while start + win <= n:
        out.append((start, start + win, w))  # (start, end, window_id)
        start += hop  # Move to next window position
        w += 1        # Increment window counter

    return out

def is_distance_like_metric(name: str) -> bool:
    """Determine if a metric represents a distance-like quantity.

    Distance-like metrics can be normalized by inter-ocular distance,
    while angular or scale metrics should not be normalized.

    Args:
        name: Metric name to classify

    Returns:
        True if metric represents distance/displacement, False for angles/scales

    Note:
        Used to determine which metrics should be scaled by inter-ocular distance
        for cross-participant normalization.
    """
    # Exclude angular measurements and scale factors from distance normalization
    if name in ("head_rotation_rad", "head_scale"):
        return False

    # By default, assume metrics are distance-like (position, aperture, displacement)
    return True

def linear_metrics(x: np.ndarray, fps: float) -> Tuple[float, float, float]:
    """Compute time-domain metrics from a signal segment.

    Calculates velocity, acceleration, and variability measures that capture
    the dynamic characteristics of pose features.

    Args:
        x: Signal values for a time window
        fps: Sampling rate in frames per second

    Returns:
        Tuple of (mean_abs_velocity, mean_abs_acceleration, rms_variability)
        Returns (NaN, NaN, NaN) if insufficient data points

    Note:
        - Velocity: rate of change (first derivative)
        - Acceleration: rate of velocity change (second derivative)
        - RMS: root-mean-square deviation from mean (variability measure)
        - Requires at least 3 data points for meaningful computation
    """
    # Need at least 3 points: x[0], x[1], x[2] to compute vel[0], vel[1] and acc[0]
    if len(x) < 3:
        return np.nan, np.nan, np.nan

    # Time step between samples
    dt = 1.0 / fps

    # Compute first derivative (velocity) using finite differences
    vel = np.diff(x) / dt  # Change in position per unit time

    # Compute second derivative (acceleration) from velocity
    acc = np.diff(vel) / dt  # Change in velocity per unit time

    # Calculate summary statistics
    mean_abs_vel = float(np.mean(np.abs(vel)))        # Mean absolute velocity
    mean_abs_acc = float(np.mean(np.abs(acc)))        # Mean absolute acceleration
    rms = float(np.sqrt(np.mean((x - np.mean(x))**2))) # Root-mean-square variability

    return mean_abs_vel, mean_abs_acc, rms

def window_features(metric_map: Dict[str, np.ndarray],
                    interocular: np.ndarray,
                    fps: int,
                    win: int,
                    hop: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Extract windowed features from time series data.

    Applies sliding windows to multi-variate time series and computes
    summary statistics (mean) for each window. Handles missing data
    by excluding windows with NaN values.

    Args:
        metric_map: Dictionary mapping metric names to time series arrays
        interocular: Array of inter-ocular distances for normalization
        fps: Sampling rate (frames per second) - currently unused
        win: Window size in frames
        hop: Step size between windows

    Returns:
        Tuple of:
        - DataFrame with windowed features (one row per window)
        - Dictionary counting dropped windows per metric due to missing data

    Note:
        Windows containing any NaN values are excluded from analysis.
        Each window gets metadata: window_index, t_start_frame, t_end_frame
    """
    # Determine sequence length from first metric (assume all same length)
    n = len(next(iter(metric_map.values()))) if metric_map else 0

    # Generate sliding window indices
    windows = windows_indices(n, win, hop)

    rows = []  # Store results for each window
    drops = {k: 0 for k in metric_map.keys()}  # Track dropped windows per metric

    # Process each window
    for (s, e, widx) in windows:
        # Initialize row with window metadata
        row = {"window_index": widx, "t_start_frame": s, "t_end_frame": e}

        # Process each metric within this window
        for key, series in metric_map.items():
            seg = series[s:e]  # Extract window segment

            # Check for missing data or empty segments
            if np.any(~np.isfinite(seg)) or len(seg) == 0:
                row[key] = np.nan  # Mark as missing
                drops[key] += 1    # Count dropped window
            else:
                # Compute mean value for this window
                row[key] = float(np.mean(seg))

        # Process inter-ocular distance for this window
        seg_io = interocular[s:e]
        if np.all(np.isfinite(seg_io)) and len(seg_io) > 0:
            row["interocular_mean"] = float(np.mean(seg_io))
        else:
            row["interocular_mean"] = np.nan

        rows.append(row)

    # Convert to DataFrame
    dfw = pd.DataFrame(rows)
    return dfw, drops
