from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from .config import CFG, SCIPY_AVAILABLE

# Local import only when available
if SCIPY_AVAILABLE:
    from scipy.signal import butter, filtfilt  # type: ignore

def find_nan_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
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
    """Interpolate NaN values only for runs shorter than max_run.

    Args:
        series: Input series with potential NaN values
        max_run: Maximum consecutive NaN values to interpolate

    Returns:
        Series with short NaN runs interpolated, long runs kept as NaN
    """
    x = series.astype(float).copy()
    nan_mask = x.isna().values
    runs = find_nan_runs(nan_mask)
    allow = np.zeros_like(nan_mask, dtype=bool)
    for s, e in runs:
        if (e - s) <= max_run:
            allow[s:e] = True
    y_interp = x.copy()
    if allow.any():
        disallowed = nan_mask & (~allow)
        temp = y_interp.copy()
        # Simplified: directly set disallowed positions to NaN
        temp[disallowed] = np.nan  # ensure disallowed NaNs remain NaN
        temp = temp.interpolate(method="linear", limit=None, limit_direction="both")
        y_interp[allow] = temp[allow]
        y_interp[~allow & nan_mask] = np.nan
    return y_interp

def butterworth_segment_filter(series: pd.Series, order: int, cutoff_hz: float, fs: float) -> pd.Series:
    """
    Low-pass Butterworth filtering applied per contiguous finite (non-NaN/inf) segment.
    Segments with length <= padlen are left UNFILTERED to avoid filtfilt ValueError.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for Butterworth filtering.")

    # Work on a float copy; keep the original index for return
    x = series.astype(float).values.copy()
    n = len(x)

    # Design digital Butterworth LPF
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    wn = max(0.0, min(0.999, wn))          # clamp normalized cutoff to (0, 1)
    b, a = butter(order, wn, btype='low', analog=False)

    # filtfilt's default pad length: 3 * (max(len(a), len(b)) - 1)
    padlen = 3 * (max(len(a), len(b)) - 1)

    # Scan through contiguous finite (valid) runs
    i = 0
    while i < n:
        # Skip invalid samples (NaN or inf)
        while i < n and not np.isfinite(x[i]):
            i += 1
        if i >= n:
            break

        # Find end of this valid segment [i, j)
        j = i
        while j < n and np.isfinite(x[j]):
            j += 1

        seg_len = j - i
        # Only filter if segment is long enough for filtfilt
        if seg_len > padlen:
            try:
                x[i:j] = filtfilt(b, a, x[i:j])  # filter in place
            except ValueError:
                # Extremely rare guard: if SciPy still complains, leave unfiltered
                pass
        # else: too short → leave as-is (skip instead of crashing)

        i = j  # advance to next segment

    return pd.Series(x, index=series.index, dtype=float)
