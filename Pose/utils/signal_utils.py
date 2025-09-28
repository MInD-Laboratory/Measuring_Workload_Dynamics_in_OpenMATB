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
        temp[~(~disallowed)] = np.nan  # ensure disallowed NaNs remain NaN
        temp = temp.interpolate(method="linear", limit=None, limit_direction="both")
        y_interp[allow] = temp[allow]
        y_interp[~allow & nan_mask] = np.nan
    return y_interp

def butterworth_segment_filter(series: pd.Series, order: int, cutoff_hz: float, fs: float) -> pd.Series:
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for Butterworth filtering.")
    x = series.astype(float).values.copy()
    nyq = fs / 2.0
    wn = min(0.999, cutoff_hz / nyq)
    b, a = butter(order, wn, btype='low', analog=False)
    padlen = 3 * (max(len(a), len(b)) - 1)

    start = 0
    while start < len(x):
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
        start = end
    return pd.Series(x, index=series.index, dtype=float)
