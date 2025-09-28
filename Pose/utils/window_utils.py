from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def windows_indices(n: int, win: int, hop: int) -> List[Tuple[int,int,int]]:
    out = []
    w = 0
    start = 0
    while start + win <= n:
        out.append((start, start + win, w))
        start += hop
        w += 1
    return out

def is_distance_like_metric(name: str) -> bool:
    if name in ("head_rotation_rad", "head_scale"):
        return False
    return True

def linear_metrics(x: np.ndarray, fps: float) -> Tuple[float, float, float]:
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    dt = 1.0 / fps
    vel = np.diff(x) / dt
    acc = np.diff(vel) / dt
    mean_abs_vel = float(np.mean(np.abs(vel)))
    mean_abs_acc = float(np.mean(np.abs(acc)))
    rms = float(np.sqrt(np.mean((x - np.mean(x))**2)))
    return mean_abs_vel, mean_abs_acc, rms

def window_features(metric_map: Dict[str, np.ndarray],
                    interocular: np.ndarray,
                    fps: int,
                    win: int,
                    hop: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
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
        seg_io = interocular[s:e]
        row["interocular_mean"] = float(np.mean(seg_io)) if np.all(np.isfinite(seg_io)) and len(seg_io) else np.nan
        rows.append(row)
    dfw = pd.DataFrame(rows)
    return dfw, drops
