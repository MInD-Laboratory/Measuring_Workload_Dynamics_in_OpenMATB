from __future__ import annotations
import math
import numpy as np
from typing import Tuple

def procrustes_frame_to_template(frame_xy: np.ndarray, templ_xy: np.ndarray, available_mask: np.ndarray) -> Tuple[bool, float, float, float, np.ndarray, np.ndarray]:
    idx = np.where(available_mask)[0]
    if idx.size < 3:
        return False, np.nan, np.nan, np.nan, np.full((2,2), np.nan), np.full_like(frame_xy, np.nan)

    X = frame_xy[idx, :]
    Y = templ_xy[idx, :]

    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    Xc = X - muX
    Yc = Y - muY

    C = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    varX = (Xc**2).sum()
    s = (S.sum()) / varX if varX > 0 else 1.0
    t = (muY.T - s * R @ muX.T).reshape(2)
    Xall = frame_xy.copy()
    Xall_centered = Xall - muX
    Xtrans = (s * (R @ Xall_centered.T)).T + muY
    return True, float(s), float(t[0]), float(t[1]), R, Xtrans

def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.atan2(dy, dx)
