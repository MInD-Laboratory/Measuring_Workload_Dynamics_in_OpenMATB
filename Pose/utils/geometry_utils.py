"""Geometric transformation utilities for pose alignment and analysis.

Provides functions for Procrustes analysis to align facial landmarks
to reference templates, and basic geometric calculations.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

def procrustes_frame_to_template(frame_xy: np.ndarray, templ_xy: np.ndarray, available_mask: np.ndarray) -> Tuple[bool, float, float, float, float, np.ndarray, np.ndarray]:
    """Align frame landmarks to template using Procrustes analysis.

    Performs Procrustes superimposition to find the optimal rigid transformation
    (rotation, translation, and uniform scaling) that aligns the source frame
    landmarks to a reference template. Uses only landmarks marked as available.

    Args:
        frame_xy: Source landmarks to align, shape (n_landmarks, 2)
        templ_xy: Target template landmarks, shape (n_landmarks, 2)
        available_mask: Boolean mask indicating which landmarks are valid, shape (n_landmarks,)

    Returns:
        Tuple containing:
        - success: True if alignment succeeded, False if insufficient landmarks
        - sx: Scaling factor in x direction
        - sy: Scaling factor in y direction
        - tx: Translation in x direction
        - ty: Translation in y direction
        - R: 2x2 rotation matrix
        - Xtrans: Transformed landmarks after alignment, shape (n_landmarks, 2)

    Note:
        Requires at least 3 valid landmarks for alignment.
        Uses SVD to find optimal rotation matrix.
        Handles reflection by ensuring rotation has positive determinant.

    Potential Issues:
        - Division by zero if varX is 0 (all points coincident) - handled but scale defaults to 1.0
        - Numerical instability with nearly collinear points
    """
    # Find indices of available (valid) landmarks
    idx = np.where(available_mask)[0]

    # Need at least 3 points for meaningful alignment
    if idx.size < 3:
        # Return failure flag and NaN values for all outputs (sx, sy, tx, ty, R, Xtrans)
        return False, np.nan, np.nan, np.nan, np.nan, np.full((2,2), np.nan), np.full_like(frame_xy, np.nan)

    # Extract only available landmarks from both frame and template
    X = frame_xy[idx, :]  # Source points
    Y = templ_xy[idx, :]  # Target points

    # Center both point sets by subtracting their centroids
    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    Xc = X - muX
    Yc = Y - muY

    # Compute best linear transform A (2x2)
    A = np.linalg.lstsq(Xc, Yc, rcond=None)[0]  # minimizes ||Xc A - Yc||^2

    # Polar decomposition: A = R * M, where R is rotation, M is symmetric
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt

    # Compute the "scaling" component in the coordinate frame of R
    M = R.T @ A
    # Extract anisotropic scales (no shear)
    sx, sy = np.diag(M)

    # Recompose: rotation + anisotropic scale
    S = np.diag([sx, sy])

    # Compute translation
    t = muY.T - R @ S @ muX.T

    # Apply to all points
    Xall_centered = frame_xy - muX
    Xtrans = (R @ S @ Xall_centered.T).T + muY

    # Return success flag and transformation parameters
    return True, float(sx), float(sy), float(t[0]), float(t[1]), R, Xtrans

def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate the angle between two 2D points.

    Computes the angle of the vector from p1 to p2 relative to the positive x-axis.
    Uses atan2 for proper quadrant handling.

    Args:
        p1: Starting point, shape (2,) with [x, y] coordinates
        p2: Ending point, shape (2,) with [x, y] coordinates

    Returns:
        Angle in radians from -π to π, where:
        - 0 means p2 is directly to the right of p1
        - π/2 means p2 is directly above p1
        - -π/2 means p2 is directly below p1
        - ±π means p2 is directly to the left of p1

    Note:
        Result is in mathematical convention (counter-clockwise positive).
        To convert to degrees: angle_degrees = math.degrees(angle_radians)

    Potential Issues:
        - No check if p1 and p2 are the same point (returns 0 in this case)
        - Assumes 2D points, no validation of input shape
    """
    # Calculate displacement vector from p1 to p2
    dx = float(p2[0] - p1[0])  # Change in x
    dy = float(p2[1] - p1[1])  # Change in y

    # atan2 handles all quadrants correctly and division by zero
    return math.atan2(dy, dx)
