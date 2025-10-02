"""Geometric transformation utilities for pose alignment and analysis.

Provides functions for Procrustes analysis to align facial landmarks
to reference templates, and basic geometric calculations.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

def procrustes_frame_to_template(frame_xy: np.ndarray, templ_xy: np.ndarray, available_mask: np.ndarray) -> Tuple[bool, float, float, float, np.ndarray, np.ndarray]:
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
        - scale: Scaling factor applied to align shapes
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
        # Return failure flag and NaN values for all outputs
        return False, np.nan, np.nan, np.nan, np.full((2,2), np.nan), np.full_like(frame_xy, np.nan)

    # Extract only available landmarks from both frame and template
    X = frame_xy[idx, :]  # Source points
    Y = templ_xy[idx, :]  # Target points

    # Center both point sets by subtracting their centroids
    muX = X.mean(axis=0, keepdims=True)  # Source centroid
    muY = Y.mean(axis=0, keepdims=True)  # Target centroid
    Xc = X - muX  # Centered source
    Yc = Y - muY  # Centered target

    # Compute cross-covariance matrix between centered point sets
    C = Xc.T @ Yc  # 2x2 matrix

    # Find optimal rotation using Singular Value Decomposition
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T  # Rotation matrix

    # Handle reflection case - ensure rotation has positive determinant
    if np.linalg.det(R) < 0:
        # Flip sign of second row of Vt to get proper rotation
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    # Calculate optimal uniform scaling factor
    varX = (Xc**2).sum()  # Total variance of centered source points
    # Scale is ratio of cross-covariance to source variance
    # Handle edge case: if variance is very small (points nearly coincident),
    # use scale=1.0 to avoid division issues and unrealistic scaling
    if varX < 1e-10:  # Numerical tolerance for near-zero variance
        s = 1.0
        import warnings
        warnings.warn("Source points have near-zero variance, using scale=1.0")
    else:
        s = (S.sum()) / varX

    # Calculate translation vector (shift after rotation and scaling)
    t = (muY.T - s * R @ muX.T).reshape(2)

    # Apply transformation to ALL landmarks (not just available ones)
    Xall = frame_xy.copy()
    Xall_centered = Xall - muX  # Center using same centroid as fitting
    # Apply scaling, rotation, then translation
    Xtrans = (s * (R @ Xall_centered.T)).T + muY

    # Return success flag and transformation parameters
    return True, float(s), float(t[0]), float(t[1]), R, Xtrans

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
