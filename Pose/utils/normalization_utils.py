"""Normalization utilities for facial landmark coordinates.

Provides functions to normalize landmark coordinates relative to screen dimensions
and calculate inter-ocular distance for scale normalization.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from .preprocessing_utils import find_real_colname

def normalize_to_screen(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Normalize pixel coordinates to screen-relative coordinates [0, 1].

    Converts absolute pixel coordinates to relative coordinates by dividing
    x-coordinates by screen width and y-coordinates by screen height.
    This makes coordinates independent of screen resolution.

    Args:
        df: DataFrame containing x and y coordinate columns
        width: Screen width in pixels
        height: Screen height in pixels

    Returns:
        DataFrame with normalized coordinates where:
        - x values range from 0 (left edge) to 1 (right edge)
        - y values range from 0 (top edge) to 1 (bottom edge)

    Note:
        - Identifies x/y columns by checking if column name starts with 'x' or 'y' (case-insensitive)
        - Creates a copy to avoid modifying the original DataFrame

    Potential Issues:
        - Assumes all columns starting with 'x'/'y' are coordinates (might catch unintended columns)
        - No validation that width/height are positive
        - No check for coordinates outside screen bounds
    """
    # Create a copy to avoid modifying original data
    out = df.copy()

    # Find all x and y coordinate columns based on naming convention
    # More strict pattern: must be 'x' or 'y' followed by a digit
    import re
    x_cols = [c for c in out.columns if re.match(r'^[xX]\d+', c)]
    y_cols = [c for c in out.columns if re.match(r'^[yY]\d+', c)]

    # Warn if no coordinate columns found
    if not x_cols and not y_cols:
        import warnings
        warnings.warn("No coordinate columns found matching pattern x<digit> or y<digit>")

    # Normalize x coordinates by screen width (0 = left, 1 = right)
    out[x_cols] = out[x_cols] / float(width)

    # Normalize y coordinates by screen height (0 = top, 1 = bottom)
    out[y_cols] = out[y_cols] / float(height)

    return out

def interocular_series(df: pd.DataFrame, conf_prefix: Optional[str] = None) -> pd.Series:
    """
    Calculate inter-ocular distance for each frame.

    Uses **landmarks 37 (right eye)** and **46 (left eye)** in 1-based
    OpenPose indexing. Inter-ocular distance is computed as
    the Euclidean distance between these two points.
    """
    # Get list of all columns for column name searching
    cols = list(df.columns)

    # Find actual column names for eye corner landmarks (37: left eye, 46: right eye)
    x37_col = find_real_colname("x", 37, cols)  # Left eye corner x
    y37_col = find_real_colname("y", 37, cols)  # Left eye corner y
    x46_col = find_real_colname("x", 46, cols)  # Right eye corner x
    y46_col = find_real_colname("y", 46, cols)  # Right eye corner y

    # Return NaN series if any required column is missing
    if not (x37_col and y37_col and x46_col and y46_col):
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Extract and convert coordinate values to float, coercing errors to NaN
    x37 = pd.to_numeric(df[x37_col], errors="coerce").astype(float)
    y37 = pd.to_numeric(df[y37_col], errors="coerce").astype(float)
    x46 = pd.to_numeric(df[x46_col], errors="coerce").astype(float)
    y46 = pd.to_numeric(df[y46_col], errors="coerce").astype(float)

    # If confidence prefix provided, check confidence values
    if conf_prefix:
        c37_col = find_real_colname(conf_prefix, 37, cols)  # Confidence for landmark 37
        c46_col = find_real_colname(conf_prefix, 46, cols)  # Confidence for landmark 46

        if c37_col and c46_col:
            # Get confidence values
            c37 = pd.to_numeric(df[c37_col], errors="coerce").astype(float)
            c46 = pd.to_numeric(df[c46_col], errors="coerce").astype(float)

            # Use config threshold or default to 0.3
            from .config import CFG
            conf_thresh = getattr(CFG, 'CONF_THRESH', 0.3)

            # Mask coordinates where confidence is below threshold
            low_conf_mask = (c37 < conf_thresh) | (c46 < conf_thresh)
            x37[low_conf_mask] = np.nan
            y37[low_conf_mask] = np.nan
            x46[low_conf_mask] = np.nan
            y46[low_conf_mask] = np.nan

    # Calculate Euclidean distance between eye corners
    distance = np.sqrt((x46 - x37) ** 2 + (y46 - y37) ** 2)

    # Validate that distances are reasonable (in normalized coordinates)
    # Typical inter-ocular distance is 5-15% of screen width in normalized coords
    # Flag suspicious values but don't modify them (let downstream handle)
    import warnings
    if isinstance(distance, pd.Series):
        # Check for any suspiciously small or large values
        too_small = (distance < 0.01) & distance.notna()
        too_large = (distance > 0.5) & distance.notna()
        if too_small.any():
            warnings.warn(f"Found {too_small.sum()} frames with suspiciously small inter-ocular distance (<0.01)")
        if too_large.any():
            warnings.warn(f"Found {too_large.sum()} frames with suspiciously large inter-ocular distance (>0.5)")

    return distance
