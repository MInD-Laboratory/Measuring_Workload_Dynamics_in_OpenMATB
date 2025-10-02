"""Data preprocessing and manipulation utilities for pose analysis.

This module contains functions for processing, filtering, and transforming
landmark data from pose estimation outputs.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

from .config import CFG


# ---------- Filename and metadata parsing ------------------------------------
def parse_participant_trial(filename: str) -> Tuple[str, int]:
    """Parse participant ID and trial number from pose filename.

    Args:
        filename: Pose filename (e.g., '3101_02_pose.csv')

    Returns:
        Tuple of (participant_id, trial_number)

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    base = Path(filename).name
    # Match pattern: PPPP_TT where PPPP is participant ID and TT is trial number
    m = re.match(r"^(\d{4})_(\d{2}).*\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern 'PPPP_TT_*.csv': {base}")
    return m.group(1), int(m.group(2))


def load_participant_info(participant_info_path: str) -> pd.DataFrame:
    """Load participant information including condition order.

    Args:
        participant_info_path: Path to participant_info.csv file

    Returns:
        DataFrame with participant IDs and session conditions
    """
    df = pd.read_csv(participant_info_path)
    # Clean participant ID to ensure it's a string
    df['Participant ID'] = df['Participant ID'].astype(str)
    return df


def create_condition_mapping(participant_info: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """Create a mapping from participant ID and trial number to condition.

    Args:
        participant_info: DataFrame from load_participant_info()

    Returns:
        Nested dict: {participant_id: {trial_number: condition}}
        Example: mapping['3101'][1] returns 'L'
    """
    condition_map = {}

    for _, row in participant_info.iterrows():
        participant_id = str(row['Participant ID'])

        # Create trial mapping for this participant
        trial_map = {}

        # Parse each session (Session01, Session02, Session03)
        for session_num in [1, 2, 3]:
            session_col = f'session0{session_num}'
            if session_col in row and not pd.isna(row[session_col]) and row[session_col] != '-':
                session_value = str(row[session_col])

                # Extract condition letter (L, M, or H) - first character
                if session_value:
                    condition = session_value[0]  # L, M, or H
                    # Map trial number (session number) to condition
                    trial_map[session_num] = condition

        condition_map[participant_id] = trial_map

    return condition_map


def get_condition_for_file(filename: str, condition_map: Dict[str, Dict[int, str]]) -> str:
    """Get the condition (L/M/H) for a given pose file.

    Args:
        filename: Pose filename
        condition_map: Mapping from create_condition_mapping()

    Returns:
        Condition letter ('L', 'M', or 'H')

    Raises:
        ValueError: If no condition mapping found
    """
    participant_id, trial_num = parse_participant_trial(filename)

    if participant_id not in condition_map:
        raise ValueError(f"No condition mapping for participant {participant_id}")

    if trial_num not in condition_map[participant_id]:
        raise ValueError(f"No condition mapping for participant {participant_id} trial {trial_num}")

    return condition_map[participant_id][trial_num]


# ---------- Column detection and selection -----------------------------------
def detect_conf_prefix_case_insensitive(columns: List[str]) -> str:
    """Detect the confidence column prefix from available columns.

    Checks for common confidence prefixes: 'prob', 'c', 'confidence'

    Args:
        columns: List of column names to search

    Returns:
        The detected confidence prefix

    Raises:
        ValueError: If no confidence columns found
    """
    cols_low = [c.lower() for c in columns]
    for prefix in ("prob", "c", "confidence"):
        if any(col.startswith(prefix) for col in cols_low):
            return prefix
    raise ValueError("Confidence prefix not found (expected 'prob*', 'c*', or 'confidence*').")


def find_real_colname(prefix: str, i: int, columns: List[str]) -> Optional[str]:
    """Find the actual column name for a given prefix and index.

    Handles case-insensitive matching and partial matches.

    Args:
        prefix: Column prefix (e.g., 'x', 'y', 'prob')
        i: Landmark index number
        columns: List of available column names

    Returns:
        Actual column name if found, None otherwise
    """
    target = f"{prefix}{i}".lower()
    # First try exact match
    for col in columns:
        if col.lower() == target:
            return col
    # Then try prefix match
    for col in columns:
        if col.lower().startswith(target):
            return col
    return None


def lm_triplet_colnames(i: int, conf_prefix: str, columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get the (x, y, confidence) column names for a landmark.

    Args:
        i: Landmark index
        conf_prefix: Confidence column prefix
        columns: Available column names

    Returns:
        Tuple of (x_col, y_col, conf_col) names, or None for missing columns
    """
    return (
        find_real_colname("x", i, columns),
        find_real_colname("y", i, columns),
        find_real_colname(conf_prefix, i, columns),
    )


# ---------- Landmark selection and filtering ---------------------------------
def relevant_indices() -> List[int]:
    """Get the list of relevant landmark indices from configuration.

    Combines all landmark groups defined in config:
    - Procrustes reference points
    - Head rotation points
    - Mouth points
    - Center face points
    - Blink detection points (both eyes)
    - Pupil points

    Returns:
        Sorted list of unique landmark indices
    """
    s = set()
    s.update(CFG.PROCRUSTES_REF)
    s.update(CFG.HEAD_ROT)
    s.update(CFG.MOUTH)
    s.update(CFG.CENTER_FACE)
    s.update(CFG.BLINK_L_TOP)
    s.update(CFG.BLINK_L_BOT)
    s.update(CFG.BLINK_R_TOP)
    s.update(CFG.BLINK_R_BOT)
    s.update([69, 70])  # pupils
    return sorted(s)


def filter_df_to_relevant(df: pd.DataFrame, conf_prefix: str, indices: List[int]) -> pd.DataFrame:
    """Filter DataFrame to keep only relevant landmark columns.

    Args:
        df: Input DataFrame with all landmarks
        conf_prefix: Confidence column prefix
        indices: List of landmark indices to keep

    Returns:
        DataFrame with only the (x, y, conf) columns for specified landmarks

    Raises:
        ValueError: If no relevant columns found
    """
    kept: List[str] = []
    cols = list(df.columns)

    for i in indices:
        x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
        if x and y and c:
            kept.extend([x, y, c])

    if not kept:
        raise ValueError("No relevant triplets found.")

    return df.loc[:, kept].copy()


# ---------- Confidence-based masking -----------------------------------------
def confidence_mask(df_reduced: pd.DataFrame, conf_prefix: str, indices: List[int], thr: float) -> Tuple[pd.DataFrame, Dict]:
    """Apply confidence masking to landmark data.

    Sets coordinates to NaN where confidence is below threshold.

    Args:
        df_reduced: DataFrame with landmark data
        conf_prefix: Confidence column prefix
        indices: Landmark indices to process
        thr: Confidence threshold (0-1)

    Returns:
        Tuple of:
        - DataFrame with low-confidence values masked
        - Dictionary with masking statistics
    """
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

        # Get confidence values and identify low-confidence frames
        conf = pd.to_numeric(dfm[c], errors="coerce")
        low = conf < thr
        low = low.fillna(False)

        # Count values before masking
        pre_x = dfm[x].notna()
        pre_y = dfm[y].notna()
        considered = int((pre_x | pre_y).sum()) * 2  # x and y coordinates
        masked = int(((pre_x | pre_y) & low).sum()) * 2

        # Apply masking
        if low.any():
            dfm.loc[low, [x, y, c]] = np.nan

        # Store statistics for this landmark
        per_lm[i] = {
            "frames_total": int(n_frames),
            "frames_low_conf": int(low.sum()),
            "coords_considered": considered,
            "coords_masked": masked,
            "pct_frames_low_conf": (int(low.sum()) / n_frames * 100.0) if n_frames else 0.0
        }

        total_considered += considered
        total_masked += masked

    # Overall statistics
    overall = {
        "frames": int(n_frames),
        "n_landmarks_considered": len(per_lm),
        "total_coord_values": int(total_considered),
        "total_coords_masked": int(total_masked),
        "pct_coords_masked": (total_masked / total_considered * 100.0) if total_considered else 0.0
    }

    return dfm, {"per_landmark": per_lm, "overall": overall}