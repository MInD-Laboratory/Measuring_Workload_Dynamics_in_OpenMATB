"""I/O utilities for reading and writing pose analysis data.

This module handles file operations including reading CSVs, writing results,
and managing directory structures. Data processing logic has been moved to
preprocessing_utils.py for better separation of concerns.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import sys
import json
import numpy as np
import pandas as pd

from .config import CFG



# ---------- Directory and file management ------------------------------------
def ensure_dirs() -> None:
    """Create all necessary output directories if they don't exist."""
    base = Path(CFG.OUT_BASE)
    for d in ["reduced", "masked", "interp_filtered", "norm_screen", "templates", "features", "linear_metrics"]:
        (base / d).mkdir(parents=True, exist_ok=True)


def list_csvs(dir_path: str) -> List[Path]:
    """List all CSV files in a directory.

    Args:
        dir_path: Directory path to search

    Returns:
        Sorted list of CSV file paths, empty list if directory doesn't exist
    """
    p = Path(dir_path)
    return sorted([f for f in p.glob("*.csv")]) if p.exists() else []


def load_raw_files() -> List[Path]:
    """Load list of raw pose CSV files from configured directory.

    Filters out participant info file and returns only pose data files.

    Returns:
        List of pose CSV file paths

    Raises:
        SystemExit: If no pose CSV files found in RAW_DIR
    """
    p = Path(CFG.RAW_DIR)
    # Get all CSV files except the participant info file
    files = sorted([f for f in p.glob("*.csv")
                   if f.exists() and f.name != CFG.PARTICIPANT_INFO_FILE])

    if not files:
        print(f"No pose CSV files found in RAW_DIR: {CFG.RAW_DIR}")
        sys.exit(1)
    return files


# ---------- Filename transformation utilities --------------------------------
def get_output_filename(input_filename: str, participant: str, condition: str, suffix: str = "") -> str:
    """Generate output filename with condition instead of trial number.

    Transforms filenames from '3101_02_pose.csv' format to '3101_M_pose.csv' format.

    Args:
        input_filename: Original input filename
        participant: Participant ID
        condition: Condition letter (L, M, or H)
        suffix: Optional suffix to add before .csv (e.g., '_reduced')

    Returns:
        Output filename with condition replacing trial number

    Examples:
        >>> get_output_filename('3101_02_pose.csv', '3101', 'M', '_reduced')
        '3101_M_reduced.csv'
    """
    # Use participant and condition to build new filename
    return f"{participant}_{condition}{suffix}.csv"


def load_participant_info_file() -> Path:
    """Locate the participant info file.

    Uses the filename from CFG.PARTICIPANT_INFO_FILE configuration.

    Returns:
        Path to participant info file

    Raises:
        FileNotFoundError: If participant info file not found
    """
    # Try in RAW_DIR first
    raw_dir_path = Path(CFG.RAW_DIR) / CFG.PARTICIPANT_INFO_FILE
    if raw_dir_path.exists():
        return raw_dir_path

    # Try parent directory
    parent_path = Path(CFG.RAW_DIR).parent / CFG.PARTICIPANT_INFO_FILE
    if parent_path.exists():
        return parent_path

    raise FileNotFoundError(
        f"{CFG.PARTICIPANT_INFO_FILE} not found in {CFG.RAW_DIR} or parent directory"
    )


# ---------- File writing operations -------------------------------------------
def write_per_frame_metrics(out_root: Path, source: str, participant: str, condition: str,
                            perframe: Dict[str, np.ndarray], interocular: np.ndarray, n_frames: int) -> None:
    """Write per-frame metrics for a single trial.
    
    Writes individual trial CSV: <out_root>/per_frame/<source>/<participant>_<condition>_perframe.csv
    
    Args:
        out_root: Root output directory
        source: Source identifier (e.g., 'procrustes_global')
        participant: Participant ID
        condition: Experimental condition
        perframe: Dictionary of metric_name -> array of per-frame values
        interocular: Array of inter-ocular distances
        n_frames: Number of frames
    """
    out_dir = out_root / "per_frame" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build per-trial DataFrame
    df_pf = pd.DataFrame({
        "participant": participant,
        "condition": condition,
        "frame": np.arange(n_frames, dtype=int),
        "interocular": interocular
    })
    
    for k, arr in perframe.items():
        df_pf[k] = arr
    
    # Write individual trial CSV
    out_path = out_dir / f"{participant}_{condition}_perframe.csv"
    df_pf.to_csv(out_path, index=False)
    
# ---------- File writing operations -------------------------------------------
# def write_per_frame_metrics(out_root: Path, source: str, participant: str, condition: str,
#                             perframe: Dict[str, np.ndarray], interocular: np.ndarray, n_frames: int) -> None:
#     """Write per-frame metrics for a single trial and update combined CSV.

#     Writes two files:
#     - Individual trial CSV: <out_root>/per_frame/<source>/<participant>_<condition>_perframe.csv
#     - Combined CSV (append): <out_root>/per_frame/<source>/all_perframe.csv

#     Args:
#         out_root: Root output directory
#         source: Source identifier (e.g., 'procrustes_global')
#         participant: Participant ID
#         condition: Experimental condition
#         perframe: Dictionary of metric_name -> array of per-frame values
#         interocular: Array of inter-ocular distances
#         n_frames: Number of frames
#     """
#     out_dir = out_root / "per_frame" / source
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Build per-trial DataFrame
#     df_pf = pd.DataFrame({
#         "participant": participant,
#         "condition": condition,
#         "frame": np.arange(n_frames, dtype=int),
#         "interocular": interocular
#     })
#     for k, arr in perframe.items():
#         df_pf[k] = arr

#     # Write individual trial CSV
#     out_path = out_dir / f"{participant}_{condition}_perframe.csv"
#     df_pf.to_csv(out_path, index=False)

#     # Append to (or create) combined CSV for this source
#     #combined_path = out_dir / "all_perframe.csv"

#     # If combined doesn't exist, create it
#     #if not combined_path.exists():
#     #    df_pf.to_csv(combined_path, index=False)
#     #    return

#     # Combined exists -> append with column alignment
#     try:
#         # Read just the header to check columns
#         existing_cols = list(pd.read_csv(combined_path, nrows=0).columns)

#         # Get union of columns (to handle both missing and new columns)
#         all_cols = list(existing_cols)
#         new_cols = [c for c in df_pf.columns if c not in existing_cols]

#         if new_cols:
#             # New columns detected - need to rewrite entire file
#             import warnings
#             warnings.warn(f"New columns detected in per-frame data: {new_cols}. Rewriting combined file.")

#             # Read existing data and add new columns
#             df_existing = pd.read_csv(combined_path)
#             all_cols.extend(new_cols)

#             # Align both dataframes to same columns
#             df_existing = df_existing.reindex(columns=all_cols, fill_value=np.nan)
#             df_pf = df_pf.reindex(columns=all_cols, fill_value=np.nan)

#             # Combine and save
#             df_combined = pd.concat([df_existing, df_pf], ignore_index=True)
#             df_combined.to_csv(combined_path, index=False)
#         else:
#             # No new columns - simple append with column alignment
#             df_pf = df_pf.reindex(columns=existing_cols, fill_value=np.nan)
#             df_pf.to_csv(combined_path, mode="a", header=False, index=False)

#     except Exception as e:
#         # If anything goes wrong, at least save the individual file
#         import warnings
#         warnings.warn(f"Failed to update combined CSV: {e}. Individual file was saved successfully.")


def save_json_summary(path: Path, payload: dict) -> None:
    """Save a dictionary as formatted JSON file.

    Args:
        path: Output file path
        payload: Dictionary to save as JSON
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)