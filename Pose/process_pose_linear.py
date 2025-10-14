"""
Pose Processing Pipeline - Clean, Well-Structured Implementation

This pipeline processes facial pose data through 8 clearly defined steps:
1) Load raw OpenPose CSVs (x1,y1,prob1,...,x70,y70,prob70) from RAW_DIR
2) Filter to relevant keypoints (your sets)                [RUN_FILTER]
3) Mask low-confidence (conf < CONF_THRESH) to NaN         [RUN_MASK]
4) Interpolate short gaps (≤ MAX_INTERP_RUN) + Butterworth [RUN_INTERP_FILTER]
5) Normalize to screen size (2560×1440)                   [RUN_NORM]
6) Build templates (global + per-participant)             [RUN_TEMPLATES]
7) Features:
   A) Procrustes vs global template (windowed 60s, 50% overlap)
   B) Procrustes vs participant template (same)
   C) Original (no Procrustes), same windowing           [RUN_FEATURES_*]
   - Per-metric: drop windows containing any NaNs
   - Save three CSVs: procrustes_global, procrustes_participant, original
8) Interocular scaling + linear metrics (vel, acc, RMS)    [RUN_LINEAR]
   - Save three CSVs for linear metrics corresponding to step 7 outputs

Key Features:
- Skips steps 1-5 if condition-based normalized files already exist (unless OVERWRITE=True)
- Clear separation of concerns with dedicated functions for each step
- Comprehensive error handling and progress reporting
- Preserves all original functionality while improving readability

A JSON summary is saved with:
  - config & flags,
  - per-file masking stats,
  - windows dropped (total & per metric) per route,
  - template info,
  - any errors encountered.
"""

# Imports
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import configuration and flags
from utils.config import CFG, SCIPY_AVAILABLE
from utils.config import (
    RUN_FILTER, RUN_MASK, RUN_INTERP_FILTER, RUN_NORM, RUN_TEMPLATES,
    RUN_FEATURES_PROCRUSTES_GLOBAL, RUN_FEATURES_PROCRUSTES_PARTICIPANT,
    RUN_FEATURES_ORIGINAL, RUN_LINEAR,
    SAVE_REDUCED, SAVE_MASKED, SAVE_INTERP_FILTERED, SAVE_NORM,
    SAVE_PER_FRAME_PROCRUSTES_GLOBAL, SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT,
    SAVE_PER_FRAME_ORIGINAL,
    OVERWRITE, OVERWRITE_TEMPLATES, SCALE_BY_INTEROCULAR
)

# Import utility functions
from utils.io_utils import (
    ensure_dirs, load_raw_files, save_json_summary,
    get_output_filename
)
from utils.preprocessing_utils import (
    detect_conf_prefix_case_insensitive,
    relevant_indices, filter_df_to_relevant, confidence_mask
)
from utils.signal_utils import interpolate_run_limited, butterworth_segment_filter
from utils.normalization_utils import normalize_to_screen, interocular_series
from utils.features_utils import (
    procrustes_features_for_file, original_features_for_file,
    compute_linear_from_perframe_dir
)
from utils.window_utils import window_features
from utils.io_utils import write_per_frame_metrics


# ================================================================================================
# HELPER FUNCTIONS
# ================================================================================================

def parse_pid_cond(filename: str) -> Tuple[str, str]:
    """
    Expect filenames like '3101_H.csv' or '410_L.csv'.
    Returns ('3101', 'H').
    """
    stem = Path(filename).stem
    # Allow extra underscores after condition, keep first two tokens
    parts = stem.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Filename must be 'participantID_condition(.csv)'; got: {filename}")
    pid, cond = parts[0], parts[1]
    return pid, cond

def check_steps_1_5_complete(files: List[Path]) -> bool:
    """Check if all output files from steps 1-5 already exist with condition-based names.

    Args:
        files: List of input pose CSV files
        condition_map: Mapping from participant/trial to condition

    Returns:
        True if all steps 1-5 outputs exist, False otherwise
    """
    if not SAVE_NORM:
        return False  # Can't check completion if we're not saving normalized files

    norm_dir = Path(CFG.OUT_BASE) / "norm_screen"
    if not norm_dir.exists():
        return False

    # Check if all normalized condition-based files exist
    for fp in files:
        try:
            pid, cond = parse_pid_cond(fp.name)
            out_name = get_output_filename(fp.name, pid, cond, "_norm")
            if not (norm_dir / out_name).exists():
                return False # Missing file means steps 1-5 incomplete
        except Exception:
            return False # Can't parse filename or find condition

    return True  # All files exist


def load_existing_normalized_data(files: List[Path]) -> Tuple[Dict, Dict]:
    """Load existing normalized data for steps 6-8.

    Args:
        files: List of input pose CSV files
        condition_map: Mapping from participant/trial to condition

    Returns:
        Tuple of (perfile_data, perfile_meta) dictionaries
    """
    perfile_data = {}
    perfile_meta = {}

    print("  Loading existing normalized data...")
    for fp in tqdm(files, desc="Loading normalized", unit="file"):
        pid, cond = parse_pid_cond(fp.name)
        out_name = get_output_filename(fp.name, pid, cond, "_norm")
        norm_path = Path(CFG.OUT_BASE) / "norm_screen" / out_name
        df_norm = pd.read_csv(norm_path)
        perfile_data[fp.name] = {"norm": df_norm}
        perfile_meta[fp.name] = {"participant": pid, "condition": cond}
    return perfile_data, perfile_meta


def load_existing_templates() -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load existing templates from disk for step 7-8.

    Returns:
        Tuple of (global_template, participant_templates)
    """
    template_dir = Path(CFG.OUT_BASE) / "templates"

    # Load global template
    global_template = None
    global_template_path = template_dir / "global_template.csv"
    if global_template_path.exists():
        global_template = pd.read_csv(global_template_path)
        print(f"  Loaded global template: {global_template.shape}")
    else:
        print(f"  Warning: Global template not found at {global_template_path}")

    # Load participant templates
    participant_templates = {}
    if template_dir.exists():
        for template_path in template_dir.glob("participant_*_template.csv"):
            # Extract participant ID from filename: participant_<PID>_template.csv
            pid = template_path.stem.replace("participant_", "").replace("_template", "")
            participant_templates[pid] = pd.read_csv(template_path)
        print(f"  Loaded {len(participant_templates)} participant templates")
    else:
        print(f"  Warning: Template directory not found at {template_dir}")

    return global_template, participant_templates


# ================================================================================================
# STEP 1: LOAD RAW DATA
# ================================================================================================

def step_1_load_raw_data(files: List[Path]) -> Dict[str, pd.DataFrame]:
    """Step 1: Load raw OpenPose CSV files.

    Args:
        files: List of pose CSV file paths

    Returns:
        Dictionary mapping filename to raw DataFrame
    """
    print("\n" + "="*80)
    print("STEP 1: LOAD RAW DATA")
    print("="*80)

    raw_data = {}

    for fp in tqdm(files, desc="Loading raw CSVs", unit="file"):
        try:
            df_raw = pd.read_csv(fp)
            raw_data[fp.name] = df_raw
            print(f"Loaded {fp.name}: {len(df_raw)} frames, {len(df_raw.columns)} columns")
        except Exception as e:
            print(f" Failed to load {fp.name}: {e}")
            continue

    print(f"\nStep 1 Complete: Loaded {len(raw_data)} files")
    return raw_data


# ================================================================================================
# STEP 2: FILTER TO RELEVANT KEYPOINTS
# ================================================================================================

def step_2_filter_keypoints(raw_data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
    """Step 2: Filter to relevant keypoints and detect confidence prefixes.

    Args:
        raw_data: Dictionary of raw DataFrames
        condition_map: Mapping from participant/trial to condition

    Returns:
        Tuple of (filtered_data, metadata) dictionaries
    """
    print("\n" + "="*80)
    print("STEP 2: FILTER TO RELEVANT KEYPOINTS")
    print("="*80)
    if not RUN_FILTER:
        print("RUN_FILTER=False. Cannot proceed safely.")
        sys.exit(1)

    filtered_data, metadata = {}, {}
    indices = relevant_indices()
    print(f"Filtering to {len(indices)} relevant landmarks: {indices}")

    for filename, df_raw in tqdm(raw_data.items(), desc="Filtering keypoints", unit="file"):
        try:
             # Detect confidence prefix
            conf_prefix = detect_conf_prefix_case_insensitive(list(df_raw.columns))
            df_reduced = filter_df_to_relevant(df_raw, conf_prefix, indices)
            filtered_data[filename] = df_reduced

            pid, cond = parse_pid_cond(filename)
            metadata[filename] = {
                "participant": pid,
                "condition": cond,
                "conf_prefix": conf_prefix,
                "original_columns": len(df_raw.columns),
                "filtered_columns": len(df_reduced.columns)
            }

            if SAVE_REDUCED:
                out_name = get_output_filename(filename, pid, cond, "_reduced")
                out_path = Path(CFG.OUT_BASE) / "reduced" / out_name
                if OVERWRITE or not out_path.exists():
                    df_reduced.to_csv(out_path, index=False)

            print(f"{filename}: {metadata[filename]['original_columns']} → {metadata[filename]['filtered_columns']} columns")
        except Exception as e:
            print(f"Failed to filter {filename}: {e}")
            continue

    print(f"\nStep 2 Complete: Filtered {len(filtered_data)} files")
    return filtered_data, metadata



# ================================================================================================
# STEP 3: MASK LOW-CONFIDENCE LANDMARKS
# ================================================================================================

def step_3_mask_low_confidence(filtered_data: Dict[str, pd.DataFrame],
                              metadata: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    """Step 3: Mask low-confidence landmarks to NaN.

    Args:
        filtered_data: Dictionary of filtered DataFrames
        metadata: File metadata dictionary

    Returns:
        Tuple of (masked_data, masking_stats) dictionaries
    """
    print("\n" + "="*80)
    print("STEP 3: MASK LOW-CONFIDENCE LANDMARKS")
    print("="*80)

    if not RUN_MASK:
        print("RUN_MASK=False. Cannot proceed safely.")
        sys.exit(1)

    masked_data = {}
    masking_stats = {}
    indices = relevant_indices()

    print(f"Masking landmarks with confidence < {CFG.CONF_THRESH}")

    for filename, df_filtered in tqdm(filtered_data.items(), desc="Masking low confidence", unit="file"):
        try:
            conf_prefix = metadata[filename]["conf_prefix"]

            # Apply confidence masking
            df_masked, stats = confidence_mask(df_filtered, conf_prefix, indices, CFG.CONF_THRESH)
            masked_data[filename] = df_masked
            masking_stats[filename] = stats["overall"]

            # Save masked file if requested
            if SAVE_MASKED:
                pid = metadata[filename]["participant"]
                cond = metadata[filename]["condition"]
                out_name = get_output_filename(filename, pid, cond, "_masked")
                out_path = Path(CFG.OUT_BASE) / "masked" / out_name
                if OVERWRITE or not out_path.exists():
                    df_masked.to_csv(out_path, index=False)

            pct_masked = stats["overall"]["pct_coords_masked"]
            print(f"{filename}: {pct_masked:.1f}% coordinates masked")

        except Exception as e:
            print(f"Failed to mask {filename}: {e}")
            continue

    print(f"\nStep 3 Complete: Masked {len(masked_data)} files")
    return masked_data, masking_stats


# ================================================================================================
# STEP 4: INTERPOLATE AND FILTER
# ================================================================================================

def step_4_interpolate_filter(masked_data: Dict[str, pd.DataFrame],
                             metadata: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    """Step 4: Interpolate short gaps and apply Butterworth filter.

    Args:
        masked_data: Dictionary of masked DataFrames
        metadata: File metadata dictionary

    Returns:
        Dictionary of interpolated and filtered DataFrames
    """
    print("\n" + "="*80)
    print("STEP 4: INTERPOLATE SHORT GAPS AND BUTTERWORTH FILTER")
    print("="*80)

    if not RUN_INTERP_FILTER:
        print("RUN_INTERP_FILTER=False. Cannot proceed safely.")
        sys.exit(1)

    if not SCIPY_AVAILABLE:
        print("SciPy is required for RUN_INTERP_FILTER. Install scipy or disable this step.")
        sys.exit(1)

    interp_filtered_data = {}

    print(f"Interpolating gaps ≤ {CFG.MAX_INTERP_RUN} frames, then Butterworth filter")
    print(f"Filter: Order {CFG.FILTER_ORDER}, Cutoff {CFG.CUTOFF_HZ} Hz, Sampling {CFG.FPS} Hz")

    for filename, df_masked in tqdm(masked_data.items(), desc="Interpolate/Filter", unit="file"):
        try:
            df_processed = df_masked.copy()

            # Process each coordinate column
            coord_cols = [col for col in df_processed.columns
                         if col.lower().startswith(('x', 'y'))]

            for col in coord_cols:
                # Interpolate short NaN runs
                df_processed[col] = interpolate_run_limited(df_processed[col], CFG.MAX_INTERP_RUN)

                # Apply Butterworth filter to contiguous segments
                df_processed[col] = butterworth_segment_filter(
                    df_processed[col], CFG.FILTER_ORDER, CFG.CUTOFF_HZ, CFG.FPS
                )

            interp_filtered_data[filename] = df_processed

            # Save processed file if requested
            if SAVE_INTERP_FILTERED:
                pid = metadata[filename]["participant"]
                cond = metadata[filename]["condition"]
                out_name = get_output_filename(filename, pid, cond, "_interp_filt")
                out_path = Path(CFG.OUT_BASE) / "interp_filtered" / out_name
                if OVERWRITE or not out_path.exists():
                    df_processed.to_csv(out_path, index=False)

            print(f"{filename}: Processed {len(coord_cols)} coordinate columns")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue

    print(f"\nStep 4 Complete: Processed {len(interp_filtered_data)} files")
    return interp_filtered_data


# ================================================================================================
# STEP 5: NORMALIZE TO SCREEN COORDINATES
# ================================================================================================

def step_5_normalize_coordinates(interp_filtered_data: Dict[str, pd.DataFrame],
                                metadata: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    """Step 5: Normalize coordinates to screen size (0-1 range).

    Args:
        interp_filtered_data: Dictionary of processed DataFrames
        metadata: File metadata dictionary

    Returns:
        Dictionary of normalized DataFrames
    """
    print("\n" + "="*80)
    print("STEP 5: NORMALIZE TO SCREEN COORDINATES")
    print("="*80)

    if not RUN_NORM:
        print("RUN_NORM=False. Templates and features require normalized coordinates.")
        sys.exit(1)

    normalized_data = {}

    print(f"Normalizing to screen size: {CFG.IMG_WIDTH} × {CFG.IMG_HEIGHT}")

    for filename, df_processed in tqdm(interp_filtered_data.items(), desc="Normalizing", unit="file"):
        try:
            # Normalize coordinates
            df_norm = normalize_to_screen(df_processed, CFG.IMG_WIDTH, CFG.IMG_HEIGHT)
            normalized_data[filename] = df_norm

            # Save normalized file if requested
            if SAVE_NORM:
                pid = metadata[filename]["participant"]
                cond = metadata[filename]["condition"]
                out_name = get_output_filename(filename, pid, cond, "_norm")
                out_path = Path(CFG.OUT_BASE) / "norm_screen" / out_name
                if OVERWRITE or not out_path.exists():
                    df_norm.to_csv(out_path, index=False)

            print(f"{filename}: Normalized to [0,1] range")

        except Exception as e:
            print(f"Failed to normalize {filename}: {e}")
            continue

    print(f"\nStep 5 Complete: Normalized {len(normalized_data)} files")
    return normalized_data


# ================================================================================================
# STEP 6: BUILD TEMPLATES (GLOBAL + PER-PARTICIPANT)
# ================================================================================================

def step_6_build_templates(normalized_data: Dict[str, pd.DataFrame],
                          metadata: Dict[str, Dict]) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Step 6: Build global and per-participant templates.

    Args:
        normalized_data: Dictionary of normalized DataFrames
        metadata: File metadata dictionary

    Returns:
        Tuple of (global_template, participant_templates)
    """
    print("\n" + "="*80)
    print("STEP 6: BUILD TEMPLATES (GLOBAL + PER-PARTICIPANT)")
    print("="*80)

    if not RUN_TEMPLATES:
        print("RUN_TEMPLATES=False. Skipping template generation.")
        return None, {}

    def compute_template_across_files(file_names: List[str]) -> pd.DataFrame:
        """Average template builder - computes mean coordinates across all frames from specified files."""
        if not file_names:
            return pd.DataFrame()

        # Reference column order from first file
        cols = normalized_data[file_names[0]].columns

        # Collect normalized coordinates from all specified files
        accum = [normalized_data[name][cols].astype(float) for name in file_names]

        # Stack all frames across files
        big = pd.concat(accum, axis=0, ignore_index=True)

        # Find x and y columns
        x_cols = [c for c in cols if c.lower().startswith("x")]
        y_cols = [c for c in cols if c.lower().startswith("y")]

        # Create single-row template frame
        templ = pd.DataFrame(index=[0], columns=x_cols + y_cols, dtype=float)

        # Mean x and y per landmark across all frames/files
        templ[x_cols] = big[x_cols].mean(axis=0, skipna=True).values
        templ[y_cols] = big[y_cols].mean(axis=0, skipna=True).values

        return templ

    global_template = None
    participant_templates = {}
    template_dir = Path(CFG.OUT_BASE) / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    # Group files by participant for participant-specific templates
    part_to_files = {}
    for filename in normalized_data.keys():
        pid = metadata[filename]["participant"]
        part_to_files.setdefault(pid, []).append(filename)

    # Build global template
    print("Building global template from all participants...")
    global_template_path = template_dir / "global_template.csv"

    if global_template_path.exists() and not OVERWRITE_TEMPLATES:
        print("Loading existing global template")
        global_template = pd.read_csv(global_template_path)
    else:
        print("Computing new global template...")
        try:
            all_filenames = list(normalized_data.keys())
            global_template = compute_template_across_files(all_filenames)
            global_template.to_csv(global_template_path, index=False)
            print(f"Global template saved: {global_template.shape}")
        except Exception as e:
            print(f"Error building global template: {e}")

    # Build per-participant templates
    print("Building per-participant templates...")
    for pid, filenames in tqdm(part_to_files.items(), desc="Participant templates"):
        template_path = template_dir / f"participant_{pid}_template.csv"

        if template_path.exists() and not OVERWRITE_TEMPLATES:
            participant_templates[pid] = pd.read_csv(template_path)
        else:
            try:
                template = compute_template_across_files(filenames)
                template.to_csv(template_path, index=False)
                participant_templates[pid] = template
            except Exception as e:
                print(f"Error building template for {pid}: {e}")

    print(f"\nStep 6 Complete: Built {len(participant_templates)} participant templates")
    return global_template, participant_templates


# ================================================================================================
# STEP 7: EXTRACT FEATURES (PROCRUSTES + ORIGINAL)
# ================================================================================================

def step_7_extract_features(normalized_data: Dict[str, pd.DataFrame],
                           metadata: Dict[str, Dict],
                           global_template: Optional[pd.DataFrame],
                           participant_templates: Dict[str, pd.DataFrame]) -> None:
    """Step 7: Extract windowed features using different normalization approaches.

    Args:
        normalized_data: Dictionary of normalized DataFrames
        metadata: File metadata dictionary
        global_template: Global Procrustes template (if available)
        participant_templates: Per-participant templates
    """
    print("\n" + "="*80)
    print("STEP 7: EXTRACT WINDOWED FEATURES")
    print("="*80)

    print(f"Window size: {CFG.WINDOW_SECONDS}s, Overlap: {CFG.WINDOW_OVERLAP*100}%")

    # Window parameters
    win = CFG.WINDOW_SECONDS * CFG.FPS  # samples per window
    hop = int(win * (1.0 - CFG.WINDOW_OVERLAP)) or max(1, win // 2)  # window hop
    rel_idxs = relevant_indices()  # relevant landmark indices

    # Feature storage
    feat_dir = Path(CFG.OUT_BASE) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Row buffers for aggregated window features
    procrustes_global_rows = []
    procrustes_part_rows = []
    original_rows = []

    # Drop counters
    procrustes_global_drops_agg = {}
    procrustes_part_drops_agg = {}
    original_drops_agg = {}

    # Process each file
    for filename in tqdm(normalized_data.keys(), desc="Computing features", unit="file"):
        print(f"\nProcessing {filename}...")
        pid, trial = metadata[filename]["participant"], metadata[filename]["condition"]
        df_norm = normalized_data[filename]

        # A) Procrustes features vs global template
        if RUN_FEATURES_PROCRUSTES_GLOBAL and global_template is not None:
            try:
                feats = procrustes_features_for_file(df_norm, global_template, rel_idxs)
                io = interocular_series(df_norm, metadata[filename].get("conf_prefix")).values
                n_frames = len(io)
                if SAVE_PER_FRAME_PROCRUSTES_GLOBAL:
                    write_per_frame_metrics(feat_dir, "procrustes_global", pid, trial, feats, io, n_frames)
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "participant", pid)
                dfw.insert(0, "source", "procrustes_global")
                procrustes_global_rows.append(dfw)

                for k, v in drops.items():
                    procrustes_global_drops_agg[k] = procrustes_global_drops_agg.get(k, 0) + v

            except Exception as e:
                print(f"Error processing {filename} (global): {e}")

            # B) Procrustes features vs participant template
        if RUN_FEATURES_PROCRUSTES_PARTICIPANT and pid in participant_templates:
            try:
                template = participant_templates[pid]
                feats = procrustes_features_for_file(df_norm, template, rel_idxs)
                io = interocular_series(df_norm, metadata[filename].get("conf_prefix")).values
                n_frames = len(io)

                if SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT:
                    write_per_frame_metrics(feat_dir, "procrustes_participant", pid, trial, feats, io, n_frames)
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "participant", pid)
                dfw.insert(0, "source", "procrustes_participant")
                procrustes_part_rows.append(dfw)

                for k, v in drops.items():
                    procrustes_part_drops_agg[k] = procrustes_part_drops_agg.get(k, 0) + v

            except Exception as e:
                print(f"Error processing {filename} (participant): {e}")

        # C) Original features (no Procrustes)
        if RUN_FEATURES_ORIGINAL:
            try:
                feats = original_features_for_file(df_norm)
                io = interocular_series(df_norm, metadata[filename].get("conf_prefix")).values
                n_frames = len(io)

                if SAVE_PER_FRAME_ORIGINAL:
                    write_per_frame_metrics(feat_dir, "original", pid, trial, feats, io, n_frames)

                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "participant", pid)
                dfw.insert(0, "source", "original")
                original_rows.append(dfw)

                for k, v in drops.items():
                    original_drops_agg[k] = original_drops_agg.get(k, 0) + v

            except Exception as e:
                print(f"Error processing {filename} (original): {e}")

    # Save aggregated windowed features
    if procrustes_global_rows:
        global_df = pd.concat(procrustes_global_rows, ignore_index=True)
        global_path = feat_dir / "procrustes_global.csv"
        global_df.to_csv(global_path, index=False)
        print(f"Saved global Procrustes features: {len(global_df)} windows")

    if procrustes_part_rows:
        part_df = pd.concat(procrustes_part_rows, ignore_index=True)
        part_path = feat_dir / "procrustes_participant.csv"
        part_df.to_csv(part_path, index=False)
        print(f"Saved participant Procrustes features: {len(part_df)} windows")

    if original_rows:
        orig_df = pd.concat(original_rows, ignore_index=True)
        orig_path = feat_dir / "original.csv"
        orig_df.to_csv(orig_path, index=False)
        print(f"Saved original features: {len(orig_df)} windows")

    print("\nStep 7 Complete: Feature extraction finished")


# ================================================================================================
# STEP 8: COMPUTE LINEAR METRICS
# ================================================================================================

def step_8_compute_linear_metrics() -> None:
    """Step 8: Compute linear metrics (velocity, acceleration, RMS) from per-frame data."""

    print("\n" + "="*80)
    print("STEP 8: COMPUTE LINEAR METRICS")
    print("="*80)

    if not RUN_LINEAR:
        print("RUN_LINEAR=False. Skipping linear metrics computation.")
        return

    feat_dir = Path(CFG.OUT_BASE) / "features"
    lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"
    lm_dir.mkdir(parents=True, exist_ok=True)

    # Process each feature type
    feature_types = []
    if RUN_FEATURES_PROCRUSTES_GLOBAL:
        feature_types.append("procrustes_global")
    if RUN_FEATURES_PROCRUSTES_PARTICIPANT:
        feature_types.append("procrustes_participant")
    if RUN_FEATURES_ORIGINAL:
        feature_types.append("original")

    for feature_type in feature_types:
        print(f"\n8) Computing linear metrics for {feature_type}...")

        try:
            per_frame_dir = feat_dir / "per_frame" / feature_type
            csv_files = [f for f in per_frame_dir.glob("*.csv") if not f.name.startswith("all_")]

            if per_frame_dir.exists() and csv_files:
                out_path = lm_dir / f"{feature_type}_linear.csv"
                compute_linear_from_perframe_dir(
                    per_frame_dir, out_path, CFG.FPS, CFG.WINDOW_SECONDS,
                    CFG.WINDOW_OVERLAP,scale_by_interocular=(feature_type == "original")
                )
                print(f"Linear metrics computed for {feature_type}: {out_path}")
            else:
                print(f"Per-frame directory not found or empty (after filtering): {per_frame_dir}")

        except Exception as e:
            print(f"Error computing linear metrics for {feature_type}: {e}")

    print("\nStep 8 Complete: Linear metrics computation finished")


# ================================================================================================
# MAIN PIPELINE FUNCTION
# ================================================================================================
def run_pose_processing_pipeline(start_step: int = 1) -> None:
    """Main pipeline function that orchestrates all 8 steps without participant CSVs.

    Args:
        start_step: Step number to start from (1-8). Previous steps will be skipped.
    """

    print("="*80)
    print("POSE PROCESSING PIPELINE")
    print("="*80)

    # Print configuration
    print("Configuration:")
    print(f"  RAW_DIR: {CFG.RAW_DIR}")
    print(f"  OUT_BASE: {CFG.OUT_BASE}")
    print(f"  CONF_THRESH: {CFG.CONF_THRESH}")
    print(f"  OVERWRITE: {OVERWRITE}")
    print(f"  START_STEP: {start_step}")

    # Ensure output directories exist
    ensure_dirs()

    # Load file list (expects names like '3101_H.csv')
    files = load_raw_files()
    print(f"✓ Found {len(files)} pose CSV files")

    # Initialize variables that may be needed later
    normalized_data = None
    perfile_meta = None
    global_template = None
    participant_templates = {}

    # ============================================================================
    # STEPS 1-5: Data preprocessing
    # ============================================================================
    if start_step <= 5:
        # Steps 1–5: either reuse existing normalized outputs or (re)compute them
        if not OVERWRITE and start_step == 1 and check_steps_1_5_complete(files):
            print(f"\nSteps 1-5 already complete (found all {len(files)} normalized condition-based files)")
            print("Loading existing data and proceeding to steps 6-8...")

            # Load existing normalized data + metadata (participant, condition parsed from filename)
            perfile_data, perfile_meta = load_existing_normalized_data(files)
            normalized_data = {fname: dct["norm"] for fname, dct in perfile_data.items()}

        else:
            if OVERWRITE:
                print("\nOVERWRITE=True: Running steps 1-5 regardless of existing files")
            else:
                print(f"\nRunning steps {start_step}-5...")

            # 1) Load raw data
            if start_step <= 1:
                raw_data = step_1_load_raw_data(files)
                if not raw_data:
                    print("No data loaded. Exiting.")
                    return
            else:
                print(f"\n[SKIP] Step 1: Starting from step {start_step}")
                raw_data = {}

            # 2) Filter to relevant keypoints
            if start_step <= 2:
                filtered_data, perfile_meta = step_2_filter_keypoints(raw_data)
                if not filtered_data:
                    print("No data after filtering. Exiting.")
                    return
            else:
                print(f"\n[SKIP] Step 2: Starting from step {start_step}")
                filtered_data = {}

            # 3) Mask low-confidence landmarks
            if start_step <= 3:
                masked_data, _ = step_3_mask_low_confidence(filtered_data, perfile_meta)
                if not masked_data:
                    print("No data after masking. Exiting.")
                    return
            else:
                print(f"\n[SKIP] Step 3: Starting from step {start_step}")
                masked_data = {}

            # 4) Interpolate short gaps + Butterworth filter
            if start_step <= 4:
                interp_filtered_data = step_4_interpolate_filter(masked_data, perfile_meta)
                if not interp_filtered_data:
                    print("No data after interpolation/filtering. Exiting.")
                    return
            else:
                print(f"\n[SKIP] Step 4: Starting from step {start_step}")
                interp_filtered_data = {}

            # 5) Normalize to screen coordinates
            if start_step <= 5:
                normalized_data = step_5_normalize_coordinates(interp_filtered_data, perfile_meta)
                if not normalized_data:
                    print("No data after normalization. Exiting.")
                    return
    else:
        # Starting from step 6 or later - load existing normalized data
        print(f"\n[SKIP] Steps 1-5: Starting from step {start_step}")
        print("Loading existing normalized data from disk...")
        perfile_data, perfile_meta = load_existing_normalized_data(files)
        normalized_data = {fname: dct["norm"] for fname, dct in perfile_data.items()}

    # ============================================================================
    # STEP 6: Build templates
    # ============================================================================
    if start_step <= 6:
        global_template, participant_templates = step_6_build_templates(normalized_data, perfile_meta)
    else:
        # Starting from step 7 or 8 - load existing templates
        print(f"\n[SKIP] Step 6: Starting from step {start_step}")
        print("Loading existing templates from disk...")
        global_template, participant_templates = load_existing_templates()

    # ============================================================================
    # STEP 7: Extract features
    # ============================================================================
    if start_step <= 7:
        step_7_extract_features(normalized_data, perfile_meta, global_template, participant_templates)
    else:
        print(f"\n[SKIP] Step 7: Starting from step {start_step}")

    # ============================================================================
    # STEP 8: Compute linear metrics
    # ============================================================================
    if start_step <= 8:
        step_8_compute_linear_metrics()
    else:
        print(f"\n[SKIP] Step 8: Invalid start_step {start_step}")

    # Save processing summary
    summary = {
        "config": {k: v for k, v in CFG.__dict__.items() if not k.startswith('_')},
        "flags": {
            "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK,
            "RUN_INTERP_FILTER": RUN_INTERP_FILTER, "RUN_NORM": RUN_NORM,
            "RUN_TEMPLATES": RUN_TEMPLATES,
            "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
            "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
            "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL,
            "RUN_LINEAR": RUN_LINEAR,
            "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
            "START_STEP": start_step
        },
        "files_processed": len(files),
        "participants": len({meta["participant"] for meta in perfile_meta.values()}) if perfile_meta else 0,
        "conditions": len({meta["condition"] for meta in perfile_meta.values()}) if perfile_meta else 0
    }

    summary_path = Path(CFG.OUT_BASE) / "processing_summary.json"
    save_json_summary(summary_path, summary)

    print("\n" + "="*80)
    print("POSE PROCESSING PIPELINE COMPLETE!")
    print("="*80)
    print(f"Processing summary saved: {summary_path}")


# ================================================================================================
# COMMAND LINE INTERFACE
# ================================================================================================

if __name__ == "__main__":
    """Command line interface for the pose processing pipeline."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Pose Processing Pipeline - Clean 8-step implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1) Load raw OpenPose CSVs
  2) Filter to relevant keypoints
  3) Mask low-confidence landmarks
  4) Interpolate gaps + Butterworth filter
  5) Normalize to screen coordinates
  6) Build templates (global + per-participant)
  7) Extract windowed features (Procrustes + original)
  8) Compute linear metrics (velocity, acceleration, RMS)

The pipeline automatically skips steps 1-5 if condition-based normalized
files already exist (unless --overwrite is specified).

Use --start-step to skip earlier steps:
  --start-step 6  : Skip steps 1-5, load normalized data from disk
  --start-step 7  : Skip steps 1-6, load normalized data + templates from disk
  --start-step 8  : Skip steps 1-7, only compute linear metrics from per-frame data

Examples:
  python process_pose_linear.py                  # Run all steps (auto-skip if data exists)
  python process_pose_linear.py --start-step 8   # Only recompute linear metrics
  python process_pose_linear.py --overwrite      # Force reprocess steps 1-5
        """
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force reprocessing of steps 1-5 even if outputs exist"
    )

    parser.add_argument(
        "--start-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=1,
        help="Start pipeline from specified step (1-8). Steps before this will be skipped and data loaded from disk where needed."
    )

    args = parser.parse_args()

    # Apply command line overrides
    if args.overwrite:
        import utils.config as config_module
        config_module.OVERWRITE = True
        globals()["OVERWRITE"] = True

    # Run the pipeline
    run_pose_processing_pipeline(start_step=args.start_step)