"""Configuration module for pose estimation and analysis pipeline.

Provides centralized configuration management for all processing parameters,
including directories, filtering settings, face landmark indices, and processing flags.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

# Check SciPy availability for optional filtering operations
# SciPy is used for Butterworth filtering and signal processing
try:
    from scipy.signal import butter, filtfilt  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:
    # SciPy not available - filtering operations will be skipped
    SCIPY_AVAILABLE = False

@dataclass
class Config:
    """Main configuration class containing all processing parameters.

    Attributes:
        RAW_DIR: Directory containing raw input data (absolute path, can be set via POSE_RAW_DIR env var)
        OUT_BASE: Base directory for processed output (absolute path, can be set via POSE_OUT_BASE env var)
        PARTICIPANT_INFO_FILE: Filename of participant info CSV (can be set via PARTICIPANT_INFO_FILE env var)
        FPS: Frames per second of input video data
        IMG_WIDTH: Width of input images in pixels
        IMG_HEIGHT: Height of input images in pixels
        CONF_THRESH: Minimum confidence threshold for landmark detection (0-1)
        MAX_INTERP_RUN: Maximum number of consecutive frames to interpolate
        FILTER_ORDER: Order of Butterworth filter
        CUTOFF_HZ: Cutoff frequency in Hz for low-pass filter
        WINDOW_SECONDS: Size of sliding window in seconds for feature extraction
        WINDOW_OVERLAP: Overlap fraction between consecutive windows (0-1)
        PROCRUSTES_REF: Landmark indices for Procrustes alignment reference points
        BLINK_L_TOP: Landmark indices for left eye upper eyelid
        BLINK_L_BOT: Landmark indices for left eye lower eyelid
        BLINK_R_TOP: Landmark indices for right eye upper eyelid
        BLINK_R_BOT: Landmark indices for right eye lower eyelid
        HEAD_ROT: Landmark indices for calculating head rotation (left/right eye corners)
        MOUTH: Landmark indices for mouth corners
        CENTER_FACE: Landmark indices for central face region (nose bridge area)
    """
    # Directory paths - can be overridden by environment variables
    # Use paths relative to the Pose directory (where this config file is located)
    _BASE_DIR: str = str(Path(__file__).parent.parent)  # Pose directory
    RAW_DIR: str = os.getenv("POSE_RAW_DIR", str(Path(_BASE_DIR) / "data" / "raw_data" / "baseline_pose")) # set to either "experimental_pose" or "baseline_pose"
    OUT_BASE: str = os.getenv("POSE_OUT_BASE", str(Path(_BASE_DIR) / "data" / "processed_data" / "baseline")) # set to either "experimental" or "baseline"

    # Participant info file - can be overridden by environment variable
    PARTICIPANT_INFO_FILE: str = os.getenv("PARTICIPANT_INFO_FILE", "participant_info.csv")

    # Video/image parameters
    FPS: int = 60  # Sampling rate of video capture
    IMG_WIDTH: int = 2560  # Image width in pixels
    IMG_HEIGHT: int = 1440  # Image height in pixels

    # Detection and filtering parameters
    CONF_THRESH: float = 0.30  # Minimum confidence for valid landmark
    MAX_INTERP_RUN: int = 60  # Max consecutive frames to interpolate (1 second at 60fps)
    FILTER_ORDER: int = 4  # Butterworth filter order
    CUTOFF_HZ: float = 10.0  # Low-pass filter cutoff frequency

    # Windowing parameters for feature extraction
    WINDOW_SECONDS: int = 60  # Window size in seconds
    WINDOW_OVERLAP: float = 0.5  # 50% overlap between windows

    # Facial landmark indices for specific features
    # Based on MediaPipe face mesh topology
    PROCRUSTES_REF: tuple[int, ...] = (30, 31, 37, 46)  # Stable points for alignment
    BLINK_L_TOP: tuple[int, ...] = (38, 39)  # Left eye upper lid
    BLINK_L_BOT: tuple[int, ...] = (41, 42)  # Left eye lower lid
    BLINK_R_TOP: tuple[int, ...] = (44, 45)  # Right eye upper lid
    BLINK_R_BOT: tuple[int, ...] = (47, 48)  # Right eye lower lid
    HEAD_ROT: tuple[int, int] = (37, 46)  # Eye corners for head rotation
    MOUTH: tuple[int, int] = (63, 67)  # Mouth corners
    CENTER_FACE: tuple[int, ...] = tuple(range(28, 36 + 1))  # Nose bridge region (28-36)

# Global configuration instance
CFG = Config()

# ---------------------- PROCESSING FLAGS -------------------------------------
# These flags control which processing steps are executed in the pipeline

# Core processing steps
RUN_FILTER          = True  # Apply Butterworth low-pass filter to smooth signals
RUN_MASK            = True  # Mask low-confidence landmarks
RUN_INTERP_FILTER   = True  # Interpolate and filter masked regions
RUN_NORM            = True  # Normalize coordinates (centering and scaling)
RUN_TEMPLATES       = True  # Generate participant-specific templates
RUN_LINEAR          = True  # Run linear regression analysis

# Feature extraction for different normalization methods
RUN_FEATURES_PROCRUSTES_GLOBAL      = True  # Extract features using global Procrustes
RUN_FEATURES_PROCRUSTES_PARTICIPANT = True  # Extract features using participant-specific Procrustes
RUN_FEATURES_ORIGINAL               = True  # Extract features from original coordinates

# Normalization options
SCALE_BY_INTEROCULAR = True  # Scale by inter-ocular distance (eye corner distance)

# ---------------------- OUTPUT FLAGS ------------------------------------------
# Control what intermediate data is saved to disk

# Save intermediate processing stages
SAVE_REDUCED            = True  # Save reduced landmark set
SAVE_MASKED             = True  # Save after confidence masking
SAVE_INTERP_FILTERED    = True  # Save after interpolation and filtering
SAVE_NORM               = True  # Save normalized coordinates

# Save per-frame features for different normalizations
SAVE_PER_FRAME_PROCRUSTES_GLOBAL      = True  # Save frame-level features (global)
SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT = True  # Save frame-level features (participant)
SAVE_PER_FRAME_ORIGINAL               = True  # Save frame-level features (original)

# File overwrite behavior
OVERWRITE               = False   # Overwrite existing processed files
OVERWRITE_TEMPLATES     = False  # Preserve existing templates (don't regenerate)
