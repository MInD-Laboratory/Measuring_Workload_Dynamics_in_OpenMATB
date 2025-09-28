from __future__ import annotations
from dataclasses import dataclass
import os

# Toggle SciPy availability in one place
try:
    from scipy.signal import butter, filtfilt  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

@dataclass
class Config:
    RAW_DIR: str = os.getenv("POSE_RAW_DIR", "data/raw/experimental")
    OUT_BASE: str = os.getenv("POSE_OUT_BASE", "data/processed/experimental")
    FPS: int = 60
    IMG_WIDTH: int = 2560
    IMG_HEIGHT: int = 1440
    CONF_THRESH: float = 0.30
    MAX_INTERP_RUN: int = 60
    FILTER_ORDER: int = 4
    CUTOFF_HZ: float = 10.0
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5
    PROCRUSTES_REF: tuple[int, ...] = (30, 31, 37, 46)
    BLINK_L_TOP: tuple[int, ...] = (38, 39)
    BLINK_L_BOT: tuple[int, ...] = (41, 42)
    BLINK_R_TOP: tuple[int, ...] = (44, 45)
    BLINK_R_BOT: tuple[int, ...] = (47, 48)
    HEAD_ROT: tuple[int, int] = (37, 46)
    MOUTH: tuple[int, int] = (63, 67)
    CENTER_FACE: tuple[int, ...] = tuple(range(28, 36 + 1))  # 28..36

CFG = Config()

# ---------------------- FLAGS -------------------------------------------------
RUN_FILTER          = True
RUN_MASK            = True
RUN_INTERP_FILTER   = True
RUN_NORM            = True
RUN_TEMPLATES       = True
RUN_FEATURES_PROCRUSTES_GLOBAL      = True
RUN_FEATURES_PROCRUSTES_PARTICIPANT = True
RUN_FEATURES_ORIGINAL               = True
RUN_LINEAR          = True
SCALE_BY_INTEROCULAR = True

SAVE_REDUCED            = True
SAVE_MASKED             = True
SAVE_INTERP_FILTERED    = True
SAVE_NORM               = True

SAVE_PER_FRAME_PROCRUSTES_GLOBAL      = True
SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT = True
SAVE_PER_FRAME_ORIGINAL               = True

OVERWRITE               = True
OVERWRITE_TEMPLATES     = False
