from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import sys
import json
import numpy as np
import pandas as pd

from .config import CFG, OVERWRITE

# ---------- Files / paths ----------------------------------------------------
def ensure_dirs() -> None:
    base = Path(CFG.OUT_BASE)
    for d in ["reduced","masked","interp_filtered","norm_screen","templates","features","linear_metrics"]:
        (base / d).mkdir(parents=True, exist_ok=True)

def list_csvs(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    return sorted([f for f in p.glob("*.csv")]) if p.exists() else []

def load_raw_files() -> List[Path]:
    files = list_csvs(CFG.RAW_DIR)
    if not files:
        print(f"No CSV files found in RAW_DIR: {CFG.RAW_DIR}")
        sys.exit(1)
    return files

def parse_participant_condition(filename: str) -> Tuple[str, str]:
    base = Path(filename).name
    m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match '<participant>_<cond>.csv': {base}")
    return m.group(1), m.group(2)

# ---------- Column detection / selection -------------------------------------
def detect_conf_prefix_case_insensitive(columns: List[str]) -> str:
    cols_low = [c.lower() for c in columns]
    for prefix in ("prob", "c", "confidence"):
        if any(col.startswith(prefix) for col in cols_low):
            return prefix
    raise ValueError("Confidence prefix not found (expected 'prob*', 'c*', or 'confidence*').")

def find_real_colname(prefix: str, i: int, columns: List[str]) -> Optional[str]:
    target = f"{prefix}{i}".lower()
    for col in columns:
        if col.lower() == target:
            return col
    for col in columns:
        if col.lower().startswith(target):
            return col
    return None

def lm_triplet_colnames(i: int, conf_prefix: str, columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        find_real_colname("x", i, columns),
        find_real_colname("y", i, columns),
        find_real_colname(conf_prefix, i, columns),
    )

def relevant_indices() -> List[int]:
    s = set()
    s.update(CFG.PROCRUSTES_REF)
    s.update(CFG.HEAD_ROT)
    s.update(CFG.MOUTH)
    s.update(CFG.CENTER_FACE)
    s.update(CFG.BLINK_L_TOP); s.update(CFG.BLINK_L_BOT)
    s.update(CFG.BLINK_R_TOP); s.update(CFG.BLINK_R_BOT)
    s.update([69, 70]) # pupils
    return sorted(s)

def filter_df_to_relevant(df: pd.DataFrame, conf_prefix: str, indices: List[int]) -> pd.DataFrame:
    kept: List[str] = []
    cols = list(df.columns)
    for i in indices:
        x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
        if x and y and c:
            kept.extend([x, y, c])
    if not kept:
        raise ValueError("No relevant triplets found.")
    return df.loc[:, kept].copy()

def confidence_mask(df_reduced: pd.DataFrame, conf_prefix: str, indices: List[int], thr: float) -> Tuple[pd.DataFrame, Dict]:
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
        conf = pd.to_numeric(dfm[c], errors="coerce")
        low = conf < thr
        low = low.fillna(False)

        pre_x = dfm[x].notna()
        pre_y = dfm[y].notna()
        considered = int((pre_x | pre_y).sum()) * 2
        masked = int(((pre_x | pre_y) & low).sum()) * 2

        if low.any():
            dfm.loc[low, [x, y, c]] = np.nan

        per_lm[i] = {
            "frames_total": int(n_frames),
            "frames_low_conf": int(low.sum()),
            "coords_considered": considered,
            "coords_masked": masked,
            "pct_frames_low_conf": (int(low.sum()) / n_frames * 100.0) if n_frames else 0.0
        }
        total_considered += considered
        total_masked += masked

    overall = {
        "frames": int(n_frames),
        "n_landmarks_considered": len(per_lm),
        "total_coord_values": int(total_considered),
        "total_coords_masked": int(total_masked),
        "pct_coords_masked": (total_masked / total_considered * 100.0) if total_considered else 0.0
    }
    return dfm, {"per_landmark": per_lm, "overall": overall}

# ---------- Small IO helpers -------------------------------------------------
def write_per_frame_metrics(out_root: Path, source: str, participant: str, condition: str,
                            perframe: Dict[str, np.ndarray], interocular: np.ndarray, n_frames: int) -> None:
    """
    Write per-frame metrics for a single trial AND append that trial to a combined CSV for the source.

    - Individual trial CSV (unchanged): <out_root>/per_frame/<source>/<participant>_<condition>_perframe.csv
    - Combined CSV (new):             <out_root>/per_frame/<source>/all_perframe.csv
    """
    out_dir = out_root / "per_frame" / source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-trial DataFrame (same columns as before)
    df_pf = pd.DataFrame({
        "participant": participant,
        "condition": condition,
        "frame": np.arange(n_frames, dtype=int),
        "interocular": interocular
    })
    for k, arr in perframe.items():
        df_pf[k] = arr

    # --- 1) write individual trial CSV (keeps previous behavior) ---
    out_path = out_dir / f"{participant}_{condition}_perframe.csv"
    df_pf.to_csv(out_path, index=False)

    # --- 2) append (or create) combined CSV for this source ---
    combined_path = out_dir / "all_perframe.csv"

    # If combined doesn't exist, write header + rows
    if not combined_path.exists():
        df_pf.to_csv(combined_path, index=False)
        return

    # Combined exists -> check headers
    existing_cols = list(pd.read_csv(combined_path, nrows=0).columns)

    # If df_pf has any new columns not in existing, we must rewrite the combined file
    new_cols = [c for c in df_pf.columns if c not in existing_cols]

    if not new_cols:
        # Align df_pf to existing column order (add missing existing cols as NaN if needed)
        for c in existing_cols:
            if c not in df_pf.columns:
                df_pf[c] = np.nan
        df_pf = df_pf[existing_cols]  # ensure same order
        df_pf.to_csv(combined_path, mode="a", header=False, index=False)
    else:
        # There are new columns: read existing, add new cols (NaN), concat, and overwrite combined file
        df_existing = pd.read_csv(combined_path)
        for c in new_cols:
            df_existing[c] = np.nan
        # Ensure consistent column order: existing columns first (now including new ones), then any remaining
        final_cols = list(df_existing.columns)
        # Append the new trial rows with columns aligned
        df_pf = df_pf.reindex(columns=final_cols, fill_value=np.nan)
        df_combined = pd.concat([df_existing, df_pf], ignore_index=True)
        df_combined.to_csv(combined_path, index=False)

def save_json_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
