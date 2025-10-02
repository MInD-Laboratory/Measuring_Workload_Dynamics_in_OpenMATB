from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path

from .config import CFG, SCALE_BY_INTEROCULAR
from .preprocessing_utils import detect_conf_prefix_case_insensitive, relevant_indices, find_real_colname
from .normalize_utils import interocular_series
from .geometry_utils import procrustes_frame_to_template, angle_between_points
from .window_utils import windows_indices, is_distance_like_metric, linear_metrics

# --------- Small helpers used by features -----------------------------------
def blink_aperture_from_points(eye_top: np.ndarray, eye_bot: np.ndarray) -> float:
    """
    Compute the blink aperture (vertical eye opening) from eye landmarks.

    Parameters
    ----------
    eye_top : np.ndarray
        Array of 2D coordinates (x,y) for landmarks along the upper eyelid.
    eye_bot : np.ndarray
        Array of 2D coordinates (x,y) for landmarks along the lower eyelid.

    Returns
    -------
    float
        Vertical distance between the average y-position of the top eyelid
        and the average y-position of the bottom eyelid. Smaller values
        indicate a closed/blinking eye; larger values indicate a more open eye.
    """
    # Average the upper eyelid landmarks to get a single "top" point
    top_mean = eye_top.mean(axis=0)
    # Average the lower eyelid landmarks to get a single "bottom" point
    bot_mean = eye_bot.mean(axis=0)
    # Return the absolute vertical distance between top and bottom
    return float(abs(top_mean[1] - bot_mean[1]))

def mouth_aperture(p63: np.ndarray, p67: np.ndarray) -> float:
    """
    Compute the mouth aperture (mouth opening) from two key landmarks.

    Parameters
    ----------
    p63 : np.ndarray
        2D coordinate (x,y) of landmark 63 (usually left mouth corner).
    p67 : np.ndarray
        2D coordinate (x,y) of landmark 67 (usually right mouth corner).

    Returns
    -------
    float
        Euclidean distance between landmarks 63 and 67.
        Represents how open the mouth is (larger = more open).
    """
    # Compute Euclidean distance (straight-line length) between the two points
    return float(np.linalg.norm(p67 - p63))

# --------- Per-file Procrustes features -------------------------------------
def procrustes_features_for_file(df_norm: pd.DataFrame,
                                 template_df: pd.DataFrame,
                                 rel_idxs: List[int]) -> Dict[str, np.ndarray]:
    # Get column names for all x and y coordinates in this file
    cols = list(df_norm.columns)
    xs, ys = [], []
    conf_prefix = detect_conf_prefix_case_insensitive(cols)  # handle confidence-score prefixes
    for i in rel_idxs:
        xs.append(find_real_colname("x", i, cols))  # find column name for x of landmark i
        ys.append(find_real_colname("y", i, cols))  # find column name for y of landmark i

    # Extract template coordinates (used for Procrustes alignment)
    templ_xy = np.column_stack([template_df[xs].values[0], template_df[ys].values[0]])
    n = len(df_norm)  # number of frames

    # Allocate arrays to store per-frame feature values
    head_rot = np.full(n, np.nan, float)      # head rotation angle (rad)
    head_tx  = np.full(n, np.nan, float)      # head translation x
    head_ty  = np.full(n, np.nan, float)      # head translation y
    head_s   = np.full(n, np.nan, float)      # head scaling factor
    head_motion_mag = np.full(n, np.nan, float)  # combined motion magnitude
    blink_ap = np.full(n, np.nan, float)      # blink aperture (eye opening)
    mouth_ap = np.full(n, np.nan, float)      # mouth aperture
    pupil_dx = np.full(n, np.nan, float)      # pupil x-offset (averaged L/R)
    pupil_dy = np.full(n, np.nan, float)      # pupil y-offset (averaged L/R)
    pupil_av = np.full(n, np.nan, float)      # pupil scalar distance metric

    # Helper: return index of landmark if present, else -1
    def idx_of(lmk: int) -> int:
        return rel_idxs.index(lmk) if lmk in rel_idxs else -1

    # Landmark groups for eyes and mouth
    L_top_idxs = [idx_of(38), idx_of(39)]     # top eyelid (left)
    L_bot_idxs = [idx_of(41), idx_of(42)]     # bottom eyelid (left)
    R_top_idxs = [idx_of(44), idx_of(45)]     # top eyelid (right)
    R_bot_idxs = [idx_of(47), idx_of(48)]     # bottom eyelid (right)
    left_eye_ring  = [idx_of(i) for i in [37,38,39,40,41,42] if idx_of(i) >= 0]   # full left eye contour
    right_eye_ring = [idx_of(i) for i in [43,44,45,46,47,48] if idx_of(i) >= 0]   # full right eye contour
    mouth_pair = (idx_of(63), idx_of(67))     # vertical mouth landmarks
    eye_pair = (idx_of(37), idx_of(46))       # left/right eye corners (for head rotation)

    # Process frame by frame
    for t in range(n):
        # Extract x,y for this frame into an array
        fx, fy = [], []
        for xc, yc in zip(xs, ys):
            fx.append(df_norm.iloc[t, df_norm.columns.get_loc(xc)] if xc else np.nan)
            fy.append(df_norm.iloc[t, df_norm.columns.get_loc(yc)] if yc else np.nan)
        frame_xy = np.column_stack([np.asarray(fx, float), np.asarray(fy, float)])

        # Mask for valid points (finite in both frame and template)
        available = np.isfinite(frame_xy).all(axis=1) & np.isfinite(templ_xy).all(axis=1)

        # Align current frame to template using Procrustes
        ok, s, tx, ty, R, Xtrans = procrustes_frame_to_template(frame_xy, templ_xy, available)
        if not ok:  # skip if alignment failed
            continue

        # Save head-level features
        head_s[t] = s
        head_tx[t] = tx
        head_ty[t] = ty
        head_motion_mag[t] = math.sqrt(tx*tx + ty*ty + (s - 1.0)**2)  # combined magnitude

        # Compute head rotation using eye corners
        i37, i46 = eye_pair
        if i37 >= 0 and i46 >= 0 and np.isfinite(Xtrans[i37]).all() and np.isfinite(Xtrans[i46]).all():
            head_rot[t] = angle_between_points(Xtrans[i37], Xtrans[i46])

        # Helper: safely return 2 points if both valid
        def safe_points(idxs: List[int]) -> Optional[np.ndarray]:
            pts = [Xtrans[i] for i in idxs if i >= 0 and np.isfinite(Xtrans[i]).all()]
            return np.vstack(pts) if len(pts) == 2 else None

        # Blink aperture from eyelid points (avg of left/right eyes)
        Ltop = safe_points(L_top_idxs); Lbot = safe_points(L_bot_idxs)
        Rtop = safe_points(R_top_idxs); Rbot = safe_points(R_bot_idxs)
        vals = []
        if Ltop is not None and Lbot is not None:
            vals.append(blink_aperture_from_points(Ltop, Lbot))
        if Rtop is not None and Rbot is not None:
            vals.append(blink_aperture_from_points(Rtop, Rbot))
        if vals:
            blink_ap[t] = float(np.mean(vals))

        # Mouth aperture (distance between two lip landmarks)
        m63, m67 = mouth_pair
        if m63 >= 0 and m67 >= 0 and np.isfinite(Xtrans[m63]).all() and np.isfinite(Xtrans[m67]).all():
            mouth_ap[t] = mouth_aperture(Xtrans[m63], Xtrans[m67])

        # Helper: compute eye center (mean of ≥3 valid contour points)
        def eye_center(idxs: List[int]) -> Optional[np.ndarray]:
            pts = [Xtrans[i] for i in idxs if i >= 0 and np.isfinite(Xtrans[i]).all()]
            if len(pts) >= 3:
                return np.vstack(pts).mean(axis=0)
            return None

        # Get centers of left and right eyes
        cL = eye_center(left_eye_ring)
        cR = eye_center(right_eye_ring)

        # Landmark indices for pupils
        i69 = rel_idxs.index(69) if 69 in rel_idxs else -1
        i70 = rel_idxs.index(70) if 70 in rel_idxs else -1

        offsets = []  # store per-eye (dx, dy)
        mags = []     # store per-eye magnitudes
        if i69 >= 0 and cL is not None and np.isfinite(Xtrans[i69]).all():
            dx, dy = Xtrans[i69] - cL   # left pupil offset from left eye center
            offsets.append((dx, dy))
            mags.append(float(np.linalg.norm([dx, dy])))  # left pupil distance magnitude
        if i70 >= 0 and cR is not None and np.isfinite(Xtrans[i70]).all():
            dx, dy = Xtrans[i70] - cR   # right pupil offset from right eye center
            offsets.append((dx, dy))
            mags.append(float(np.linalg.norm([dx, dy])))  # right pupil distance magnitude

        if offsets:
            # Average offsets across left/right pupils → keep x and y separate
            pupil_dx[t] = float(np.mean([o[0] for o in offsets]))
            pupil_dy[t] = float(np.mean([o[1] for o in offsets]))
            # Average magnitude across eyes → scalar "pupil_metric"
            pupil_av[t] = float(np.mean(mags))

    # Return dictionary of all features for this file
    return {
        "head_rotation_rad": head_rot,
        "head_tx": head_tx,
        "head_ty": head_ty,
        "head_scale": head_s,
        "head_motion_mag": head_motion_mag,
        "blink_aperture": blink_ap,
        "mouth_aperture": mouth_ap,
        "pupil_dx": pupil_dx,         # averaged horizontal pupil offset
        "pupil_dy": pupil_dy,         # averaged vertical pupil offset
        "pupil_metric": pupil_av      # scalar pupil distance magnitude
    }

# --------- Per-file "original" features -------------------------------------
def original_features_for_file(df_norm: pd.DataFrame) -> Dict[str, np.ndarray]:
    # Cache column names for faster lookups
    cols = list(df_norm.columns)

    # Helper: fetch a landmark column by index+axis; return NaN series if missing
    def col(i: int, axis: str) -> pd.Series:
        c = find_real_colname(axis, i, cols)                    # resolve actual column name (handles casing/prefixes)
        return df_norm[c].astype(float) if c else pd.Series([np.nan]*len(df_norm))  # cast to float or NaNs

    # Number of frames in this file
    n = len(df_norm)

    # Outer eye corners (37: left outer, 46: right outer) for head rotation
    x37, y37 = col(37, "x"), col(37, "y")
    x46, y46 = col(46, "x"), col(46, "y")

    # Head rotation (angle of vector 37→46)
    head_rot = np.full(n, np.nan, float)                        # preallocate with NaNs
    vdx = (x46 - x37).values                                    # Δx between eye corners
    vdy = (y46 - y37).values                                    # Δy between eye corners
    valid = np.isfinite(vdx) & np.isfinite(vdy)                  # frames where both deltas are finite
    head_rot[valid] = np.arctan2(vdy[valid], vdx[valid])         # atan2 for robust angle in radians

    # Helper: simple average of two Series (elementwise)
    def avg2(s1, s2): return (s1 + s2) / 2.0

    # Eyelid heights (top vs bottom) for blink aperture per eye
    Ltop = avg2(col(38,"y"), col(39,"y"))                        # left eye: average top lid y
    Lbot = avg2(col(41,"y"), col(42,"y"))                        # left eye: average bottom lid y
    Rtop = avg2(col(44,"y"), col(45,"y"))                        # right eye: average top lid y
    Rbot = avg2(col(47,"y"), col(48,"y"))                        # right eye: average bottom lid y

    # Blink aperture (average of available eyes)
    blink = np.full(n, np.nan, float)          # final combined aperture (length n)

    L_ok = Ltop.notna() & Lbot.notna()         # frames where left eyelid pair is valid
    R_ok = Rtop.notna() & Rbot.notna()         # frames where right eyelid pair is valid

    # Build full-length left/right apertures (NaN where invalid)
    l_ap = np.full(n, np.nan, float)
    r_ap = np.full(n, np.nan, float)
    if L_ok.any():
        l_ap[L_ok.values] = np.abs(Ltop[L_ok].values - Lbot[L_ok].values)
    if R_ok.any():
        r_ap[R_ok.values] = np.abs(Rtop[R_ok].values - Rbot[R_ok].values)

    # Combine per frame
    both  = np.isfinite(l_ap) & np.isfinite(r_ap)   # both eyes valid
    onlyL = np.isfinite(l_ap) & ~np.isfinite(r_ap)  # only left valid
    onlyR = np.isfinite(r_ap) & ~np.isfinite(l_ap)  # only right valid

    blink[both]  = (l_ap[both] + r_ap[both]) / 2.0  # average when both present
    blink[onlyL] = l_ap[onlyL]                      # fallback to left-only
    blink[onlyR] = r_ap[onlyR]                      # fallback to right-only

    # Mouth aperture (distance between landmarks 63 and 67)
    x63, y63 = col(63,"x"), col(63,"y")
    x67, y67 = col(67,"x"), col(67,"y")
    mouth = np.sqrt((x67 - x63)**2 + (y67 - y63)**2).values       # Euclidean distance per frame

    # Eye center helper: return (cx, cy) arrays as nanmeans over provided landmark IDs
    def eye_center_xy(ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        xs = [col(i,"x").values for i in ids if find_real_colname("x", i, cols)]  # gather x arrays
        ys = [col(i,"y").values for i in ids if find_real_colname("y", i, cols)]  # gather y arrays
        if not xs or not ys:
            return np.full(n, np.nan), np.full(n, np.nan)          # no data → NaNs
        x_mat = np.vstack(xs); y_mat = np.vstack(ys)                # shape: (#ids, n)
        return np.nanmean(x_mat, axis=0), np.nanmean(y_mat, axis=0) # nan-robust mean across landmarks

    # Geometric eye centers from ring landmarks (left: 37–42, right: 43–48)
    cLx, cLy = eye_center_xy([37,38,39,40,41,42])
    cRx, cRy = eye_center_xy([43,44,45,46,47,48])

    # Pupil landmark coordinates (69 left pupil, 70 right pupil)
    pLx = col(69, "x").values if find_real_colname("x", 69, cols) else np.full(n, np.nan)
    pLy = col(69, "y").values if find_real_colname("y", 69, cols) else np.full(n, np.nan)
    pRx = col(70, "x").values if find_real_colname("x", 70, cols) else np.full(n, np.nan)
    pRy = col(70, "y").values if find_real_colname("y", 70, cols) else np.full(n, np.nan)

    # Pupil offsets relative to eye centers (per eye)
    dxL = pLx - cLx                                               # left pupil horizontal offset
    dyL = pLy - cLy                                               # left pupil vertical offset
    dxR = pRx - cRx                                               # right pupil horizontal offset
    dyR = pRy - cRy                                               # right pupil vertical offset

    # Average L/R offsets separately for x and y (nan-robust)
    # Stack into 2×n and take nanmean across the first axis; if both are NaN → result NaN
    pupil_dx = np.nanmean(np.vstack([dxL, dxR]), axis=0)          # averaged x-offset
    pupil_dy = np.nanmean(np.vstack([dyL, dyR]), axis=0)          # averaged y-offset

    # Pupil magnitude per eye (Euclidean), then average L/R to one scalar
    dL = np.sqrt(dxL**2 + dyL**2)                                 # left magnitude
    dR = np.sqrt(dxR**2 + dyR**2)                                 # right magnitude
    pupil = np.where(np.isfinite(dL) & np.isfinite(dR),           # both valid → average
                     (dL + dR) / 2.0,
                     np.where(np.isfinite(dL), dL,                # only left valid → left
                              np.where(np.isfinite(dR), dR, np.nan)))  # only right valid → right, else NaN
    
    # Center-face magnitude (dispersion of central face landmarks around their mean)
    # Collect x and y arrays for the center-face landmarks (one row per landmark, columns = frames)
    nose_x = [col(i, "x").values for i in CFG.CENTER_FACE]        
    nose_y = [col(i, "y").values for i in CFG.CENTER_FACE]        

    # Stack into 2D arrays (shape: #points x #frames). If empty, make empty arrays.
    nose_x = np.vstack(nose_x) if len(nose_x) else np.empty((0, n))
    nose_y = np.vstack(nose_y) if len(nose_y) else np.empty((0, n))

    # Preallocate outputs
    cfm = np.full(n, np.nan, float)        # overall RMS magnitude
    cfm_x = np.full(n, np.nan, float)      # RMS spread in X direction only
    cfm_y = np.full(n, np.nan, float)      # RMS spread in Y direction only

    if nose_x.size and nose_y.size:
        # Compute per-point means across frames (shape: #points x 1)
        mean_x = np.nanmean(nose_x, axis=1, keepdims=True)
        mean_y = np.nanmean(nose_y, axis=1, keepdims=True)

        # Deviations from each point’s own mean (so we’re measuring *spread*, not absolute position)
        dx = nose_x - mean_x   # x deviations, same shape (#points x #frames)
        dy = nose_y - mean_y   # y deviations

        # Euclidean distance per point, per frame
        dists = np.sqrt(dx**2 + dy**2)

        # Root-mean-square across points → gives 1D series over frames
        cfm = np.sqrt(np.nanmean(dists**2, axis=0))   # total magnitude spread
        cfm_x = np.sqrt(np.nanmean(dx**2, axis=0))    # horizontal spread only
        cfm_y = np.sqrt(np.nanmean(dy**2, axis=0))    # vertical spread only

    # Return all per-frame features (arrays length n)
    return {
        "head_rotation_rad": head_rot,        # head rotation angle (radians)
        "blink_aperture": blink,              # eyelid opening (avg across eyes)
        "mouth_aperture": mouth,              # lip opening distance
        "pupil_dx": pupil_dx,                 # averaged horizontal pupil offset (L/R)
        "pupil_dy": pupil_dy,                 # averaged vertical pupil offset (L/R)
        "pupil_metric": pupil,                # averaged magnitude of pupil offset (L/R)
        "center_face_magnitude": cfm,          # central-face dispersion (stability proxy)
        "center_face_x": cfm_x,
        "center_face_y": cfm_y
    }

# --------- Linear-from-perframe helper --------------------------------------
def compute_linear_from_perframe_dir(per_frame_dir: Path,
                                     out_csv: Path,
                                     fps: int,
                                     window_seconds: int,
                                     window_overlap: float,
                                     scale_by_interocular: bool = True) -> Dict[str, int]:
    """
    Compute linear (time-domain) metrics from per-frame pose/behavioral data
    stored in individual CSV files.

    Parameters
    ----------
    per_frame_dir : Path
        Directory containing per-frame CSVs (one per trial/participant/condition).
    out_csv : Path
        Output CSV path for aggregated windowed metrics.
    fps : int
        Sampling rate (frames per second).
    window_seconds : int
        Length of each analysis window in seconds.
    window_overlap : float
        Fraction of overlap between consecutive windows (0.0 = no overlap, 0.5 = 50% overlap).
    scale_by_interocular : bool, default=True
        Whether to normalize distance-like metrics by interocular distance
        (to compensate for scaling differences between participants).

    Returns
    -------
    Dict[str, int]
        Dictionary mapping metric name → number of dropped windows due to invalid data.
    """
    rows = []                       # Collect results for all windows across all files
    drops_agg: Dict[str, int] = {}  # Count of dropped windows per metric
    files = sorted(per_frame_dir.glob("*.csv"))  # All per-frame CSV files in directory

    # --- Process each per-frame CSV file ---
    for pf in files:
        df = pd.read_csv(pf)

        # Participant ID and condition labels, if present
        pid = str(df["participant"].iloc[0]) if "participant" in df.columns and len(df) else "NA"
        cond = str(df["condition"].iloc[0]) if "condition" in df.columns and len(df) else "NA"

        # Metric columns = everything except metadata columns
        metric_cols = [c for c in df.columns if c not in ("participant","condition","frame","interocular")]

        # Interocular distance (used for normalization if available)
        io = df["interocular"].to_numpy(float) if "interocular" in df.columns else np.full(len(df), np.nan)

        # --- Scale metrics if appropriate ---
        scaled = {}
        for k in metric_cols:
            arr = df[k].to_numpy(float)
            if scale_by_interocular and is_distance_like_metric(k) and np.isfinite(io).any():
                # Normalize distance-like metrics by interocular distance
                # Protect against division by zero or very small values
                with np.errstate(divide='ignore', invalid='ignore'):
                    scaled_arr = arr / io
                    # Replace inf/nan from division with original unscaled values
                    bad_mask = ~np.isfinite(scaled_arr) | (io < 1e-6)
                    scaled_arr[bad_mask] = arr[bad_mask]
                    scaled[k] = scaled_arr
            else:
                scaled[k] = arr

        # --- Compute windowing parameters ---
        win = window_seconds * fps                        # Window size in frames
        hop = int(win * (1.0 - window_overlap))           # Step size (with overlap)
        hop = max(1, hop)                                 # Ensure hop is at least 1
        n = len(df)                                       # Total number of frames in trial

        # --- Slide windows across the sequence ---
        for (s, e, widx) in windows_indices(n, win, hop):
            # Base metadata for this window
            base = {
                "source": per_frame_dir.name,
                "participant": pid,
                "condition": cond,
                "window_index": widx,
                "t_start_frame": s,
                "t_end_frame": e
            }

            # --- Process each metric within the window ---
            for k, arr in scaled.items():
                seg = arr[s:e]  # Segment of data for this metric

                if np.any(~np.isfinite(seg)) or len(seg) < 3:
                    # Drop if window has NaNs or is too short
                    drops_agg[k] = drops_agg.get(k, 0) + 1
                    base[f"{k}_mean_abs_vel"] = np.nan
                    base[f"{k}_mean_abs_acc"] = np.nan
                    base[f"{k}_rms"] = np.nan
                else:
                    # Compute velocity, acceleration, RMS from segment
                    v, a, r = linear_metrics(seg.astype(float), fps)
                    base[f"{k}_mean_abs_vel"] = v
                    base[f"{k}_mean_abs_acc"] = a
                    base[f"{k}_rms"] = r

            rows.append(base)  # Store results for this window

    # --- Write results to CSV ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return drops_agg
