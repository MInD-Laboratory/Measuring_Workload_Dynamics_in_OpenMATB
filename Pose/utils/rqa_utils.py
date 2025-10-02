from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from rqa.utils import rqa_utils_cpp, norm_utils

def _compute_rqa(series: np.ndarray, p: Dict[str, Any]) -> Tuple[Dict[str, float], int]:
    """Run RQA on one 1D window; returns (stats, err_code)."""
    data_norm = norm_utils.normalize_data(series, p.get("norm", 1))
    ds        = rqa_utils_cpp.rqa_dist(data_norm, data_norm,
                                       dim=p.get("eDim", 4), lag=p.get("tLag", 20))
    _, rs, _, err = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=p.get("rescaleNorm", 1),
        rad=p.get("radius", 0.15),
        diag_ignore=p.get("tw", 2),
        minl=p.get("minl", 4),
        rqa_mode="auto",
    )
    return rs, err

def _sliding_windows(n: int, win_len: int, step: int):
    i = 0
    while i + win_len <= n:
        yield i, i + win_len
        i += step

def _resolve_meta(g: pd.DataFrame, rid: str) -> Tuple[str, str]:
    """Return (participant, condition) from df columns if present; else parse from recording_id."""
    pid = g["participant"].iloc[0] if "participant" in g.columns else ""
    cond = g["condition"].iloc[0] if "condition" in g.columns else ""
    if (not pid or not cond) and isinstance(rid, str) and "_" in rid:
        parts = rid.split("_", 1)
        pid = pid or parts[0]
        cond = cond or parts[1]
    return str(pid), str(cond)

def rqa_over_df_all_features(
    df_all: pd.DataFrame,
    features: List[str],
    fps: int,
    win_seconds: int = 60,
    overlap_frac: float = 0.5,
    params: Optional[Dict[str, Any]] = None,
    out_csv: Optional[str] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Run windowed RQA over the given df_all for each selected feature, per recording_id.
    - df_all must contain columns: recording_id, and the selected feature columns.
    - Optional participant/condition columns are used if present (else parsed from recording_id).

    Returns a tidy DataFrame; optionally writes CSV to out_csv.
    """
    if params is None:
        params = {
            "norm": 1, "eDim": 4, "tLag": 20, "rescaleNorm": 1,
            "radius": 0.15, "tw": 2, "minl": 4,
        }

    # Sanity: features present
    cols_present = [c for c in features if c in df_all.columns]
    if not cols_present:
        raise ValueError("None of the selected features are in df_all. Check SELECTED_FEATURES and NORM_KIND.")

    win_len = int(win_seconds * fps)
    step    = max(1, int(win_len * (1.0 - overlap_frac)))

    results: List[Dict[str, Any]] = []
    groups = df_all.groupby("recording_id", sort=False)

    outer = groups if not progress else tqdm(groups, desc="Recordings")
    for rid, g in outer:
        pid, cond = _resolve_meta(g, rid)
        # keep only numeric, let per-feature selection drive what we compute
        for feat in cols_present:
            x = pd.to_numeric(g[feat], errors="coerce").to_numpy()
            n = len(x)
            if n < win_len:
                continue
            inner = _sliding_windows(n, win_len, step)
            inner = inner if not progress else tqdm(inner, leave=False, desc=f"{rid}:{feat}")
            for w_idx, (s, e) in enumerate(inner):
                win = x[s:e]
                if not np.isfinite(win).all():
                    continue  # skip windows with NaNs (from masking)
                try:
                    rs, err = _compute_rqa(win, params)
                except Exception as ex:
                    if progress:
                        tqdm.write(f"‼️ {rid} {feat} w{w_idx}: exception {ex}")
                    continue
                if err != 0:
                    if progress:
                        tqdm.write(f"‼️ {rid} {feat} w{w_idx}: error code {err}")
                    continue
                results.append({
                    "recording_id":      rid,
                    "participant":       pid,
                    "condition":         cond,
                    "feature":           feat,
                    "window_index":      w_idx,
                    "window_start":      s,
                    "window_end":        e,
                    "perc_recur":        float(rs.get("perc_recur", np.nan)),
                    "perc_determ":       float(rs.get("perc_determ", np.nan)),
                    "maxl_found":        float(rs.get("maxl_found", np.nan)),
                    "mean_line_length":  float(rs.get("mean_line_length", np.nan)),
                    "std_line_length":   float(rs.get("std_line_length", np.nan)),
                    "entropy":           float(rs.get("entropy", np.nan)),
                    "laminarity":        float(rs.get("laminarity", np.nan)),
                    "trapping_time":     float(rs.get("trapping_time", np.nan)),
                    "vmax":              float(rs.get("vmax", np.nan)),
                    "divergence":        float(rs.get("divergence", np.nan)),
                    "trend_lower_diag":  float(rs.get("trend_lower_diag", np.nan)),
                    "trend_upper_diag":  float(rs.get("trend_upper_diag", np.nan)),
                })

    df_out = pd.DataFrame(results)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        print(f"[RQA] wrote {len(df_out)} rows → {out_csv}")
    else:
        print(f"[RQA] produced {len(df_out)} rows (not written)")

    return df_out

def rqa_cross_over_df(
    df_all: pd.DataFrame,
    feat1: str,
    feat2: str,
    fps: int,
    win_seconds: int = 60,
    overlap_frac: float = 0.5,
    params: Optional[Dict[str, Any]] = None,
    out_csv: Optional[str] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Windowed cross-RQA across recordings in df_all for two numeric feature columns.

    Args:
      df_all: dataframe containing recording_id, optional participant/condition, and feature columns.
      feat1, feat2: column names in df_all to run CRQA between.
      fps, win_seconds, overlap_frac: windowing params (win length = win_seconds * fps).
      params: rqa params dict (keys: norm,eDim,tLag,rescaleNorm,radius,tw,minl).
      out_csv: if provided, save resulting tidy DataFrame to this path.
      progress: show tqdm progress.

    Returns:
      DataFrame with rows: recording_id, participant, condition, feature1, feature2,
      window_index, window_start, window_end, and RQA metric columns.
    """
    if params is None:
        params = {"norm": 1, "eDim": 4, "tLag": 20, "rescaleNorm": 1,
                  "radius": 0.15, "tw": 2, "minl": 4}

    if feat1 not in df_all.columns or feat2 not in df_all.columns:
        raise ValueError(f"Features not present in df_all: {feat1}, {feat2}")

    win_len = int(win_seconds * fps)
    step = max(1, int(win_len * (1.0 - overlap_frac)))

    METRIC_COLS = [
        "perc_recur", "perc_determ", "maxl_found", "mean_line_length", "std_line_length",
        "entropy", "laminarity", "trapping_time", "vmax", "divergence",
        "trend_lower_diag", "trend_upper_diag"
    ]

    results: List[Dict[str, Any]] = []
    groups = df_all.groupby("recording_id", sort=False)

    outer = groups if not progress else tqdm(groups, desc="Recordings (CRQA)")
    for rid, g in outer:
        pid = g["participant"].iloc[0] if "participant" in g.columns else ""
        cond = g["condition"].iloc[0] if "condition" in g.columns else ""
        x = pd.to_numeric(g[feat1], errors="coerce").to_numpy()
        y = pd.to_numeric(g[feat2], errors="coerce").to_numpy()
        n = len(x)
        if n < win_len or len(y) < win_len:
            continue
        # sliding windows
        i = 0
        inner_iter = []
        while i + win_len <= n:
            inner_iter.append((i, i + win_len))
            i += step
        inner = inner_iter if not progress else tqdm(inner_iter, leave=False, desc=f"{rid}:{feat1}×{feat2}")
        for w_idx, (s, e) in enumerate(inner):
            xw = x[s:e]; yw = y[s:e]
            if not (np.isfinite(xw).all() and np.isfinite(yw).all()):
                continue
            try:
                xn = norm_utils.normalize_data(xw, params.get("norm", 1))
                yn = norm_utils.normalize_data(yw, params.get("norm", 1))
                ds = rqa_utils_cpp.rqa_dist(xn, yn, dim=params.get("eDim", 4), lag=params.get("tLag", 20))
                _, rs, _, err = rqa_utils_cpp.rqa_stats(
                    ds["d"],
                    rescale=params.get("rescaleNorm", 1),
                    rad=params.get("radius", 0.15),
                    diag_ignore=params.get("tw", 0),
                    minl=params.get("minl", 4),
                    rqa_mode="cross",
                )
            except Exception as ex:
                if progress:
                    tqdm.write(f"‼️ {rid} {feat1}×{feat2} w{w_idx}: exception {ex}")
                continue
            if err != 0:
                if progress:
                    tqdm.write(f"‼️ {rid} {feat1}×{feat2} w{w_idx}: error code {err}")
                continue

            row: Dict[str, Any] = {
                "recording_id": rid,
                "participant": pid,
                "condition": cond,
                "feature1": feat1,
                "feature2": feat2,
                "window_index": w_idx,
                "window_start": s,
                "window_end": e,
            }
            for k in METRIC_COLS:
                row[k] = float(rs.get(k, np.nan))
            results.append(row)

    df_out = pd.DataFrame(results)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        if progress:
            print(f"[CRQA] wrote {len(df_out)} rows → {out_csv}")
    else:
        if progress:
            print(f"[CRQA] produced {len(df_out)} rows (not written)")

    return df_out
