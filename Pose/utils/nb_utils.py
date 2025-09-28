# utils/nb_utils.py
from __future__ import annotations
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List
import re
from typing import Iterable, Optional

# ---------- disk status ----------
def outputs_exist(base: str | Path) -> dict:
    base = Path(base)
    per_frame = {
        "procrustes_global": list((base / "features" / "per_frame" / "procrustes_global").glob("*.csv")),
        "procrustes_participant": list((base / "features" / "per_frame" / "procrustes_participant").glob("*.csv")),
        "original": list((base / "features" / "per_frame" / "original").glob("*.csv")),
    }
    linear = {
        "procrustes_global": (base / "linear_metrics" / "procrustes_global_linear.csv").exists(),
        "procrustes_participant": (base / "linear_metrics" / "procrustes_participant_linear.csv").exists(),
        "original": (base / "linear_metrics" / "original_linear.csv").exists(),
    }
    any_per_frame = any(len(v) > 0 for v in per_frame.values())
    any_linear = any(linear.values())
    return {"per_frame": per_frame, "linear": linear,
            "any_per_frame": any_per_frame, "any_linear": any_linear}

# ---------- file picking ----------
def pick_norm_file(out_base: str | Path, sample_norm: str | Path | None = None) -> Path:
    if sample_norm:
        return Path(sample_norm)
    norm_dir = Path(out_base) / "norm_screen"
    files = sorted(norm_dir.glob("*_norm.csv"))
    if not files:
        raise FileNotFoundError(f"No normalized CSVs in {norm_dir}. Run the pipeline first.")
    return files[0]

# ---------- column access ----------
def find_col(df: pd.DataFrame, axis: str, i: int) -> str | None:
    c1, c2 = f"{axis}{i}", f"{axis.upper()}{i}"
    return c1 if c1 in df.columns else (c2 if c2 in df.columns else None)

def series_num(df: pd.DataFrame, axis: str, i: int, n: int) -> pd.Series:
    c = find_col(df, axis, i)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series([np.nan]*n)

# ---------- slicing ----------
def slice_first_seconds(df: pd.DataFrame, fps: int, seconds: int) -> pd.DataFrame:
    n = len(df)
    end = min(n, fps * seconds)
    return df.iloc[:end].reset_index(drop=True)

# ---------- metrics & plotting ----------
META_COLS = {"source","participant","condition","window_index","t_start_frame","t_end_frame"}

def ensure_condition_order(df: pd.DataFrame, cond_order=("L","M","H")) -> pd.DataFrame:
    if "condition" in df.columns:
        df["condition"] = pd.Categorical(df["condition"], categories=list(cond_order), ordered=True)
    return df

def candidate_metric_cols(df: pd.DataFrame) -> List[str]:
    num_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    priority, others = [], []
    for c in num_cols:
        lc = c.lower()
        if lc.endswith("_mean_abs_vel") or lc.endswith("_mean_abs_acc") or lc.endswith("_rms"):
            priority.append(c)
        else:
            others.append(c)
    return priority + others

def default_metric(cols: List[str]) -> str | None:
    if not cols: return None
    lowers = [c.lower() for c in cols]
    prefs = [
        "blink_aperture_rms", "mouth_aperture_rms", "center_face_magnitude_rms",
        "blink_aperture_mean_abs_vel", "mouth_aperture_mean_abs_vel",
    ]
    for exact in prefs:
        if exact in lowers:
            return cols[lowers.index(exact)]
    for i, lc in enumerate(lowers):
        if lc.endswith("_rms"):
            return cols[i]
    return cols[0]

def sem(series) -> float:
    s = pd.Series(series).astype(float)
    return s.std(ddof=1) / np.sqrt(max(s.count(), 1))

def bar_by_condition(df: pd.DataFrame, metric: str, cond_order=("L","M","H"),
                     colors=("#4474b5","#fbe79b","#de3a2c"), title_suffix: str = ""):
    df = ensure_condition_order(df, cond_order)
    grouped = df.groupby("condition")[metric].agg(["mean", sem]).reindex(cond_order)
    idx = np.arange(len(cond_order))
    means = grouped["mean"].to_numpy(dtype=float)
    errs  = grouped["sem"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(idx, means, yerr=errs, capsize=4, width=0.75, color=list(colors), edgecolor="black", alpha=0.9)
    ax.set_xticks(idx); ax.set_xticklabels(cond_order)
    ax.set_xlabel("Condition"); ax.set_ylabel("Mean ± SEM")
    ttl = f"{metric} by Condition" + (f" — {title_suffix}" if title_suffix else "")
    ax.set_title(ttl); ax.set_xlim(-0.5, len(cond_order)-0.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.25)
    for x, m in zip(idx, means):
        if np.isfinite(m):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig, ax

def load_rqa_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    # drop windowing columns
    df = df.drop(columns=[c for c in ["window_start", "window_end"] if c in df.columns])
    return ensure_condition_order(df)

# ---------- stats ----------
def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected = {}
    for i, (lbl, p) in enumerate(items, start=1):
        corrected[lbl] = min(p * (m - i + 1), 1.0)
    return {k: corrected[k] for k in pvals.keys()}

_NORM_SUBDIRS = {
    "original": "original",
    "procrustes_global": "procrustes_global",
    "procrustes_participant": "procrustes_participant",
}

_META_COLS = {"participant", "condition", "frame", "interocular"}

def _per_frame_dir(out_base: str | Path, norm_kind: str) -> Path:
    if norm_kind not in _NORM_SUBDIRS:
        raise ValueError(f"Unknown norm_kind '{norm_kind}'. "
                         f"Use one of {list(_NORM_SUBDIRS.keys())}.")
    return Path(out_base) / "features" / "per_frame" / _NORM_SUBDIRS[norm_kind]

def available_norm_kinds(out_base: str | Path) -> list[str]:
    kinds = []
    for k in _NORM_SUBDIRS:
        d = _per_frame_dir(out_base, k)
        if d.exists() and any(d.glob("*.csv")):
            kinds.append(k)
    return kinds

def list_per_frame_files(out_base: str | Path, norm_kind: str) -> list[Path]:
    d = _per_frame_dir(out_base, norm_kind)
    if not d.exists():
        return []
    return sorted(d.glob("*.csv"))

def _parse_pid_cond_from_name(p: Path) -> tuple[str, str]:
    m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)_perframe\.csv$", p.name)
    return (m.group(1), m.group(2)) if m else ("NA", "NA")

def _matches_filters(pid: str, cond: str,
                     participants: Optional[Iterable[str]],
                     conditions: Optional[Iterable[str]]) -> bool:
    ok_pid = True if participants is None else str(pid) in set(map(str, participants))
    ok_cond = True if conditions is None else str(cond) in set(map(str, conditions))
    return ok_pid and ok_cond

def load_all_per_frame(out_base: str | Path,
                       norm_kind: str,
                       participants: Optional[Iterable[str]] = None,
                       conditions: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load and concatenate ALL per-frame CSVs for a route (optionally filtered).
    Adds a 'recording_id' column from the filename stem and enforces meta columns if present.
    """
    paths = list_per_frame_files(out_base, norm_kind)
    if not paths:
        raise FileNotFoundError(f"No per-frame CSVs for '{norm_kind}' under {_per_frame_dir(out_base, norm_kind)}")

    frames = []
    for p in paths:
        pid, cond = _parse_pid_cond_from_name(p)
        if not _matches_filters(pid, cond, participants, conditions):
            continue
        df = pd.read_csv(p)
        df.insert(0, "recording_id", p.stem)  # e.g., "472_H_perframe"
        # make sure participant/condition exist (pipeline writes them; still be defensive)
        if "participant" not in df.columns: df.insert(1, "participant", pid)
        if "condition" not in df.columns:   df.insert(2, "condition", cond)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No files matched filters participants={participants} conditions={conditions} for '{norm_kind}'")

    out = pd.concat(frames, axis=0, ignore_index=True)

    # ensure expected meta dtypes are clean
    for c in ["recording_id", "participant", "condition"]:
        if c in out.columns:
            out[c] = out[c].astype(str)

    # frame should be int when present
    if "frame" in out.columns:
        out["frame"] = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")

    return out

def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return candidate feature columns from a long-form per-frame table.
    Excludes typical meta columns and keeps only numeric dtype.
    """
    meta = {
        "recording_id", "participant", "condition", "route",
        "frame", "interocular", "source"
    }
    cols = []
    for c in df.columns:
        if c in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
