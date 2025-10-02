# utils/nl_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import math
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path



# ---------------- AMI  ----------------
def ami(timeseries, min_lag: int, max_lag: int):
    """
    Auto Mutual Information using histogram (equiprobable) binning per lag.
    Returns array shape (L, 2) with columns [lag, AMI].
    """
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        x = timeseries.values.flatten()
    elif isinstance(timeseries, np.ndarray):
        x = timeseries.flatten()
    else:
        raise ValueError("timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = x[np.isfinite(x)]
    length = len(x)
    if length < 2 * max_lag or max_lag < 1:
        return None  # too short for requested lags

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(max(1, min_lag), max_lag + 1)
    else:
        lag = np.arange(max(1, min_lag), max(1, length // 2))

    # scale to [0,1] (guard against constant series)
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.column_stack((lag, np.zeros_like(lag, dtype=float)))
    x = (x - lo) / (hi - lo)

    ami_values = np.zeros(len(lag), dtype=float)
    for i in tqdm(range(len(lag)), desc='Processing AMI', leave=False):
        N = length - lag[i]
        if N <= 2:
            ami_values[i] = 0.0
            continue
        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.var(x[:N], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        x1 = x[:N]
        x2 = x[lag[i]:]
        s = 0.0
        for k1 in range(1, k + 1):
            x1_lo, x1_hi = (k1 - 1) / k, k1 / k
            mask1 = (x1 > x1_lo) & (x1 <= x1_hi)
            if not mask1.any():
                continue
            px1 = mask1.sum() / N
            for k2 in range(1, k + 1):
                x2_lo, x2_hi = (k2 - 1) / k, k2 / k
                mask2 = (x2 > x2_lo) & (x2 <= x2_hi)
                if not mask2.any():
                    continue
                px2 = mask2.sum() / N
                pxy = (mask1 & mask2).sum() / N
                if pxy > 0:
                    s += pxy * np.log2(pxy / (px1 * px2))
        ami_values[i] = s

    return np.column_stack((lag, ami_values))

# ---------------- Helpers for AMI batch computation ----------------

def _valid_recordings_with_series(df_all: pd.DataFrame, feature: str, max_lag: int) -> List[Tuple[str, int]]:
    """
    Return list of (recording_id, length_of_valid_series) where the feature exists,
    is numeric, and has enough finite samples for AMI up to max_lag.
    """
    out = []
    if feature not in df_all.columns:
        return out
    for rid, g in df_all.groupby("recording_id"):
        x = pd.to_numeric(g[feature], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if len(x) >= 2 * max_lag:
            out.append((rid, len(x)))
    return out

def pick_recordings(df_all: pd.DataFrame, feature: str, max_lag: int,
                    sample_n: int, strategy: str = "longest", seed: Optional[int] = 42) -> List[str]:
    """
    Choose up to sample_n recording_ids for a feature.
    strategy: "longest" | "random"
    """
    cands = _valid_recordings_with_series(df_all, feature, max_lag)
    if not cands:
        return []
    if strategy == "random":
        rng = random.Random(seed)
        rids = [rid for rid, _ in cands]
        rng.shuffle(rids)
        return rids[:sample_n]
    # default: longest
    cands.sort(key=lambda t: t[1], reverse=True)
    return [rid for rid, _ in cands[:sample_n]]

def compute_ami_for_feature(df_all: pd.DataFrame, feature: str, min_lag: int, max_lag: int,
                            recording_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Compute AMI curves for a feature across selected recording_ids.
    Returns dict: recording_id -> DataFrame(lag, ami).
    """
    curves = {}
    for rid in recording_ids:
        g = df_all[df_all["recording_id"] == rid]
        x = pd.to_numeric(g[feature], errors="coerce")
        arr = ami(x, min_lag=min_lag, max_lag=max_lag)
        if arr is None:
            continue
        curves[rid] = pd.DataFrame({"lag": arr[:, 0].astype(int), "ami": arr[:, 1]})
    return curves

def average_ami(curves: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Average AMI over recordings (inner-join by lag).
    """
    if not curves:
        return None
    dfs = []
    for rid, dfc in curves.items():
        tmp = dfc.copy()
        tmp = tmp.rename(columns={"ami": f"ami__{rid}"})
        dfs.append(tmp)
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on="lag", how="inner")
    if base.empty:
        return None
    ami_cols = [c for c in base.columns if c.startswith("ami__")]
    base["ami_mean"] = base[ami_cols].mean(axis=1)
    base["ami_sem"]  = base[ami_cols].sem(axis=1, ddof=1)
    return base[["lag", "ami_mean", "ami_sem"]].copy()

def first_local_minimum(
    y: np.ndarray,
    min_lag: int = 1,
    lags: np.ndarray | None = None,
    smooth_win: int = 5,
    min_prominence: float = 0.05,
    rel_drop: float = 0.10,
    allow_global_fallback: bool = True,
    cutoff_bits: float | None = 1.0,   # <-- NEW: hard AMI threshold (bits). Set None to disable.
) -> Optional[int]:
    """
    Robust AMI tau finder:
      1) Prefer first lag where smoothed AMI <= cutoff_bits (default 1.0 bit).
      2) Otherwise, return first meaningful local minimum (prominence/relative-drop guarded).
      3) Otherwise, optionally return global minimum if it passes the same test.
    """
    import numpy as np

    if y is None or len(y) < 3:
        return None

    # convert to float and mask non-finite
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return None

    # simple moving-average smoothing
    if smooth_win is None or smooth_win <= 1:
        ys = y.copy()
    else:
        w = int(max(1, int(smooth_win)))
        if w % 2 == 0:
            w += 1
        pad = w // 2
        y_pad = np.pad(y, pad_width=pad, mode="edge")
        kernel = np.ones(w) / w
        ys = np.convolve(y_pad, kernel, mode="valid")

    L = len(ys)
    if min_lag < 1:
        min_lag = 1
    if min_lag >= L - 1:
        return None

    # ---- 1) Hard threshold criterion --------------------------------------
    if cutoff_bits is not None:
        for i in range(min_lag, L):
            if np.isfinite(ys[i]) and ys[i] <= float(cutoff_bits):
                return int(lags[i]) if (lags is not None and len(lags) == L) else i

    # ---- 2) Meaningful first local minimum --------------------------------
    early_n = max(1, min(min_lag, L // 4))
    early_level = np.nanmean(ys[:early_n])

    abs_thresh = float(min_prominence)
    rel_thresh = float(rel_drop) * max(abs(early_level), 1e-12)

    def is_meaningful_drop(val):
        drop = early_level - val
        return (drop >= abs_thresh) or (drop >= rel_thresh)

    for i in range(min_lag + 1, L - 1):
        if not (np.isfinite(ys[i-1]) and np.isfinite(ys[i]) and np.isfinite(ys[i+1])):
            continue
        # strict/flat local min
        if ((ys[i] < ys[i-1] and ys[i] <= ys[i+1]) or
            (ys[i] <= ys[i-1] and ys[i] < ys[i+1]) or
            (ys[i] == ys[i-1] == ys[i+1])):
            if is_meaningful_drop(ys[i]):
                return int(lags[i]) if (lags is not None and len(lags) == L) else i

    # ---- 3) Global-min fallback -------------------------------------------
    if allow_global_fallback:
        idx = int(np.nanargmin(ys))
        if idx >= min_lag and is_meaningful_drop(ys[idx]):
            return int(lags[idx]) if (lags is not None and len(lags) == L) else idx

    return None

def ami_average_across_features(
    df_all: pd.DataFrame,
    features: List[str],
    sample_n: int = 10,
    min_lag: int = 1,
    max_lag: int = 140,
    strategy: str = "longest",
    seed: int = 42,
    norm_kind: Optional[str] = None,
    plot: bool = True,
    ncols: int = 2,
    title_suffix: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute AMI per-feature (average across recordings) and a combined average across features.

    Returns a dict:
      {
        "per_feature_curves": { feature: [df_curve_rec1, df_curve_rec2, ...], ... },
        "per_feature_avg":    { feature: df_avg, ... },
        "combined_avg":       df_combined_or_None,
        "summary_lines":      [str, ...]
      }

    Notes:
      - `pick_recordings`, `compute_ami_for_feature`, and `average_ami` are expected to
        exist in utils.nl_utils and follow the repo's existing signatures.
      - combined average is equal-weight across features (mean of per-feature averaged curves).
    """
    per_feature_curves: Dict[str, List[pd.DataFrame]] = {}
    per_feature_avg: Dict[str, Optional[pd.DataFrame]] = {}
    summary_lines: List[str] = []

    if verbose:
        feat_names_short = ", ".join(features)
        print(f"Doing AMI on {sample_n} recordings per feature for: {feat_names_short} — plotting." if plot else
              f"Doing AMI on {sample_n} recordings per feature for: {feat_names_short}.")

    for feat in features:
        # pick recordings for this feature
        rids = pick_recordings(df_all, feat, max_lag=max_lag, sample_n=sample_n,
                               strategy=strategy, seed=seed)
        if not rids:
            summary_lines.append(f"[{feat}] No recordings with length ≥ {2*max_lag} frames. Skipped.")
            per_feature_curves[feat] = []
            per_feature_avg[feat] = None
            continue

        # compute AMI curves for each recording (list of DataFrames or arrays depending on repo)
        curves = compute_ami_for_feature(df_all, feat, min_lag=min_lag, max_lag=max_lag, recording_ids=rids)
        if not curves:
            summary_lines.append(f"[{feat}] compute_ami_for_feature returned empty. Skipped.")
            per_feature_curves[feat] = []
            per_feature_avg[feat] = None
            continue

        # average across recordings for this feature
        avg = average_ami(curves)
        per_feature_curves[feat] = curves
        per_feature_avg[feat] = avg

        if avg is None or (isinstance(avg, pd.DataFrame) and avg.empty):
            summary_lines.append(f"[{feat}] Computed on {len(curves)} rec(s), but averaging failed (no common lags?).")
        else:
            try:
                l0, l1 = int(avg["lag"].min()), int(avg["lag"].max())
                summary_lines.append(f"[{feat}] {len(curves)} rec(s); lags {l0}..{l1}")
            except Exception:
                summary_lines.append(f"[{feat}] {len(curves)} rec(s); averaged (lag info not available).")

    # Build combined average across the features we successfully computed
    good_avgs = {k: v for k, v in per_feature_avg.items() if v is not None and not getattr(v, "empty", False)}
    combined_avg = None

    if good_avgs:
        # choose numeric value column automatically (first numeric column other than 'lag')
        sample_df = next(iter(good_avgs.values()))
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in numeric_cols if str(c).lower() != "lag"]
        if not value_cols:
            summary_lines.append("Couldn't find numeric AMI column in averaged frames; aborting combined average.")
        else:
            value_col = value_cols[0]
            series_list = []
            for feat, df_avg in good_avgs.items():
                if value_col in df_avg.columns:
                    s = df_avg.set_index("lag")[value_col].rename(feat)
                else:
                    other_numeric = [c for c in df_avg.select_dtypes(include=[np.number]).columns if str(c).lower() != "lag"]
                    if not other_numeric:
                        continue
                    s = df_avg.set_index("lag")[other_numeric[0]].rename(feat)
                series_list.append(s)

            if series_list:
                concat = pd.concat(series_list, axis=1)  # columns = features, index = lag
                mean_series = concat.mean(axis=1, skipna=True)
                combined_avg = mean_series.reset_index()
                combined_avg.columns = ["lag", value_col]

                # add combined to the per_feature_avg mapping so plotting & downstream code see it
                per_feature_avg["Combined"] = combined_avg
                summary_lines.append(f"[Combined] Averaged across {len(series_list)} feature-averages.")
            else:
                summary_lines.append("No valid per-feature averages to combine.")
    else:
        summary_lines.append("No successful per-feature averages computed; skipping combined average.")

    if plot:
        try:
            # lazy import to avoid circular import at module import time
            from ...Pose.utils.viz_utils import plot_ami_subplots
            plot_ami_subplots(per_feature_curves, per_feature_avg, ncols=ncols, title_suffix=title_suffix)
        except Exception as e:
            summary_lines.append(f"Plot failed: {e}")

    return {
        "per_feature_curves": per_feature_curves,
        "per_feature_avg": per_feature_avg,
        "combined_avg": combined_avg,
        "summary_lines": summary_lines,
    }

# ---------------- FNN  ----------------

# ---- helpers to choose recordings  --------------------
def _valid_recordings(df_all: pd.DataFrame, feature: str, min_len: int) -> List[Tuple[str, int]]:
    """Return (recording_id, usable_length) with enough finite samples for FNN."""
    out: List[Tuple[str, int]] = []
    if feature not in df_all.columns:
        return out
    for rid, g in df_all.groupby("recording_id"):
        x = pd.to_numeric(g[feature], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if len(x) >= min_len:
            out.append((rid, len(x)))
    return out

def _pick_recordings(df_all: pd.DataFrame, feature: str, min_len: int,
                     sample_n: int, strategy: str = "longest", seed: int = 42) -> List[str]:
    cands = _valid_recordings(df_all, feature, min_len)
    if not cands:
        return []
    if strategy == "random":
        rng = np.random.default_rng(seed)
        rids = [rid for rid, _ in cands]
        rng.shuffle(rids)
        return rids[:sample_n]
    # longest
    cands.sort(key=lambda t: t[1], reverse=True)
    return [rid for rid, _ in cands[:sample_n]]

# ---- core FNN -------------------------------------------------------------
def embed_time_series(data: np.ndarray, embedding_dim: int, lag: int) -> np.ndarray:
    n = len(data) - (embedding_dim - 1) * lag
    if n <= 0:
        return np.empty((0, embedding_dim))
    out = np.empty((n, embedding_dim), dtype=float)
    for c in range(embedding_dim):
        out[:, c] = data[c * lag : c * lag + n]
    return out

def fnn(timeseries, tlag: int, min_dimension: int, max_dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dims, percent_false_neighbors) for a single 1D series."""
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        x = timeseries.values.flatten().astype(float)
    elif isinstance(timeseries, np.ndarray):
        x = timeseries.flatten().astype(float)
    else:
        raise ValueError("timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = x[np.isfinite(x)]
    if len(x) < (max_dimension + 1) * tlag + 1:
        # not enough samples, return empty
        return np.arange(min_dimension, max_dimension + 1), np.full((max_dimension - min_dimension + 1,), np.nan, dtype=float)

    dims = np.arange(min_dimension, max_dimension + 1, dtype=int)
    pct  = np.ones_like(dims, dtype=float)

    mean_x = np.mean(x)
    Ra = np.sqrt(np.mean((x - mean_x) ** 2))  # Abar in Kennel et al.

    for i, d in enumerate(tqdm(dims, desc="Processing FNN", leave=False)):
        max_l = len(x) - d * tlag
        emb = embed_time_series(x, d, tlag)
        if emb.shape[0] < 2:
            pct[i] = np.nan
            continue
        tree = KDTree(emb)
        number_false = 0
        for idx in range(max_l):
            # nearest neighbor excluding itself
            dist, nn_idx = tree.query(emb[idx], k=2)
            nearest_d = dist[1]
            nn = nn_idx[1]
            if nn >= max_l or nearest_d <= 0:
                # degenerate case
                test1 = 1.0
                test2 = 0.0
            else:
                test1 = abs(x[idx + d * tlag] - x[nn + d * tlag]) / nearest_d
                test2 = abs(x[idx + d * tlag] - x[nn + d * tlag]) / (Ra if Ra > 0 else 1e-12)
            number_false += (test1 >= 15) or (test2 >= 2)
        pct[i] = 100.0 * number_false / max_l
    return dims, pct

# ---- Orchestrator: average across recordings and features -----------------
def fnn_average_across_features(
    df_all: pd.DataFrame,
    features: List[str],
    tau: int,
    min_dim: int = 1,
    max_dim: int = 10,
    sample_n: int = 10,
    strategy: str = "longest",
    seed: int = 42,
    plot: bool = True,
    ncols: int = 2,
    title_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For each feature:
      - pick `sample_n` recordings with length >= (max_dim+1)*tau
      - run FNN per recording
      - average %FNN across recordings (mean ± SEM)
    Also returns a 'Combined' curve (mean across per-feature means).
    """
    results: Dict[str, Any] = {
        "per_feature_curves": {},   # feature -> list of %FNN arrays (one per recording)
        "per_feature_avg":   {},    # feature -> DataFrame [dim, fnn_mean, fnn_sem]
        "combined_avg":      None,  # DataFrame or None
        "summary_lines":     [],
    }

    dims_ref = np.arange(min_dim, max_dim + 1, dtype=int)
    min_len_required = (max_dim + 1) * tau

    for feat in features:
        rids = _pick_recordings(df_all, feat, min_len_required, sample_n, strategy, seed)
        if not rids:
            results["summary_lines"].append(f"[{feat}] No recordings with ≥ {(max_dim+1)*tau} samples. Skipped.")
            continue

        curves = []
        for rid in rids:
            g = df_all[df_all["recording_id"] == rid]
            s = pd.to_numeric(g[feat], errors="coerce").dropna().values
            d, p = fnn(s, tlag=tau, min_dimension=min_dim, max_dimension=max_dim)
            # guard: if returned NaNs because borderline length, skip
            if np.all(~np.isfinite(p)):
                continue
            curves.append(p)

        if not curves:
            results["summary_lines"].append(f"[{feat}] FNN produced no valid curves. Skipped.")
            continue

        mat = np.vstack(curves)
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1) / math.sqrt(np.sum(np.isfinite(mat[:, 0])))
        df_avg = pd.DataFrame({"dim": dims_ref, "fnn_mean": mean, "fnn_sem": sem})

        results["per_feature_curves"][feat] = curves
        results["per_feature_avg"][feat]    = df_avg
        results["summary_lines"].append(f"[{feat}] {len(curves)} rec(s); dims {min_dim}..{max_dim}")

    # combined average (equal-weight across features)
    good = [df for df in results["per_feature_avg"].values() if df is not None and not df.empty]
    if good:
        stack = np.stack([df["fnn_mean"].to_numpy() for df in good], axis=0)
        comb_mean = np.nanmean(stack, axis=0)
        comb_sem  = np.nanstd(stack, axis=0, ddof=1) / math.sqrt(stack.shape[0])
        results["combined_avg"] = pd.DataFrame({"dim": dims_ref, "fnn_mean": comb_mean, "fnn_sem": comb_sem})
        results["summary_lines"].append(f"[Combined] Averaged across {len(good)} feature-averages.")
    else:
        results["summary_lines"].append("No per-feature FNN averages to combine.")

    if plot:
        _plot_fnn_subplots(results["per_feature_avg"], results["combined_avg"], ncols=ncols, title_suffix=title_suffix)

    return results

# ---- simple plotting (kept local to avoid viz_utils circular import) ------
def _plot_fnn_subplots(per_feature_avg: Dict[str, pd.DataFrame],
                       combined_avg: Optional[pd.DataFrame],
                       ncols: int = 2,
                       title_suffix: Optional[str] = None):
    import matplotlib.pyplot as plt

    items = list(per_feature_avg.items())
    if combined_avg is not None:
        items.append(("Combined", combined_avg))

    if not items:
        return

    n = len(items)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows), squeeze=False)
    axit = iter(axes.ravel())

    for name, df in items:
        ax = next(axit)
        ax.plot(df["dim"], df["fnn_mean"], marker="o", label="Mean %FNN")
        ax.fill_between(df["dim"],
                        df["fnn_mean"] - df["fnn_sem"],
                        df["fnn_mean"] + df["fnn_sem"],
                        alpha=0.2, label="± SEM")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("% False Nearest Neighbours")
        title = f"{name}" + (f" · {title_suffix}" if title_suffix else "")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best")

    # hide any extra axes
    for ax in axit:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


# ---------------- Cross-AMI  ----------------

def _safe_minmax_scale(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=float)
    lo, hi = np.nanmin(a[finite]), np.nanmax(a[finite])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=float)
    out = (a - lo) / (hi - lo)
    out[~finite] = np.nan
    return out

def cross_ami(timeseries1, timeseries2, min_lag: int, max_lag: int) -> np.ndarray:
    """
    Cross Average Mutual Information (X-AMI) between timeseries1 and timeseries2.

    Returns an (L,2) array: column0 = lag, column1 = ami.
    """
    # convert inputs to 1D numpy arrays
    if isinstance(timeseries1, (pd.Series, pd.DataFrame)):
        timeseries1 = timeseries1.values.flatten()
    if isinstance(timeseries2, (pd.Series, pd.DataFrame)):
        timeseries2 = timeseries2.values.flatten()
    x = np.asarray(timeseries1).astype(float)
    y = np.asarray(timeseries2).astype(float)

    # keep only indices where both finite (aligned)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    length = min(len(x), len(y))
    if length == 0:
        return np.column_stack((np.arange(0), np.zeros(0)))

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(max(1, min_lag), max_lag + 1)
    else:
        lag = np.arange(0, max(1, length // 2))

    # normalize
    x = _safe_minmax_scale(x)
    y = _safe_minmax_scale(y)

    ami_values = np.zeros(len(lag), dtype=float)

    for i in range(len(lag)):
        L = int(lag[i])
        N = length - L
        if N <= 2:
            ami_values[i] = 0.0
            continue

        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.nanvar(x[:N], ddof=1) == 0 or np.nanvar(y[L:], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        xw = x[:N]
        yw = y[L:]

        ami_sum = 0.0
        for k1 in range(1, k + 1):
            x_lo, x_hi = (k1 - 1) / k, k1 / k
            mask_x = (xw > x_lo) & (xw <= x_hi)
            if not mask_x.any():
                continue
            px1 = mask_x.sum() / N

            for k2 in range(1, k + 1):
                y_lo, y_hi = (k2 - 1) / k, k2 / k
                mask_y = (yw > y_lo) & (yw <= y_hi)
                if not mask_y.any():
                    continue
                py2 = mask_y.sum() / N

                ppp = (mask_x & mask_y).sum() / N
                if ppp > 0 and px1 > 0 and py2 > 0:
                    ami_sum += ppp * np.log2(ppp / (px1 * py2))

        ami_values[i] = ami_sum

    return np.column_stack((lag, ami_values))


# ---------- helpers to batch across recordings / pair averaging -----------
def average_cross_ami_across_recordings(
    df_all: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    min_lag: int = 1,
    max_lag: int = 100,
    recording_col: str = "recording_id",
    require_min_length: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute cross_ami for each recording in df_all and return averaged AMI curve.

    Returns dict:
      {
        "lags": np.ndarray,
        "per_recording": { recording_id: ami_array (1D) , ... },
        "mean": np.ndarray (mean across recordings),
        "sem": np.ndarray (sem across recordings),
        "n_recordings": int
      }

    Requirements:
      - df_all must contain recording_col, and columns feat_x, feat_y.
    """
    if feat_x not in df_all.columns or feat_y not in df_all.columns:
        raise ValueError(f"Features not present in df_all: {feat_x}, {feat_y}")

    per_rec = {}
    recs = df_all.groupby(recording_col, sort=False)
    outer = recs if not progress else tqdm(recs, desc="Cross-AMI recordings")

    computed = []
    lags = None
    for rid, g in outer:
        xa = pd.to_numeric(g[feat_x], errors="coerce").values
        ya = pd.to_numeric(g[feat_y], errors="coerce").values
        # aligned finite region inside cross_ami will trim to overlapping finite region
        if require_min_length is not None:
            finite = np.isfinite(xa) & np.isfinite(ya)
            if finite.sum() < require_min_length:
                continue
        arr = cross_ami(xa, ya, min_lag=min_lag, max_lag=max_lag)
        if arr is None or arr.size == 0:
            continue
        per_rec[rid] = arr[:, 1]
        computed.append(arr[:, 1])
        if lags is None:
            lags = arr[:, 0]

    if not computed:
        return {"lags": np.array([], dtype=int), "per_recording": {}, "mean": np.array([]), "sem": np.array([]), "n_recordings": 0}

    stacked = np.vstack(computed)  # n_rec x n_lags
    mean = np.nanmean(stacked, axis=0)
    sem = np.nanstd(stacked, ddof=1, axis=0) / np.sqrt(stacked.shape[0])

    return {"lags": lags.astype(int), "per_recording": per_rec, "mean": mean, "sem": sem, "n_recordings": stacked.shape[0]}


# ---------- plotting / saving helper ------------------------------------

# ---------- core cross-AMI implementation (ported) -----------------------
def _safe_minmax_scale(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=float)
    lo, hi = np.nanmin(a[finite]), np.nanmax(a[finite])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=float)
    out = (a - lo) / (hi - lo)
    out[~finite] = np.nan
    return out

def cross_ami(timeseries1, timeseries2, min_lag: int, max_lag: int) -> np.ndarray:
    """
    Cross Average Mutual Information (X-AMI) between timeseries1 and timeseries2.

    Returns an (L,2) array: column0 = lag, column1 = ami.
    """
    # convert inputs to 1D numpy arrays
    if isinstance(timeseries1, (pd.Series, pd.DataFrame)):
        timeseries1 = timeseries1.values.flatten()
    if isinstance(timeseries2, (pd.Series, pd.DataFrame)):
        timeseries2 = timeseries2.values.flatten()
    x = np.asarray(timeseries1).astype(float)
    y = np.asarray(timeseries2).astype(float)

    # keep only indices where both finite (aligned)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    length = min(len(x), len(y))
    if length == 0:
        return np.column_stack((np.arange(0), np.zeros(0)))

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(max(1, min_lag), max_lag + 1)
    else:
        lag = np.arange(0, max(1, length // 2))

    # normalize
    x = _safe_minmax_scale(x)
    y = _safe_minmax_scale(y)

    ami_values = np.zeros(len(lag), dtype=float)

    for i in range(len(lag)):
        L = int(lag[i])
        N = length - L
        if N <= 2:
            ami_values[i] = 0.0
            continue

        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.nanvar(x[:N], ddof=1) == 0 or np.nanvar(y[L:], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        xw = x[:N]
        yw = y[L:]

        ami_sum = 0.0
        for k1 in range(1, k + 1):
            x_lo, x_hi = (k1 - 1) / k, k1 / k
            mask_x = (xw > x_lo) & (xw <= x_hi)
            if not mask_x.any():
                continue
            px1 = mask_x.sum() / N

            for k2 in range(1, k + 1):
                y_lo, y_hi = (k2 - 1) / k, k2 / k
                mask_y = (yw > y_lo) & (yw <= y_hi)
                if not mask_y.any():
                    continue
                py2 = mask_y.sum() / N

                ppp = (mask_x & mask_y).sum() / N
                if ppp > 0 and px1 > 0 and py2 > 0:
                    ami_sum += ppp * np.log2(ppp / (px1 * py2))

        ami_values[i] = ami_sum

    return np.column_stack((lag, ami_values))


# ---------- helpers to batch across recordings / pair averaging -----------
def average_cross_ami_across_recordings(
    df_all: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    min_lag: int = 1,
    max_lag: int = 100,
    recording_col: str = "recording_id",
    require_min_length: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute cross_ami for each recording in df_all and return averaged AMI curve.

    Returns dict:
      {
        "lags": np.ndarray,
        "per_recording": { recording_id: ami_array (1D) , ... },
        "mean": np.ndarray (mean across recordings),
        "sem": np.ndarray (sem across recordings),
        "n_recordings": int
      }

    Requirements:
      - df_all must contain recording_col, and columns feat_x, feat_y.
    """
    if feat_x not in df_all.columns or feat_y not in df_all.columns:
        raise ValueError(f"Features not present in df_all: {feat_x}, {feat_y}")

    per_rec = {}
    recs = df_all.groupby(recording_col, sort=False)
    outer = recs if not progress else tqdm(recs, desc="Cross-AMI recordings")

    computed = []
    lags = None
    for rid, g in outer:
        xa = pd.to_numeric(g[feat_x], errors="coerce").values
        ya = pd.to_numeric(g[feat_y], errors="coerce").values
        # aligned finite region inside cross_ami will trim to overlapping finite region
        if require_min_length is not None:
            finite = np.isfinite(xa) & np.isfinite(ya)
            if finite.sum() < require_min_length:
                continue
        arr = cross_ami(xa, ya, min_lag=min_lag, max_lag=max_lag)
        if arr is None or arr.size == 0:
            continue
        per_rec[rid] = arr[:, 1]
        computed.append(arr[:, 1])
        if lags is None:
            lags = arr[:, 0]

    if not computed:
        return {"lags": np.array([], dtype=int), "per_recording": {}, "mean": np.array([]), "sem": np.array([]), "n_recordings": 0}

    stacked = np.vstack(computed)  # n_rec x n_lags
    mean = np.nanmean(stacked, axis=0)
    sem = np.nanstd(stacked, ddof=1, axis=0) / np.sqrt(stacked.shape[0])

    return {"lags": lags.astype(int), "per_recording": per_rec, "mean": mean, "sem": sem, "n_recordings": stacked.shape[0]}


# ---------- plotting / saving helper ------------------------------------
def plot_and_save_cross_ami(lags: np.ndarray, mean_curve: np.ndarray, sem: Optional[np.ndarray] = None,
                            title: Optional[str] = None, out_path: Optional[Path] = None, show: bool = True):
    """
    Plot mean cross-AMI with optional SEM ribbon and optionally save figure.
    """
    plt.figure(figsize=(7, 3.5))
    plt.plot(lags, mean_curve, marker="o", linewidth=1)
    if sem is not None and len(sem) == len(mean_curve):
        plt.fill_between(lags, mean_curve - sem, mean_curve + sem, alpha=0.2)
    if title:
        plt.title(title)
    plt.xlabel("Lag (samples)")
    plt.ylabel("Cross AMI")
    plt.grid(True)
    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()

