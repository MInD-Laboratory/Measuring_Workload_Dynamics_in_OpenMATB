"""Notebook and analysis utilities for pose data exploration.

Provides convenience functions for interactive data analysis in Jupyter notebooks,
including file discovery, data manipulation, visualization, and statistical analysis.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List
from utils.stats_utils import pretty_metric, fmt

# ---------- Output file discovery and status checking ----------
def outputs_exist(base: str | Path) -> dict:
    """Check which pipeline outputs exist on disk.

    Scans the output directory structure to determine what analysis
    results are available from previous pipeline runs.

    Args:
        base: Base output directory path

    Returns:
        Dictionary with structure:
        {
            'per_frame': {source: [list_of_csv_files]},
            'linear': {source: boolean},
            'any_per_frame': boolean,
            'any_linear': boolean
        }

    Note:
        Useful for notebooks to determine what data is available
        without re-running expensive pipeline stages.
    """
    base = Path(base)

    # Check for per-frame feature files (individual trial CSVs)
    per_frame = {
        "procrustes_global": list((base / "features" / "per_frame" / "procrustes_global").glob("*.csv")),
        "procrustes_participant": list((base / "features" / "per_frame" / "procrustes_participant").glob("*.csv")),
        "original": list((base / "features" / "per_frame" / "original").glob("*.csv")),
    }

    # Check for aggregated linear metric files
    linear = {
        "procrustes_global": (base / "linear_metrics" / "procrustes_global_linear.csv").exists(),
        "procrustes_participant": (base / "linear_metrics" / "procrustes_participant_linear.csv").exists(),
        "original": (base / "linear_metrics" / "original_linear.csv").exists(),
    }

    # Summary flags for quick checking
    any_per_frame = any(len(v) > 0 for v in per_frame.values())
    any_linear = any(linear.values())

    return {"per_frame": per_frame, "linear": linear,
            "any_per_frame": any_per_frame, "any_linear": any_linear}

# ---------- File selection utilities ----------
def pick_norm_file(out_base: str | Path, sample_norm: str | Path | None = None) -> Path:
    """Select a normalized CSV file for analysis.

    Provides automatic file selection when exploring pipeline outputs.
    If no specific file is provided, picks the first available normalized file.

    Args:
        out_base: Base output directory
        sample_norm: Specific file path, or None for automatic selection

    Returns:
        Path to a normalized CSV file

    Raises:
        FileNotFoundError: If no normalized files exist

    Note:
        Useful for notebooks where you want to quickly load some
        example data without specifying exact filenames.
    """
    # Use specific file if provided
    if sample_norm:
        return Path(sample_norm)

    # Otherwise, find first available normalized file
    norm_dir = Path(out_base) / "norm_screen"
    files = sorted(norm_dir.glob("*_norm.csv"))

    if not files:
        raise FileNotFoundError(f"No normalized CSVs in {norm_dir}. Run the pipeline first.")

    return files[0]

# ---------- Column access utilities ----------
def find_col(df: pd.DataFrame, axis: str, i: int) -> str | None:
    """Find column name for a landmark coordinate.

    Handles case variations in column naming (x1 vs X1).

    Args:
        df: DataFrame to search
        axis: Coordinate axis ('x' or 'y')
        i: Landmark index number

    Returns:
        Column name if found, None otherwise

    Example:
        >>> find_col(df, 'x', 37)  # Returns 'x37' or 'X37' if present
    """
    c1, c2 = f"{axis}{i}", f"{axis.upper()}{i}"  # Try both cases
    return c1 if c1 in df.columns else (c2 if c2 in df.columns else None)


def series_num(df: pd.DataFrame, axis: str, i: int, n: int) -> pd.Series:
    """Extract landmark coordinate as numeric series.

    Safely extracts and converts landmark coordinates to float,
    returning NaN series if column doesn't exist.

    Args:
        df: DataFrame containing landmark data
        axis: Coordinate axis ('x' or 'y')
        i: Landmark index number
        n: Length for NaN series if column missing

    Returns:
        Numeric series with coordinate values or NaNs
    """
    c = find_col(df, axis, i)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series([np.nan]*n)

# ---------- Data slicing utilities ----------
def slice_first_seconds(df: pd.DataFrame, fps: int, seconds: int) -> pd.DataFrame:
    """Extract first N seconds of data.

    Useful for quick analysis of data beginnings or creating
    consistent-length samples for comparison.

    Args:
        df: DataFrame with time-series data
        fps: Frames per second (sampling rate)
        seconds: Number of seconds to extract

    Returns:
        DataFrame containing only the first N seconds of data

    Note:
        Resets index to start from 0 for convenience.
    """
    n = len(df)
    end = min(n, fps * seconds)  # Don't exceed available data
    return df.iloc[:end].reset_index(drop=True)

# ---------- Metrics analysis and plotting utilities ----------
# Columns that contain metadata rather than actual measurements
META_COLS = {"source","participant","condition","window_index","t_start_frame","t_end_frame"}

def ensure_condition_order(df: pd.DataFrame, cond_order=("L","M","H")) -> pd.DataFrame:
    """Ensure condition column has consistent ordering.

    Converts condition column to ordered categorical for consistent
    plotting and statistical analysis.

    Args:
        df: DataFrame with 'condition' column
        cond_order: Desired order of conditions (default: Low, Medium, High)

    Returns:
        DataFrame with condition as ordered categorical

    Note:
        Modifies the input DataFrame in-place for the condition column.
    """
    if "condition" in df.columns:
        df["condition"] = pd.Categorical(df["condition"], categories=list(cond_order), ordered=True)
    return df

def candidate_metric_cols(df: pd.DataFrame) -> List[str]:
    """Identify numeric columns suitable for analysis.

    Filters out metadata columns and prioritizes derived metrics
    (velocity, acceleration, RMS) over raw measurements.

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names, with priority metrics first

    Note:
        Priority is given to time-domain metrics (_mean_abs_vel,
        _mean_abs_acc, _rms) as these often show clearer condition effects.
    """
    # Find numeric columns that aren't metadata
    num_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

    priority, others = [], []
    for c in num_cols:
        lc = c.lower()
        # Prioritize derived time-domain metrics
        if lc.endswith("_mean_abs_vel") or lc.endswith("_mean_abs_acc") or lc.endswith("_rms"):
            priority.append(c)
        else:
            others.append(c)

    return priority + others  # Priority metrics first

def default_metric(cols: List[str]) -> str | None:
    """Select a sensible default metric for analysis.

    Chooses metrics that typically show clear condition effects,
    preferring blink and mouth RMS metrics.

    Args:
        cols: List of available metric column names

    Returns:
        Recommended metric name, or None if no columns available

    Note:
        Selection priority:
        1. Specific high-impact metrics (blink_aperture_rms, etc.)
        2. Any RMS metric
        3. First available metric
    """
    if not cols:
        return None

    lowers = [c.lower() for c in cols]

    # Preferred metrics known to show strong condition effects
    prefs = [
        "blink_aperture_rms", "mouth_aperture_rms", "center_face_magnitude_rms",
        "blink_aperture_mean_abs_vel", "mouth_aperture_mean_abs_vel",
    ]

    # First, try exact matches with preferred metrics
    for exact in prefs:
        if exact in lowers:
            return cols[lowers.index(exact)]

    # Second, try any RMS metric (good for variability analysis)
    for i, lc in enumerate(lowers):
        if lc.endswith("_rms"):
            return cols[i]

    # Finally, just use the first available metric
    return cols[0]

def sem(series) -> float:
    """Calculate standard error of the mean.

    Args:
        series: Numeric data series

    Returns:
        Standard error of the mean

    Note:
        Uses sample standard deviation (ddof=1) and protects
        against division by zero with max(count, 1).
    """
    s = pd.Series(series).astype(float)
    return s.std(ddof=1) / np.sqrt(max(s.count(), 1))

def bar_by_condition(df: pd.DataFrame, metric: str, cond_order=("L","M","H"),
                     colors=("#1f77b4","#f1c40f","#8B0000"), title_suffix: str = ""):
    """Create bar plot comparing metric across conditions.

    Generates publication-ready bar plot with error bars showing
    mean ± SEM for each experimental condition.

    Args:
        df: DataFrame with 'condition' column and metric data
        metric: Name of the metric column to plot
        cond_order: Order of conditions (default: L, M, H)
        colors: Colors for each condition bar
        title_suffix: Additional text for plot title

    Returns:
        Tuple of (figure, axis) objects

    Note:
        - Error bars represent standard error of the mean (SEM)
        - Values are displayed on top of each bar
        - Grid lines help with value reading
    """
    # Ensure consistent condition ordering
    df = ensure_condition_order(df, cond_order)

    # Calculate means and standard errors by condition
    grouped = df.groupby("condition")[metric].agg(["mean", sem]).reindex(cond_order)
    idx = np.arange(len(cond_order))  # Bar positions
    means = grouped["mean"].to_numpy(dtype=float)
    errs  = grouped["sem"].to_numpy(dtype=float)

    # Create figure and plot bars
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(idx, means, yerr=errs, capsize=4, width=0.75,
           color=list(colors), edgecolor="black", alpha=0.9)

    # Configure axes
    ax.set_xticks(idx)
    ax.set_xticklabels(cond_order)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean ± SEM")

    # Set title
    ttl = f"{metric} by Condition" + (f" — {title_suffix}" if title_suffix else "")
    ax.set_title(ttl)
    ax.set_xlim(-0.5, len(cond_order)-0.5)

    # Format y-axis and add grid
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.25)

    # Add value labels on bars
    for x, m in zip(idx, means):
        if np.isfinite(m):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom",
                   fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig, ax

# ---------- Statistical utilities ----------
def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Step-down procedure that controls family-wise error rate while
    being less conservative than standard Bonferroni correction.

    Args:
        pvals: Dictionary mapping comparison labels to raw p-values

    Returns:
        Dictionary with same keys but corrected p-values

    Note:
        This is a duplicate of the function in stats_utils.py.
        Consider importing from there instead to avoid duplication.
    """
    # Sort p-values in ascending order
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected = {}

    # Apply Holm-Bonferroni formula: p_corrected = p_raw * (m - i + 1)
    for i, (lbl, p) in enumerate(items, start=1):
        corrected[lbl] = min(p * (m - i + 1), 1.0)  # Cap at 1.0

    # Return in original key order
    return {k: corrected[k] for k in pvals.keys()}

def build_rqa_table(results, out_file):
    """Build LaTeX table with β and p-values for each contrast."""
    lines = [
        r"\begin{tabular}{llcc|cc|cc}",
        r"\toprule",
        r"Data Type & Metric & $\beta_{\text{M}}$ & $p_{\text{M}}$ & "
        r"$\beta_{\text{H}}$ & $p_{\text{H}}$ & $\beta_{\text{H--M}}$ & $p_{\text{H--M}}$ \\",
        r"\midrule"
    ]
    
    for col_name in sorted(results.keys()):
        metric_results = results[col_name]
        rows = []
        
        for metric in sorted(metric_results.keys()):
            ests, pvals, _, _ = metric_results[metric]
            
            # Extract contrasts: M-L, H-L, H-M
            b_m = ests.get(("L", "M"))
            p_m = pvals.get(("L", "M"))
            b_h = ests.get(("L", "H"))
            p_h = pvals.get(("L", "H"))
            b_hm = ests.get(("M", "H"))
            p_hm = pvals.get(("M", "H"))
            
            Bm, Pm = fmt(b_m, p_m)
            Bh, Ph = fmt(b_h, p_h)
            Bhm, Phm = fmt(b_hm, p_hm)
            
            rows.append((pretty_metric(metric), Bm, Pm, Bh, Ph, Bhm, Phm))
        
        # Write rows for this data type
        first = True
        for (metric_name, Bm, Pm, Bh, Ph, Bhm, Phm) in rows:
            left = f"\\multirow{{{len(rows)}}}{{*}}{{{col_name}}}" if first else ""
            lines.append(
                f"{left} & {metric_name} & {Bm} & {Pm} & {Bh} & {Ph} & {Bhm} & {Phm} \\\\"
            )
            first = False
        lines.append(r"\midrule")
    
    lines += [r"\bottomrule", r"\end{tabular}"]
    
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Table written to: {out_file}")