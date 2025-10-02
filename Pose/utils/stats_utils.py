# utils/stats_utils.py
"""Statistical helpers for pose linear metrics.

- Uses rpy2 -> lmerTest + emmeans for mixed models (if available).
- No session_order requirement. If missing, defaults sensible values.
- Exposes:
    - discover_linear_files(root)
    - load_session_csvs(list_of_paths)
    - run_rpy2_lmer(df, dv, feature_label)
    - build_table_with_emmeans(df, out_tex, figs_dir)
    - barplot_ax(...)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Any
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import textwrap
from collections import defaultdict

# rpy2 optional but we activate conversions if available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    # activate conversion rules (fix ContextVar warning)
    try:
        pandas2ri.activate()
    except Exception:
        # best-effort; localconverter still used in functions
        pass
    _HAVE_RPY2 = True
except Exception:
    _HAVE_RPY2 = False

# ---- defaults ----
COND_ORDER = ["L", "M", "H"]
DESIRED_ORDER = ["Vel", "Acc", "Rms"]

NON_METRIC_COLS = {"file","participant","participant_id","condition","window_index",
                   "window_start_s","window_end_s","source","method","_source_file","session"}

# -----------------------
# small helpers
# -----------------------
def split_metric_name(name: str) -> Tuple[str, str]:
    pretty = (name.replace("avg","Average").replace("dist","Amplitude")
                   .replace("mean_abs_","").replace("mean_abs","")
                   .replace("_"," ").title())
    parts = pretty.split()
    return (pretty,"") if len(parts) < 2 else (" ".join(parts[:-1]), parts[-1])

def fmt(beta: float | None, p: float | None) -> Tuple[str,str]:
    b = "--" if beta is None or (isinstance(beta,float) and not np.isfinite(beta)) else f"$\\beta = {beta:.3f}$"
    if p is None or (isinstance(p,float) and not np.isfinite(p)):
        return b, "--"
    return b, (r"$p < .001$" if p < 0.001 else f"$p = {p:.3f}$")

# -----------------------
# rpy2 wrapper: lmer + emmeans
# -----------------------
def _robust_ci_cols(ci_pd: pd.DataFrame) -> Tuple[str|None, str|None]:
    cand = list(ci_pd.columns)
    lower = next((c for c in cand if c.lower().startswith("lower")), None)
    upper = next((c for c in cand if c.lower().startswith("upper")), None)
    if lower and upper:
        return lower, upper
    lower = next((c for c in cand if "lcl" in c.lower()), None)
    upper = next((c for c in cand if "ucl" in c.lower()), None)
    return lower, upper

def run_rpy2_lmer(df: pd.DataFrame, dv: str, adjust: str = "none") -> Tuple[Dict[Tuple[str,str], float], Dict[Tuple[str,str], float], Dict[str,float], Dict[str,Tuple[float,float]]]:
    """
    Fit: dv ~ condition + window_index + (1 | participant)
    Returns:
      pairs_est (lo,hi): estimate (hi - lo)
      pairs_p   (lo,hi): p-value
      means: emmeans per condition
      cis: (lower, upper) per condition (approx if confint fails)
    """
    if not _HAVE_RPY2:
        raise ImportError("rpy2 not available. Install rpy2 and R packages lmerTest/emmeans.")

    # local imports
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    # Prepare dataframe: ensure needed cols exist, set defaults if missing
    d = df.copy()
    if "participant_id" in d.columns and "participant" not in d.columns:
        d = d.rename(columns={"participant_id":"participant"})
    if "participant" not in d.columns:
        raise ValueError("Data must contain 'participant' column")
    # ensure window_index exists
    if "window_index" not in d.columns:
        d["window_index"] = 0
    # condition
    if "condition" not in d.columns:
        raise ValueError("Data must contain 'condition' column")
    # keep only necessary cols
    cols = ["participant","condition","window_index", dv]
    dat = d[cols].dropna().copy()
    dat = dat.rename(columns={dv: "dv"})
    dat["participant"] = dat["participant"].astype(str)
    dat["condition"] = pd.Categorical(dat["condition"].astype(str).str.strip().str.upper(), categories=COND_ORDER, ordered=True)

    # center/scale window_index
    w = pd.to_numeric(dat["window_index"], errors="coerce")
    if w.isna().all():
        dat["widx_c"] = 0.0
    else:
        dat["widx_c"] = (w - np.nanmean(w)) / (np.nanstd(w) if np.nanstd(w) != 0 else 1.0)

    # push to R using localconverter
    with localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["dat"] = robjects.conversion.py2rpy(dat)

    # load packages and fit
    robjects.r('suppressPackageStartupMessages(library(lme4))')
    robjects.r('suppressPackageStartupMessages(library(lmerTest))')
    robjects.r('suppressPackageStartupMessages(library(emmeans))')
    # robust control
    robjects.r('ctrl <- lme4::lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000))')

    # formula uses window_index (we already created widx_c but emmeans on condition unaffected)
    rcode = """
        dat$participant <- factor(dat$participant)
        dat$condition <- factor(dat$condition, levels = c("L","M","H"), ordered = TRUE)
        dat$window_index <- as.numeric(dat$window_index)
        fit <- lmerTest::lmer(dv ~ condition + window_index + (1 | participant), data = dat, control = ctrl)
        emm <- emmeans::emmeans(fit, specs = ~ condition)
    """
    robjects.r(rcode)

    # pull emmeans, confint, pairs
    emm_df_r = robjects.r("as.data.frame(emm)")
    try:
        ci_df_r = robjects.r("as.data.frame(confint(emm, level = 0.95))")
    except Exception:
        ci_df_r = None
    pwc_df_r = robjects.r(f"as.data.frame(pairs(emm, adjust = '{adjust}'))")

    with localconverter(robjects.default_converter + pandas2ri.converter):
        emm_pd = robjects.conversion.rpy2py(emm_df_r)
        pwc_pd = robjects.conversion.rpy2py(pwc_df_r)
        ci_pd = robjects.conversion.rpy2py(ci_df_r) if ci_df_r is not None else pd.DataFrame()

    # means
    means = {str(r["condition"]): float(r["emmean"]) for _, r in emm_pd.iterrows()}

    # cis: robust detection
    cis = {}
    if not ci_pd.empty and "condition" in ci_pd.columns:
        lower_col, upper_col = _robust_ci_cols(ci_pd)
        if lower_col and upper_col:
            for _, r in ci_pd.iterrows():
                cis[str(r["condition"])] = (float(r[lower_col]), float(r[upper_col]))
    else:
        # fallback using SE from emm_pd
        se_col = next((c for c in emm_pd.columns if c.lower() in ("se","stderr","std.error")), None)
        if se_col:
            for _, r in emm_pd.iterrows():
                cond = str(r["condition"])
                mean = float(r["emmean"]); se = float(r[se_col])
                cis[cond] = (mean - 1.96*se, mean + 1.96*se)
        else:
            for cond in means.keys():
                cis[cond] = (float("nan"), float("nan"))

    # pairwise p and estimates
    pcol = "p.value" if "p.value" in pwc_pd.columns else next((c for c in pwc_pd.columns if c.lower().startswith("p")), None)
    pairs_est: Dict[Tuple[str,str], float] = {}
    pairs_p: Dict[Tuple[str,str], float] = {}
    order = {k: i for i,k in enumerate(COND_ORDER)}
    for _, r in pwc_pd.iterrows():
        contrast = str(r.get("contrast", ""))
        contrast = contrast.replace("–","-").replace(" - ","-")
        parts = [p.strip() for p in contrast.split("-")]
        if len(parts) != 2:
            continue
        left = next((lvl for lvl in COND_ORDER if lvl in parts[0]), None)
        right = next((lvl for lvl in COND_ORDER if lvl in parts[1]), None)
        if left is None or right is None or left == right:
            continue
        est_lr = float(r["estimate"]) if "estimate" in r and pd.notnull(r["estimate"]) else np.nan
        pv = float(r[pcol]) if (pcol and pd.notnull(r[pcol])) else np.nan
        lo, hi = (left, right) if order[left] < order[right] else (right, left)
        est_hi_minus_lo = est_lr if (left == hi and right == lo) else -est_lr
        pairs_est[(lo,hi)] = est_hi_minus_lo
        pairs_p[(lo,hi)] = pv

    return pairs_est, pairs_p, means, cis

# --------------------------
# plotting helper
# --------------------------
def barplot_ax(ax, means: List[float], sems: List[float], pvals: List[float],
               ylabel: str, metric_name: str,
               colors: List[str] | None = None,
               bar_width: float = 0.80,
               ylim_padding: Tuple[float,float] = (0.4, 0.1)):
    if colors is None:
        colors = ['#4575b4', '#ffffbf', '#d73027']
    import numpy as _np
    x = _np.arange(len(means))
    ax.bar(x, means, yerr=sems, capsize=4, color=colors, width=bar_width, edgecolor="black", edgewidth=4)
    lowers = [m - (s if not _np.isnan(s) else 0) for m,s in zip(means,sems)]
    uppers = [m + (s if not _np.isnan(s) else 0) for m,s in zip(means,sems)]
    y_min = min(lowers); y_max = max(uppers)
    y_span = y_max - y_min if y_max > y_min else 1.0
    pairs = [(0,1,pvals[0]), (0,2,pvals[1]), (1,2,pvals[2])]
    sig_pairs = [(i,j,p) for (i,j,p) in pairs if (p is not None and not np.isnan(p) and p < 0.05)]
    sig_pairs = sorted(sig_pairs, key=lambda t: (t[1]-t[0]))
    h_step = 0.2 * y_span; line_h = 0.03 * y_span; y0 = y_max + 0.04 * y_span
    for idx, (i,j,p) in enumerate(sig_pairs):
        y = y0 + idx * h_step
        ax.plot([x[i], x[i], x[j], x[j]], [y, y+line_h, y+line_h, y], lw=1.5, color='black', clip_on=False)
        stars = '***' if p < .001 else '**' if p < .01 else '*'
        ax.text((x[i]+x[j])/2, y+0.25*line_h, stars, ha='center', va='bottom', fontsize=13, fontweight='bold', color='black', clip_on=False)
    ax.set_xlim(-0.5, len(means)-0.5); ax.set_xticks([]); ax.set_ylabel("\n".join(textwrap.wrap(ylabel, width=25)), weight='bold', fontsize=12)
    ax.set_ylim(y_min - ylim_padding[0]*y_span, y_max + ylim_padding[1]*y_span + len(sig_pairs)*h_step)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_linewidth(1.4)
    ax.tick_params(axis='y', width=1.3, labelsize=11)
    for lab in ax.get_yticklabels(): lab.set_fontweight('bold')

# --------------------------
# discovery + loading
# --------------------------
def discover_linear_files(root: Path = Path("data/processed_data")) -> Dict[str, List[Path]]:
    sessions = {}
    root = Path(root)
    for session_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        lm_dir = session_dir / "linear_metrics"
        if lm_dir.exists() and any(lm_dir.glob("*.csv")):
            sessions[session_dir.name] = sorted(lm_dir.glob("*.csv"))
    return sessions

def load_session_csvs(files: List[Path]) -> pd.DataFrame:
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "participant" not in df.columns and "participant_id" in df.columns:
                df = df.rename(columns={"participant_id":"participant"})
            df["_source_file"] = str(f.name)
            parts.append(df)
        except Exception as e:
            print(f"[WARN] failed to load {f}: {e}")
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

# --------------------------
# table builder + plots (main)
# --------------------------
def build_table_with_emmeans(df: pd.DataFrame, out_tex: str | Path, figs_dir: str | Path):
    out_tex = Path(out_tex)
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # prepare df
    df = df.copy()
    df["condition"] = df["condition"].astype(str).str.strip().str.upper()
    df = df[df["condition"].isin(COND_ORDER)].copy()
    if "window_index" not in df.columns:
        df["window_index"] = 0

    metric_cols = [c for c in df.columns if c not in NON_METRIC_COLS and pd.api.types.is_numeric_dtype(df[c])]
    grouped = defaultdict(list)
    modeled = skipped = 0

    for metric in metric_cols:
        ser = df[metric].dropna()
        if ser.empty:
            skipped += 1; continue
        if not {"L","M","H"}.issubset(set(df.dropna(subset=[metric])["condition"].unique())):
            skipped += 1; continue

        tmp = df[["participant","condition","window_index",metric]].dropna().rename(columns={metric:"dv"})
        try:
            pairs_est, pairs_p, means, cis = run_rpy2_lmer(tmp, "dv", adjust="none")
        except Exception as e:
            print(f"[WARN] model failed for {metric}: {e}")
            skipped += 1
            continue

        b_m = pairs_est.get(("L","M"), np.nan); p_m = pairs_p.get(("L","M"), np.nan)
        b_h = pairs_est.get(("L","H"), np.nan); p_h = pairs_p.get(("L","H"), np.nan)
        b_hm= pairs_est.get(("M","H"), np.nan); p_hm= pairs_p.get(("M","H"), np.nan)

        Bm, Pm = fmt(b_m, p_m)
        Bh, Ph = fmt(b_h, p_h)
        Bhm, Phm = fmt(b_hm, p_hm)

        region, metric_type = split_metric_name(metric)
        grouped[region].append((metric_type, Bm, Pm, Bh, Ph, Bhm, Phm))
        modeled += 1

        # plot per metric
        conds = ["L","M","H"]
        mean_vals = [means.get(c, float("nan")) for c in conds]
        sems = []
        for c in conds:
            if c in cis and cis[c] is not None:
                lo, hi = cis[c]
                sems.append((hi - lo) / 3.92 if (not pd.isna(lo) and not pd.isna(hi)) else float("nan"))
            else:
                sems.append(float("nan"))
        pvals_for_plot = [p_m, p_h, p_hm]
        fig, ax = plt.subplots(figsize=(4,5))
        barplot_ax(ax, mean_vals, sems, pvals_for_plot, ylabel=metric.replace("_"," ").title(), metric_name=metric)
        ax.set_title(f"{metric.replace('_',' ').title()}", fontsize=11, weight="bold")
        out_svg = figs_dir / f"{metric}.svg"
        fig.savefig(out_svg, bbox_inches="tight")
        plt.close(fig)

    # write latex table
    lines = [
        r"\begin{tabular}{llcc|cc|cc}",
        r"\toprule",
        r"Region & Metric & $\beta_{\text{M}}$ & $p_{\text{M}}$ & $\beta_{\text{H}}$ & $p_{\text{H}}$ & $\beta_{\text{H--M}}$ & $p_{\text{H--M}}$ \\",
        r"\midrule"
    ]
    for region in sorted(grouped.keys()):
        rows = grouped[region]
        rows.sort(key=lambda x: DESIRED_ORDER.index(x[0]) if x[0] in DESIRED_ORDER else len(DESIRED_ORDER))
        first = True
        for (metric_type, Bm, Pm, Bh, Ph, Bhm, Phm) in rows:
            region_label = f"\\multirow{{{len(rows)}}}{{*}}{{{region}}}" if first else ""
            lines.append(f"{region_label} & {metric_type} & {Bm} & {Pm} & {Bh} & {Ph} & {Bhm} & {Phm} \\\\")
            first = False
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_tex} | modeled={modeled}, skipped={skipped}")
    return modeled, skipped
