from __future__ import annotations

import math
import numpy as np
import pandas as pd

# Optional R bridge
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    _HAVE_RPY2 = True
except Exception:
    _HAVE_RPY2 = False

DEFAULT_CONDITION_ORDER = ["L", "M", "H"]
DEFAULT_GROUP_COL = "participant"
DEFAULT_COND_COL = "condition"

DEFAULT_PRETTY = {
    "track_point_accuracy": "Tracking accuracy",
    "resman_point_accuracy": "Resource Management accuracy",
    "sysmon_point_accuracy": "System Monitoring accuracy",
    "comms_point_accuracy": "Communications accuracy",
}

TLX_SUBSCALES = [
    "mental_demand",
    "physical_demand",
    "time_pressure",
    "performance",
    "effort",
    "frustration",
]

def ensure_condition_order(df: pd.DataFrame, cond_col: str = DEFAULT_COND_COL, order=None) -> pd.DataFrame:
    if order is None:
        order = DEFAULT_CONDITION_ORDER
    out = df.copy()
    out[cond_col] = out[cond_col].astype(str).str.strip()
    out[cond_col] = pd.Categorical(out[cond_col], categories=order, ordered=True)
    return out

def erf(x: float) -> float:
    return math.erf(x)

def stats_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def fmt_p_latex(p: float) -> str:
    if (p is None) or (not np.isfinite(p)):
        return "NA"
    return "$< .001$" if p < 0.001 else f"{p:.3f}"

def fmt_b(beta: float) -> str:
    return "NA" if (beta is None or (not np.isfinite(beta))) else f"{beta:.3f}"

def pair_cell(beta: float, p: float) -> str:
    if (beta is None) or (not np.isfinite(beta)) or (p is None) or (not np.isfinite(p)):
        return "NA"
    return f"{fmt_b(beta)}, {fmt_p_latex(p)}"

# -----------------------
# rpy2 / R-based LMEs + emmeans
# -----------------------
def _require_rpy2():
    if not _HAVE_RPY2:
        raise ImportError("rpy2 is not available. Install rpy2 and ensure R has lmerTest and emmeans installed.")

def print_means(df: pd.DataFrame, dv: str, group: str = "condition"):
    means = df.groupby(group, observed=False)[dv].mean()
    print(f"Means for {dv}:")
    for cond, val in means.items():
        print(f"  {cond}: {val:.3f}")
    return means

def run_rpy2_lmer(df: pd.DataFrame, dv: str, feature_label: str):
    """
    Fits lmerTest::lmer and extracts:
      - pwc: {('L','M'): {'estimate': β, 'p': pval}, ...} from pairs(emmeans(...))
      - emm_means: {'L': μ_L, 'M': μ_M, 'H': μ_H}
      - cis: {'L': (lo, hi), ...} from confint(emmeans(...))
    Robust to emmeans column name variants (lower.CL / asymp.LCL / etc.).
    """
    # --- rpy2 / R setup ---
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    d = df.copy()
    # required / optional covariates
    participant_col = "participant_id" if "participant_id" in d.columns else "participant"
    if "session_order_numeric" not in d.columns: d["session_order_numeric"] = 0
    if "window_index" not in d.columns: d["window_index"] = 0

    use_cols = [participant_col, "condition", "session_order_numeric", "window_index", dv]
    dat = d[use_cols].dropna().rename(columns={participant_col: "participant_id", dv: "dv"})
    dat["condition"] = pd.Categorical(dat["condition"], categories=["L","M","H"], ordered=True)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_dat = robjects.conversion.py2rpy(dat)
    robjects.globalenv["dat"] = r_dat
    robjects.r("suppressPackageStartupMessages(library(lmerTest))")
    robjects.r("suppressPackageStartupMessages(library(emmeans))")
    robjects.r("dat$participant_id <- factor(dat$participant_id)")

    # model
    robjects.r("model <- lmer(dv ~ condition + session_order_numeric + window_index + (1|participant_id), data=dat)")
    print(f"\n=== {feature_label} (R lmerTest) ===")
    print(robjects.r("summary(model)"))

    # pairs(emmeans) → get estimate + p (column names vary a bit between versions)
    pwc_df = robjects.r('as.data.frame(pairs(emmeans(model, "condition")))')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pwc_pd = robjects.conversion.rpy2py(pwc_df)

    def _find_col(cols, *cands):
        cl = [c.lower() for c in cols]
        for cand in cands:
            # exact or substring match
            for i, name in enumerate(cl):
                if name == cand.lower() or cand.lower() in name:
                    return cols[i]
        return None

    contrast_col = _find_col(pwc_pd.columns, "contrast")
    est_col      = _find_col(pwc_pd.columns, "estimate", "est")
    p_col        = _find_col(pwc_pd.columns, "p.value", "pvalue", "p")

    pairwise = {}
    for _, row in pwc_pd.iterrows():
        c = str(row[contrast_col])
        g1, g2 = [s.strip() for s in c.split('-')]
        est = float(row[est_col]) if est_col in pwc_pd.columns else np.nan
        p   = float(row[p_col])   if p_col   in pwc_pd.columns   else np.nan
        pairwise[(g1, g2)] = {"estimate": est, "p": p}

    # confint(emmeans(...)) → means + CIs; column names vary by version
    ci_df = robjects.r('as.data.frame(confint(emmeans(model, "condition")))')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        ci_pd = robjects.conversion.rpy2py(ci_df)

    cond_col   = _find_col(ci_pd.columns, "condition")
    emmean_col = _find_col(ci_pd.columns, "emmean", "estimate", "response", "lsmean")
    lower_col  = _find_col(ci_pd.columns, "lower.CL", "lcl", "lower")
    upper_col  = _find_col(ci_pd.columns, "upper.CL", "ucl", "upper")

    # fallback if confint didn't include emmean explicitly
    if emmean_col is None:
        # summary(..., infer=TRUE) often includes emmean plus CIs
        summ_df = robjects.r('as.data.frame(summary(emmeans(model, "condition"), infer=c(TRUE, TRUE)))')
        with localconverter(robjects.default_converter + pandas2ri.converter):
            summ_pd = robjects.conversion.rpy2py(summ_df)
        if cond_col is None:   cond_col   = _find_col(summ_pd.columns, "condition")
        if emmean_col is None: emmean_col = _find_col(summ_pd.columns, "emmean", "estimate", "response", "lsmean")
        if lower_col is None:  lower_col  = _find_col(summ_pd.columns, "lower.CL", "lcl", "lower")
        if upper_col is None:  upper_col  = _find_col(summ_pd.columns, "upper.CL", "ucl", "upper")
        ci_pd = summ_pd  # use this table

    if cond_col is None:
        raise RuntimeError("Could not locate the 'condition' column in emmeans output.")
    emm_means, cis = {}, {}
    cond_map_r = {'1':'L','1.0':'L','L':'L','2':'M','2.0':'M','M':'M','3':'H','3.0':'H','H':'H'}
    for _, row in ci_pd.iterrows():
        key = cond_map_r.get(str(row[cond_col]), str(row[cond_col]))
        emm_means[key] = float(row[emmean_col]) if (emmean_col and emmean_col in ci_pd.columns) else np.nan
        lo = float(row[lower_col]) if (lower_col and lower_col in ci_pd.columns) else np.nan
        hi = float(row[upper_col]) if (upper_col and upper_col in ci_pd.columns) else np.nan
        cis[key] = (lo, hi)

    return pairwise, emm_means, cis
    

def _get_p_twolevel(pwc: dict, key: tuple[str, str]) -> float:
    import numpy as np
    v = pwc.get(key, pwc.get((key[1], key[0]), None))
    if v is None: return np.nan
    if isinstance(v, dict): return v.get("p", v.get("p.value", np.nan))
    try: return float(v)
    except Exception: return np.nan

def lme_contrast_table_r(
    df: pd.DataFrame,
    accuracy_cols: list[str],
    rt_cols: list[tuple[str, str]],
    label_prefix: str
) -> pd.DataFrame:
    import numpy as np
    rows = []

    def betas_from_emm(means: dict[str, float]):
        return (
            means.get("M", np.nan) - means.get("L", np.nan),  # L–M
            means.get("H", np.nan) - means.get("M", np.nan),  # M–H
            means.get("H", np.nan) - means.get("L", np.nan),  # L–H
        )

    for dv in accuracy_cols:
        pwc, emm_means, _ = run_rpy2_lmer(df, dv, feature_label=dv)
        b_lm, b_mh, b_lh = betas_from_emm(emm_means)
        p_lm = _get_p_twolevel(pwc, ("L", "M"))
        p_mh = _get_p_twolevel(pwc, ("M", "H"))
        p_lh = _get_p_twolevel(pwc, ("L", "H"))
        metric = DEFAULT_PRETTY.get(dv, dv)
        rows.append([metric, pair_cell(b_lm, p_lm), pair_cell(b_mh, p_mh), pair_cell(b_lh, p_lh)])

    for dv, pretty in rt_cols:
        pwc, emm_means, _ = run_rpy2_lmer(df, dv, feature_label=pretty)
        b_lm, b_mh, b_lh = betas_from_emm(emm_means)
        p_lm = _get_p_twolevel(pwc, ("L", "M"))
        p_mh = _get_p_twolevel(pwc, ("M", "H"))
        p_lh = _get_p_twolevel(pwc, ("L", "H"))
        rows.append([pretty, pair_cell(b_lm, p_lm), pair_cell(b_mh, p_mh), pair_cell(b_lh, p_lh)])

    return pd.DataFrame(rows, columns=["Metric", "L–M ($\beta$, p)", "M–H ($\beta$, p)", "L–H ($\beta$, p)"])


def _get_est_p(pwc: dict, key: tuple[str,str]) -> tuple[float, float]:
    est = np.nan; p = np.nan
    # Try in given order or reversed (emmeans may return 'M - L' etc.)
    if key in pwc:
        est = pwc[key]["estimate"]; p = pwc[key]["p"]
    else:
        rev = (key[1], key[0])
        if rev in pwc:
            est = -pwc[rev]["estimate"]; p = pwc[rev]["p"]
    return est, p

# ---------- Correlations & plotting helpers (Python) ----------
def within_subject_corr(df: pd.DataFrame, x: str, y: str, subject: str = DEFAULT_GROUP_COL):
    from scipy.stats import pearsonr
    sub = df.dropna(subset=[x, y, subject]).copy()
    if sub.empty:
        return np.nan, np.nan, 0
    sub[x + "_resid"] = sub[x] - sub.groupby(subject)[x].transform("mean")
    sub[y + "_resid"] = sub[y] - sub.groupby(subject)[y].transform("mean")
    xr = sub[x + "_resid"].values
    yr = sub[y + "_resid"].values
    r = np.corrcoef(xr, yr)[0, 1] if xr.size > 1 else np.nan
    n = len(xr)
    if not np.isfinite(r):
        return np.nan, np.nan, n
    from scipy.stats import pearsonr as _pearsonr
    r_s, p_s = _pearsonr(xr, yr)
    return float(r_s), float(p_s), n

def build_tlx_lme_table_r(nasa_df: pd.DataFrame) -> pd.DataFrame:
    nasa_df = nasa_df.copy()
    if "tlx_total" not in nasa_df.columns:
        nasa_df["tlx_total"] = nasa_df[TLX_SUBSCALES].sum(axis=1)

    rows = []
    for dv in TLX_SUBSCALES + ["tlx_total"]:
        pwc, _, _ = run_rpy2_lmer(nasa_df, dv, feature_label=f"NASA {dv}")
        b_lm, p_lm = _get_est_p(pwc, ("L","M"))
        b_mh, p_mh = _get_est_p(pwc, ("M","H"))
        b_lh, p_lh = _get_est_p(pwc, ("L","H"))
        pretty_name = dv if dv != "tlx_total" else "TLX total (sum)"
        rows.append([pretty_name, pair_cell(b_lm, p_lm), pair_cell(b_mh, p_mh), pair_cell(b_lh, p_lh)])
    return pd.DataFrame(rows, columns=["Metric","L–M ($\beta$, p)","M–H ($\beta$, p)","L–H ($\beta$, p)"])

def build_within_subject_corr_table(merged_df: pd.DataFrame, group_col: str = DEFAULT_GROUP_COL) -> pd.DataFrame:
    tlx_vars = TLX_SUBSCALES + ["tlx_total"]
    accs = ["track_point_accuracy","resman_point_accuracy","sysmon_point_accuracy","comms_point_accuracy"]
    rows = []; row_names = []
    pretty_acc = {
        "track_point_accuracy": "Tracking accuracy",
        "resman_point_accuracy": "Resource Mgmt accuracy",
        "sysmon_point_accuracy": "System Monitoring accuracy",
        "comms_point_accuracy": "Communications accuracy",
    }
    for acc in accs:
        row = []
        for tlx in tlx_vars:
            r, p, n = within_subject_corr(merged_df, acc, tlx, subject=group_col)
            cell = "NA" if not np.isfinite(r) or (p is None) else f"{r:.3f}, {fmt_p_latex(p)}"
            row.append(cell)
        rows.append(row)
        row_names.append(pretty_acc.get(acc, acc))
    out = pd.DataFrame(rows, columns=[c.replace("_","\\_") for c in tlx_vars], index=row_names).reset_index().rename(columns={"index":"Performance metric"})
    return out

def prep_perf_long(df: pd.DataFrame) -> pd.DataFrame:
    task_cols = {
        'track_point_accuracy': 'Tracking',
        'resman_point_accuracy': 'Resource Management',
        'sysmon_point_accuracy': 'System Monitoring',
        'comms_point_accuracy': 'Communications',
        'mean_rt': 'Mean RT'
    }
    long_df = df.melt(id_vars=['participant', 'load_level'], value_vars=task_cols.keys(), var_name='subtask', value_name='value')
    long_df['subtask'] = long_df['subtask'].map(task_cols)
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')
    return long_df.groupby(['participant', 'load_level', 'subtask'])['value'].mean().reset_index()

def summarise_perf(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(['subtask', 'load_level'])['value'].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['load_level'] = summary['load_level'].map({'Low': 'L', 'Moderate': 'M', 'High': 'H'})
    summary['load_level'] = pd.Categorical(summary['load_level'], categories=['L', 'M', 'H'], ordered=True)
    return summary

def make_wide(df_session: pd.DataFrame) -> pd.DataFrame:
    task_cols = {
        "track_point_accuracy": "Track",
        "resman_point_accuracy": "ResMan",
        "sysmon_point_accuracy": "SysMon",
        "comms_point_accuracy": "Comms",
    }
    grouped = df_session.groupby(["participant", "condition"])[list(task_cols.keys())].mean()
    wide = grouped.unstack("condition")
    wide = wide.swaplevel(0, 1, axis=1)  # outer=condition, inner=subtask
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = pd.MultiIndex.from_tuples([(c[0], task_cols.get(c[1], c[1])) for c in wide.columns])
        want_cols = pd.MultiIndex.from_product([["H","M","L"], list(task_cols.values())])
        wide = wide.reindex(columns=want_cols)
    return wide

def corr_with_pvalues(df1: pd.DataFrame, df2: pd.DataFrame | None = None):
    from scipy.stats import pearsonr
    cols1 = df1.columns
    cols2 = df2.columns if df2 is not None else cols1
    r = pd.DataFrame(index=cols1, columns=cols2, dtype=float)
    p = pd.DataFrame(index=cols1, columns=cols2, dtype=float)
    for c1 in cols1:
        x = df1[c1]
        for c2 in cols2:
            y = (df2[c2] if df2 is not None else df1[c2])
            mask = x.notna() & y.notna()
            if mask.sum() > 2:
                rv, pv = pearsonr(x[mask], y[mask])
            else:
                rv, pv = np.nan, np.nan
            r.loc[c1, c2] = rv
            p.loc[c1, c2] = pv
    return r, p
