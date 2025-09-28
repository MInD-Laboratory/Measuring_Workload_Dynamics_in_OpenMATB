# utils/stats_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

COND_ORDER = ("L", "M", "H")

def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected = {}
    for i, (lbl, p) in enumerate(items, start=1):
        corrected[lbl] = min(p * (m - i + 1), 1.0)
    return {k: corrected[k] for k in pvals.keys()}

def compare_groups_statistical(df: pd.DataFrame, metric: str, test_type: str = "auto"):
    """Omnibus across L/M/H, then Holm–Bonferroni pairwise."""
    work = df[["condition", metric]].dropna()
    groups = [work[work["condition"] == c][metric].astype(float).values
              for c in COND_ORDER if (work["condition"] == c).any()]
    group_names = [c for c in COND_ORDER if (work["condition"] == c).any()]
    k = len(groups)

    # descriptives
    desc = []
    for c in group_names:
        vals = work.loc[work["condition"] == c, metric].astype(float)
        desc.append({
            "condition": c, "n": int(vals.count()),
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "sem": float(vals.sem()), "median": float(vals.median())
        })
    desc_df = pd.DataFrame(desc).set_index("condition").reindex(COND_ORDER)

    # choose test
    if test_type == "auto":
        normal = True
        for vals in groups:
            if len(vals) >= 3:
                try:
                    _, p = stats.shapiro(vals)
                except Exception:
                    p = 1.0
                if p < 0.05:
                    normal = False
                    break
        test_type = "parametric" if normal else "nonparametric"

    # omnibus
    omni_name = None; omni_stat = float("nan"); omni_p = float("nan")
    if k >= 2:
        if test_type == "parametric":
            omni_name = "One-way ANOVA" if k > 2 else "Independent t-test"
            if k > 2:
                omni_stat, omni_p = stats.f_oneway(*groups)
            else:
                omni_stat, omni_p = stats.ttest_ind(*groups, equal_var=False)
        else:
            omni_name = "Kruskal–Wallis" if k > 2 else "Mann–Whitney U"
            if k > 2:
                omni_stat, omni_p = stats.kruskal(*groups)
            else:
                omni_stat, omni_p = stats.mannwhitneyu(*groups, alternative="two-sided")

    # pairwise with Holm–Bonferroni
    pairs = {}
    if k > 2:
        raw_p = {}
        for i in range(k):
            for j in range(i+1, k):
                a, b = groups[i], groups[j]
                label = f"{group_names[i]} vs {group_names[j]}"
                if test_type == "parametric":
                    _, p = stats.ttest_ind(a, b, equal_var=False)
                else:
                    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                raw_p[label] = float(p)
        corr = holm_bonferroni(raw_p) if raw_p else {}
        for label in raw_p:
            pairs[label] = {"p_raw": raw_p[label], "p_holm": corr[label], "significant_0.05": corr[label] < 0.05}

    return {
        "metric": metric,
        "test_type": test_type,
        "omnibus": {"name": omni_name, "stat": float(omni_stat), "p": float(omni_p)},
        "descriptives": desc_df,
        "pairwise": pd.DataFrame.from_dict(pairs, orient="index") if pairs else pd.DataFrame()
    }
