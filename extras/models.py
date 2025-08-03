import itertools
import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler

# ===============================================================
# ------------------------ CONFIG -------------------------------
# ===============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("model_output")
RESULTS_DIR.mkdir(exist_ok=True)
DRY_RUN: bool = False

N_SEEDS = 10
MODEL_TYPES = [
    "linear",
    "rqa",
    "task_performance",
    "combined_linear_rqa",
    "linear+task",
    "rqa+task",
    "combined_all",
]

CONDITION_TYPES = ["experimental","baseline","experimental"]
SPLIT_STRATEGIES = ["temporal","random","leave_participant_out"]
SELECTION_METHODS = ["forward","backward"] 
PCA_VARIANCES = [None, 0.80, 0.90, 0.95]

# ===============================================================
# --------------------- DATA HANDLING ---------------------------
# ===============================================================
def load_and_tag(file_path: Path, split_label: str) -> pd.DataFrame:
    """Read CSV and append a simple condition column."""
    df = pd.read_csv(file_path)
    print(f"[load_and_tag] Loaded '{file_path.name}' with shape {df.shape}")
    df["dataset_split"] = split_label      # new column
    return df

def prepare_datasets():
    print("[prepare_datasets] Starting dataset preparation…")

    # ---------- Linear pose ----------
    lin_pose_bsl = load_and_tag(
        DATA_DIR / "linear_pose_metrics" / "baseline_pose_metrics.csv", "baseline"
    )
    lin_pose_exp = load_and_tag(
        DATA_DIR / "linear_pose_metrics" / "experimental_pose_metrics.csv", "experimental"
    )
    lin_pose_exp["window_index"] += 3  # shift post‑baseline windows
    lin_pose = pd.concat([lin_pose_bsl, lin_pose_exp], ignore_index=True)
    lin_pose = lin_pose.rename(
        {c: f"lin_{c}" for c in lin_pose.columns 
        if c not in ["participant", "condition", "window_index", "dataset_split"]},
        axis=1,
    )

    # ---------- RQA pose (long → wide) ----------
    rqa_pose_bsl = load_and_tag(DATA_DIR / "rqa" / "baseline_pose_rqa.csv", "baseline")
    rqa_pose_exp = load_and_tag(DATA_DIR / "rqa" / "experimental_pose_rqa.csv", "experimental")
    rqa_pose_exp["window_index"] += 3
    rqa_pose = pd.concat([rqa_pose_bsl, rqa_pose_exp], ignore_index=True)
    rqa_pose = rqa_pose.rename(
        {c: f"rqa_{c}" for c in rqa_pose.columns 
        if c not in ["participant", "condition", "window_index", "column", "dataset_split"]},
        axis=1,
    )
    rqa_wide = (
        rqa_pose.pivot_table(index=["participant", "condition", "window_index"], columns="column", aggfunc="first")
    )
    rqa_wide.columns = [f"rqa_{col}" for col in rqa_wide.columns]
    rqa_wide = rqa_wide.reset_index()

    # ---------- Task performance ----------
    task_perf_bsl = load_and_tag(DATA_DIR / "performance" / "performance_bsl.csv", "baseline")
    task_perf_exp = load_and_tag(DATA_DIR / "performance" / "performance_exp.csv", "experimental")
    task_perf_exp["window_index"] += 3
    task_perf = pd.concat([task_perf_bsl, task_perf_exp], ignore_index=True)
    task_perf = task_perf.rename(
        {c: f"task_{c}" for c in task_perf.columns 
        if c not in ["participant", "condition", "window_index", "dataset_split"]},
        axis=1,
    )

    # ---------- Merge all ----------
    pose_merged = pd.merge(lin_pose, rqa_wide, on=["participant", "condition", "window_index"], how="inner")
    df = pd.merge(pose_merged, task_perf, on=["participant", "condition", "window_index"], how="inner")
    # --- ensure dataset_split is preserved cleanly ---
    ds_cols = [col for col in df.columns if col.startswith("dataset_split")]
    if ds_cols:
        df["dataset_split"] = df[ds_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[c for c in ds_cols if c != "dataset_split"])
    else:
        raise ValueError("[prepare_datasets] 'dataset_split' not found after merging")
    rqa_pose = rqa_pose.drop(columns="dataset_split", errors="ignore")
    task_perf = task_perf.drop(columns="dataset_split", errors="ignore")

    # Drop helper columns if present
    drop_cols = [c for c in df.columns if any(key in c for key in ["window_start", "window_end"])]
    if drop_cols:
        print(f"[prepare_datasets] Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Feature lists
    lin_feats = [c for c in df.columns if c.startswith("lin_")]
    rqa_feats = [c for c in df.columns if c.startswith("rqa_")]
    task_feats = [c for c in df.columns if c.startswith("task_")]

    return df.reset_index(drop=True), lin_feats, rqa_feats, task_feats
# ===============================================================
# ------------------ FEATURE SELECTION --------------------------
# ===============================================================
def _random_forest_score(X, y, groups, split, scorer, seed):
    for train_idx, test_idx in split.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = clf.predict(X.iloc[test_idx])
        return scorer(y.iloc[test_idx], y_pred)

def forward_selection(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    selected, best_score = [], -np.inf
    remaining = list(X.columns)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    scorer = balanced_accuracy_score

    while remaining:
        candidate_scores = []
        for feat in remaining:
            score = _random_forest_score(X[selected + [feat]], y, groups, split, scorer, seed)
            candidate_scores.append((score, feat))
        candidate_scores.sort(reverse=True)
        top_score, top_feat = candidate_scores[0]
        if top_score > best_score:
            best_score = top_score
            selected.append(top_feat)
            remaining.remove(top_feat)
        else:
            break
    return selected

def backward_selection(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    selected = list(X.columns)
    best_score = -np.inf
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    scorer = balanced_accuracy_score

    while len(selected) > 1:
        candidate_scores = []
        for feat in selected:
            feats_minus = [f for f in selected if f != feat]
            score = _random_forest_score(X[feats_minus], y, groups, split, scorer, seed)
            candidate_scores.append((score, feat))
        candidate_scores.sort(reverse=True)
        top_score, worst_feat = candidate_scores[0]
        if top_score >= best_score:
            best_score = top_score
            selected.remove(worst_feat)
        else:
            break
    return selected
# ===============================================================
# ---------------------- SPLIT HELPERS --------------------------
# ===============================================================
def get_random_split(seed):
    """
    Random split of rows (ignores participant groups)
    """
    return ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

def get_leave_participant_out(seed):
    """
    Group-based split: leaves ~20% of participants out in test, shuffled by seed
    """
    return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

def temporal_splits(df):
    """Yield train / test splits while **avoiding overlapping windows**.

    - Baseline windows (0‑2) are **minute 0**. We first yield a split with
      `t_cut = 2`, i.e. *only baseline* in the training set.
    - Then we add one non‑overlapping experimental minute at a time: window 3,
      5, 7, … (step = 2).
    """
    max_w = int(df["window_index"].max())

    # minute 0: baseline‑only (windows ≤2 in train, rest test)
    if (df["window_index"] <= 2).any():
        tr = df.index[df["window_index"] <= 2].to_numpy()
        te = df.index[df["window_index"] > 2].to_numpy()
        if len(te):
            yield tr, te, 2  # t_cut = 2 → minute 0

    # subsequent minutes, step by 2 to avoid overlaps
    for t in range(3, max_w + 1, 2):
        tr = df.index[df["window_index"] <= t].to_numpy()
        te = df.index[df["window_index"] > t].to_numpy()
        if len(te) == 0:
            break
        yield tr, te, t

# ===============================================================
# -------------------- MAIN PIPELINE ----------------------------
# ===============================================================
def _minute_from_tcut(t_cut):
    """Convert window-index cut to minute index (baseline minute = 0)."""
    if t_cut <= 2:
        return 0
    return ((t_cut - 3) // 2) + 1

def _fit_eval(X, y, tr_idx, te_idx, groups, sel_method, pca_var, seed, split_type, t_cut):
    minute_cut = _minute_from_tcut(t_cut) if split_type == "temporal" else None

    # ---- dry‑run shortcut ----
    if DRY_RUN:
        assert X.iloc[tr_idx].shape[1] > 0, "No features in train set"
        assert not X.iloc[tr_idx].isnull().any().any(), "NaNs in features"
        return dict(seed=seed, split_strategy=split_type, temporal_cut=minute_cut,
                    selection_method=sel_method, pca_variance=pca_var,
                    n_selected=X.shape[1], n_components=None,
                    accuracy=np.nan, balanced_accuracy=np.nan, f1=np.nan, kappa=np.nan,
                    selected_features=json.dumps(list(X.columns)))

    # ---- feature selection ----
    sel_fn = forward_selection if sel_method == "forward" else backward_selection
    selected = sel_fn(X, y, groups, seed)

    X_train, X_test = X.iloc[tr_idx][selected].copy(), X.iloc[te_idx][selected].copy()
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    n_comp = None
    if pca_var is not None:
        pca = PCA(n_components=pca_var, svd_solver="full")
        X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)
        n_comp = pca.n_components_

    clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y.iloc[tr_idx])
    y_pred = clf.predict(X_test)

    return dict(seed=seed, split_strategy=split_type, temporal_cut=minute_cut, selection_method=sel_method,
                pca_variance=pca_var, n_selected=len(selected), n_components=n_comp,
                accuracy=accuracy_score(y.iloc[te_idx], y_pred),
                balanced_accuracy=balanced_accuracy_score(y.iloc[te_idx], y_pred),
                f1=f1_score(y.iloc[te_idx], y_pred, average="weighted"),
                kappa=cohen_kappa_score(y.iloc[te_idx], y_pred),
                selected_features=json.dumps(selected))

def run_one_job(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    groups: pd.Series,
    split_strategy: str,
    selection_method: str,
    pca_variance,
    seed: int,
):

    X_full = df[feature_cols].copy()
    y = df[label_col].copy()

    # -------- Choose splitter --------
    if split_strategy == "random":
        splitter = get_random_split(seed)
        splits = splitter.split(X_full, y)
    elif split_strategy == "leave_participant_out":
        splitter = get_leave_participant_out(seed)
        splits = splitter.split(X_full, y, groups)
    elif split_strategy == "temporal":
        splits = temporal_splits(df)
    else:
        raise ValueError(split_strategy)

    results_rows = []

    # Temporal yields many splits; others one
    if split_strategy == "temporal":
        for train_idx, test_idx, t_cut in splits:
            res = _fit_eval(
                X_full,
                y,
                train_idx,
                test_idx,
                groups,
                selection_method,
                pca_variance,
                seed,
                split_strategy,
                t_cut,
            )
            results_rows.append(res)
    else:
        train_idx, test_idx = next(splits)
        res = _fit_eval(
            X_full,
            y,
            train_idx,
            test_idx,
            groups,
            selection_method,
            pca_variance,
            seed,
            split_strategy,
            None,
        )
        results_rows.append(res)

    return results_rows

def main():
    # ---------- Data ----------
    df, lin_feats, rqa_feats, task_feats = prepare_datasets()

    # Map model_type → feature list
    feat_map = {
        "linear": lin_feats,
        "rqa": rqa_feats,
        "task_performance": task_feats,
        "combined_linear_rqa": lin_feats + rqa_feats,
        "linear+task": lin_feats + task_feats,
        "rqa+task": rqa_feats + task_feats,
        "combined_all": lin_feats + rqa_feats + task_feats,
    }

    # ---------- Validate feature map ----------
    for name, feats in feat_map.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise ValueError(f"{name}: {len(missing)} features not found → {missing[:5]}")

    # ---------- Grid loop ----------
    results = []
    total_jobs = (
        len(MODEL_TYPES)
        * len(CONDITION_TYPES)
        * len(SPLIT_STRATEGIES)
        * len(SELECTION_METHODS)
        * len(PCA_VARIANCES)
        * N_SEEDS
    )
    pbar = tqdm(total=total_jobs, desc="Total jobs")

    for (
        model_type,
        condition_type,
        split_strategy,
        selection_method,
        pca_var,
        seed,
    ) in itertools.product(
        MODEL_TYPES,
        CONDITION_TYPES,
        SPLIT_STRATEGIES,
        SELECTION_METHODS,
        PCA_VARIANCES,
        range(N_SEEDS),
    ):
        # -------- Subset by condition --------
        df_sub = df if condition_type == "both" else df[df["dataset_split"] == condition_type]

        # Guard: temporal split needs both baseline & experimental
        if split_strategy == "temporal" and condition_type != "both":
            pbar.update(1)
            continue
        if split_strategy == "temporal" and df_sub["window_index"].max() <= 3:
            pbar.update(1)
            continue

        feature_cols = feat_map[model_type]
        groups = df_sub["participant"]

        run_rows = run_one_job(
            df_sub,
            feature_cols,
            label_col="condition",
            groups=groups,
            split_strategy=split_strategy,
            selection_method=selection_method,
            pca_variance=pca_var,
            seed=seed,
        )

        for row in run_rows:
            row.update(
                dict(
                    model_type=model_type,
                    condition_type=condition_type,
                )
            )
            results.append(row)

        pbar.update(1)

    pbar.close()

    # ---------- Save ----------
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "results_per_seed.csv", index=False)

    summary = (
        res_df.groupby([
            "model_type",
            "condition_type",
            "split_strategy",
            "selection_method",
            "pca_variance",
        ])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
            bal_acc_sd=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_sd=("kappa", "std"),
            n_selected_mean=("n_selected", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "results_summary.csv", index=False)

    print("\nAll done! Results saved to 'results/' directory." + (" (DRY RUN)" if DRY_RUN else ""))

if __name__ == "__main__":
    main()
