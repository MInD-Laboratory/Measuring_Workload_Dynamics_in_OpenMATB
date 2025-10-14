#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script orchestrates:
  1) Loading and merging feature CSVs (supports wide and long formats).
  2) Optional feature selection (backward elimination via permutation importance).
  3) Model training/evaluation across multiple seeds and split strategies.
  4) Learning-curve experiments with per-minute feature selection and overlap control.
  5) Results logging to JSON and a consolidated CSV.

Design choices:
- Feature selection is executed ONCE per model (or once per minute for learning curves)
  to avoid leakage, reduce variance, and cut runtime.
- RandomForest + StandardScaler + optional PCA inside a Pipeline for consistency.
- Balanced accuracy as the primary headline metric
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Quiet down noisy libraries (e.g., pandas chained assignment warnings or sklearn user warnings)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Stable RF defaults; class_weight balanced to counter skewed labels.
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,          # Let trees grow to purity; RF typically robust here.
    "class_weight": "balanced", # Reweight classes by inverse frequency.
    "n_jobs": -1,               # Use all cores.
}

# Ordered labels; used for metrics (F1, kappa, confusion matrix normalization).
LABELS = ["L", "M", "H"]

# Non-feature columns to drop before model fit. These are identifiers or targets.
ID_COLS = {
    "condition", "participant", "window_index",
    "window_start", "window_end", "minute",
    "window_start_s", "window_end_s",
    "source", "t_start_frame", "t_end_frame",  # Metadata from CSV files
}

from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance


def get_all_model_configs(
    experiment_config, feature_groups, default_config, skip_learning_curves=False
):
    """
    Expand experiment definitions into fully-resolved model configs.

    - Merges per-experiment overrides with defaults.
    - Resolves feature group names -> file paths.
    - Optionally skips "learning" sections to speed runs.

    Args:
        experiment_config (dict): Sections keyed by name, each with 'enabled' and 'experiments'.
        feature_groups (dict): Mapping group_name -> (filepath, phase) tuples.
        default_config (dict): Global defaults for models.
        skip_learning_curves (bool): If True, ignore sections whose name includes 'learning'.

    Returns:
        dict: { model_name: config dict with resolved file list }
    """
    all_models = {}
    
    for section_name, section in experiment_config.items():
        # Respect "enabled: false"
        if not section.get("enabled", True):
            continue
        
        # Fast path: allow skipping heavy learning-curve sections
        if skip_learning_curves and "learning" in section_name.lower():
            continue
        
        # Iterate declared experiments in the section
        for exp in section.get("experiments", []):
            name = exp["name"]
            
            # Shallow-merge default config with the experiment overrides
            config = default_config.copy()
            config.update(exp)
            
            # Resolve declared feature group names to actual file tuples
            config["files"] = []
            for group_name in exp["feature_groups"]:
                if group_name not in feature_groups:
                    raise ValueError(
                        f"Unknown feature group '{group_name}' in experiment '{name}'. "
                        f"Available groups: {list(feature_groups.keys())}"
                    )
                config["files"].append(feature_groups[group_name])
            
            all_models[name] = config
    
    return all_models


def get_config_suffix(config):
    """
    Generate a suffix for output filenames based on model configuration.

    This allows different configurations to generate separate output files,
    preventing overwrites when running with different settings.

    Args:
        config (dict): Model configuration dictionary

    Returns:
        str: Suffix string like "_backward_hyp" or "_forward_pca" or "_none"
    """
    if not config:
        return ""

    parts = []

    # Feature selection method
    feat_sel = config.get("feature_selection", None)
    if feat_sel:
        parts.append(str(feat_sel))
    else:
        parts.append("none")

    # Hyperparameter tuning
    if config.get("tune_hyperparameters", False):
        parts.append("hyp")

    # PCA
    if config.get("use_pca", False):
        parts.append("pca")

    return "_" + "_".join(parts) if parts else ""


def check_model_complete(model_name, output_dir, config=None):
    """
    Check if an experiment's JSON output already exists (used to resume/skip).

    Args:
        model_name (str): Unique experiment name.
        output_dir (str | Path): Directory where JSON results are saved.
        config (dict, optional): Model configuration to generate filename suffix.

    Returns:
        bool: True if {model_name}{suffix}.json exists -> considered "complete".
    """
    suffix = get_config_suffix(config) if config else ""
    output_path = Path(output_dir) / f"{model_name}{suffix}.json"
    return output_path.exists()


def prompt_user_action():
    """
    CLI prompt for how to handle existing results when re-running.

    Returns:
        str: One of {'overwrite','continue','skip','cancel'}.
    """
    print("\nWhat would you like to do?")
    print("  [o] Overwrite all existing results")
    print("  [c] Continue incomplete experiments (resume)")
    print("  [s] Skip existing results, only run new experiments")
    print("  [x] Cancel and exit")
    
    while True:
        choice = input("\nChoice [o/c/s/x]: ").strip().lower()
        if choice == 'o':
            return 'overwrite'
        elif choice == 'c':
            return 'continue'
        elif choice == 's':
            return 'skip'
        elif choice == 'x':
            return 'cancel'
        else:
            print("Invalid choice. Please enter o, c, s, or x.")

# ============================================================================
#  Feature Selection
# ============================================================================

def prefilter_low_variance_and_corr(X, var_thresh=1e-8, corr_thresh=0.95):
    """
    Aggressive prefilter: drop near-constant features and highly correlated pairs.
    
    Args:
        X: Feature matrix (DataFrame)
        var_thresh: Variance threshold for removing near-constant features
        corr_thresh: Correlation threshold (lower = more aggressive, e.g., 0.90-0.95)
        
    Returns:
        DataFrame with filtered features
    """
    # Remove near-zero variance features
    vt = VarianceThreshold(threshold=var_thresh)
    X_filtered = pd.DataFrame(
        vt.fit_transform(X), 
        index=X.index,
        columns=X.columns[vt.get_support()]
    )
    
    if X_filtered.shape[1] <= 1:
        return X_filtered
    
    # Remove highly correlated features
    corr = X_filtered.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] >= corr_thresh].tolist()
        to_drop.update(high_corr)
    
    keep_cols = [c for c in X_filtered.columns if c not in to_drop]
    return X_filtered[keep_cols]


def backward_elimination_permutation(
    X, y,
    cv_folds=5,
    n_repeats=5,
    threshold_percentile=25,  # Remove bottom 25% of features at a time
    min_features=5,
    random_state=0
):
    """
    Permutation importance-based elimination that removes multiple features at once.
    
    Strategy:
    - Use permutation importance (more reliable than gini importance)
    - Remove bottom X% of features at each iteration
    - Continue until performance degrades
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target labels
        cv_folds: Number of CV folds
        n_repeats: Number of permutation repeats
        threshold_percentile: Remove features below this importance percentile
        min_features: Minimum features to retain
        random_state: Random seed
        
    Returns:
        tuple: (selected_features, best_score)
    """
    # Prefilter
    X_filtered = prefilter_low_variance_and_corr(X, corr_thresh=0.95)
    features = list(X_filtered.columns)
    
    if len(features) <= min_features:
        return features, 0.0
    
    rf = RandomForestClassifier(**RF_PARAMS, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    def cv_score(feat_list):
        return cross_val_score(
            rf, X_filtered[feat_list], y, cv=cv, 
            scoring="balanced_accuracy", n_jobs=-1
        ).mean()
    
    best_score = cv_score(features)
    best_features = features.copy()
    
    iteration = 0
    while len(features) > min_features:
        iteration += 1
        
        # Compute permutation importance across CV folds
        importances = []
        for train_idx, val_idx in cv.split(X_filtered[features], y):
            X_train = X_filtered[features].iloc[train_idx]
            X_val = X_filtered[features].iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            rf.set_params(random_state=random_state + iteration)
            rf.fit(X_train, y_train)
            
            perm_imp = permutation_importance(
                rf, X_val, y_val,
                scoring="balanced_accuracy",
                n_repeats=n_repeats,
                random_state=random_state + iteration,
                n_jobs=-1
            )
            importances.append(perm_imp.importances_mean)
        
        # Average importances across folds
        mean_importance = np.mean(importances, axis=0)
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': mean_importance
        }).sort_values('importance')
        
        # Remove bottom percentile
        n_to_remove = max(1, int(len(features) * (threshold_percentile / 100)))
        n_to_remove = min(n_to_remove, len(features) - min_features)
        
        if n_to_remove == 0:
            break
        
        features_to_remove = importance_df['feature'].iloc[:n_to_remove].tolist()
        candidate_features = [f for f in features if f not in features_to_remove]
        
        # Evaluate
        candidate_score = cv_score(candidate_features)
        
        print(f"  Iteration {iteration}: {len(features)} -> {len(candidate_features)} features, "
              f"score: {candidate_score:.4f} (best: {best_score:.4f})")
        
        # Update best if improved
        if candidate_score >= best_score - 0.005:  # Allow small drops
            if candidate_score > best_score:
                best_score = candidate_score
                best_features = candidate_features.copy()
            features = candidate_features
        else:
            # Performance degraded significantly, stop
            break
    
    return best_features, best_score


def backward_elimination_rf_fast(X, y, cv_folds=5, random_state=0):
    """
    Thin wrapper for permutation-only backward elimination used in learning curves.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Labels.
        cv_folds (int): Cross-validation folds inside feature selection.
        random_state (int): Seed.

    Returns:
        tuple[list[str], float]: (selected_features, best_cv_score)
    """
    features, score = backward_elimination_permutation(
        X, y,
        cv_folds=cv_folds,
        n_repeats=3,          # reduce runtime
        threshold_percentile=20,
        min_features=5,
        random_state=random_state,
    )

    return features, score


def tune_rf_hyperparameters(X, y, n_iter=50, cv_folds=3, random_state=0):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.

    Searches over key RF parameters that affect model complexity and performance:
    - max_features: Number of features to consider for each split
    - min_samples_split: Minimum samples required to split an internal node
    - min_samples_leaf: Minimum samples required at each leaf node
    - max_depth: Maximum depth of trees
    - n_estimators: Number of trees in the forest

    Args:
        X (pd.DataFrame): Feature matrix
        y (np.ndarray): Target labels
        n_iter (int): Number of parameter settings sampled (default: 50)
        cv_folds (int): Number of cross-validation folds (default: 3)
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Best hyperparameters found by RandomizedSearchCV
    """
    from scipy.stats import randint, uniform

    # Define hyperparameter search space
    param_distributions = {
        'max_features': ['sqrt', 'log2', None],  # None means use all features
        'min_samples_split': randint(2, 20),      # Minimum samples to split a node
        'min_samples_leaf': randint(1, 10),       # Minimum samples at leaf
        'max_depth': [None, 10, 20, 30, 50],      # Maximum tree depth
        'n_estimators': randint(100, 500),        # Number of trees
    }

    # Base RF with fixed parameters
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )

    # Randomized search
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    print(f"  Tuning RF hyperparameters ({n_iter} iterations, {cv_folds}-fold CV)...")
    search.fit(X, y)

    print(f"  Best CV score: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    return search.best_params_


# ============================================================================
#  DATA LOADING & PREPROCESSING
# ============================================================================

def normalize_window_columns(df):
    """
    Standardize window boundary column names.

    Multiple CSV formats are supported; this harmonizes 'window_start_s'/'window_end_s'
    into 'window_start'/'window_end' so downstream logic can assume consistent names.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Same data with normalized column names (copy if renamed).
    """
    rename_map = {}
    if "window_start_s" in df.columns and "window_start" not in df.columns:
        rename_map["window_start_s"] = "window_start"
    if "window_end_s" in df.columns and "window_end" not in df.columns:
        rename_map["window_end_s"] = "window_end"
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def load_and_merge_features(file_list, skip_every=None):
    """
    Load multiple feature CSVs and merge them on key identifiers.

    Supports:
      - Wide format: one row per window with many feature columns.
      - Long format: rows identified by 'feature' name + value columns; pivoted to wide.
    Also supports downsampling windows (skip_every) to reduce temporal overlap.

    Args:
        file_list (list[tuple]): List of (filepath, phase) tuples; 'phase' is not used here
                                 but kept for compatibility (e.g., 'pre' baseline vs 'main').
        skip_every (int | None): Keep only every Nth window when 'window_index' exists.

    Returns:
        pd.DataFrame: Merged wide-format dataset with harmonized time/minute columns.

    Raises:
        FileNotFoundError: If any CSV path is missing.
    """
    dfs_to_merge = []
    
    for filepath, phase in file_list:
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"Feature file not found: {filepath}\n"
                f"Please generate feature files before running pipeline."
            )
        
        df = pd.read_csv(filepath)
        df = normalize_window_columns(df)
        
        # Optional downsampling: useful if sliding windows are dense/overlapping.
        if skip_every is not None and "window_index" in df.columns:
            df = df[df["window_index"] % skip_every == 0].copy()
        
        # Ensure join keys are string-typed where appropriate to avoid merge mismatches.
        if "participant" in df.columns:
            df["participant"] = df["participant"].astype(str)
        if "condition" in df.columns:
            df["condition"] = df["condition"].astype(str)
        
        # Derive 'minute' for coarser alignment when exact window indices differ.
        if "window_start" in df.columns:
            df["minute"] = (df["window_start"] / 60).round().astype(int)
        
        # Long-format detection: presence of a 'feature' column
        if "feature" in df.columns:
            # Build index for pivot; keep whatever time/ID columns exist.
            index_cols = [c for c in ["participant", "condition", "window_start", 
                                      "window_end", "window_index", "minute"] 
                          if c in df.columns]
            
            # Everything else except 'feature' and known non-value columns is a value to pivot
            value_cols = [c for c in df.columns if c not in index_cols 
                          and c not in ["feature", "norm_method"]]
            
            # Pivot to wide format: MultiIndex columns (value_name, feature) -> flattened after
            df = df.pivot_table(
                index=index_cols,
                columns="feature",
                values=value_cols
            )
            
            # Flatten multi-level columns: e.g., ('mean','HR') -> 'mean_HR'
            df.columns = ['_'.join(str(c) for c in col).strip() 
                          for col in df.columns.values]
            df = df.reset_index()
        
        dfs_to_merge.append(df)
    
    # Merge logic
    if len(dfs_to_merge) == 1:
        merged = dfs_to_merge[0]
    else:
        # Compute intersection of columns to find viable merge keys shared across files
        common_cols = set(dfs_to_merge[0].columns)
        for df in dfs_to_merge[1:]:
            common_cols = common_cols & set(df.columns)
        
        # Prefer exact alignment on window_index; fallback to 'minute' if needed
        if "window_index" in common_cols:
            merge_keys = ["participant", "condition", "window_index"]
        elif "minute" in common_cols:
            merge_keys = ["participant", "condition", "minute"]
        else:
            # As a last resort, derive 'minute' from 'window_start' where available
            for i, df in enumerate(dfs_to_merge):
                if "window_start" in df.columns and "minute" not in df.columns:
                    dfs_to_merge[i]["minute"] = (df["window_start"] / 60).round().astype(int)
            merge_keys = ["participant", "condition", "minute"]
        
        # Start merge from the first df, carry its timing columns to avoid duplicates
        merged = dfs_to_merge[0]
        timing_cols = [c for c in ["window_start", "window_end", "window_index"] 
                       if c in merged.columns]
        
        for df in dfs_to_merge[1:]:
            # Drop duplicate timing columns from other dfs (retain from the first df)
            cols_to_drop = [c for c in df.columns 
                            if c in timing_cols and c not in merge_keys]
            df_clean = df.drop(columns=cols_to_drop, errors="ignore")
            
            merged = pd.merge(
                merged,
                df_clean,
                on=merge_keys,
                how="inner",         # strict join to keep only rows present in all sources
                suffixes=("", "_dup")
            )
            
            # Clean any duplicate columns created by merge suffixing
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            merged = merged.drop(columns=dup_cols, errors="ignore")
    
    # Ensure both 'window_start' and 'minute' exist for downstream learning curves.
    if "window_start" not in merged.columns and "minute" in merged.columns:
        merged["window_start"] = merged["minute"] * 60  # coarse back-fill
    if "minute" not in merged.columns and "window_start" in merged.columns:
        merged["minute"] = (merged["window_start"] / 60).round().astype(int)
    
    return merged.reset_index(drop=True)


def drop_identifier_columns(df):
    """
    Remove non-feature columns prior to modeling.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with ID_COLS removed (silently ignores missing).
    """
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def make_train_test_split(df, split_strategy, random_state=0):
    """
    Split dataset into train/test according to strategy.

    - 'random': stratified 80/20 across all rows (cross-task).
    - 'participant': leave ~20% of participants out entirely (cross-participant).

    Args:
        df (pd.DataFrame): Full merged dataset with 'condition' and 'participant'.
        split_strategy (str): 'random' or 'participant'.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

    Raises:
        ValueError: For unsupported split_strategy.
    """
    y = df["condition"].values
    
    if split_strategy == "random":
        # Stratified on label to preserve class ratios in both splits.
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )
    
    elif split_strategy == "participant":
        # Leave-participant-out split:
        # Hold out a subset of participants (~20%) so the model generalizes to unseen subjects.
        participants = df["participant"].unique()
        rng = np.random.default_rng(random_state)
        rng.shuffle(participants)
        
        n_test = max(1, int(0.2 * len(participants)))
        test_participants = set(participants[:n_test])
        
        is_test = df["participant"].isin(test_participants)
        train_idx = df.index[~is_test]
        test_idx = df.index[is_test]
    
    else:
        raise ValueError(
            f"Unknown split_strategy: '{split_strategy}'. "
            f"Must be 'random' or 'participant'."
        )
    
    return df.loc[train_idx], df.loc[test_idx]


def fit_and_evaluate(df, split_strategy, seed, config, selected_features=None, tuned_rf_params=None):
    """
    Train and evaluate a single RF pipeline for a given seed and split.

    Note:
    - Feature selection MUST be done upstream to avoid target leakage and to amortize cost.
      This function only applies a provided feature subset.

    Args:
        df (pd.DataFrame): Full merged dataset.
        split_strategy (str): 'random' or 'participant'.
        seed (int): Random seed.
        config (dict): Model config controlling scaler/PCA usage.
        selected_features (list[str] | None): Pre-selected feature names.
        tuned_rf_params (dict | None): Tuned RF hyperparameters from RandomizedSearchCV.

    Returns:
        dict: { 'metrics': {...}, 'cm': confusion_matrix (normalized true) }
    """
    # Split once per seed
    train_df, test_df = make_train_test_split(df, split_strategy, random_state=seed)

    # Extract targets
    y_train = train_df["condition"].values
    y_test  = test_df["condition"].values

    # Extract features: drop IDs and the target; keep only engineered columns
    X_train = drop_identifier_columns(train_df).drop(columns=["condition"], errors="ignore")
    X_test  = drop_identifier_columns(test_df).drop(columns=["condition"], errors="ignore")

    # Enforce pre-selected feature set if provided (avoids per-seed drift/leakage)
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    # else: use all available feature columns

    # Build pipeline components conditionally
    pipeline_steps = []

    if config.get("use_scaler", True):
        pipeline_steps.append(("scaler", StandardScaler()))

    if config.get("use_pca", False):
        # PCA requires non-NaN data, so add imputation before PCA
        # Use median imputation (more robust to outliers than mean)
        pipeline_steps.append(("imputer", SimpleImputer(strategy='median')))

        # Keep components that explain 'pca_variance' of variance
        pca_variance = config.get("pca_variance", 0.95)
        pipeline_steps.append(("pca", PCA(n_components=pca_variance)))

    # RF config per-seed to vary bootstrap randomness, etc.
    rf_kwargs = dict(RF_PARAMS)

    # Apply tuned hyperparameters if provided
    if tuned_rf_params:
        rf_kwargs.update(tuned_rf_params)

    rf_kwargs["random_state"] = seed
    rf_kwargs.setdefault("n_jobs", -1)
    pipeline_steps.append(("rf", RandomForestClassifier(**rf_kwargs)))

    pipe = Pipeline(pipeline_steps)

    # Fit and predict
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Core metrics; use LABELS to ensure consistent averaging/ordering.
    metrics = {
        "test_acc": accuracy_score(y_test, y_pred),
        "test_bal_acc": balanced_accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, labels=LABELS, average="weighted"),
        "test_kappa": cohen_kappa_score(y_test, y_pred, labels=LABELS),
    }

    # Per-class metrics: precision, recall, F1 for L, M, H
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average=None, zero_division=0
    )

    for i, label in enumerate(LABELS):
        metrics[f"test_precision_{label}"] = precision[i]
        metrics[f"test_recall_{label}"] = recall[i]
        metrics[f"test_f1_{label}"] = f1[i]

    # Confusion matrix normalized by true label counts; expressed in %
    cm = confusion_matrix(y_test, y_pred, labels=LABELS, normalize="true") * 100.0

    return {"metrics": metrics, "cm": cm}


def run_single_model(name, config, output_dir, force=False, resume=False):
    """
    Run a single experiment:
      - Load/merge features
      - (Optionally) perform ONE feature selection on seed=0 training split
      - Evaluate across N seeds using the same selected features
      - Save aggregated metrics & confusion matrix

    Args:
        name (str): Experiment name.
        config (dict): Experiment config (includes 'files', 'split_strategy', etc.).
        output_dir (str | Path): Output directory for JSON + log CSV.
        force (bool): If True, overwrite existing JSON.
        resume (bool): If True, skip completed models unless incomplete.

    Notes:
        - Using seed=0 for feature selection is a pragmatic compromise:
          stable subset, no per-seed leakage, reduced cost.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = get_config_suffix(config)
    output_path = output_dir / f"{name}{suffix}.json"

    # Respect existing results unless explicitly forcing/resuming
    if output_path.exists() and not force and not resume:
        print(f"[SKIP] {name}: already complete")
        return
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Features: {' + '.join([Path(f[0]).stem for f in config['files']])}")
    print(f"Split: {config['split_strategy']}")
    print(f"{'='*60}")
    
    # Load and merge declared feature groups
    merged = load_and_merge_features(
        config["files"],
        skip_every=config.get("skip_every")
    )
    print(f"Loaded data: {merged.shape}")

    # -----------------------------
    # OPTIONAL HYPERPARAMETER TUNING
    # -----------------------------
    tuned_rf_params = {}
    if config.get("tune_hyperparameters", False):
        print("\nPerforming hyperparameter tuning (once for all seeds)...")
        # Tune on the training fold of seed=0 to avoid test leakage
        train_df, _ = make_train_test_split(merged, config["split_strategy"], random_state=0)
        y_train = train_df["condition"].values
        X_train = drop_identifier_columns(train_df).drop(columns=["condition"], errors="ignore")

        # Tune hyperparameters
        tuned_rf_params = tune_rf_hyperparameters(
            X_train, y_train,
            n_iter=config.get("tune_n_iter", 50),
            cv_folds=config.get("tune_cv_folds", 3),
            random_state=0
        )

        # Update RF_PARAMS for this experiment
        print(f"\n  Using tuned parameters for all seeds in this experiment.")

    # -----------------------------
    # ONE-TIME FEATURE SELECTION
    # -----------------------------
    selected_features = None
    if config.get("feature_selection") == "backward":
        print("Performing feature selection (once for all seeds)...")
        # Selection performed only on the training fold of seed=0 to avoid test leakage
        train_df, _ = make_train_test_split(merged, config["split_strategy"], random_state=0)
        y_train = train_df["condition"].values
        X_train = drop_identifier_columns(train_df).drop(columns=["condition"], errors="ignore")
        
        # Use the permutation-based backward elimination (fast, robust for RF)
        selected_features, score = backward_elimination_permutation(
            X_train, y_train, 
            n_repeats=3,            # repeat permutation few times for speed
            threshold_percentile=20,# keep features above this importance percentile
            min_features=5,         # never drop below this baseline
            random_state=0
        )
        print(f"  Selected {len(selected_features)}/{len(X_train.columns)} features (score: {score:.4f})")
    
    # Evaluate across multiple seeds with the fixed feature subset
    n_seeds = config.get("n_seeds", 20)
    all_metrics = []
    all_cms = []
    
    for seed in tqdm(range(n_seeds), desc=f"{name}"):
        result = fit_and_evaluate(
            merged,
            split_strategy=config["split_strategy"],
            seed=seed,
            config=config,
            selected_features=selected_features,  # fixed across seeds
            tuned_rf_params=tuned_rf_params if tuned_rf_params else None
        )

        all_metrics.append(result["metrics"])
        all_cms.append(result["cm"])
    
    # Aggregate metrics across seeds
    metrics_df = pd.DataFrame(all_metrics)

    aggregated_metrics = {
        "test_acc_mean": float(metrics_df["test_acc"].mean()),
        "test_acc_std": float(metrics_df["test_acc"].std(ddof=1)),
        "test_bal_acc_mean": float(metrics_df["test_bal_acc"].mean()),
        "test_bal_acc_std": float(metrics_df["test_bal_acc"].std(ddof=1)),
        "test_f1_mean": float(metrics_df["test_f1"].mean()),
        "test_f1_std": float(metrics_df["test_f1"].std(ddof=1)),
        "test_kappa_mean": float(metrics_df["test_kappa"].mean()),
        "test_kappa_std": float(metrics_df["test_kappa"].std(ddof=1)),
    }

    # Aggregate per-class metrics
    for label in LABELS:
        aggregated_metrics[f"test_precision_{label}_mean"] = float(metrics_df[f"test_precision_{label}"].mean())
        aggregated_metrics[f"test_precision_{label}_std"] = float(metrics_df[f"test_precision_{label}"].std(ddof=1))
        aggregated_metrics[f"test_recall_{label}_mean"] = float(metrics_df[f"test_recall_{label}"].mean())
        aggregated_metrics[f"test_recall_{label}_std"] = float(metrics_df[f"test_recall_{label}"].std(ddof=1))
        aggregated_metrics[f"test_f1_{label}_mean"] = float(metrics_df[f"test_f1_{label}"].mean())
        aggregated_metrics[f"test_f1_{label}_std"] = float(metrics_df[f"test_f1_{label}"].std(ddof=1))
    
    # Average the confusion matrices elementwise (still percentages)
    cm_avg = np.mean(np.stack(all_cms, axis=0), axis=0)
    
    # Persist results (omit raw file list for brevity/noise)
    results = {
        "name": name,
        "config": {k: v for k, v in config.items() if k != "files"},
        "metrics": aggregated_metrics,
        "confusion_matrix": cm_avg.tolist(),
        "selected_features": selected_features if selected_features else [],
        "tuned_rf_params": tuned_rf_params if tuned_rf_params else {},
        "n_features": len(selected_features) if selected_features else len(merged.columns) - len(ID_COLS),
        "n_seeds": n_seeds,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Completed: {name}")
    print(f"  Balanced Accuracy: {aggregated_metrics['test_bal_acc_mean']:.4f} "
          f"± {aggregated_metrics['test_bal_acc_std']:.4f}")
    print(f"  Features used: {results['n_features']}")
    
    # Also append/update a flat CSV log for quick comparison across experiments
    log_to_csv(name, results, output_dir)

# ============================================================================
#  Learning Curves
# ============================================================================

def run_learning_curve_experiment(config, feature_groups, default_config, output_dir, force=False, resume=False):
    """
    Execute a learning curve experiment with incremental time (minutes).

    Two-phase design:
      PHASE 1: For each minute 'm', perform feature selection ONCE using data
               available up to that minute (plus baseline if provided).
               Optionally tune hyperparameters once per minute.
      PHASE 2: For each seed and minute, evaluate with the pre-selected features
               while purging overlapping windows near the split to avoid leakage.

    Key implementation details:
      - 'w_idx' is assumed to increment per window; window_cutoff = minute * 2
        implies 2 windows/min (adjust if your data differs).
      - test_parity ensures train/test windows alternate parity to reduce overlap.
      - 'adaptive_*' normalization modes prevent test-time knowledge of test stats.
      - Supports resume: loads partial results and continues from last completed seed.
    """
    name = config["name"]
    suffix = get_config_suffix(config)
    output_path = Path(output_dir) / f"{name}{suffix}.json"
    checkpoint_path = Path(output_dir) / f"{name}{suffix}_checkpoint.json"

    if output_path.exists() and not force and not resume:
        print(f"[SKIP] {name}: already complete")
        return
    
    print(f"\n{'='*60}")
    print(f"Running Learning Curves: {name}")
    print(f"{'='*60}")

    # Determine if baseline data is used (either concatenation or aggregates)
    baseline_concat_groups = config.get("baseline_concatenate_groups", [])
    baseline_agg_groups = config.get("baseline_aggregate_groups", [])
    has_baseline_concat = len(baseline_concat_groups) > 0
    has_baseline_agg = len(baseline_agg_groups) > 0
    has_any_baseline = has_baseline_concat or has_baseline_agg

    # Normalization strategy for learning curves
    norm_mode = config.get("normalization_mode", "standard")
    valid_modes = ["standard", "adaptive_per_trial", "adaptive_global"]
    if norm_mode not in valid_modes:
        raise ValueError(f"normalization_mode must be one of {valid_modes}, got '{norm_mode}'")

    # Baseline concatenation implies we *cannot* do adaptive normalization (would leak baseline distribution)
    if has_baseline_concat and norm_mode != "standard":
        raise ValueError(
            f"Adaptive normalization modes can only be used without baseline_concatenate_groups."
        )

    # Feature selection method (usually 'backward' here)
    selection_method = config.get("feature_selection", default_config.get("feature_selection"))

    # Hyperparameter tuning settings
    tune_hyperparameters = config.get("tune_hyperparameters", default_config.get("tune_hyperparameters", False))

    print(f"Normalization mode: {norm_mode}")
    print(f"Feature selection: {selection_method if selection_method else 'None'}")
    print(f"Hyperparameter tuning: {tune_hyperparameters}")

    # -------------------------
    # Load baseline concatenation data (trial windows)
    # -------------------------
    if has_baseline_concat:
        baseline_concat_files = [feature_groups[g] for g in baseline_concat_groups]
        baseline_concat_df = load_and_merge_features(baseline_concat_files, skip_every=config.get("skip_every"))
        print(f"Baseline concatenation data (trial windows): {baseline_concat_df.shape}")

        X_baseline_concat = drop_identifier_columns(baseline_concat_df).drop(columns=["condition"], errors="ignore")
        y_baseline_concat = baseline_concat_df["condition"].values
    else:
        X_baseline_concat = None
        y_baseline_concat = None

    # -------------------------
    # Load baseline aggregate features (participant-level stats)
    # -------------------------
    if has_baseline_agg:
        baseline_agg_files = [feature_groups[g] for g in baseline_agg_groups]
        baseline_agg_df = load_and_merge_features(baseline_agg_files, skip_every=config.get("skip_every"))
        print(f"Baseline aggregate features (participant stats): {baseline_agg_df.shape}")
    else:
        baseline_agg_df = None

    if not has_any_baseline:
        print("No baseline data - will start from minute 1")
    else:
        if has_baseline_agg:
            print("Baseline aggregates present - can start from minute 0!")
    
    # -------------------------
    # Load experimental data
    # -------------------------
    exp_files = [feature_groups[g] for g in config["experimental_groups"]]
    exp_df = load_and_merge_features(exp_files, skip_every=config.get("skip_every"))
    print(f"Experimental data: {exp_df.shape}")

    # Merge baseline aggregates as COLUMNS (participant-level features added to each window)
    if has_baseline_agg:
        # Merge on participant and condition to add baseline stats to each experimental window
        merge_keys = ["participant", "condition"]
        exp_df = pd.merge(exp_df, baseline_agg_df, on=merge_keys, how="left", suffixes=("", "_baseline_dup"))

        # Clean any duplicate columns
        dup_cols = [c for c in exp_df.columns if c.endswith("_baseline_dup")]
        exp_df = exp_df.drop(columns=dup_cols, errors="ignore")

        print(f"After merging baseline aggregates: {exp_df.shape}")

    # We rely on window_index to define temporal cutoffs; enforce presence
    if "window_index" not in exp_df.columns:
        raise ValueError("Expected 'window_index' in experimental data.")
    exp_df["w_idx"] = exp_df["window_index"].astype(int)

    # Grouping keys to define disjoint entities (participant/trial/session)
    GROUP_COLS = [c for c in ["participant", "trial_id", "trial", "session"] if c in exp_df.columns]
    if not GROUP_COLS:
        # Create a dummy grouping column if none exist
        exp_df["__all__"] = 1
        GROUP_COLS = ["__all__"]
    
    # Experimental schedule
    minutes = config["minutes"]
    n_seeds = config.get("n_seeds", default_config.get("n_seeds", 20))

    # Storage for per-minute metrics across seeds
    results = {
        m: {
            "BalancedAcc": [],
            "F1": [],
            "Kappa": [],
            "n_features": [],
        }
        for m in minutes
    }

    # Resume logic: load checkpoint if it exists
    start_seed = 0
    minute_features = {}
    minute_tuned_params = {}

    if resume and checkpoint_path.exists():
        print(f"\n[RESUME] Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        # Restore progress
        results = checkpoint["results"]
        minute_features = checkpoint.get("selected_features_per_minute", {})
        minute_tuned_params = checkpoint.get("tuned_params_per_minute", {})
        start_seed = checkpoint.get("last_completed_seed", 0) + 1

        print(f"  Resuming from seed {start_seed}/{n_seeds}")

        if start_seed >= n_seeds:
            print(f"  All seeds already complete, finalizing...")
            start_seed = n_seeds  # Will skip Phase 2 loop
    
    # ============================================================================
    # PHASE 1: Feature selection and hyperparameter tuning ONCE PER MINUTE
    # ============================================================================
    # Skip Phase 1 if resuming (features/params already loaded from checkpoint)
    if not (resume and minute_features):
        print("\n" + "="*60)
        print("PHASE 1: Feature Selection & Hyperparameter Tuning (once per minute)")
        print("="*60)

        for minute in minutes:
            # window_cutoff defines how much training time we allow (2 windows per minute assumed)
            window_cutoff = minute * 2
            mask_train = exp_df["w_idx"] < window_cutoff

            if minute == 0:
                # Minute 0 requires some baseline data (either concatenation or aggregates)
                if not has_any_baseline:
                    continue

                # CASE A: Using baseline aggregate features only (from experimental windows)
                if has_baseline_agg and not has_baseline_concat:
                    # Use experimental windows at minute 0, but they only have baseline agg features
                    # (experimental temporal features haven't accumulated yet)
                    # Select only baseline aggregate columns for minute 0
                    exp_baseline_cols = [c for c in exp_df.columns if "_baseline_" in c]
                    if not exp_baseline_cols:
                        print(f"Warning: No baseline aggregate columns found at minute 0")
                        continue

                    # Get all experimental windows but use only baseline aggregate features
                    X_for_selection = exp_df[exp_baseline_cols]
                    y_for_selection = exp_df["condition"].values

                # CASE B: Using baseline concatenation (trial windows)
                elif has_baseline_concat:
                    X_for_selection = X_baseline_concat
                    y_for_selection = y_baseline_concat

            else:
                exp_train = exp_df[mask_train]
                if exp_train.empty:
                    continue

                X_exp_train = drop_identifier_columns(exp_train).drop(columns=["condition"], errors="ignore")
                y_exp_train = exp_train["condition"].values

                # Optionally concatenate baseline windows with experimental windows
                if has_baseline_concat:
                    X_for_selection = pd.concat([X_baseline_concat, X_exp_train], ignore_index=True)
                    y_for_selection = np.concatenate([y_baseline_concat, y_exp_train])
                else:
                    X_for_selection = X_exp_train
                    y_for_selection = y_exp_train

            # Execute feature selection once for this minute's available data
            if selection_method == "backward":
                print(f"\nMinute {minute}: Selecting features...")
                selected_features, fs_score = backward_elimination_rf_fast(
                    X_for_selection,
                    y_for_selection,
                    cv_folds=5,
                    random_state=42,
                )
                print(f"  → Selected {len(selected_features)} features (score: {fs_score:.4f})")
                minute_features[minute] = selected_features
            else:
                # No FS: use whatever columns exist at selection time
                minute_features[minute] = list(X_for_selection.columns)

            # Hyperparameter tuning for this minute (if enabled)
            if tune_hyperparameters:
                print(f"  Tuning hyperparameters for minute {minute}...")
                tuned_params = tune_rf_hyperparameters(
                    X_for_selection[minute_features[minute]] if minute in minute_features else X_for_selection,
                    y_for_selection,
                    n_iter=config.get("tune_n_iter", default_config.get("tune_n_iter", 30)),
                    cv_folds=config.get("tune_cv_folds", default_config.get("tune_cv_folds", 5)),
                    random_state=42
                )
                minute_tuned_params[minute] = tuned_params
            else:
                minute_tuned_params[minute] = {}
    else:
        print(f"\n[RESUME] Using cached feature selection and hyperparameters from checkpoint")
    
    # ============================================================================
    # PHASE 2: Evaluate across all seeds using pre-selected features
    # ============================================================================
    print("\n" + "="*60)
    print(f"PHASE 2: Evaluation ({n_seeds} seeds per minute)")
    print("="*60 + "\n")

    for seed in tqdm(range(start_seed, n_seeds), desc=f"{name}", initial=start_seed, total=n_seeds):
        for minute in minutes:
            window_cutoff = minute * 2

            # Parity trick: hold out alternating windows (by parity) after cutoff
            # so adjacent windows don't leak information via overlap.
            test_parity = window_cutoff % 2
            mask_test = (exp_df["w_idx"] >= window_cutoff) & ((exp_df["w_idx"] % 2) == test_parity)
            mask_train = exp_df["w_idx"] < window_cutoff

            # Purge near neighbors in train that are adjacent to test windows (±1 index)
            # to aggressively avoid overlap leakage.
            if mask_test.any():
                test_keys = exp_df.loc[mask_test, GROUP_COLS + ["w_idx"]].copy()
                nbr_minus = test_keys.copy(); nbr_minus["w_idx"] -= 1
                nbr_plus  = test_keys.copy();  nbr_plus["w_idx"]  += 1
                banned = pd.concat([test_keys, nbr_minus, nbr_plus], ignore_index=True)
                banned = banned[banned["w_idx"] >= 0]

                ban_idx = pd.MultiIndex.from_frame(banned[GROUP_COLS + ["w_idx"]])
                train_frame = exp_df.loc[mask_train, GROUP_COLS + ["w_idx"]]
                train_idx = pd.MultiIndex.from_frame(train_frame)
                to_purge = train_idx.isin(ban_idx)
                purge_index = exp_df.loc[mask_train].index[to_purge]
                if len(purge_index) > 0:
                    mask_train.loc[purge_index] = False
            
            if mask_test.sum() == 0:
                # Nothing to test at this minute for this seed/grouping
                continue
            
            exp_train = exp_df[mask_train]
            exp_test = exp_df[mask_test]

            # Build training set (optionally include baseline)
            if minute == 0:
                if not has_any_baseline:
                    continue

                # CASE A: Using baseline aggregate features only
                if has_baseline_agg and not has_baseline_concat:
                    # Use experimental windows but only baseline aggregate features
                    exp_baseline_cols = [c for c in exp_df.columns if "_baseline_" in c]
                    if not exp_baseline_cols:
                        continue

                    X_train = exp_df[exp_baseline_cols]
                    y_train = exp_df["condition"].values

                # CASE B: Using baseline concatenation (trial windows)
                elif has_baseline_concat:
                    X_train = X_baseline_concat
                    y_train = y_baseline_concat

            else:
                if exp_train.empty:
                    continue

                X_exp_train = drop_identifier_columns(exp_train).drop(columns=["condition"], errors="ignore")
                y_exp_train = exp_train["condition"].values

                # Optionally concatenate baseline windows with experimental windows
                if has_baseline_concat:
                    X_train = pd.concat([X_baseline_concat, X_exp_train], ignore_index=True)
                    y_train = np.concatenate([y_baseline_concat, y_exp_train])
                else:
                    X_train = X_exp_train
                    y_train = y_exp_train
            
            # Prepare test matrices
            X_test = drop_identifier_columns(exp_test).drop(columns=["condition"], errors="ignore")
            y_test = exp_test["condition"].values
            
            # Align features by intersection (defensive against rare column drift)
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            
            # Apply the pre-selected feature subset for this minute
            if minute in minute_features:
                selected_features = minute_features[minute]
                # Intersect again to be safe
                selected_features = [f for f in selected_features if f in common_features]
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
            
            # -----------------------
            # Normalization options
            # -----------------------
            if norm_mode == "standard":
                # Fit scaler on training only; transform test with same params.
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            elif norm_mode == "adaptive_per_trial":
                # Per-(participant, condition) normalization using training stats
                # to handle distribution shifts across trials.
                train_with_meta = exp_train[["participant", "condition"]].copy()
                test_with_meta = exp_test[["participant", "condition"]].copy()
                
                X_train_scaled = np.zeros_like(X_train.values, dtype=float)
                X_test_scaled = np.zeros_like(X_test.values, dtype=float)
                
                train_groups = train_with_meta.groupby(["participant", "condition"]).groups
                
                for (participant, condition), _ in train_groups.items():
                    trial_train_mask = (train_with_meta["participant"] == participant) & \
                                      (train_with_meta["condition"] == condition)
                    trial_train_data = X_train.values[trial_train_mask]
                    
                    train_mean = np.mean(trial_train_data, axis=0)
                    train_std = np.std(trial_train_data, axis=0)
                    train_std[train_std == 0] = 1.0  # avoid divide-by-zero
                    
                    X_train_scaled[trial_train_mask] = (trial_train_data - train_mean) / train_std
                    
                    # Apply same stats to matching test trials (if any)
                    trial_test_mask = (test_with_meta["participant"] == participant) & \
                                     (test_with_meta["condition"] == condition)
                    
                    if trial_test_mask.any():
                        trial_test_data = X_test.values[trial_test_mask]
                        X_test_scaled[trial_test_mask] = (trial_test_data - train_mean) / train_std
                
                # Fallback: for test groups unseen in training, use global train stats
                test_groups = test_with_meta.groupby(["participant", "condition"]).groups
                for (participant, condition), _ in test_groups.items():
                    if (participant, condition) not in train_groups:
                        global_mean = np.mean(X_train.values, axis=0)
                        global_std = np.std(X_train.values, axis=0)
                        global_std[global_std == 0] = 1.0
                        
                        trial_test_mask = (test_with_meta["participant"] == participant) & \
                                         (test_with_meta["condition"] == condition)
                        trial_test_data = X_test.values[trial_test_mask]
                        X_test_scaled[trial_test_mask] = (trial_test_data - global_mean) / global_std
            
            elif norm_mode == "adaptive_global":
                # One global set of stats computed from training only
                train_mean = np.mean(X_train.values, axis=0)
                train_std = np.std(X_train.values, axis=0)
                train_std[train_std == 0] = 1.0
                
                X_train_scaled = (X_train.values - train_mean) / train_std
                X_test_scaled = (X_test.values - train_mean) / train_std
            
            # -----------------------
            # Classifier fit/eval
            # -----------------------
            from sklearn.ensemble import RandomForestClassifier
            rf_kwargs = dict(RF_PARAMS)

            # Apply tuned hyperparameters for this minute if available
            if minute in minute_tuned_params and minute_tuned_params[minute]:
                rf_kwargs.update(minute_tuned_params[minute])

            rf_kwargs["random_state"] = seed
            rf_kwargs.setdefault("n_jobs", -1)

            clf = RandomForestClassifier(**rf_kwargs)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            
            # Store metrics for this (seed, minute)
            from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
            results[minute]["BalancedAcc"].append(balanced_accuracy_score(y_test, y_pred))
            results[minute]["F1"].append(f1_score(y_test, y_pred, labels=LABELS, average="weighted"))
            results[minute]["Kappa"].append(cohen_kappa_score(y_test, y_pred, labels=LABELS))
            results[minute]["n_features"].append(len(X_train.columns))

        # Save checkpoint after each seed completes
        if resume or force:
            checkpoint_data = {
                "name": name,
                "config": config,
                "has_baseline_concat": has_baseline_concat,
                "has_baseline_agg": has_baseline_agg,
                "normalization_mode": norm_mode,
                "feature_selection": selection_method,
                "tune_hyperparameters": tune_hyperparameters,
                "minutes": minutes,
                "results": results,
                "selected_features_per_minute": {str(m): minute_features.get(m, []) for m in minutes},
                "tuned_params_per_minute": {str(m): minute_tuned_params.get(m, {}) for m in minutes},
                "last_completed_seed": seed,
                "n_seeds": n_seeds,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
    
    # Save the full learning-curve payload (including selected features and tuned params per minute)
    import json
    from datetime import datetime
    output_data = {
        "name": name,
        "config": config,
        "has_baseline_concat": has_baseline_concat,
        "has_baseline_agg": has_baseline_agg,
        "normalization_mode": norm_mode,
        "feature_selection": selection_method,
        "tune_hyperparameters": tune_hyperparameters,
        "minutes": minutes,
        "results": results,
        "selected_features_per_minute": {
            str(m): minute_features.get(m, []) for m in minutes
        },
        "tuned_params_per_minute": {
            str(m): minute_tuned_params.get(m, {}) for m in minutes
        },
        "n_seeds": n_seeds,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n✓ Completed: {name}")
    
    # Console summary of BA curve (mean ± std), plus average feature count used
    print(f"\nLearning Curve Summary (Balanced Accuracy) - {norm_mode}:")
    print(f"{'Minute':<10} {'Mean':<10} {'Std':<10} {'N':<10} {'Features':<15}")
    print("-" * 60)
    for minute in minutes:
        if results[minute]["BalancedAcc"]:
            mean_acc = np.mean(results[minute]["BalancedAcc"])
            std_acc = np.std(results[minute]["BalancedAcc"], ddof=1)
            n_samples = len(results[minute]["BalancedAcc"])
            mean_feats = np.mean(results[minute]["n_features"])
            std_feats = np.std(results[minute]["n_features"], ddof=1) if len(results[minute]["n_features"]) > 1 else 0
            print(f"{minute:<10} {mean_acc*100:.2f}%    {std_acc*100:.2f}%    {n_samples:<10} "
                  f"{mean_feats:.1f} ± {std_feats:.1f}")

    # Log to experiment_log.csv for easy comparison with main RF models
    # Use the final minute's metrics as representative performance
    if minutes and results[minutes[-1]]["BalancedAcc"]:
        final_minute = minutes[-1]
        log_results = {
            "name": name,
            "config": {
                "split_strategy": "learning_curve",
                "normalization_mode": norm_mode,
                "final_minute": final_minute,
            },
            "metrics": {
                "test_bal_acc_mean": np.mean(results[final_minute]["BalancedAcc"]),
                "test_bal_acc_std": np.std(results[final_minute]["BalancedAcc"], ddof=1),
                "test_f1_mean": np.mean(results[final_minute]["F1"]),
                "test_f1_std": np.std(results[final_minute]["F1"], ddof=1),
                "test_kappa_mean": np.mean(results[final_minute]["Kappa"]),
                "test_kappa_std": np.std(results[final_minute]["Kappa"], ddof=1),
            },
            "n_features": int(np.mean(results[final_minute]["n_features"])),
            "n_seeds": n_seeds,
            "timestamp": datetime.now().isoformat(),
        }
        log_to_csv(name, log_results, output_dir)


# ============================================================================
#  RESULTS MANAGEMENT
# ============================================================================

def log_to_csv(name, results, output_dir):
    """
    Append/update a compact CSV log for quick experiment comparisons.

    Args:
        name (str): Experiment name.
        results (dict): Results payload saved to JSON.
        output_dir (str | Path): Folder where 'experiment_log.csv' lives/should live.

    Behavior:
        - Removes any existing row with the same experiment_name before appending,
          so the CSV reflects the latest run for each experiment.
    """
    log_path = Path(output_dir) / "experiment_log.csv"
    
    # Flatten a subset of metrics/metadata into one row
    row = {
        "experiment_name": name,
        "split_strategy": results["config"].get("split_strategy", "unknown"),
        "n_features": results.get("n_features", 0),
        "n_seeds": results.get("n_seeds", 0),
        "test_bal_acc_mean": results["metrics"]["test_bal_acc_mean"],
        "test_bal_acc_std": results["metrics"]["test_bal_acc_std"],
        "test_f1_mean": results["metrics"]["test_f1_mean"],
        "test_f1_std": results["metrics"]["test_f1_std"],
        "test_kappa_mean": results["metrics"]["test_kappa_mean"],
        "test_kappa_std": results["metrics"]["test_kappa_std"],
        "timestamp": results.get("timestamp", ""),
    }
    
    df = pd.DataFrame([row])
    
    if log_path.exists():
        existing = pd.read_csv(log_path)
        # De-duplicate by experiment_name (keep only latest)
        existing = existing[existing["experiment_name"] != name]
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(log_path, index=False)


def print_summary(output_dir):
    """
    Pretty-print a quick summary of all experiments from the CSV log.

    Args:
        output_dir (str | Path): Directory containing 'experiment_log.csv'

    Side effects:
        Prints a table: Experiment | Split | Balanced Accuracy (mean ± std)
    """
    log_path = Path(output_dir) / "experiment_log.csv"
    
    if not log_path.exists():
        print("No experiments logged yet.")
        return
    
    df = pd.read_csv(log_path)
    
    print("\nEXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<40} {'Split':<15} {'Bal Acc':<20}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        bal_acc = f"{row['test_bal_acc_mean']*100:.1f}% ± {row['test_bal_acc_std']*100:.1f}"
        print(f"{row['experiment_name']:<40} {row['split_strategy']:<15} {bal_acc:<20}")
    
    print("=" * 80)
