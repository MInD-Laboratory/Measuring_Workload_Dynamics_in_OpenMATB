#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the Random Forest workload detection pipeline.

This module contains all core functionality for:
  - Data loading and preprocessing
  - Feature selection (forward/backward elimination)
  - Model training and evaluation
  - Results management and logging
  - Learning curve analysis

Functions are organized into sections:
  1. Configuration & Setup
  2. Data Loading & Preprocessing  
  3. Feature Selection
  4. Model Training & Evaluation
  5. Results Management
  6. Learning Curves
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
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)

warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# 1. CONFIGURATION & SETUP
# ============================================================================

# Random Forest hyperparameters (fixed for all experiments)
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# Class labels for workload conditions
LABELS = ["L", "M", "H"]

# Columns that should not be used as features
ID_COLS = {
    "condition", "participant", "window_index",
    "window_start", "window_end", "minute",
    "window_start_s", "window_end_s",  # Alternative naming
}


def get_all_model_configs(
    experiment_config, feature_groups, default_config, skip_learning_curves=False
):
    """
    Generate all model configurations from experiment definitions.
    
    Combines experiment specifications with default settings and resolves
    feature group mappings.
    
    Args:
        experiment_config: Dictionary of experiment sections
        feature_groups: Mapping of feature group names to file paths
        default_config: Default model configuration
        skip_learning_curves: If True, skip learning curve experiments
        
    Returns:
        dict: Mapping of model names to full configurations
    """
    all_models = {}
    
    for section_name, section in experiment_config.items():
        # Skip disabled sections
        if not section.get("enabled", True):
            continue
        
        # Skip learning curves if requested
        if skip_learning_curves and "learning" in section_name.lower():
            continue
        
        # Process each experiment in this section
        for exp in section.get("experiments", []):
            name = exp["name"]
            
            # Build configuration by merging defaults with experiment-specific settings
            config = default_config.copy()
            config.update(exp)
            
            # Resolve feature groups to file paths
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


def check_model_complete(model_name, output_dir):
    """
    Check if a model has been fully run.
    
    A model is considered complete if its output JSON exists.
    
    Args:
        model_name: Name of the model
        output_dir: Directory containing results
        
    Returns:
        bool: True if model results exist
    """
    output_path = Path(output_dir) / f"{model_name}.json"
    return output_path.exists()


def prompt_user_action():
    """
    Prompt user for action when existing results are found.
    
    Returns:
        str: User choice ('overwrite', 'continue', 'skip', or 'cancel')
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
# 2. DATA LOADING & PREPROCESSING
# ============================================================================

def normalize_window_columns(df):
    """
    Normalize window column names across different CSV formats.
    
    Some CSVs use 'window_start_s' while others use 'window_start'.
    This function standardizes to 'window_start' and 'window_end'.
    
    Args:
        df: DataFrame with window columns
        
    Returns:
        DataFrame with normalized column names
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
    Load and merge multiple feature CSV files.
    
    Handles both wide-format (one row per window) and long-format
    (multiple rows per window) CSV files. Automatically pivots long-format
    data to wide format and merges on participant, condition, and window timing.
    
    Args:
        file_list: List of (filepath, phase) tuples
                   phase is 'pre' for baseline or 'main' for experimental
        skip_every: If provided, only keep every Nth window (for reducing overlap)
        
    Returns:
        DataFrame with merged features in wide format
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
        
        # Apply window skipping if requested (e.g., to reduce overlap)
        if skip_every is not None and "window_index" in df.columns:
            df = df[df["window_index"] % skip_every == 0].copy()
        
        # Ensure participant and condition are strings
        if "participant" in df.columns:
            df["participant"] = df["participant"].astype(str)
        if "condition" in df.columns:
            df["condition"] = df["condition"].astype(str)
        
        # Add minute column if window_start exists
        if "window_start" in df.columns:
            df["minute"] = (df["window_start"] / 60).round().astype(int)
        
        # Check if this is long-format data (has 'feature' column)
        if "feature" in df.columns:
            # Pivot from long to wide format
            # Assume we have columns like: participant, condition, window_start, 
            # window_end, feature, mean, rms, min, max
            
            index_cols = [c for c in ["participant", "condition", "window_start", 
                                      "window_end", "window_index", "minute"] 
                          if c in df.columns]
            
            value_cols = [c for c in df.columns if c not in index_cols 
                          and c not in ["feature", "norm_method"]]
            
            df = df.pivot_table(
                index=index_cols,
                columns="feature",
                values=value_cols
            )
            
            # Flatten multi-level column names
            df.columns = ['_'.join(str(c) for c in col).strip() 
                          for col in df.columns.values]
            df = df.reset_index()
        
        dfs_to_merge.append(df)
    
    # Merge all DataFrames
    if len(dfs_to_merge) == 1:
        merged = dfs_to_merge[0]
    else:
        # Determine merge keys - use what's available across all DataFrames
        # Priority: participant, condition, window_index (most reliable)
        # Fallback: participant, condition, minute
        
        # Check which columns are common across all dataframes
        common_cols = set(dfs_to_merge[0].columns)
        for df in dfs_to_merge[1:]:
            common_cols = common_cols & set(df.columns)
        
        # Determine best merge keys
        if "window_index" in common_cols:
            merge_keys = ["participant", "condition", "window_index"]
        elif "minute" in common_cols:
            merge_keys = ["participant", "condition", "minute"]
        else:
            # Add minute to all dataframes if window_start exists
            for i, df in enumerate(dfs_to_merge):
                if "window_start" in df.columns and "minute" not in df.columns:
                    dfs_to_merge[i]["minute"] = (df["window_start"] / 60).round().astype(int)
            merge_keys = ["participant", "condition", "minute"]
        
        # Start with first dataframe
        merged = dfs_to_merge[0]
        
        # Keep track of timing columns from first dataframe
        timing_cols = [c for c in ["window_start", "window_end", "window_index"] 
                       if c in merged.columns]
        
        for df in dfs_to_merge[1:]:
            # Drop duplicate timing columns from subsequent dataframes (keep from first)
            cols_to_drop = [c for c in df.columns 
                            if c in timing_cols and c not in merge_keys]
            df_clean = df.drop(columns=cols_to_drop, errors="ignore")
            
            merged = pd.merge(
                merged,
                df_clean,
                on=merge_keys,
                how="inner",
                suffixes=("", "_dup")
            )
            
            # Remove duplicate columns created by merge
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            merged = merged.drop(columns=dup_cols, errors="ignore")
    
    # Ensure window_start and minute columns exist for learning curves
    if "window_start" not in merged.columns and "minute" in merged.columns:
        # Reconstruct window_start from minute (approximate)
        merged["window_start"] = merged["minute"] * 60
    
    if "minute" not in merged.columns and "window_start" in merged.columns:
        merged["minute"] = (merged["window_start"] / 60).round().astype(int)
    
    return merged.reset_index(drop=True)


def drop_identifier_columns(df):
    """
    Remove columns that shouldn't be used as features.
    
    Includes participant IDs, condition labels, window indices, etc.
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        DataFrame with only feature columns
    """
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def make_train_test_split(df, split_strategy, random_state=0):
    """
    Split data into train and test sets using specified strategy.
    
    Args:
        df: DataFrame with features and labels
        split_strategy: Either 'random' (cross-task) or 'participant' (cross-participant)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    y = df["condition"].values
    
    if split_strategy == "random":
        # Random 80/20 split, stratified by condition
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )
    
    elif split_strategy == "participant":
        # Leave-participant-out: hold out ~20% of participants
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


# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================

def backward_elimination_rf(X, y, cv_folds=5, tol=1e-4, random_state=0):
    """
    Perform backward elimination using Random Forest feature importances.
    
    Iteratively removes the least important feature as long as cross-validated
    performance doesn't degrade beyond tolerance threshold.
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target labels
        cv_folds: Number of cross-validation folds
        tol: Tolerance for performance degradation
        random_state: Random seed
        
    Returns:
        tuple: (selected_features, best_score)
            selected_features: List of selected feature names
            best_score: Best balanced accuracy achieved
    """
    features = X.columns.tolist()
    rf = RandomForestClassifier(**RF_PARAMS, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Baseline performance with all features
    best_score = cross_val_score(
        rf, X[features], y, cv=cv, scoring="balanced_accuracy"
    ).mean()
    best_features = features.copy()
    
    # Progress bar for elimination iterations
    pbar = tqdm(
        range(len(features) - 1),
        desc="  Feature selection",
        leave=False,
        position=1
    )
    
    # Iteratively remove least important feature
    for _ in pbar:
        if len(features) <= 1:
            break
        
        # Train model to get feature importances
        rf.fit(X[features], y)
        
        # Find least important feature
        least_important_idx = int(np.argmin(rf.feature_importances_))
        least_important = features[least_important_idx]
        
        # Test performance without this feature
        candidate_features = [f for f in features if f != least_important]
        candidate_score = cross_val_score(
            rf, X[candidate_features], y, cv=cv, scoring="balanced_accuracy"
        ).mean()
        
        # Update progress bar
        pbar.set_postfix({
            "n_features": len(candidate_features),
            "score": f"{candidate_score:.4f}"
        })
        
        # Accept removal if performance doesn't degrade
        if candidate_score + tol >= best_score:
            best_score = candidate_score
            best_features = candidate_features
            features = candidate_features
        else:
            # Performance degraded, stop elimination
            break
    
    pbar.close()
    return best_features, best_score


# ============================================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================================

def select_features_once(df, split_strategy, config, seed=0):
    """
    Perform feature selection once using a single seed.
    
    Feature selection is computationally expensive, so we do it once
    and reuse the selected features across all random seeds.
    
    Args:
        df: DataFrame with features and labels
        split_strategy: How to split train/test
        config: Model configuration dict
        seed: Random seed for feature selection (default: 0)
        
    Returns:
        list: Selected feature names
    """
    # Split data
    train_df, _ = make_train_test_split(df, split_strategy, random_state=seed)
    
    # Extract labels and features
    y_train = train_df["condition"].values
    X_train = drop_identifier_columns(train_df).drop(columns=["condition"], errors="ignore")
    
    # Get feature selection method from config
    selection_method = config.get("feature_selection")
    
    if selection_method == "backward":
        # Backward elimination
        selected_features, score = backward_elimination_rf(
            X_train, y_train, random_state=seed
        )
        print(f"    Backward elimination: {len(selected_features)}/{len(X_train.columns)} "
              f"features (score: {score:.4f})")
    
    elif selection_method == "forward":
        # Forward selection (not implemented yet - placeholder)
        print(f"    Forward selection not implemented, using all {len(X_train.columns)} features")
        selected_features = list(X_train.columns)
    
    elif selection_method is None:
        # No feature selection
        selected_features = list(X_train.columns)
        print(f"    Using all {len(selected_features)} features (no selection)")
    
    else:
        raise ValueError(
            f"Unknown feature_selection method: '{selection_method}'. "
            f"Must be 'backward', 'forward', or None."
        )
    
    # Ensure we have at least some features
    if not selected_features:
        print("    WARNING: No features selected, using all features")
        selected_features = list(X_train.columns)
    
    return selected_features


def fit_and_evaluate(df, split_strategy, seed, config):
    """
    Fit and evaluate with PER-SEED feature selection.
    Feature selection now happens fresh for each seed.
    """
    # Split data
    train_df, test_df = make_train_test_split(df, split_strategy, random_state=seed)

    # Labels
    y_train = train_df["condition"].values
    y_test  = test_df["condition"].values

    # Features
    X_train = drop_identifier_columns(train_df).drop(columns=["condition"], errors="ignore")
    X_test  = drop_identifier_columns(test_df).drop(columns=["condition"], errors="ignore")

    # *** FEATURE SELECTION MOVED HERE (per-seed) ***
    selection_method = config.get("feature_selection")
    
    if selection_method == "backward":
        selected_features, _ = backward_elimination_rf(X_train, y_train, random_state=seed)
    elif selection_method == "forward":
        # Forward selection not implemented yet
        selected_features = list(X_train.columns)
    elif selection_method is None:
        selected_features = list(X_train.columns)
    else:
        raise ValueError(f"Unknown feature_selection method: '{selection_method}'")
    
    # Ensure we have at least some features
    if not selected_features:
        selected_features = list(X_train.columns)
    
    # Apply selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Pipeline
    pipeline_steps = []

    if config.get("use_scaler", True):
        pipeline_steps.append(("scaler", StandardScaler()))

    if config.get("use_pca", False):
        pca_variance = config.get("pca_variance", 0.95)
        pipeline_steps.append(("pca", PCA(n_components=pca_variance)))

    rf_kwargs = dict(RF_PARAMS)
    rf_kwargs["random_state"] = seed
    rf_kwargs.setdefault("n_jobs", -1)
    pipeline_steps.append(("rf", RandomForestClassifier(**rf_kwargs)))

    pipe = Pipeline(pipeline_steps)

    # Fit / predict
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Metrics
    metrics = {
        "test_acc": accuracy_score(y_test, y_pred),
        "test_bal_acc": balanced_accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, labels=LABELS, average="weighted"),
        "test_kappa": cohen_kappa_score(y_test, y_pred, labels=LABELS),
    }

    cm = confusion_matrix(y_test, y_pred, labels=LABELS, normalize="true") * 100.0
    
    # *** RETURN SELECTED FEATURES TOO ***
    return {"metrics": metrics, "cm": cm, "selected_features": selected_features}



def run_single_model(name, config, output_dir, force=False, resume=False):
    """
    Run a single model configuration across multiple random seeds.
    Now does feature selection PER-SEED instead of once globally.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{name}.json"
    
    # Check if already complete
    if output_path.exists() and not force and not resume:
        print(f"[SKIP] {name}: already complete")
        return
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Features: {' + '.join([Path(f[0]).stem for f in config['files']])}")
    print(f"Split: {config['split_strategy']}")
    print(f"{'='*60}")
    
    # Load and merge features
    merged = load_and_merge_features(
        config["files"],
        skip_every=config.get("skip_every")
    )
    print(f"Loaded data: {merged.shape}")
    
    # *** REMOVE select_features_once() call ***
    # Feature selection now happens inside fit_and_evaluate() per-seed
    
    # Run across multiple seeds
    n_seeds = config.get("n_seeds", 20)
    all_metrics = []
    all_cms = []
    all_selected_features = []  # Track which features each seed selected
    
    for seed in tqdm(range(n_seeds), desc=f"{name}"):
        result = fit_and_evaluate(
            merged,
            split_strategy=config["split_strategy"],
            seed=seed,
            config=config  # *** Pass config instead of selected_features ***
        )
        
        all_metrics.append(result["metrics"])
        all_cms.append(result["cm"])
        all_selected_features.append(result["selected_features"])
    
    # Aggregate results across seeds
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
    
    # Average confusion matrix across seeds
    cm_avg = np.mean(np.stack(all_cms, axis=0), axis=0)
    
    # *** Compute feature selection statistics ***
    # Find features selected by majority of seeds (>50%)
    from collections import Counter
    feature_counts = Counter()
    for features in all_selected_features:
        feature_counts.update(features)
    
    total_seeds = len(all_selected_features)
    common_features = [f for f, count in feature_counts.items() if count > total_seeds / 2]
    
    # Save results
    results = {
        "name": name,
        "config": {k: v for k, v in config.items() if k != "files"},
        "metrics": aggregated_metrics,
        "confusion_matrix": cm_avg.tolist(),
        "selected_features_common": common_features,  # Features selected by >50% of seeds
        "n_features_mean": float(np.mean([len(f) for f in all_selected_features])),
        "n_features_std": float(np.std([len(f) for f in all_selected_features], ddof=1)),
        "n_seeds": n_seeds,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Completed: {name}")
    print(f"  Balanced Accuracy: {aggregated_metrics['test_bal_acc_mean']:.4f} "
          f"± {aggregated_metrics['test_bal_acc_std']:.4f}")
    print(f"  Features selected: {results['n_features_mean']:.1f} ± {results['n_features_std']:.1f}")
    
    # Log to CSV
    log_to_csv(name, results, output_dir)


# ============================================================================
# 5. RESULTS MANAGEMENT
# ============================================================================

def log_to_csv(name, results, output_dir):
    """
    Log experiment results to a CSV file for easy comparison.
    
    Args:
        name: Experiment name
        results: Results dictionary
        output_dir: Directory containing experiment_log.csv
    """
    log_path = Path(output_dir) / "experiment_log.csv"
    
    # Prepare row
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
    
    # Append to CSV
    df = pd.DataFrame([row])
    
    if log_path.exists():
        existing = pd.read_csv(log_path)
        # Remove old entry for this experiment if it exists
        existing = existing[existing["experiment_name"] != name]
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(log_path, index=False)


def print_summary(output_dir):
    """
    Print summary of all experiments.
    
    Args:
        output_dir: Directory containing experiment_log.csv
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


# ============================================================================
# 6. LEARNING CURVES
# ============================================================================
# def run_learning_curve_experiment(config, feature_groups, default_config, output_dir, force=False):
#     """
#     Run learning curve analysis with incremental training data.
    
#     Learning curves show how model performance changes as we add more
#     training data. This is useful for understanding:
#       - How much data is needed for good performance
#       - Whether baseline data helps
#       - How quickly the model adapts to new conditions
    
#     The experiment:
#       1. Optionally starts with baseline data (minute 0)
#       2. Incrementally adds experimental data (minutes 1, 2, 3, ...)
#       3. Tests on the remaining experimental data
#       4. Repeats across multiple random seeds
    
#     If no baseline_groups specified, starts at minute 1 instead of minute 0.
    
#     Args:
#         config: Learning curve configuration dict with:
#             - name: Experiment name
#             - baseline_groups: List of baseline feature groups (optional, can be empty)
#             - experimental_groups: List of experimental feature groups (required)
#             - minutes: List of cutoff minutes to test
#             - skip_every: Window skipping parameter (optional)
#             - n_seeds: Number of random seeds (optional)
#         feature_groups: Mapping of group names to file paths
#         default_config: Default model settings
#         output_dir: Directory to save results
#         force: If True, overwrite existing results
#     """
#     name = config["name"]
#     output_path = Path(output_dir) / f"{name}.json"
    
#     # Check if already complete
#     if output_path.exists() and not force:
#         print(f"[SKIP] {name}: already complete")
#         return
    
#     print(f"\n{'='*60}")
#     print(f"Running Learning Curves: {name}")
#     print(f"{'='*60}")
    
#     # Check if we have baseline data
#     has_baseline = config.get("baseline_groups") and len(config["baseline_groups"]) > 0
    
#     # Load baseline data if specified
#     if has_baseline:
#         baseline_files = [feature_groups[g] for g in config["baseline_groups"]]
#         baseline_df = load_and_merge_features(baseline_files, skip_every=config.get("skip_every"))
#         print(f"Baseline data: {baseline_df.shape}")
        
#         # Extract features for baseline (drop ID columns)
#         X_baseline = drop_identifier_columns(baseline_df).drop(columns=["condition"], errors="ignore")
#         y_baseline = baseline_df["condition"].values
#     else:
#         print("No baseline data - will start from minute 1")
#         X_baseline = None
#         y_baseline = None
    
#     # Load experimental data
#     exp_files = [feature_groups[g] for g in config["experimental_groups"]]
#     exp_df = load_and_merge_features(exp_files, skip_every=config.get("skip_every"))
#     print(f"Experimental data: {exp_df.shape}")

#     if "window_index" not in exp_df.columns:
#         raise ValueError("Expected 'window_index' in experimental data.")
#     exp_df["w_idx"] = exp_df["window_index"].astype(int)

#     GROUP_COLS = [c for c in ["participant", "trial_id", "trial", "session"] if c in exp_df.columns]
#     if not GROUP_COLS:
#         exp_df["__all__"] = 1
#         GROUP_COLS = ["__all__"]
    
#     # Settings
#     minutes = config["minutes"]
#     n_seeds = config.get("n_seeds", default_config.get("n_seeds", 20))
    
#     # Results storage: results[minute][metric] = [scores across seeds]
#     results = {
#         m: {
#             "BalancedAcc": [],
#             "F1": [],
#             "Kappa": [],
#         }
#         for m in minutes
#     }
    
#     # Run across seeds
#     for seed in tqdm(range(n_seeds), desc=f"{name}"):
#         for minute in minutes:
#             # Convert minutes to window index
#             # With 60s windows and 50% overlap (30s step), window_index = minute * 2
#             window_cutoff = minute * 2

#             # build non-overlapping TEST via parity, then purge TRAIN neighbors
#             # TEST: all windows >= cutoff that match parity (avoid test-test overlap)
#             test_parity = window_cutoff % 2
#             mask_test = (exp_df["w_idx"] >= window_cutoff) & ((exp_df["w_idx"] % 2) == test_parity)

#             # TRAIN: all windows before cutoff
#             mask_train = exp_df["w_idx"] < window_cutoff

#             # If there are test windows, purge any overlapping TRAIN windows (neighbors ±1 in same group)
#             if mask_test.any():
#                 test_keys = exp_df.loc[mask_test, GROUP_COLS + ["w_idx"]].copy()
#                 nbr_minus = test_keys.copy(); nbr_minus["w_idx"] = nbr_minus["w_idx"] - 1
#                 nbr_plus  = test_keys.copy();  nbr_plus["w_idx"]  = nbr_plus["w_idx"]  + 1
#                 banned = pd.concat([test_keys, nbr_minus, nbr_plus], ignore_index=True)
#                 banned = banned[banned["w_idx"] >= 0]

#                 ban_idx = pd.MultiIndex.from_frame(banned[GROUP_COLS + ["w_idx"]])
#                 train_frame = exp_df.loc[mask_train, GROUP_COLS + ["w_idx"]]
#                 train_idx = pd.MultiIndex.from_frame(train_frame)
#                 to_purge = train_idx.isin(ban_idx)
#                 # clear overlapped train windows
#                 purge_index = exp_df.loc[mask_train].index[to_purge]
#                 if len(purge_index) > 0:
#                     mask_train.loc[purge_index] = False
#             # ------------------------------------------------------------------------------- # <<<
            
#             # Skip if no test data
#             if mask_test.sum() == 0:
#                 continue
            
#             # Prepare experimental training data
#             exp_train = exp_df[mask_train]
#             exp_test = exp_df[mask_test]
            
#             # Build training set based on whether we have baseline and experimental data
#             if minute == 0:
#                 # At minute 0, only use baseline data (if available)
#                 if not has_baseline:
#                     # No baseline and no experimental data yet - skip
#                     continue
#                 X_train = X_baseline
#                 y_train = y_baseline
            
#             else:
#                 # minute > 0: use experimental data (and optionally baseline)
#                 if exp_train.empty:
#                     # No experimental training data yet - skip
#                     continue
                
#                 X_exp_train = drop_identifier_columns(exp_train).drop(columns=["condition"], errors="ignore")
#                 y_exp_train = exp_train["condition"].values
                
#                 if has_baseline:
#                     # Combine baseline + experimental
#                     X_train = pd.concat([X_baseline, X_exp_train], ignore_index=True)
#                     y_train = np.concatenate([y_baseline, y_exp_train])
#                 else:
#                     # Only experimental data
#                     X_train = X_exp_train
#                     y_train = y_exp_train
            
#             # Prepare test data
#             X_test = drop_identifier_columns(exp_test).drop(columns=["condition"], errors="ignore")
#             y_test = exp_test["condition"].values
            
#             # Ensure same features in train and test
#             common_features = list(set(X_train.columns) & set(X_test.columns))
#             X_train = X_train[common_features]
#             X_test = X_test[common_features]
            
#             # Build and train pipeline
#             # Build classifier kwargs safely
#             rf_kwargs = dict(RF_PARAMS)         # don't mutate the global
#             rf_kwargs["random_state"] = seed
#             rf_kwargs.setdefault("n_jobs", -1)  # only set if not already present

#             # Pipeline
#             pipe = Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("rf", RandomForestClassifier(**rf_kwargs)),
#             ])

            
#             pipe.fit(X_train, y_train)
#             y_pred = pipe.predict(X_test)
            
#             # Compute metrics
#             results[minute]["BalancedAcc"].append(balanced_accuracy_score(y_test, y_pred))
#             results[minute]["F1"].append(f1_score(y_test, y_pred, labels=LABELS, average="weighted"))
#             results[minute]["Kappa"].append(cohen_kappa_score(y_test, y_pred, labels=LABELS))
    
#     # Save results
#     output_data = {
#         "name": name,
#         "config": config,
#         "has_baseline": has_baseline,
#         "minutes": minutes,
#         "results": results,
#         "n_seeds": n_seeds,
#         "timestamp": datetime.now().isoformat(),
#     }
    
#     with open(output_path, "w") as f:
#         json.dump(output_data, f, indent=2)
    
#     print(f"✓ Completed: {name}")
    
#     # Print summary
#     print("\nLearning Curve Summary (Balanced Accuracy):")
#     print(f"{'Minute':<10} {'Mean':<10} {'Std':<10} {'N':<10}")
#     print("-" * 40)
#     for minute in minutes:
#         if results[minute]["BalancedAcc"]:
#             mean_acc = np.mean(results[minute]["BalancedAcc"])
#             std_acc = np.std(results[minute]["BalancedAcc"], ddof=1)
#             n_samples = len(results[minute]["BalancedAcc"])
#             print(f"{minute:<10} {mean_acc*100:.2f}%    {std_acc*100:.2f}%    {n_samples:<10}")#!/usr/bin/env python3


def run_learning_curve_experiment(config, feature_groups, default_config, output_dir, force=False):
    """
    Run learning curve analysis with incremental training data.
    NOW WITH PER-SEED FEATURE SELECTION.
    
    Learning curves show how model performance changes as we add more
    training data. This is useful for understanding:
      - How much data is needed for good performance
      - Whether baseline data helps
      - How quickly the model adapts to new conditions
    
    The experiment:
      1. Optionally starts with baseline data (minute 0)
      2. Incrementally adds experimental data (minutes 1, 2, 3, ...)
      3. Tests on the remaining experimental data
      4. Repeats across multiple random seeds
      5. Feature selection done per-seed (not globally)
    
    If no baseline_groups specified, starts at minute 1 instead of minute 0.
    
    Normalization modes (only applied when has_baseline=False):
      - "standard": Fit scaler on train, transform both train and test (default sklearn)
      - "adaptive_per_trial": Compute mean/std per (participant, condition) from train,
                              apply those stats to transform test data for that trial
      - "adaptive_global": Compute mean/std globally from all train data,
                          apply to all test data
    
    Args:
        config: Learning curve configuration dict with:
            - name: Experiment name
            - baseline_groups: List of baseline feature groups (optional, can be empty)
            - experimental_groups: List of experimental feature groups (required)
            - minutes: List of cutoff minutes to test
            - skip_every: Window skipping parameter (optional)
            - n_seeds: Number of random seeds (optional)
            - normalization_mode: "standard", "adaptive_per_trial", or "adaptive_global" (optional)
            - feature_selection: "backward", "forward", or None (optional)
        feature_groups: Mapping of group names to file paths
        default_config: Default model settings
        output_dir: Directory to save results
        force: If True, overwrite existing results
    """
    name = config["name"]
    output_path = Path(output_dir) / f"{name}.json"
    
    # Check if already complete
    if output_path.exists() and not force:
        print(f"[SKIP] {name}: already complete")
        return
    
    print(f"\n{'='*60}")
    print(f"Running Learning Curves: {name}")
    print(f"{'='*60}")
    
    # Check if we have baseline data
    has_baseline = config.get("baseline_groups") and len(config["baseline_groups"]) > 0
    
    # Get normalization mode (default to standard)
    norm_mode = config.get("normalization_mode", "standard")
    valid_modes = ["standard", "adaptive_per_trial", "adaptive_global"]
    if norm_mode not in valid_modes:
        raise ValueError(f"normalization_mode must be one of {valid_modes}, got '{norm_mode}'")
    
    # Adaptive modes only work without baseline
    if has_baseline and norm_mode != "standard":
        raise ValueError(
            f"Adaptive normalization modes ('{norm_mode}') can only be used when "
            f"baseline_groups is empty or not specified. Set has_baseline=False."
        )
    
    # Get feature selection method
    selection_method = config.get("feature_selection", default_config.get("feature_selection"))
    
    print(f"Normalization mode: {norm_mode}")
    print(f"Feature selection: {selection_method if selection_method else 'None'}")
    
    # Load baseline data if specified
    if has_baseline:
        baseline_files = [feature_groups[g] for g in config["baseline_groups"]]
        baseline_df = load_and_merge_features(baseline_files, skip_every=config.get("skip_every"))
        print(f"Baseline data: {baseline_df.shape}")
        
        # Extract features for baseline (drop ID columns)
        X_baseline = drop_identifier_columns(baseline_df).drop(columns=["condition"], errors="ignore")
        y_baseline = baseline_df["condition"].values
    else:
        print("No baseline data - will start from minute 1")
        X_baseline = None
        y_baseline = None
    
    # Load experimental data
    exp_files = [feature_groups[g] for g in config["experimental_groups"]]
    exp_df = load_and_merge_features(exp_files, skip_every=config.get("skip_every"))
    print(f"Experimental data: {exp_df.shape}")

    if "window_index" not in exp_df.columns:
        raise ValueError("Expected 'window_index' in experimental data.")
    exp_df["w_idx"] = exp_df["window_index"].astype(int)

    GROUP_COLS = [c for c in ["participant", "trial_id", "trial", "session"] if c in exp_df.columns]
    if not GROUP_COLS:
        exp_df["__all__"] = 1
        GROUP_COLS = ["__all__"]
    
    # Settings
    minutes = config["minutes"]
    n_seeds = config.get("n_seeds", default_config.get("n_seeds", 20))
    
    # Results storage: results[minute][metric] = [scores across seeds]
    results = {
        m: {
            "BalancedAcc": [],
            "F1": [],
            "Kappa": [],
            "n_features": [],  # Track number of features selected per seed
        }
        for m in minutes
    }
    
    # Run across seeds
    for seed in tqdm(range(n_seeds), desc=f"{name}"):
        for minute in minutes:
            # Convert minutes to window index
            # With 60s windows and 50% overlap (30s step), window_index = minute * 2
            window_cutoff = minute * 2

            # build non-overlapping TEST via parity, then purge TRAIN neighbors
            # TEST: all windows >= cutoff that match parity (avoid test-test overlap)
            test_parity = window_cutoff % 2
            mask_test = (exp_df["w_idx"] >= window_cutoff) & ((exp_df["w_idx"] % 2) == test_parity)

            # TRAIN: all windows before cutoff
            mask_train = exp_df["w_idx"] < window_cutoff

            # If there are test windows, purge any overlapping TRAIN windows (neighbors ±1 in same group)
            if mask_test.any():
                test_keys = exp_df.loc[mask_test, GROUP_COLS + ["w_idx"]].copy()
                nbr_minus = test_keys.copy(); nbr_minus["w_idx"] = nbr_minus["w_idx"] - 1
                nbr_plus  = test_keys.copy();  nbr_plus["w_idx"]  = nbr_plus["w_idx"]  + 1
                banned = pd.concat([test_keys, nbr_minus, nbr_plus], ignore_index=True)
                banned = banned[banned["w_idx"] >= 0]

                ban_idx = pd.MultiIndex.from_frame(banned[GROUP_COLS + ["w_idx"]])
                train_frame = exp_df.loc[mask_train, GROUP_COLS + ["w_idx"]]
                train_idx = pd.MultiIndex.from_frame(train_frame)
                to_purge = train_idx.isin(ban_idx)
                # clear overlapped train windows
                purge_index = exp_df.loc[mask_train].index[to_purge]
                if len(purge_index) > 0:
                    mask_train.loc[purge_index] = False
            
            # Skip if no test data
            if mask_test.sum() == 0:
                continue
            
            # Prepare experimental training data
            exp_train = exp_df[mask_train]
            exp_test = exp_df[mask_test]
            
            # Build training set based on whether we have baseline and experimental data
            if minute == 0:
                # At minute 0, only use baseline data (if available)
                if not has_baseline:
                    # No baseline and no experimental data yet - skip
                    continue
                X_train = X_baseline
                y_train = y_baseline
            
            else:
                # minute > 0: use experimental data (and optionally baseline)
                if exp_train.empty:
                    # No experimental training data yet - skip
                    continue
                
                X_exp_train = drop_identifier_columns(exp_train).drop(columns=["condition"], errors="ignore")
                y_exp_train = exp_train["condition"].values
                
                if has_baseline:
                    # Combine baseline + experimental
                    X_train = pd.concat([X_baseline, X_exp_train], ignore_index=True)
                    y_train = np.concatenate([y_baseline, y_exp_train])
                else:
                    # Only experimental data
                    X_train = X_exp_train
                    y_train = y_exp_train
            
            # Prepare test data
            X_test = drop_identifier_columns(exp_test).drop(columns=["condition"], errors="ignore")
            y_test = exp_test["condition"].values
            
            # Ensure same features in train and test
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            
            # *** FEATURE SELECTION PER-SEED ***
            if selection_method == "backward":
                selected_features, _ = backward_elimination_rf(
                    X_train, y_train, random_state=seed
                )
            elif selection_method == "forward":
                # Forward selection not implemented
                selected_features = list(X_train.columns)
            elif selection_method is None:
                selected_features = list(X_train.columns)
            else:
                raise ValueError(f"Unknown feature_selection method: '{selection_method}'")
            
            if not selected_features:
                selected_features = list(X_train.columns)
            
            # Apply selected features
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            
            # Apply normalization based on mode
            if norm_mode == "standard":
                # Standard: fit on train, transform both
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            elif norm_mode == "adaptive_per_trial":
                # Per-trial adaptive: compute stats per (participant, condition) from train
                # Apply those stats to test data for that trial
                
                # We need to keep track of participant and condition for both train and test
                # Add them back temporarily
                train_with_meta = exp_train[["participant", "condition"]].copy()
                test_with_meta = exp_test[["participant", "condition"]].copy()
                
                # Initialize scaled arrays
                X_train_scaled = np.zeros_like(X_train.values, dtype=float)
                X_test_scaled = np.zeros_like(X_test.values, dtype=float)
                
                # Get unique (participant, condition) combinations in training data
                train_groups = train_with_meta.groupby(["participant", "condition"]).groups
                
                for (participant, condition), train_indices in train_groups.items():
                    # Get training data for this trial
                    trial_train_mask = (train_with_meta["participant"] == participant) & \
                                      (train_with_meta["condition"] == condition)
                    trial_train_data = X_train.values[trial_train_mask]
                    
                    # Compute mean and std from training data
                    train_mean = np.mean(trial_train_data, axis=0)
                    train_std = np.std(trial_train_data, axis=0)
                    train_std[train_std == 0] = 1.0  # Avoid division by zero
                    
                    # Transform training data for this trial
                    X_train_scaled[trial_train_mask] = (trial_train_data - train_mean) / train_std
                    
                    # Find corresponding test data for this trial
                    trial_test_mask = (test_with_meta["participant"] == participant) & \
                                     (test_with_meta["condition"] == condition)
                    
                    if trial_test_mask.any():
                        trial_test_data = X_test.values[trial_test_mask]
                        # Apply training stats to test data
                        X_test_scaled[trial_test_mask] = (trial_test_data - train_mean) / train_std
                
                # Handle any test trials that don't have corresponding training data
                # (use global stats as fallback)
                test_groups = test_with_meta.groupby(["participant", "condition"]).groups
                for (participant, condition), test_indices in test_groups.items():
                    if (participant, condition) not in train_groups:
                        # Fallback to global training stats
                        global_mean = np.mean(X_train.values, axis=0)
                        global_std = np.std(X_train.values, axis=0)
                        global_std[global_std == 0] = 1.0
                        
                        trial_test_mask = (test_with_meta["participant"] == participant) & \
                                         (test_with_meta["condition"] == condition)
                        trial_test_data = X_test.values[trial_test_mask]
                        X_test_scaled[trial_test_mask] = (trial_test_data - global_mean) / global_std
            
            elif norm_mode == "adaptive_global":
                # Global adaptive: compute stats from all training data
                # Apply to all test data
                train_mean = np.mean(X_train.values, axis=0)
                train_std = np.std(X_train.values, axis=0)
                train_std[train_std == 0] = 1.0  # Avoid division by zero
                
                # Transform both train and test using training stats
                X_train_scaled = (X_train.values - train_mean) / train_std
                X_test_scaled = (X_test.values - train_mean) / train_std
            
            # Build and train classifier
            rf_kwargs = dict(RF_PARAMS)
            rf_kwargs["random_state"] = seed
            rf_kwargs.setdefault("n_jobs", -1)
            
            clf = RandomForestClassifier(**rf_kwargs)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            
            # Compute metrics
            results[minute]["BalancedAcc"].append(balanced_accuracy_score(y_test, y_pred))
            results[minute]["F1"].append(f1_score(y_test, y_pred, labels=LABELS, average="weighted"))
            results[minute]["Kappa"].append(cohen_kappa_score(y_test, y_pred, labels=LABELS))
            results[minute]["n_features"].append(len(selected_features))
    
    # Save results
    output_data = {
        "name": name,
        "config": config,
        "has_baseline": has_baseline,
        "normalization_mode": norm_mode,
        "feature_selection": selection_method,
        "minutes": minutes,
        "results": results,
        "n_seeds": n_seeds,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Completed: {name}")
    
    # Print summary
    print(f"\nLearning Curve Summary (Balanced Accuracy) - {norm_mode}:")
    print(f"{'Minute':<10} {'Mean':<10} {'Std':<10} {'N':<10} {'Features':<15}")
    print("-" * 60)
    for minute in minutes:
        if results[minute]["BalancedAcc"]:
            mean_acc = np.mean(results[minute]["BalancedAcc"])
            std_acc = np.std(results[minute]["BalancedAcc"], ddof=1)
            n_samples = len(results[minute]["BalancedAcc"])
            mean_feats = np.mean(results[minute]["n_features"])
            std_feats = np.std(results[minute]["n_features"], ddof=1)
            print(f"{minute:<10} {mean_acc*100:.2f}%    {std_acc*100:.2f}%    {n_samples:<10} "
                  f"{mean_feats:.1f} ± {std_feats:.1f}")