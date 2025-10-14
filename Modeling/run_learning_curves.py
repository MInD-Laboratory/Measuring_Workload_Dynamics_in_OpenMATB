#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learning Curve Experiments for Workload Detection

This script runs temporal prediction experiments to evaluate model performance
as training duration increases. Tests how quickly the model can learn to predict
cognitive load with limited training data.

Two main split strategies:
  1. Random Split: Stratified 80/20 split across all windows
  2. Leave-Participant-Out: Hold out ~20% of participants entirely

Usage:
    python run_learning_curves.py [--force] [--dry-run]

Configuration:
    Edit the LEARNING_CURVES_CONFIG dictionary below to define which experiments to run.
    Default model settings are defined in DEFAULT_MODEL_CONFIG.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import helper utilities from pipeline_utils module
from pipeline_utils import (
    run_learning_curve_experiment,
    prompt_user_action,
    get_config_suffix
)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Output directory for all results
OUTPUT_DIR = Path("model_output") / "lc_models"

# Default settings applied to all models unless overridden
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 10,                    # Number of random seeds for reliability
    "feature_selection": "None",  # Options: "forward", "backward", None
    "selection_level": "metric",      # Options: "metric", "group"
    "selection_stopping": "drop",     # Options: "drop", "best"
    "use_pca": False,                 # Whether to apply PCA after feature selection
    "pca_variance": 0.95,             # Variance to retain if using PCA
    "write_cm": True,                 # Whether to save confusion matrices
    "tune_hyperparameters": False,    # Whether to tune RF hyperparameters (once per minute)
    "tune_n_iter": 30,                # Number of parameter settings sampled for tuning
    "tune_cv_folds": 5,               # Number of CV folds for hyperparameter tuning
}


# ============================================================================
# LEARNING CURVES CONFIGURATION
# ============================================================================
LEARNING_CURVES_CONFIG = {

    "enabled": True,  # Set to True to run learning curve experiments
    "description": "Learning curves testing different baseline approaches and feature combinations",
    "experiments": [

        # ==========================================================
        # EXPERIMENTAL ONLY (no baseline data)
        # Starts at minute 1 - only experimental features available
        # ==========================================================
        {
            "name": "lc_exp_only_linear",
            "baseline_concatenate_groups": [],  # No baseline trial windows
            "baseline_aggregate_groups": [],     # No baseline aggregate features
            "experimental_groups": ["procrustes_linear_exp"],
            "minutes": list(range(1, 8)),
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        {
            "name": "lc_exp_only_linear_perf",
            "baseline_concatenate_groups": [],
            "baseline_aggregate_groups": [],
            "experimental_groups": ["procrustes_linear_exp", "performance_exp"],
            "minutes": list(range(1, 8)),
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        # ==========================================================
        # BASELINE AGGREGATES (participant-level features)
        # Starts at minute 0 - can predict from baseline individual differences alone
        # Then adds experimental features as they accumulate
        # ==========================================================
        {
            "name": "lc_baseline_agg_linear",
            "baseline_concatenate_groups": [],
            "baseline_aggregate_groups": ["baseline_linear_procrustes"],
            "experimental_groups": ["procrustes_linear_exp"],
            "minutes": list(range(0, 8)),  # Can start at minute 0!
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        {
            "name": "lc_baseline_agg_linear_perf",
            "baseline_concatenate_groups": [],
            "baseline_aggregate_groups": ["baseline_linear_procrustes", "baseline_performance"],
            "experimental_groups": ["procrustes_linear_exp", "performance_exp"],
            "minutes": list(range(0, 8)),  # Can start at minute 0!
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        {
            "name": "lc_baseline_agg_all",
            "baseline_concatenate_groups": [],
            "baseline_aggregate_groups": ["baseline_linear_procrustes", "baseline_rqa_procrustes", "baseline_performance"],
            "experimental_groups": ["procrustes_linear_exp", "procrustes_rqa_exp", "performance_exp"],
            "minutes": list(range(0, 8)),
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        # ==========================================================
        # BASELINE CONCATENATION (baseline trial windows)
        # Starts at minute 0 - trains on separate baseline trial data
        # Then adds experimental windows for minutes 1+
        # ==========================================================
        {
            "name": "lc_baseline_concat_linear",
            "baseline_concatenate_groups": ["procrustes_linear_bsl"],
            "baseline_aggregate_groups": [],
            "experimental_groups": ["procrustes_linear_exp"],
            "minutes": list(range(0, 8)),  # Can start at minute 0 with baseline windows
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        # ==========================================================
        # COMBINED: Both baseline approaches
        # Baseline windows + baseline aggregates + experimental features
        # ==========================================================
        {
            "name": "lc_baseline_both_linear",
            "baseline_concatenate_groups": ["procrustes_linear_bsl"],
            "baseline_aggregate_groups": ["baseline_linear_procrustes"],
            "experimental_groups": ["procrustes_linear_exp"],
            "minutes": list(range(0, 8)),
            "skip_every": 2,
            "normalization_mode": "standard",
        },
        {
            "name": "lc_baseline_both_performance_linear",
            "baseline_concatenate_groups": ["procrustes_linear_bsl", "performance_bsl"],
            "baseline_aggregate_groups": ["baseline_linear_procrustes", "baseline_performance"],
            "experimental_groups": ["procrustes_linear_exp", "performance_exp"],
            "minutes": list(range(0, 8)),
            "skip_every": 2,
            "normalization_mode": "standard",
        },

        # ==========================================================
        # ADAPTIVE NORMALIZATION (commented out for speed)
        # Uncomment to test adaptive normalization modes
        # Note: Adaptive modes cannot use baseline_concatenate_groups
        # ==========================================================
        # {
        #     "name": "lc_adaptive_per_trial",
        #     "baseline_concatenate_groups": [],
        #     "baseline_aggregate_groups": [],
        #     "experimental_groups": ["procrustes_linear_exp"],
        #     "minutes": list(range(1, 8)),
        #     "skip_every": 2,
        #     "normalization_mode": "adaptive_per_trial",
        # },
        # {
        #     "name": "lc_agg_adaptive_per_trial",
        #     "baseline_concatenate_groups": [],
        #     "baseline_aggregate_groups": ["baseline_linear_procrustes", "baseline_performance"],
        #     "experimental_groups": ["procrustes_linear_exp"],
        #     "minutes": list(range(1, 8)),
        #     "skip_every": 2,
        #     "normalization_mode": "adaptive_per_trial",
        # },

    ],
}


# ============================================================================
# FEATURE GROUP DEFINITIONS
# Map feature group names to (filepath, phase) tuples
# ============================================================================

FEATURE_GROUPS = {
    # ========================================================================
    # EXPERIMENTAL FEATURES (used in learning curves)
    # ========================================================================

    # Linear pose features - different normalizations
    "procrustes_linear_exp": ("../Pose/data/processed_data/experimental/linear_metrics/procrustes_global_linear.csv", "main"),
    "none_linear_exp": ("../Pose/data/processed_data/experimental/linear_metrics/original_linear.csv", "main"),

    # RQA features - different normalizations
    "procrustes_rqa_exp": ("../Pose/data/rqa/experimental_procrustes_global_rqa_crqa.csv", "main"),
    "none_rqa_exp": ("../Pose/data/rqa/experimental_original_rqa_crqa.csv", "main"),

    # Task Performance metrics
    "performance_exp": ("../performance/data/out/performance_exp.csv", "main"),

    # ========================================================================
    # Concatenation BASELINE APPROACH (concatenation of baseline trial windows)
    # ========================================================================
    "procrustes_linear_bsl": ("../Pose/data/processed_data/baseline/linear_metrics/procrustes_global_linear.csv", "pre"),
    "none_linear_bsl": ("../Pose/data/processed_data/baseline/linear_metrics/original_linear.csv", "pre"),
    "procrustes_rqa_bsl": ("../Pose/data/rqa/baseline_procrustes_global_rqa_crqa.csv", "pre"),
    "none_rqa_bsl": ("../Pose/data/rqa/baseline_original_rqa_crqa.csv", "pre"),
    "performance_bsl": ("../performance/data/out/performance_bsl.csv", "pre"),

    # ========================================================================
    # Aggregates BASELINE APPROACH (aggregates - min/max/range across L/M/H conditions)
    # Generated by: python prepare_baseline_features.py
    # ========================================================================
    "baseline_performance": ("baseline_features/performance_baseline.csv", "main"),
    "baseline_linear_procrustes": ("baseline_features/linear_procrustes_baseline.csv", "main"),
    "baseline_linear_original": ("baseline_features/linear_original_baseline.csv", "main"),
    "baseline_rqa_procrustes": ("baseline_features/rqa_procrustes_baseline.csv", "main"),
    "baseline_rqa_original": ("baseline_features/rqa_original_baseline.csv", "main"),
}

# ============================================================================
# CLEANUP STAGED DATA FILES
# ============================================================================
def clean_staged_data(filepath: Path):
    """Remove metadata columns if they exist."""
    try:
        df = pd.read_csv(filepath)
        drop_cols = ["source", "t_start_frame", "t_end_frame", "window_start", "window_end"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df.to_csv(filepath, index=False)
    except Exception as e:
        print(f"[WARN] Could not clean {filepath}: {e}")


# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run learning curve experiments for workload detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_learning_curves.py                # Run all learning curve experiments
  python run_learning_curves.py --force        # Overwrite all existing results
  python run_learning_curves.py --dry-run      # Show what would be run without executing
        """
    )

    # Overwrite existing results
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite all existing results and recompute from scratch"
    )

    # Resume incomplete experiments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete experiments from last checkpoint"
    )

    # Dry run (no execution)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )

    return parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution."""
    args = parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LEARNING CURVE EXPERIMENTS - WORKLOAD DETECTION")
    print("=" * 80)

    # Note: No longer cleaning source files - we load directly from source directories
    # The load_and_merge_features function handles column filtering automatically

    # --------------------------------------------------------------
    # 1. Collect learning curve experiments
    # --------------------------------------------------------------
    lc_experiments = LEARNING_CURVES_CONFIG["experiments"] if LEARNING_CURVES_CONFIG["enabled"] else []

    # --------------------------------------------------------------
    # 2. Display planned runs
    # --------------------------------------------------------------
    if lc_experiments:
        print(f"\n[INFO] Configured {len(lc_experiments)} learning curve experiments")
        print(f"\n  {LEARNING_CURVES_CONFIG.get('description', '')}")
        for exp in lc_experiments:
            print(f"    - {exp['name']}")
    else:
        print("\n[WARN] No learning curve experiments configured to run!")
        return

    # --------------------------------------------------------------
    # 3. Handle dry-run mode
    # --------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] No experiments will be executed.")
        return

    # --------------------------------------------------------------
    # 4. Handle existing results and user prompt
    # --------------------------------------------------------------
    if not args.force and not args.resume:
        existing_experiments = []
        for exp in lc_experiments:
            # Merge with defaults to get correct config suffix
            merged_config = DEFAULT_MODEL_CONFIG.copy()
            merged_config.update(exp)
            suffix = get_config_suffix(merged_config)
            if (OUTPUT_DIR / f"{exp['name']}{suffix}.json").exists():
                existing_experiments.append(exp["name"])

        if existing_experiments:
            print(f"\n[INFO] Found {len(existing_experiments)} existing results")
            action = prompt_user_action()

            if action == "overwrite":
                args.force = True
            elif action == "continue":
                args.resume = True
            elif action == "skip":
                pass
            else:  # cancel
                print("Exiting.")
                return

    # --------------------------------------------------------------
    # 5. Run Learning Curves
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING LEARNING CURVE EXPERIMENTS")
    print("=" * 80)

    experiments_to_run = []
    for lc_config in lc_experiments:
        name = lc_config["name"]
        # Merge with defaults to get correct config suffix
        merged_config = DEFAULT_MODEL_CONFIG.copy()
        merged_config.update(lc_config)
        suffix = get_config_suffix(merged_config)
        output_path = OUTPUT_DIR / f"{name}{suffix}.json"

        if args.force or not output_path.exists():
            experiments_to_run.append(lc_config)
        else:
            print(f"✓ Skipping (complete): {name}")

    # Execute experiments
    if experiments_to_run:
        print(f"\n→ Running {len(experiments_to_run)} experiments...")
        for lc_config in tqdm(experiments_to_run, desc="Experiments"):
            name = lc_config["name"]
            try:
                run_learning_curve_experiment(
                    config=lc_config,
                    feature_groups=FEATURE_GROUPS,
                    default_config=DEFAULT_MODEL_CONFIG,
                    output_dir=OUTPUT_DIR,
                    force=args.force,
                    resume=args.resume,
                )
            except Exception as e:
                print(f"\n✗ Error running {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n✓ All learning curve experiments already complete!")

    # --------------------------------------------------------------
    # 6. Print Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✓ LEARNING CURVE EXPERIMENTS COMPLETE")
    print("=" * 80)

    # Count completed experiments
    completed = 0
    for exp in lc_experiments:
        merged_config = DEFAULT_MODEL_CONFIG.copy()
        merged_config.update(exp)
        suffix = get_config_suffix(merged_config)
        if (OUTPUT_DIR / f"{exp['name']}{suffix}.json").exists():
            completed += 1
    print(f"\nCompleted: {completed}/{len(lc_experiments)} experiments")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Open the visualization notebook to view learning curves")
    print("  2. Analyze how performance improves with training duration")


# Entry point
if __name__ == "__main__":
    main()
