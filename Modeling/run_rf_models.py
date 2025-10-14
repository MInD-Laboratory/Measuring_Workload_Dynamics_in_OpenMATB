#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest Models for Workload Detection

This script runs overall Random Forest classifier experiments across:
  - Feature types (linear pose, RQA/nonlinear, performance)
  - Validation strategies (random split, leave-participant-out)

Two main split strategies:
  1. Random Split: Stratified 80/20 split across all windows
  2. Leave-Participant-Out: Hold out ~20% of participants entirely

Usage:
    python run_rf_models.py [--force] [--resume] [--dry-run]

Configuration:
    Edit the EXPERIMENT_CONFIG dictionary below to define which experiments to run.
    Default model settings are defined in DEFAULT_MODEL_CONFIG.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import helper utilities from pipeline_utils module
from pipeline_utils import (
    get_all_model_configs,
    run_single_model,
    check_model_complete,
    print_summary,
    prompt_user_action
)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Output directory for all results
OUTPUT_DIR = Path("model_output") / "rf_models"

# Default settings applied to all models unless overridden
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 15,                    # Number of random seeds for reliability
    "feature_selection": "None",  # Options: "forward", "backward", None
    "selection_level": "metric",      # Options: "metric", "group"
    "selection_stopping": "drop",     # Options: "drop", "best"
    "use_pca": False,                 # Options: True or False; Whether to apply PCA after feature selection
    "pca_variance": 0.95,             # Variance to retain if using PCA
    "write_cm": True,                 # Options: True or False; Whether to save confusion matrices
    "tune_hyperparameters": False,     # Options: True or False; Whether to tune RF hyperparameters with RandomizedSearchCV
    "tune_n_iter": 30,                # Number of parameter settings sampled for tuning
    "tune_cv_folds": 5,               # Number of CV folds for hyperparameter tuning
}


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_CONFIG = {
    # ==========================================================
    # 1. Linear vs RQA vs Combined (Procrustes & None)
    # ==========================================================
    "feature_comparison": {
        "enabled": True, # True or False
        "description": "Compare Procrustes vs None for linear, RQA, and combined features",
        "experiments": [
            # -------------------------------
            # Linear only
            # -------------------------------
            {"name": "linear_procrustes_random", "feature_groups": ["procrustes_linear_exp"], "split_strategy": "random"},
            {"name": "linear_procrustes_participant", "feature_groups": ["procrustes_linear_exp"], "split_strategy": "participant"},
            {"name": "linear_none_random", "feature_groups": ["none_linear_exp"], "split_strategy": "random"},
            {"name": "linear_none_participant", "feature_groups": ["none_linear_exp"], "split_strategy": "participant"},

            # # -------------------------------
            # # RQA only
            # # -------------------------------
            {"name": "rqa_procrustes_random", "feature_groups": ["procrustes_rqa_exp"], "split_strategy": "random"},
            {"name": "rqa_procrustes_participant", "feature_groups": ["procrustes_rqa_exp"], "split_strategy": "participant"},
            {"name": "rqa_none_random", "feature_groups": ["none_rqa_exp"], "split_strategy": "random"},
            {"name": "rqa_none_participant", "feature_groups": ["none_rqa_exp"], "split_strategy": "participant"},

            # # # -------------------------------
            # # # Combined linear + RQA
            # # # -------------------------------
            {"name": "combined_procrustes_random", "feature_groups": ["procrustes_linear_exp", "procrustes_rqa_exp"], "split_strategy": "random"},
            {"name": "combined_procrustes_participant", "feature_groups": ["procrustes_linear_exp", "procrustes_rqa_exp"], "split_strategy": "participant"},
            {"name": "combined_none_random", "feature_groups": ["none_linear_exp", "none_rqa_exp"], "split_strategy": "random"},
            {"name": "combined_none_participant", "feature_groups": ["none_linear_exp", "none_rqa_exp"], "split_strategy": "participant"},
        ],
    },

    # ==========================================================
    # 2. Performance Metrics (alone and combined with pose/RQA)
    # ==========================================================
    "performance_metrics": {
        "enabled": True,  # True or False
        "description": "Evaluate performance metrics alone and combined with linear/RQA features",
        "experiments": [
            # -------------------------------
            # Performance only
            # -------------------------------
            {"name": "performance_random", "feature_groups": ["performance_exp"], "split_strategy": "random"},
            {"name": "performance_participant", "feature_groups": ["performance_exp"], "split_strategy": "participant"},

            # -------------------------------
            # Linear + Performance
            # -------------------------------
            {"name": "linear_perf_procrustes_random", "feature_groups": ["procrustes_linear_exp", "performance_exp"], "split_strategy": "random"},
            {"name": "linear_perf_procrustes_participant", "feature_groups": ["procrustes_linear_exp", "performance_exp"], "split_strategy": "participant"},
            {"name": "linear_perf_none_random", "feature_groups": ["none_linear_exp", "performance_exp"], "split_strategy": "random"},
            {"name": "linear_perf_none_participant", "feature_groups": ["none_linear_exp", "performance_exp"], "split_strategy": "participant"},

            # -------------------------------
            # RQA + Performance
            # -------------------------------
            {"name": "rqa_perf_procrustes_random", "feature_groups": ["procrustes_rqa_exp", "performance_exp"], "split_strategy": "random"},
            {"name": "rqa_perf_procrustes_participant", "feature_groups": ["procrustes_rqa_exp", "performance_exp"], "split_strategy": "participant"},
            {"name": "rqa_perf_none_random", "feature_groups": ["none_rqa_exp", "performance_exp"], "split_strategy": "random"},
            {"name": "rqa_perf_none_participant", "feature_groups": ["none_rqa_exp", "performance_exp"], "split_strategy": "participant"},

            # -------------------------------
            # Linear + RQA + Performance (all features)
            # -------------------------------
            {"name": "all_procrustes_random", "feature_groups": ["procrustes_linear_exp", "procrustes_rqa_exp", "performance_exp"], "split_strategy": "random"},
            {"name": "all_procrustes_participant", "feature_groups": ["procrustes_linear_exp", "procrustes_rqa_exp", "performance_exp"], "split_strategy": "participant"},
            {"name": "all_none_random", "feature_groups": ["none_linear_exp", "none_rqa_exp", "performance_exp"], "split_strategy": "random"},
            {"name": "all_none_participant", "feature_groups": ["none_linear_exp", "none_rqa_exp", "performance_exp"], "split_strategy": "participant"},
        ],
    },

    # ==========================================================
    # 3. BASELINE COMPARISON EXPERIMENTS
    # Test whether baseline aggregates improve classification
    # Note: Model A (exp only) already exists as linear_procrustes_* in feature_comparison
    # ==========================================================
    "baseline_comparison": {
        "enabled": True, # True or False
        "description": "Test Model B variants (exp + different baseline aggregates). Compare against linear_procrustes_* from feature_comparison.",
        "experiments": [
            # ========================================
            # Model B1: Experimental + baseline performance aggregates
            # ========================================
            {"name": "modelB1_perf_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance"],
             "split_strategy": "random"},

            {"name": "modelB1_perf_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance"],
             "split_strategy": "participant"},

            # ========================================
            # Model B2: Experimental + baseline performance + baseline linear
            # ========================================
            {"name": "modelB2_perf_linear_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance", "baseline_linear_procrustes"],
             "split_strategy": "random"},

            {"name": "modelB2_perf_linear_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance", "baseline_linear_procrustes"],
             "split_strategy": "participant"},

            # ========================================
            # Model B3: Experimental + baseline performance + baseline RQA
            # ========================================
            {"name": "modelB3_perf_rqa_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance", "baseline_rqa_procrustes"],
             "split_strategy": "random"},

            {"name": "modelB3_perf_rqa_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance", "baseline_rqa_procrustes"],
             "split_strategy": "participant"},

            # ========================================
            # Model B4: Experimental + all baseline aggregates
            # ========================================
            {"name": "modelB4_all_baseline_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance", "baseline_linear_procrustes", "baseline_rqa_procrustes"],
             "split_strategy": "random"},

            {"name": "modelB4_all_baseline_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_performance", "baseline_linear_procrustes", "baseline_rqa_procrustes"],
             "split_strategy": "participant"},

             # ========================================
            # Model B5: Experimental + baseline linear
            # ========================================
            {"name": "modelB5_linear_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_linear_procrustes"],
             "split_strategy": "random"},

            {"name": "modelB5_linear_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_linear_procrustes"],
             "split_strategy": "participant"},

            # ========================================
            # Model B6: Experimental + baseline RQA
            # ========================================
            {"name": "modelB6_rqa_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_rqa_procrustes"],
             "split_strategy": "random"},

            {"name": "modelB6_rqa_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_rqa_procrustes"],
             "split_strategy": "participant"},

            # ========================================
            # Model B7: Experimental + + baseline linear + baseline RQA
            # ========================================
            {"name": "modelB7_linear_rqa_random",
             "feature_groups": ["procrustes_linear_exp", "baseline_linear_procrustes", "baseline_rqa_procrustes"],
             "split_strategy": "random"},

            {"name": "modelB7_linear_rqa_participant",
             "feature_groups": ["procrustes_linear_exp", "baseline_linear_procrustes", "baseline_rqa_procrustes"],
             "split_strategy": "participant"},
        ],
    },
}


# ============================================================================
# FEATURE GROUP DEFINITIONS
# Map feature group names to (filepath, phase) tuples
# ============================================================================

FEATURE_GROUPS = {
    # ========================================================================
    # MODEL A: Experimental features only (original approach)
    # ========================================================================

    # Linear pose features - different normalizations
    "procrustes_linear_exp": ("../Pose/data/processed_data/experimental/linear_metrics/procrustes_global_linear.csv", "main"),
    "procrustes_linear_bsl": ("../Pose/data/processed_data/baseline/linear_metrics/procrustes_global_linear.csv", "pre"),
    "none_linear_exp": ("../Pose/data/processed_data/experimental/linear_metrics/original_linear.csv", "main"),
    "none_linear_bsl": ("../Pose/data/processed_data/baseline/linear_metrics/original_linear.csv", "pre"),

    # RQA features - different normalizations
    "procrustes_rqa_exp": ("../Pose/data/rqa/experimental_procrustes_global_rqa_crqa.csv", "main"),
    "procrustes_rqa_bsl": ("../Pose/data/rqa/baseline_procrustes_global_rqa_crqa.csv", "pre"),
    "none_rqa_exp": ("../Pose/data/rqa/experimental_original_rqa_crqa.csv", "main"),
    "none_rqa_bsl": ("../Pose/data/rqa/baseline_original_rqa_crqa.csv", "pre"),

    # Task Performance metrics
    "performance_exp": ("../performance/data/out/performance_exp.csv", "main"),
    "performance_bsl": ("../performance/data/out/performance_bsl.csv", "pre"),

    # ========================================================================
    # MODEL B: Baseline aggregates (min, max, range across L/M/H baseline conditions)
    # These files contain ONLY baseline aggregate features (to be combined with exp features)
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
        description="Run Random Forest workload detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rf_models.py                # Run all enabled experiments
  python run_rf_models.py --force        # Overwrite all existing results
  python run_rf_models.py --resume       # Resume incomplete experiments
  python run_rf_models.py --dry-run      # Show what would be run without executing
        """
    )

    # Overwrite existing results
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite all existing results and recompute from scratch"
    )

    # Primary continuation flag
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Continue incomplete experiments (resume at seed level)"
    )

    # Backwards compatibility alias: --continue (maps to same dest)
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden from help output
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
    print("RANDOM FOREST WORKLOAD DETECTION - OVERALL MODELS")
    print("=" * 80)

    # Note: No longer cleaning source files - we load directly from source directories
    # The load_and_merge_features function handles column filtering automatically


    # --------------------------------------------------------------
    # 1. Generate all model configurations
    # --------------------------------------------------------------
    all_models = get_all_model_configs(
        EXPERIMENT_CONFIG,
        FEATURE_GROUPS,
        DEFAULT_MODEL_CONFIG,
        skip_learning_curves=False  # Not relevant for this script
    )

    # --- Normalize all_models to a dict[name -> config] ---
    def _to_name_config_dict(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            if not obj:
                return {}
            # Case A: list of dicts with a 'name' key
            if isinstance(obj[0], dict):
                out = {}
                for m in obj:
                    if "name" not in m:
                        raise ValueError("Model config dict missing 'name' key.")
                    name = m["name"]
                    cfg = {k: v for k, v in m.items() if k != "name"}
                    out[name] = cfg
                return out
            # Case B: list of (name, config) tuples
            try:
                return dict(obj)
            except Exception as e:
                raise TypeError(
                    "get_all_model_configs must return a dict, a list of dicts with 'name', "
                    "or a list of (name, config) pairs."
                ) from e
        raise TypeError("Unsupported type for all_models: {}".format(type(obj)))

    all_models = _to_name_config_dict(all_models)

    # --------------------------------------------------------------
    # 2. Display planned runs
    # --------------------------------------------------------------
    if all_models:
        print(f"\n[INFO] Configured {len(all_models)} RF models")
        for section_key, section in EXPERIMENT_CONFIG.items():
            if section.get("enabled", True):
                print(f"\n  {section_key}: {section.get('description', '')}")
                for exp in section.get("experiments", []):
                    print(f"    - {exp['name']}")
    else:
        print("\n[WARN] No models configured to run!")
        return

    # --------------------------------------------------------------
    # 3. Handle dry-run mode
    # --------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] No models will be executed.")
        return

    # --------------------------------------------------------------
    # 4. Handle existing results and user prompt
    # --------------------------------------------------------------
    if not args.force and not args.resume:
        existing_models = []
        if isinstance(all_models, dict) and all_models:
            existing_models = [
                name for name, config in all_models.items()
                if check_model_complete(name, OUTPUT_DIR, config)
            ]

        if existing_models:
            print(f"\n[INFO] Found {len(existing_models)} existing results")
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
    # 5. Run Models
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING RF MODELS")
    print("=" * 80)

    # Filter models that need to run
    models_to_run = []
    for name, config in all_models.items():
        if args.force or not check_model_complete(name, OUTPUT_DIR, config):
            models_to_run.append((name, config))
        else:
            print(f"✓ Skipping (complete): {name}")

    # Execute models
    if models_to_run:
        print(f"\n→ Running {len(models_to_run)} models...")
        for name, config in tqdm(models_to_run, desc="Models"):
            try:
                run_single_model(
                    name=name,
                    config=config,
                    output_dir=OUTPUT_DIR,
                    force=args.force,
                    resume=args.resume,
                )
            except Exception as e:
                print(f"\n✗ Error running {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n✓ All models already complete!")

    # --------------------------------------------------------------
    # 6. Print Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✓ RF MODELS COMPLETE")
    print("=" * 80)

    print_summary(OUTPUT_DIR)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Open the visualization notebook to view results")
    print("  2. Check experiment_log.csv for detailed metrics")


# Entry point
if __name__ == "__main__":
    main()
