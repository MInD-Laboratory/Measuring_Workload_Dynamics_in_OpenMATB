#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest Pipeline for Workload Detection

This pipeline runs Random Forest classifiers across different:
  - Normalization methods (none, minmax, zscore)
  - Feature types (linear pose, RQA/nonlinear, performance)
  - Validation strategies (random, participant)
  - Training configurations (with/without baseline data)
  - Learning curves (incremental training data)

Usage:
    python run_pipeline.py [--force] [--resume] [--learning-curves] [--help]

Note:
    `--continue` is kept as a backwards-compatible alias for `--resume`.

Configuration:
    Edit the EXPERIMENT_CONFIG dictionary below to define which experiments to run.
    Default model settings are defined in DEFAULT_MODEL_CONFIG.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Import helper utilities from your pipeline_utils module
from pipeline_utils import (
    get_all_model_configs,
    run_single_model,
    run_learning_curve_experiment,
    check_model_complete,
    print_summary,
    prompt_user_action
)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Output directory for all results
OUTPUT_DIR = Path("model_output")

# Default settings applied to all models unless overridden
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 20,                    # Number of random seeds for reliability
    "feature_selection": "backward",  # Options: "forward", "backward", None
    "selection_level": "metric",      # Options: "metric", "group"
    "selection_stopping": "drop",     # Options: "drop", "best"
    "use_pca": False,                 # Whether to apply PCA after feature selection
    "pca_variance": 0.95,             # Variance to retain if using PCA
    "write_cm": True,                 # Whether to save confusion matrices
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# Define which experiments to run
# ============================================================================

EXPERIMENT_CONFIG = {
    # ========================================
    # Section 1: Normalization Comparison
    # Compare different normalization methods on pose features
    # ========================================
    "normalization_comparison": {
        "enabled": True,
        "description": "Compare none/minmax/zscore normalization on linear pose",
        "experiments": [
            # Random split
            {"name": "linear_pose_none_random", "feature_groups": ["linear_pose_exp_none"], "split_strategy": "random"},
            {"name": "linear_pose_minmax_random", "feature_groups": ["linear_pose_exp_minmax"], "split_strategy": "random"},
            {"name": "linear_pose_zscore_random", "feature_groups": ["linear_pose_exp_zscore"], "split_strategy": "random"},
            # Participant split
            {"name": "linear_pose_none_participant", "feature_groups": ["linear_pose_exp_none"], "split_strategy": "participant"},
            {"name": "linear_pose_minmax_participant", "feature_groups": ["linear_pose_exp_minmax"], "split_strategy": "participant"},
            {"name": "linear_pose_zscore_participant", "feature_groups": ["linear_pose_exp_zscore"], "split_strategy": "participant"},
        ]
    },

    "normalization_comparison_rqa": {
        "enabled": True,
        "description": "Compare none/minmax/zscore normalization on linear pose",
        "experiments": [
            # Random split
            {"name": "linear_pose_none_random", "feature_groups": ["rqa_pose_exp_none"], "split_strategy": "random"},
            {"name": "linear_pose_minmax_random", "feature_groups": ["rqa_pose_exp_minmax"], "split_strategy": "random"},
            {"name": "linear_pose_zscore_random", "feature_groups": ["rqa_pose_exp_zscore"], "split_strategy": "random"},
            # Participant split
            {"name": "linear_pose_none_participant", "feature_groups": ["rqa_pose_exp_none"], "split_strategy": "participant"},
            {"name": "linear_pose_minmax_participant", "feature_groups": ["rqa_pose_exp_minmax"], "split_strategy": "participant"},
            {"name": "linear_pose_zscore_participant", "feature_groups": ["rqa_pose_exp_zscore"], "split_strategy": "participant"},
        ]
    },

    # ========================================
    # Section 2: Feature Type Comparison
    # Compare linear vs nonlinear vs combined features
    # ========================================
    # "feature_comparison": {
    #     "enabled": True,
    #     "description": "Compare linear, RQA, and combined features",
    #     "experiments": [
    #         # Linear features only
    #         {"name": "linear_pose_exp_random", "feature_groups": ["linear_pose_exp_zscore"], "split_strategy": "random"},
    #         {"name": "linear_pose_exp_participant", "feature_groups": ["linear_pose_exp_zscore"], "split_strategy": "participant"},

    #         # RQA features only
    #         {"name": "rqa_pose_exp_random", "feature_groups": ["rqa_pose_exp_zscore"], "split_strategy": "random"},
    #         {"name": "rqa_pose_exp_participant", "feature_groups": ["rqa_pose_exp_zscore"], "split_strategy": "participant"},

    #         # Combined linear + RQA
    #         {"name": "linear_rqa_exp_random", "feature_groups": ["linear_pose_exp_zscore", "rqa_pose_exp_zscore"], "split_strategy": "random"},
    #         {"name": "linear_rqa_exp_participant", "feature_groups": ["linear_pose_exp_zscore", "rqa_pose_exp_zscore"], "split_strategy": "participant"},
    #     ]
    # },

    # ========================================
    # Section 3: Performance Metrics
    # Test performance metrics alone or with pose features
    # ========================================
    "performance_metrics": {
        "enabled": True,
        "description": "Evaluate performance metrics",
        "experiments": [
            {"name": "performance_exp_random", "feature_groups": ["performance_exp"], "split_strategy": "random"},
            {"name": "performance_exp_participant", "feature_groups": ["performance_exp"], "split_strategy": "participant"},
        ]
    },

    # ========================================
    # Section 4: Baseline Data Impact
    # Add baseline data to experimental data
    # ========================================
    # "baseline_impact": {
    #     "enabled": False,  # Enable if you have baseline data
    #     "description": "Test impact of adding baseline data",
    #     "experiments": [
    #         {
    #             "name": "linear_exp_bsl_pose",
    #             "feature_groups": ["linear_pose_exp_zscore", "linear_pose_bsl_zscore"],
    #             "split_strategy": "participant",
    #             "n_seeds": 10,  # Override default seed count
    #         },
    #         {
    #             "name": "linear_exp_bsl_perf",
    #             "feature_groups": ["linear_pose_exp_zscore", "performance_bsl"],
    #             "split_strategy": "participant",
    #             "n_seeds": 10,
    #         },
    #     ]
    # },
}

# ============================================================================
# LEARNING CURVES CONFIGURATION
# ============================================================================
LEARNING_CURVES_CONFIG = {
    "enabled": True,  # Set to True to run learning curves
    "description": "Incremental training with growing data",
    
    # Define learning curve experiments
    "experiments": [
        # Linear pose + performance WITH baseline
        {
            "name": "lc_linear_perf_with_baseline",
            "baseline_groups": ["linear_pose_bsl_zscore", "performance_bsl"],
            "experimental_groups": ["linear_pose_exp_zscore", "performance_exp"],
            "minutes": list(range(0, 8)),  # 0-7 minutes (starts at 0 with baseline)
            "skip_every": 2,
            "n_seeds": 20,
        },
        
        # Linear pose + performance WITHOUT baseline
        {
            "name": "lc_linear_perf_no_baseline",
            "baseline_groups": [],  # Empty = no baseline
            "experimental_groups": ["linear_pose_exp_zscore", "performance_exp"],
            "minutes": list(range(1, 8)),  # 1-7 minutes (starts at 1, no baseline)
            "skip_every": 2,
            "n_seeds": 20,
        },
        
        # Linear pose ONLY WITH baseline
        {
            "name": "lc_linear_only_with_baseline",
            "baseline_groups": ["linear_pose_bsl_zscore"],
            "experimental_groups": ["linear_pose_exp_zscore"],
            "minutes": list(range(0, 8)),
            "skip_every": 2,
            "n_seeds": 20,
        },

        {
            "name": "lc_linear_only_wo_baseline",
            "baseline_groups": [],
            "experimental_groups": ["linear_pose_exp_zscore"],
            "minutes": list(range(1, 8)),
            "skip_every": 2,
            "n_seeds": 20,
        },
        
        # # RQA + performance WITH baseline
        # {
        #     "name": "lc_rqa_perf_with_baseline",
        #     "baseline_groups": ["rqa_pose_bsl_zscore", "performance_bsl"],
        #     "experimental_groups": ["rqa_pose_exp_zscore", "performance_exp"],
        #     "minutes": list(range(0, 8)),
        #     "skip_every": 2,
        #     "n_seeds": 20,
        # },
        
        # # ALL features (linear + RQA + performance) WITH baseline
        # {
        #     "name": "lc_all_features_with_baseline",
        #     "baseline_groups": ["linear_pose_bsl_zscore", "rqa_pose_bsl_zscore", "performance_bsl"],
        #     "experimental_groups": ["linear_pose_exp_zscore", "rqa_pose_exp_zscore", "performance_exp"],
        #     "minutes": list(range(0, 8)),
        #     "skip_every": 2,
        #     "n_seeds": 20,
        # },
        
        # # ALL features WITHOUT baseline
        # {
        #     "name": "lc_all_features_no_baseline",
        #     "baseline_groups": [],
        #     "experimental_groups": ["linear_pose_exp_zscore", "rqa_pose_exp_zscore", "performance_exp"],
        #     "minutes": list(range(1, 8)),
        #     "skip_every": 2,
        #     "n_seeds": 20,
        # },
    ]
}

# ============================================================================
# FEATURE GROUP DEFINITIONS
# Map feature group names to (filepath, phase) tuples
# ============================================================================

FEATURE_GROUPS = {
    # Linear pose features - different normalizations
    "linear_pose_exp_none": ("_staged_data/linear_pose/linear_original_original_exp.csv", "main"),
    "linear_pose_exp_minmax": ("_staged_data/linear_pose/linear_minmax_original_exp.csv", "main"),
    "linear_pose_exp_zscore": ("_staged_data/linear_pose/linear_zscore_original_exp.csv", "main"),

    "linear_pose_bsl_none": ("_staged_data/linear_pose/linear_original_original_bsl.csv", "pre"),
    "linear_pose_bsl_minmax": ("_staged_data/linear_pose/linear_minmax_original_bsl.csv", "pre"),
    "linear_pose_bsl_zscore": ("_staged_data/linear_pose/linear_zscore_original_bsl.csv", "pre"),

    # RQA features - different normalizations
    "rqa_pose_exp_none": ("_staged_data/rqa_pose/exp_original_original_rqa.csv", "main"),
    #"rqa_pose_exp_minmax": ("exp_minmax_rqa.csv", "main"),
    #"rqa_pose_exp_zscore": ("exp_zscore_rqa.csv", "main"),

    #"rqa_pose_bsl_none": ("bsl_none_rqa.csv", "pre"),
    #"rqa_pose_bsl_minmax": ("bsl_minmax_rqa.csv", "pre"),
    #"rqa_pose_bsl_zscore": ("bsl_zscore_rqa.csv", "pre"),

    # Performance metrics (no normalization needed)
    "performance_exp": ("_staged_data/performance/performance_exp.csv", "main"),
    "performance_bsl": ("_staged_data/performance/performance_bsl.csv", "pre"),
}

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Random Forest workload detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run all enabled experiments
  python run_pipeline.py --force            # Overwrite all existing results
  python run_pipeline.py --resume           # Resume incomplete experiments
  python run_pipeline.py --learning-curves  # Only run learning curves
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

    # Learning curves only
    parser.add_argument(
        "--learning-curves",
        action="store_true",
        help="Only run learning curve experiments (skip regular models)"
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
    print("RANDOM FOREST WORKLOAD DETECTION PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------------
    # 1. Generate all model configurations (excluding learning curves)
    # --------------------------------------------------------------
    all_models = get_all_model_configs(
        EXPERIMENT_CONFIG,
        FEATURE_GROUPS,
        DEFAULT_MODEL_CONFIG,
        skip_learning_curves=args.learning_curves
    )

    # --------------------------------------------------------------
    # 2. Collect learning curve experiments
    # --------------------------------------------------------------
    lc_experiments = []
    if LEARNING_CURVES_CONFIG["enabled"] and not args.learning_curves:
        lc_experiments = LEARNING_CURVES_CONFIG["experiments"]
    elif args.learning_curves:
        lc_experiments = LEARNING_CURVES_CONFIG["experiments"]
        all_models = []  # Skip regular models

    # --- BEGIN: normalize all_models to a dict[name -> config] ---
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
    # 3. Display planned runs
    # --------------------------------------------------------------
    if all_models:
        print(f"\n[INFO] Configured {len(all_models)} regular models")
        for section_key, section in EXPERIMENT_CONFIG.items():
            if section.get("enabled", True):
                print(f"\n  {section_key}: {section.get('description', '')}")
                for exp in section.get("experiments", []):
                    print(f"    - {exp['name']}")

    if lc_experiments:
        print(f"\n[INFO] Configured {len(lc_experiments)} learning curve experiments")
        for exp in lc_experiments:
            print(f"    - {exp['name']}")

    # --------------------------------------------------------------
    # 4. Handle dry-run mode
    # --------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] No models will be executed.")
        return

    # --------------------------------------------------------------
    # 5. Handle existing results and user prompt
    # --------------------------------------------------------------
    if not args.force and not args.resume:
        existing_models = []
        if isinstance(all_models, dict) and all_models:
            existing_models = [
                name for name in all_models.keys()
                if check_model_complete(name, OUTPUT_DIR)
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
    # 6. Run Regular Models
    # --------------------------------------------------------------
    if all_models:
        print("\n" + "=" * 80)
        print("RUNNING REGULAR MODELS")
        print("=" * 80)

        # Filter models that need to run
        models_to_run = []
        for name, config in all_models.items():
            if args.force or not check_model_complete(name, OUTPUT_DIR):
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
                        resume=args.resume,  # renamed param
                    )
                except Exception as e:
                    print(f"\n✗ Error running {name}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("\n✓ All regular models already complete!")

    # --------------------------------------------------------------
    # 7. Run Learning Curves
    # --------------------------------------------------------------
    if lc_experiments:
        print("\n" + "=" * 80)
        print("RUNNING LEARNING CURVE EXPERIMENTS")
        print("=" * 80)

        for lc_config in lc_experiments:
            name = lc_config["name"]
            output_path = OUTPUT_DIR / f"{name}.json"

            if args.force or not output_path.exists():
                print(f"\n→ Running: {name}")
                try:
                    run_learning_curve_experiment(
                        config=lc_config,
                        feature_groups=FEATURE_GROUPS,
                        default_config=DEFAULT_MODEL_CONFIG,
                        output_dir=OUTPUT_DIR,
                        force=args.force,
                    )
                except Exception as e:
                    print(f"✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"✓ Skipping (complete): {name}")

    # --------------------------------------------------------------
    # 8. Print Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)

    print_summary(OUTPUT_DIR)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Open the visualization notebook to view results")
    print("  2. Check experiment_log.csv for detailed metrics")


# Entry point
if __name__ == "__main__":
    main()
