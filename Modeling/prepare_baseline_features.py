#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare Baseline Features for Random Forest Models

This script computes participant-level baseline aggregates and creates new feature files
that can be merged with experimental data.

Baseline features capture individual differences by computing range, min, and max
across the three baseline conditions (L, M, H) for each participant.

  Model A: Experimental only (no baseline needed - already exists)
  Model B: Experimental + baseline aggregates (range, min, max across L/M/H baseline conditions)

Output files are created in: Modeling/baseline_features/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths (source data)
INPUT_PATHS = {
    "linear": {
        "procrustes_exp": "../Pose/data/processed_data/experimental/linear_metrics/procrustes_global_linear.csv",
        "procrustes_bsl": "../Pose/data/processed_data/baseline/linear_metrics/procrustes_global_linear.csv",
        "original_exp": "../Pose/data/processed_data/experimental/linear_metrics/original_linear.csv",
        "original_bsl": "../Pose/data/processed_data/baseline/linear_metrics/original_linear.csv",
    },
    "rqa": {
        "procrustes_exp": "../Pose/data/rqa/experimental_procrustes_global_rqa_crqa.csv",
        "procrustes_bsl": "../Pose/data/rqa/baseline_procrustes_global_rqa_crqa.csv",
        "original_exp": "../Pose/data/rqa/experimental_original_rqa_crqa.csv",
        "original_bsl": "../Pose/data/rqa/baseline_original_rqa_crqa.csv",
    },
    "performance": {
        "performance_exp": "../performance/data/out/performance_exp.csv",
        "performance_bsl": "../performance/data/out/performance_bsl.csv",
    }
}

# Output directory
OUTPUT_DIR = Path("baseline_features")

# Metadata columns to exclude from feature computation
METADATA_COLS = {"participant", "condition", "window_index", "window_start", "window_end",
                 "minute", "window_start_s", "window_end_s",
                 "source", "t_start_frame", "t_end_frame"}  # CSV metadata columns


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_columns(df):
    """Extract feature columns (exclude metadata)."""
    return [c for c in df.columns if c not in METADATA_COLS]


def compute_participant_baseline_aggregates(baseline_df):
    """
    Compute per-participant baseline statistics across L/M/H conditions.

    For each participant, compute min, max, and range across all baseline windows
    (spanning all three baseline conditions: L, M, H).

    Returns:
        DataFrame with columns: participant, {feature}_baseline_min, {feature}_baseline_max, {feature}_baseline_range
    """
    feature_cols = get_feature_columns(baseline_df)

    # Group by participant and compute statistics across all baseline conditions
    agg_dict = {}
    for col in feature_cols:
        agg_dict[f"{col}_baseline_min"] = (col, 'min')
        agg_dict[f"{col}_baseline_max"] = (col, 'max')
        # Range will be computed after aggregation

    baseline_agg = baseline_df.groupby("participant").agg(**agg_dict).reset_index()

    # Compute range = max - min for each feature
    for col in feature_cols:
        baseline_agg[f"{col}_baseline_range"] = (
            baseline_agg[f"{col}_baseline_max"] - baseline_agg[f"{col}_baseline_min"]
        )

    return baseline_agg


def create_baseline_feature_file(exp_df, baseline_agg):
    """
    Create baseline aggregate feature file.

    Returns ONLY the baseline aggregates (min, max, range) aligned with experimental windows.
    These can be combined with experimental features in run_rf_models.py.

    Args:
        exp_df: Experimental data (windows x features) - used only for structure/metadata
        baseline_agg: Baseline aggregates (participant x baseline_features)

    Returns:
        DataFrame with participant, condition, window_index + baseline aggregate features only
    """
    # Select only metadata columns from experimental data
    metadata_cols = ["participant", "condition", "window_index"]
    exp_metadata = exp_df[metadata_cols].copy()

    # Merge baseline aggregates (broadcasts to all windows)
    result = exp_metadata.merge(baseline_agg, on="participant", how="left")

    return result


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_feature_type(feature_type, paths):
    """
    Process one feature type (linear, rqa, or performance).

    Creates Model B feature files (experimental + baseline aggregates).
    """
    print(f"\n{'='*70}")
    print(f"Processing: {feature_type.upper()}")
    print(f"{'='*70}")

    results = {}

    # Process each alignment/variant
    for key, exp_path in paths.items():
        if not key.endswith("_exp"):
            continue

        # Get corresponding baseline path
        bsl_key = key.replace("_exp", "_bsl")
        if bsl_key not in paths:
            print(f"  [SKIP] No baseline for {key}")
            continue

        bsl_path = paths[bsl_key]

        # Check if files exist
        if not Path(exp_path).exists():
            print(f"  [SKIP] Missing experimental: {exp_path}")
            continue
        if not Path(bsl_path).exists():
            print(f"  [SKIP] Missing baseline: {bsl_path}")
            continue

        print(f"\n  Processing: {key}")

        # Load data
        exp_df = pd.read_csv(exp_path)
        bsl_df = pd.read_csv(bsl_path)

        print(f"    Experimental: {exp_df.shape}")
        print(f"    Baseline: {bsl_df.shape}")

        # Ensure participant is string type
        exp_df["participant"] = exp_df["participant"].astype(str)
        bsl_df["participant"] = bsl_df["participant"].astype(str)

        # Compute participant-level baseline aggregates (min, max, range)
        baseline_agg = compute_participant_baseline_aggregates(bsl_df)
        print(f"    Baseline aggregates: {baseline_agg.shape[0]} participants")

        # Create baseline feature file (aggregates only, aligned with experimental windows)
        baseline_features = create_baseline_feature_file(exp_df, baseline_agg)
        print(f"    Baseline feature file: {baseline_features.shape}")

        # Store results
        variant_name = key.replace("_exp", "")
        results[f"{variant_name}_baseline"] = baseline_features

    return results


def main():
    """Main execution."""
    print("=" * 70)
    print("BASELINE FEATURE PREPARATION FOR RF MODELS")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Process each feature type
    for feature_type, paths in INPUT_PATHS.items():
        try:
            results = process_feature_type(feature_type, paths)

            # Save results
            for variant_name, df in results.items():
                # Avoid double-prefixing (e.g., "performance_performance_baseline")
                if variant_name.startswith(f"{feature_type}_"):
                    output_file = OUTPUT_DIR / f"{variant_name}.csv"
                    output_key = variant_name
                else:
                    output_file = OUTPUT_DIR / f"{feature_type}_{variant_name}.csv"
                    output_key = f"{feature_type}_{variant_name}"

                df.to_csv(output_file, index=False)
                print(f"\n  ✓ Saved: {output_file}")
                all_results[output_key] = output_file

        except Exception as e:
            print(f"\n  ✗ Error processing {feature_type}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated {len(all_results)} feature files:")
    for name, path in all_results.items():
        print(f"  - {name}: {path}")

    print("\n" + "="*70)
    print("USAGE IN run_rf_models.py:")
    print("="*70)
    print("""
# Model A: Experimental only (current - no changes needed)
{"name": "linear_exp_only",
 "feature_groups": ["procrustes_linear_exp"]}

# Model B: Experimental + baseline aggregates (min, max, range)
# Test with different baseline feature types:
{"name": "with_baseline_perf",
 "feature_groups": ["procrustes_linear_exp", "performance_model_b"]}

{"name": "with_baseline_perf_linear",
 "feature_groups": ["procrustes_linear_exp", "performance_model_b", "linear_procrustes_model_b"]}

{"name": "with_baseline_perf_rqa",
 "feature_groups": ["procrustes_linear_exp", "performance_model_b", "rqa_procrustes_model_b"]}

{"name": "with_baseline_all",
 "feature_groups": ["procrustes_linear_exp", "performance_model_b", "linear_procrustes_model_b", "rqa_procrustes_model_b"]}
    """)


if __name__ == "__main__":
    main()
