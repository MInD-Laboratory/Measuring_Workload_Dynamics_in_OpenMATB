# Random Forest Workload Detection

Machine learning pipeline for classifying cognitive workload conditions using Random Forest classifiers trained on multimodal behavioral features from pose dynamics, recurrence quantification analysis, and task performance metrics.

## Overview

This pipeline trains Random Forest models to distinguish between low, moderate, and high cognitive load conditions. The system supports multiple feature types, validation strategies, feature selection methods, and includes functionality for learning curve analysis to evaluate model performance with varying amounts of training data.

Key capabilities:
- Multiple feature types (linear pose metrics, RQA/CRQA, task performance)
- Two validation strategies (random split, leave-participant-out)
- Automated feature selection (backward/forward elimination)
- Optional PCA dimensionality reduction
- Hyperparameter tuning via RandomizedSearchCV
- Learning curve experiments for temporal prediction analysis
- Baseline feature integration to capture individual differences

## Directory Structure

```
Modeling/
├── run_rf_models.py              # Main experiment runner
├── run_learning_curves.py        # Learning curve experiments
├── prepare_baseline_features.py  # Generate baseline aggregate features
├── pipeline_utils.py             # Core utility functions
├── visualize_results.py          # Results visualization
├── README.md                     # This file
│
├── baseline_features/            # Baseline aggregates (generated)
│   ├── performance_baseline.csv
│   ├── linear_procrustes_baseline.csv
│   └── rqa_procrustes_baseline.csv
│
├── model_output/                 # Results (auto-generated)
│   ├── rf_models/                # Random Forest experiment results
│   │   ├── *.json                # Individual experiment results
│   │   └── experiment_log.csv    # Summary of RF experiments
│   ├── lc_models/                # Learning curve results
│   │   ├── *.json                # Learning curve results per experiment
│   │   ├── *_checkpoint.json     # Resume checkpoints (during runs)
│   │   └── experiment_log.csv    # Summary of learning curves
│   └── confusion_matrices/       # Per-experiment confusion matrices
│
└── figs/                         # Visualizations (auto-generated)
    ├── feature_comparison/
    ├── performance_metrics/
    ├── baseline_comparison/
    └── learning_curves/
```

## Input Data

Feature files are located in parent directories and included in the repository:

**Pose features** (`../Pose/data/processed_data/`):
- `experimental/linear_metrics/procrustes_global_linear.csv` - Procrustes-aligned linear features
- `experimental/linear_metrics/original_linear.csv` - Non-aligned linear features
- `baseline/linear_metrics/` - Corresponding baseline files

**RQA features** (`../Pose/data/rqa/`):
- `experimental_procrustes_global_rqa_crqa.csv` - RQA metrics on aligned poses
- `experimental_original_rqa_crqa.csv` - RQA metrics on non-aligned poses
- `baseline_*_rqa_crqa.csv` - Corresponding baseline files

**Performance metrics** (`../performance/data/out/`):
- `performance_exp.csv` - Task accuracy and response times
- `performance_bsl.csv` - Baseline performance

All feature files include:
- `participant` - Participant ID
- `condition` - Workload level (L, M, H)
- `window_index` - Window number
- Feature columns (metrics computed per 60-second window)

## Quick Start

### 1. Generate Baseline Features (Optional) - files may alreday exist

To test models that incorporate individual baseline variability:

```bash
python prepare_baseline_features.py
```

This computes participant-level aggregates (min, max, range) across baseline conditions and saves to `baseline_features/`. These features capture individual differences in baseline ability that may moderate workload effects.

### 2. Basic Random Forest Experiments

Run the main experiment suite with default configuration:
```bash
cd Modeling
python run_rf_models.py
```

This executes all enabled experiment sections:
- Feature comparison (linear vs RQA vs combined)
- Performance metrics integration
- Baseline feature comparison

Results are saved to `model_output/rf_models/` with detailed JSON files per experiment and a consolidated `experiment_log.csv`.

### 3. Learning Curves

To evaluate how model performance improves with increasing training data:

```bash
python run_learning_curves.py
```

### 4. Visualize Results

Generate publication-quality figures from experiment results:

```bash
python visualize_results.py
```

Creates SVG plots comparing model performance across experiments, saved to `figs/`.

## Configuration

### Main Experiments (run_rf_models.py)

The experiment configuration is organized into sections that can be independently enabled/disabled:

**feature_comparison**: Tests linear features, RQA features, and their combination
**performance_metrics**: Evaluates impact of adding task performance metrics
**baseline_comparison**: Tests whether baseline aggregates improve classification

Default model settings (configurable in run_rf_models.py):
```python
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 15,                      # Random seeds for reliability (configurable)
    "feature_selection": "None",        # Options: "backward", "forward", "None"
    "use_pca": False,                   # PCA after selection
    "tune_hyperparameters": False,      # RandomizedSearchCV (set to True to enable)
    "tune_n_iter": 30,                  # Hyperparameter search iterations
    "write_cm": True,                   # Save confusion matrices
}
```

**Configuration-Based Output Files**: Experiments with different configurations generate separate output files to prevent overwrites. Filename format: `experiment_name_<config_suffix>.json` where suffix includes feature selection method, `_hyp` (if hyperparameter tuning enabled), and `_pca` (if PCA enabled). Examples:
- `linear_procrustes_random_backward_hyp.json` (backward selection + hyperparameter tuning)
- `linear_procrustes_random_none.json` (no feature selection, no tuning)

Each experiment specifies:
- `name` - Unique experiment identifier
- `feature_groups` - List of feature sets to combine
- `split_strategy` - "random" (80/20 stratified) or "participant" (leave-participant-out)

### Learning Curves (run_learning_curves.py)

Learning curve experiments test temporal prediction: "Given M minutes of training data, can I predict workload for the remaining trial?"

**Three Data Groups** (clarifies different baseline approaches):
- `baseline_concatenate_groups` - Baseline trial windows concatenated to training data
- `baseline_aggregate_groups` - Participant-level aggregate features (min/max/range across baseline conditions)
- `experimental_groups` - Experimental trial features

**Key Features**:
- Can start at **minute 0** when baseline aggregate features are present
- Tests: "Can I predict workload from baseline individual differences alone?"
- `minutes` - Training durations to test (e.g., 0-7 minutes)
- `skip_every` - Window stride to handle 50% overlap
- `normalization_mode` - "standard" or adaptive normalization strategies

Default settings use 10 seeds and conservative configuration for faster execution. Results saved to `model_output/lc_models/`.

### Baseline Features (prepare_baseline_features.py)

Computes three aggregates per feature across baseline L/M/H conditions:
- `*_baseline_min` - Minimum value across baseline
- `*_baseline_max` - Maximum value across baseline
- `*_baseline_range` - Range (max - min)

These aggregates are aligned with experimental windows and can be merged with experimental features during modeling.

## Split Strategies

**Random Split (80/20 stratified)**:
- Splits all windows randomly while preserving condition distribution
- Tests: Can the model classify unseen windows from seen participants?
- Easier generalization task

**Leave-Participant-Out**:
- Holds out approximately 20% of participants entirely
- Tests: Can the model generalize to completely new people?
- Harder generalization task, tests true individual differences

## Feature Selection

The pipeline implements backward elimination with cross-validated permutation importance:

1. Pre-filtering: Remove low-variance and highly correlated features (r > 0.95)
2. Iterative removal: Drop least important feature, retrain, evaluate
3. Stopping: Continue until performance drops below threshold

Feature selection can operate at two levels:
- `metric` - Select individual features
- `group` - Select entire feature groups (all features from a source)

Selection is performed once per random seed to assess stability.

## Model Architecture

**Classifier**: Random Forest with 300 trees
- `class_weight='balanced'` - Handles class imbalance
- `max_depth=None` - Unlimited depth
- Optional hyperparameter tuning via RandomizedSearchCV

**Pipeline**: StandardScaler → Optional PCA → Random Forest

**Evaluation**: 5-fold stratified cross-validation per seed (20 seeds default)

**Metrics**:
- Balanced accuracy (primary metric)
- F1-score (macro-averaged)
- Cohen's kappa
- Per-class precision, recall, F1
- Confusion matrices

## Output Files

### JSON Results (model_output/rf_models/ or model_output/lc_models/)

Each experiment produces a detailed JSON file:
```json
{
  "name": "experiment_name",
  "config": {
    "split_strategy": "random",
    "n_seeds": 20,
    "feature_selection": "backward"
  },
  "metrics": {
    "test_bal_acc_mean": 0.75,
    "test_bal_acc_std": 0.03,
    "test_f1_mean": 0.74,
    "test_kappa_mean": 0.62
  },
  "confusion_matrix_mean": [[...], [...], [...]],
  "selected_features": ["feature1", "feature2", ...],
  "n_features": 42
}
```

### CSV Logs (model_output/rf_models/ or model_output/lc_models/)

**experiment_log.csv**: One row per experiment with all metrics for easy comparison (separate log files in each subdirectory)

**Columns include**:
- Experiment name and configuration
- Mean and std for balanced accuracy, F1, kappa
- Number of features selected
- Number of seeds

### Confusion Matrices (model_output/confusion_matrices/)

Individual confusion matrix JSON files saved per experiment when `write_cm: True`.

### Visualizations (figs/)

Publication-ready SVG plots organized by experiment type:
- Bar charts comparing model performance
- Confusion matrix heatmaps
- Learning curves with error bars
- Feature importance plots

## Learning Curves Details

Learning curve experiments answer: "How much training data is needed to achieve good performance?"

**Two-phase design**:
1. Phase 1: Feature selection and hyperparameter tuning once per minute
2. Phase 2: Evaluate across all seeds using selected features/parameters

**Temporal purging**: Removes overlapping windows between train and test sets to prevent data leakage (critical with 50% window overlap).

**Outputs**:
- JSON file per experiment with performance at each training duration
- Results also logged to `experiment_log.csv` using final minute metrics
- Can be visualized with `visualize_results.py`

**Resume capability**: Experiments save checkpoints after each seed completes. Re-run with `--resume` flag to continue interrupted experiments.

## Baseline Feature Integration

Baseline features test whether individual differences in baseline ability improve workload classification.

**Hypothesis**: Participants with different baseline ranges/variability may respond differently to workload.

**Model variants**:
- Model A: Experimental features only (baseline comparison)
- Model B variants: Experimental + baseline aggregates (performance, pose, RQA, or all)

**Workflow**:
1. Run `prepare_baseline_features.py` to generate aggregates
2. Enable baseline_comparison section in `run_rf_models.py`
3. Compare Model A vs Model B balanced accuracy

**Interpretation**:
- Model B > Model A: Baseline variability captures useful individual differences
- No difference: Within-subject design already accounts for individual differences

## Command Line Options

**run_rf_models.py**:
```bash
python run_rf_models.py              # Run all enabled experiments
python run_rf_models.py --force      # Overwrite existing results
python run_rf_models.py --dry-run    # Preview experiments without running
```

**run_learning_curves.py**:
```bash
python run_learning_curves.py           # Run learning curve experiments
python run_learning_curves.py --force   # Overwrite existing results
python run_learning_curves.py --resume  # Resume from checkpoint
python run_learning_curves.py --dry-run # Preview experiments
```

## Dependencies

**Core requirements**:
- numpy
- pandas
- scikit-learn (Random Forest, cross-validation, metrics)
- scipy (signal processing)
- matplotlib (visualizations)
- seaborn (heatmaps)
- tqdm (progress bars)

All dependencies are included in the root `requirements.txt`.

## Reproducibility

**Random seeds**: All experiments use multiple random seeds (default 20) to assess result stability.

**Feature selection**: Selection is repeated per seed to measure feature stability across folds.

**Hyperparameter tuning**: When enabled, uses RandomizedSearchCV with fixed random state for reproducibility.

**Outputs**: All results include full configuration, selected features, and per-seed metrics.

**Checkpointing**: Learning curves save progress after each seed to enable resumption after interruptions.

All results are deterministic given the same random seed and data.
