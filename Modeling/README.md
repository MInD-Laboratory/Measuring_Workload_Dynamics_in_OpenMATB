# Random Forest Workload Detection Pipeline

A comprehensive, flexible pipeline for training and evaluating Random Forest classifiers on multimodal workload detection data with support for:

- **Multiple normalization methods** (none, min-max, z-score)
- **Different feature types** (linear pose, RQA/nonlinear, performance metrics)
- **Validation strategies** (random cross-task, leave-participant-out)
- **Feature selection** (backward/forward elimination with optional PCA)
- **Learning curves** (incremental training data analysis)
- **Baseline data integration** (combining pre-task and experimental data)

## 📁 File Structure

```
.
├── run_pipeline.py          # Main pipeline runner
├── pipeline_utils.py        # Core utility functions (Parts 1 & 2)
├── visualize_results.py     # Results visualization notebook/script
├── README.md                # This file
│
├── data/                    # Input feature files (create this)
│   ├── exp_zscore_linear.csv
│   ├── exp_zscore_rqa.csv
│   ├── performance_exp.csv
│   └── ...
│
├── model_output/            # Results (auto-generated)
│   ├── *.json              # Individual experiment results
│   ├── experiment_log.csv  # Summary of all experiments
│   └── summary_table.csv   # Detailed metrics table
│
└── figs/                    # Visualizations (auto-generated)
    ├── normalization/
    ├── features/
    ├── performance/
    ├── baseline/
    └── learning_curves/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Prepare Your Data

Place your feature CSV files in the appropriate directory. Each CSV should have:

**Required columns:**
- `participant` - Participant ID (string or int)
- `condition` - Workload condition label (e.g., "L", "M", "H")
- `window_start` - Window start time in seconds
- `window_end` - Window end time in seconds

**Feature columns:**
- Any number of numeric feature columns
- Long format (with `feature` column) or wide format supported

**Example wide format:**
```csv
participant,condition,window_index,feature1,feature2,...
P01,L,0,30,0.45,0.23,...
P01,L,30,60,0.48,0.21,...
```

**Example long format:**
```csv
participant,condition,window_indexfeature,mean,rms,min,max
P01,L,0,30,blink_rate,0.45,0.52,0.1,0.9
P01,L,0,30,pupil_size,3.2,3.5,2.1,4.3
```

### 3. Configure Experiments

Edit `run_pipeline.py` to define your experiments:

```python
# Update FEATURE_GROUPS to point to your data files
FEATURE_GROUPS = {
    "linear_pose_exp_zscore": ("data/exp_zscore_linear.csv", "main"),
    "rqa_pose_exp_zscore": ("data/exp_zscore_rqa.csv", "main"),
    # ... add your files or run with given files
}

# Enable/disable experiment sections
EXPERIMENT_CONFIG = {
    "normalization_comparison": {
        "enabled": True,  # Set to False to skip
        # ...
    },
    # ...
}
```

### 4. Run the Pipeline

```bash
# Run all enabled experiments
python run_pipeline.py

# Force overwrite existing results
python run_pipeline.py --force

# Continue incomplete experiments
python run_pipeline.py --continue

# Only run learning curves
python run_pipeline.py --learning-curves

# Preview what will run (no execution)
python run_pipeline.py --dry-run
```

### 5. Visualize Results

```bash
# Run visualization script
python visualize_results.py

# Or use as Jupyter notebook
jupyter notebook visualize_results.py
```

## ⚙️ Configuration Options

### Default Model Settings

Configure in `run_pipeline.py`:

```python
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 20,                    # Number of random seeds
    "feature_selection": "backward",   # "forward", "backward", or None
    "selection_level": "metric",       # "metric" or "group"
    "selection_stopping": "drop",      # "drop" or "best"
    "use_pca": False,                  # Apply PCA after selection
    "pca_variance": 0.95,              # Variance to retain
    "write_cm": True,                  # Save confusion matrices
}
```

### Per-Experiment Overrides

Override defaults for specific experiments:

```python
{
    "name": "my_experiment",
    "feature_groups": ["linear_pose_exp_zscore"],
    "split_strategy": "random",
    "n_seeds": 10,        # Override: use fewer seeds
    "use_pca": True,      # Override: enable PCA
}
```

### Learning Curves

Configure incremental training experiments:

```python
LEARNING_CURVES_CONFIG = {
    "enabled": True,
    "experiments": [
        {
            "name": "learning_curves_linear_random",
            "baseline_groups": ["linear_pose_bsl_zscore"],
            "experimental_groups": ["linear_pose_exp_zscore"],
            "split_strategy": "random",
            "minutes": list(range(0, 8)),  # 0-7 minutes
            "skip_every": 2,  # Use every 2nd window, overlapping windows
        }
    ]
}
```

## 📊 Output Files

### JSON Results

Each experiment generates a JSON file with:

```json
{
  "name": "experiment_name",
  "config": {...},
  "metrics": {
    "test_bal_acc_mean": 0.75,
    "test_bal_acc_std": 0.03,
    "test_f1_mean": 0.74,
    ...
  },
  "confusion_matrix": [[...], [...], [...]],
  "selected_features": ["feature1", "feature2", ...],
  "n_features": 42,
  "n_seeds": 20
}
```

### CSV Logs

- `experiment_log.csv` - One row per experiment with all metrics
- `summary_table.csv` - Detailed metrics for export

### Visualizations

All plots saved as SVG for publication quality:

- Normalization comparison bar charts
- Feature type comparison plots
- Confusion matrices (heatmaps)
- Learning curves with error bars
- Baseline impact plots

## 🔧 Advanced Usage

### Custom Feature Selection

Implement your own selection in `pipeline_utils.py`:

```python
def custom_selection(X, y, random_state=0):
    """Your custom feature selection logic."""
    # ... implementation
    return selected_features, score
```

### Adding New Metrics

Extend `fit_and_evaluate()` in `pipeline_utils.py`:

```python
metrics = {
    "test_acc": accuracy_score(y_test, y_pred),
    "test_bal_acc": balanced_accuracy_score(y_test, y_pred),
    # Add your metric:
    "test_precision": precision_score(y_test, y_pred, average="weighted"),
}
```

### Resume Incomplete Runs

The pipeline automatically resumes interrupted runs:

```bash
# Pipeline interrupted? Just run again
python run_pipeline.py

# Or explicitly continue
python run_pipeline.py --continue
```

## 📈 Example Workflows

### Workflow 1: Compare Normalizations

```python
# In run_pipeline.py, enable only normalization comparison
EXPERIMENT_CONFIG = {
    "normalization_comparison": {"enabled": True},
    "feature_comparison": {"enabled": False},
    # ...
}
```

```bash
python run_pipeline.py
python visualize_results.py
# Check figs/normalization/
```

### Workflow 2: Test Feature Combinations

```python
# Enable feature comparison
EXPERIMENT_CONFIG = {
    "normalization_comparison": {"enabled": False},
    "feature_comparison": {"enabled": True},
    # ...
}
```

### Workflow 3: Full Pipeline with Learning Curves

```python
# Enable everything
EXPERIMENT_CONFIG = {
    "normalization_comparison": {"enabled": True},
    "feature_comparison": {"enabled": True},
    "performance_metrics": {"enabled": True},
    "baseline_impact": {"enabled": True},
}

LEARNING_CURVES_CONFIG = {"enabled": True}
```

```bash
python run_pipeline.py
python visualize_results.py
```

## 🐛 Troubleshooting

### "File not found" errors

- Check that `FEATURE_GROUPS` paths point to existing CSV files
- Ensure data directory structure matches configuration

### Memory issues with many seeds

- Reduce `n_seeds` in `DEFAULT_MODEL_CONFIG`
- Use `--continue` to process seeds incrementally

### Feature selection takes too long

- Set `feature_selection: None` to disable
- Reduce cross-validation folds in `backward_elimination_rf()`

### Plots not showing

- Check that output directories exist
- Verify matplotlib backend: `export MPLBACKEND=Agg`

## 📚 API Reference

### Key Functions in `pipeline_utils.py`

#### Data Loading
- `load_and_merge_features(file_list, skip_every=None)` - Load and merge CSVs
- `normalize_window_columns(df)` - Standardize column names
- `make_train_test_split(df, split_strategy, random_state)` - Split data

#### Feature Selection
- `backward_elimination_rf(X, y, ...)` - Backward feature elimination
- `select_features_once(df, split_strategy, config, seed)` - One-time selection

#### Training & Evaluation
- `fit_and_evaluate(df, split_strategy, seed, selected_features, config)` - Train model
- `run_single_model(name, config, output_dir, ...)` - Full model pipeline

#### Learning Curves
- `run_learning_curve_experiment(config, ...)` - Incremental training analysis

#### Results
- `log_to_csv(name, results, output_dir)` - Log to experiment_log.csv
- `print_summary(output_dir)` - Print results summary
