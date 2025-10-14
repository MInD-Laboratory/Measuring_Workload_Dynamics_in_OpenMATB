# Major Changes Log - Random Forest Modeling Pipeline

This file documents significant changes to the Random Forest modeling pipeline for workload detection.

---

## 2025-10-14: Session Updates

### 1. PCA Implementation Fix (pipeline_utils.py)

**Problem**: PCA was failing with "Input X contains NaN" error when enabled.

**Solution**: Added SimpleImputer before PCA in the pipeline
- Added `from sklearn.impute import SimpleImputer` (line 34)
- Inserted imputer step before PCA using median strategy (lines 670-672)
- Pipeline order now: StandardScaler → SimpleImputer (conditional) → PCA (conditional) → RF

**File**: `pipeline_utils.py`

**Lines modified**: 34, 670-672

**Status**: ✓ Fixed - PCA now handles missing values properly

---

### 2. Performance Metrics Section Expansion (run_rf_models.py)

**Change**: Expanded performance_metrics experiments from 2 to 14 experiments

**New experiments added**:
- Performance only (2 experiments: random/participant)
- Linear + Performance (4 experiments: procrustes/original × random/participant)
- RQA + Performance (4 experiments: procrustes/original × random/participant)
- All features combined (4 experiments: procrustes/original × random/participant)

**Purpose**: Systematically test whether performance metrics improve classification when combined with pose and/or RQA features

**File**: `run_rf_models.py`

**Lines modified**: 97-134

**Status**: ✓ Complete

---

### 3. Baseline Experiments Complete Restructure

#### 3a. Changed Baseline Feature Computation (prepare_baseline_features.py)

**Major conceptual change**:
- **Before**: Computed mean/std across all baseline windows → created change scores (experimental - baseline_mean)
- **After**: Compute min/max/range across three baseline conditions (L, M, H) → captures individual variability profile

**Key changes**:
1. **compute_participant_baseline_aggregates()** (lines 66-93):
   - Changed from computing mean/std to computing min/max/range
   - Range = max - min for each feature

2. **create_baseline_feature_file()** (lines 96-117):
   - Renamed from `create_model_b_features()`
   - Now creates separate files with ONLY baseline aggregates (not merged with experimental)
   - Contains: participant, condition, window_index + baseline aggregates

3. **Removed Model C entirely**:
   - Deleted `create_model_c_features()` function
   - No longer creates change scores, ratios, or normalized change features

4. **Updated output naming** (line 180):
   - Files now named `*_baseline.csv` instead of `*_model_b.csv` and `*_model_c.csv`
   - Example: `performance_baseline.csv`, `linear_procrustes_baseline.csv`

**File**: `prepare_baseline_features.py`

**Lines modified**: 4-17 (docstring), 66-117 (functions), 170-180 (processing), 218-239 (usage docs)

**Status**: ✓ Complete

---

#### 3b. Restructured Baseline Experiments (run_rf_models.py)

**Major changes**:

1. **Updated FEATURE_GROUPS section** (lines 233-243):
   - Changed from `model_b` and `model_c` naming to `baseline_*` naming
   - Updated file paths to `*_baseline.csv`
   - Removed all Model C feature group definitions
   - New groups: `baseline_performance`, `baseline_linear_procrustes`, `baseline_rqa_procrustes`, etc.

2. **Restructured baseline_comparison experiments** (lines 136-199):
   - **Removed**: Model A experiments (avoid duplication with feature_comparison)
   - **Removed**: All Model C experiments (change scores, ratios)
   - **Added**: 4 Model B variants testing different baseline feature combinations:
     - **Model B1**: Experimental + baseline performance (2 exp)
     - **Model B2**: Experimental + baseline performance + linear (2 exp)
     - **Model B3**: Experimental + baseline performance + RQA (2 exp)
     - **Model B4**: Experimental + all baseline aggregates (2 exp)
   - Total: 8 experiments (down from 12)

3. **Updated section description** (line 143):
   - Notes that Model A already exists as `linear_procrustes_*` in feature_comparison
   - Clarifies comparison should be against existing experiments

**Purpose**: Test whether individual baseline variability profiles (min/max/range across L/M/H) improve workload classification

**File**: `run_rf_models.py`

**Lines modified**: 136-199 (experiments), 233-243 (feature groups)

**Status**: ✓ Complete

---

## Summary of Current Experiment Configuration

With all sections enabled (`run_rf_models.py`):

### Enabled Experiments:
1. **feature_comparison**: 12 experiments
   - Linear only (4: procrustes/original × random/participant)
   - RQA only (4: procrustes/original × random/participant)
   - Combined linear+RQA (4: procrustes/original × random/participant)

2. **performance_metrics**: 14 experiments
   - Performance only (2)
   - Linear + Performance (4)
   - RQA + Performance (4)
   - All features (4)

3. **baseline_comparison**: 8 experiments
   - Model B1: exp + baseline performance (2)
   - Model B2: exp + baseline perf + linear (2)
   - Model B3: exp + baseline perf + RQA (2)
   - Model B4: exp + all baseline (2)

**Total: 34 experiments**

### Model Configuration:
- `n_seeds`: 20
- `feature_selection`: "backward"
- `use_pca`: False
- `tune_hyperparameters`: True
- `tune_n_iter`: 50

---

## 2025-10-14: Learning Curve Fixes (Priority 1 and 2)

### 4. Learning Curve Improvements

Comprehensive overhaul of learning curve functionality to match main RF models and add critical missing features.

#### 4a. Added Hyperparameter Tuning Support

**Problem**: Learning curves lacked hyperparameter tuning, causing inconsistent model quality with main experiments

**Solution**:
1. **run_learning_curves.py** (lines 51-53):
   - Added `tune_hyperparameters`, `tune_n_iter`, `tune_cv_folds` to DEFAULT_MODEL_CONFIG
   - Defaults: `tune_hyperparameters: False`, `tune_n_iter: 30`, `tune_cv_folds: 5`

2. **pipeline_utils.py** (lines 925-929, 1053-1065):
   - Added hyperparameter tuning in Phase 1 (once per minute)
   - Stores tuned params per minute in `minute_tuned_params` dict
   - Applies tuned params in Phase 2 during model training (lines 1214-1216)

**Impact**: Learning curves now support same hyperparameter optimization as main models

**Status**: ✓ Complete

---

#### 4b. Added Resume Functionality with Seed-Level Checkpointing

**Problem**: No way to resume interrupted experiments, causing massive time loss on crashes (20 seeds × 8 minutes × N experiments)

**Solution**:
1. **run_learning_curves.py** (lines 268-273, 329-347, 377-378):
   - Added `--resume` flag to argument parser
   - Pass `resume` parameter to `run_learning_curve_experiment()`
   - Updated user prompt to support resume action

2. **pipeline_utils.py** (lines 876, 896, 980-1000):
   - Modified function signature to accept `resume` parameter
   - Added checkpoint loading logic at startup
   - Loads `results`, `minute_features`, `minute_tuned_params`, and `start_seed` from checkpoint

3. **Phase 1 Skip Logic** (lines 1006-1067):
   - Wrapped Phase 1 in `if not (resume and minute_features)` block
   - Skips feature selection/tuning if resuming with cached results
   - Prints resume status message

4. **Phase 2 Checkpointing** (lines 1076, 1232-1249):
   - Changed loop to start from `start_seed` instead of 0
   - Saves checkpoint after EVERY seed completes
   - Checkpoint includes all results, features, params, and progress marker

5. **Cleanup** (lines 1276-1278):
   - Deletes checkpoint file after successful completion

**Impact**: Can now resume from any seed after crash/interruption. Checkpoint saved every ~1-2 minutes per experiment.

**Status**: ✓ Complete

---

#### 4c. Updated Baseline Experiments to Use New Aggregate Approach

**Problem**: Learning curves used deprecated baseline concatenation method, incompatible with new Model B approach

**Solution**:
1. **Updated FEATURE_GROUPS** (lines 204-238):
   - Added new baseline aggregate feature groups: `baseline_performance`, `baseline_linear_procrustes`, `baseline_rqa_procrustes`, etc.
   - Kept old baseline groups for backward compatibility (marked as deprecated)
   - Added clear section comments distinguishing old vs new approaches

2. **Restructured Experiments** (lines 62-152):
   - Set `enabled: False` by default (user must opt-in)
   - Reduced default `n_seeds` from 20 to 10 for faster runtime
   - Commented out most old baseline concatenation experiments
   - Added 2 new experiments using baseline aggregates:
     - `lc_linear_with_baseline_agg`: Linear + baseline linear aggregates
     - `lc_linear_perf_with_baseline_agg`: Linear + performance + both baseline aggregates
   - Commented out adaptive normalization experiments (can be re-enabled)

3. **Reduced Total Experiments**: 13 → 3 active (most commented for speed)

**Impact**: Learning curves now compatible with new baseline aggregate approach. Much faster default configuration.

**Status**: ✓ Complete

---

#### 4d. Added experiment_log.csv Logging

**Problem**: Learning curve results only saved to JSON, not comparable with main models in experiment_log.csv

**Solution** (lines 1296-1319):
- Added CSV logging at end of `run_learning_curve_experiment()`
- Uses final minute's metrics as representative performance
- Logs to same `experiment_log.csv` as main RF models
- Includes: balanced accuracy, F1, kappa (mean ± std), n_features, n_seeds

**Impact**: Can now compare learning curves with main models in single CSV table

**Status**: ✓ Complete

---

#### 4e. Experiment Count Reduction and Better Configuration

**Changes**:
1. Default `n_seeds`: 20 → 10 (50% faster)
2. Active experiments: 13 → 3 (commented out non-essential ones)
3. Set `enabled: False` by default (user must explicitly enable)
4. Kept all experiments available but commented for easy re-enabling

**Impact**: Much faster default runtime while preserving all functionality

**Status**: ✓ Complete

---

#### 4f. Improved Progress Tracking

**Changes** (line 1076):
- Updated tqdm progress bar to show:
  - `initial=start_seed` (correct progress on resume)
  - `total=n_seeds` (accurate remaining estimate)
- Better visibility into (experiment, seed) progress

**Status**: ✓ Complete

---

### Summary of Learning Curve Fixes

**Files Modified**:
1. `run_learning_curves.py` - Added tuning config, resume flag, updated experiments, reduced count
2. `pipeline_utils.py` - Implemented tuning, checkpointing, resume, CSV logging

**Key Improvements**:
- ✓ Hyperparameter tuning (once per minute in Phase 1)
- ✓ Resume from any seed after interruption
- ✓ Seed-level checkpointing (saved after every seed)
- ✓ New baseline aggregate experiments
- ✓ CSV logging for easy comparison
- ✓ Reduced default experiment count (13 → 3 active)
- ✓ Better progress tracking with tqdm

**Backward Compatibility**:
- Old baseline concatenation experiments preserved but commented
- All original experiments available, just need to uncomment
- Default config is now conservative (fewer experiments, disabled by default)

---

## 2025-10-14: Baseline Feature File Naming Fix

### 5. Fixed Duplicate Prefix in Performance Baseline Filename

**Problem**: prepare_baseline_features.py was generating `performance_performance_baseline.csv` instead of `performance_baseline.csv`

**Root Cause**: When processing performance features:
- Key = "performance_exp" → variant_name = "performance" → results key = "performance_baseline"
- Output filename = f"{feature_type}_{variant_name}" = f"performance_performance_baseline"
- This didn't affect linear/rqa because variant names (procrustes/original) differ from feature types

**Solution**: Added logic to detect and avoid double-prefixing (lines 204-210):
```python
# Avoid double-prefixing (e.g., "performance_performance_baseline")
if variant_name.startswith(f"{feature_type}_"):
    output_file = OUTPUT_DIR / f"{variant_name}.csv"
    output_key = variant_name
else:
    output_file = OUTPUT_DIR / f"{feature_type}_{variant_name}.csv"
    output_key = f"{feature_type}_{variant_name}"
```

**Impact**:
- Performance baseline file now correctly named `performance_baseline.csv`
- Matches expected path in run_rf_models.py line 225
- Model B experiments can now run successfully

**File**: `prepare_baseline_features.py`

**Lines modified**: 204-210 (added conditional logic for output filename)

**Status**: ✓ Fixed - All 5 baseline files now generated with correct naming

---

## 2025-10-14: Repository Configuration Update

### 6. Updated .gitignore to Allow Final Feature Files

**Problem**: Feature files needed for RF modeling were being ignored, preventing users from running models without reprocessing all data

**Solution**: Updated both root `.gitignore` and `Pose/.gitignore` to allow final feature CSV files:

**Changes to `Pose/.gitignore`** (lines 93-105):
```
# Ignore raw data and intermediate processing files
data/raw_data/
data/processed_data/**
!data/processed_data/experimental/
!data/processed_data/experimental/linear_metrics/
!data/processed_data/experimental/linear_metrics/*.csv
!data/processed_data/baseline/
!data/processed_data/baseline/linear_metrics/
!data/processed_data/baseline/linear_metrics/*.csv

# Allow final RQA feature files only
data/rqa/**
!data/rqa/*_rqa_crqa.csv
```

**Changes to root `.gitignore`** (lines 23-40):
- Added exceptions for linear_metrics CSV files
- Added exceptions for RQA feature files (`*_rqa_crqa.csv`)
- Added exceptions for `Modeling/baseline_features/*.csv`

**Impact**:
- Users can now clone repo and run RF models without reprocessing raw data
- 15 final feature files now uploadable (6 linear, 4 RQA, 5 baseline)
- Intermediate processing files remain ignored
- Performance data files already tracked

**Status**: ✓ Complete

---

## Files Modified Summary

**Session 1** (Baseline restructure):
1. `Modeling/pipeline_utils.py` - SimpleImputer for PCA
2. `Modeling/run_rf_models.py` - Performance metrics expansion, baseline restructure
3. `Modeling/prepare_baseline_features.py` - Min/max/range aggregates

**Session 2** (Learning curve fixes):
4. `Modeling/run_learning_curves.py` - Tuning config, resume, experiments update
5. `Modeling/pipeline_utils.py` - Tuning implementation, checkpointing, CSV logging

**Session 3** (File naming and repo config):
6. `Modeling/prepare_baseline_features.py` - Fixed performance file naming
7. `.gitignore` - Allow final feature files
8. `Pose/.gitignore` - Allow final feature files
9. `mjr_changes.md` - Consolidated changelog (this file)

---


## Notes

- PCA is currently disabled but implementation is fixed and ready if needed
- All baseline experiments now compare against existing `linear_procrustes_*` experiments
- Baseline aggregates capture individual variability across L/M/H baseline conditions
- No redundant experiments - Model A not duplicated in baseline_comparison
- Learning curves disabled by default - set `enabled: True` in run_learning_curves.py to use
- Learning curve checkpoints enable resuming from interruptions
- Both main RF models and learning curves now log to same experiment_log.csv
- Final feature files are now tracked in git for easy model execution

---

# APPENDIX: Baseline Feature Experiments Documentation

This section provides detailed documentation on the baseline feature experiments methodology.

## Hypothesis

Individual differences in baseline ability might moderate workload effects:
- Someone with good baseline range/variability might respond differently to workload
- Someone with poor baseline stability might show larger changes under the same workload
- Baseline aggregates (min, max, range across L/M/H baseline conditions) could capture individual differences

## Model Variants

### Model A: Experimental Only (Baseline for Comparison)
- **Already exists as**: `linear_procrustes_random` and `linear_procrustes_participant` in feature_comparison section
- **Features**: Only experimental window features (linear pose features)
- **What it tests**: Can we classify L/M/H from pose features during task?
- **Limitation**: Ignores individual differences in baseline ability

### Model B Variants: Experimental + Baseline Aggregates
All Model B variants use experimental features plus different combinations of baseline aggregates (min, max, range across baseline L/M/H conditions).

#### Model B1: Experimental + Baseline Performance
- **Features**: Experimental linear + baseline performance aggregates
- **What it tests**: Does knowing someone's baseline task performance range help?
- **Example features**: `pupil_dx_mean` (exp) + `track_point_accuracy_baseline_min`, `track_point_accuracy_baseline_max`, `track_point_accuracy_baseline_range`

#### Model B2: Experimental + Baseline Performance + Linear
- **Features**: Experimental linear + baseline performance + baseline linear aggregates
- **What it tests**: Does adding baseline pose feature ranges help?
- **Example features**: Experimental features + performance aggregates + `pupil_dx_baseline_min`, `pupil_dx_baseline_max`, `pupil_dx_baseline_range`

#### Model B3: Experimental + Baseline Performance + RQA
- **Features**: Experimental linear + baseline performance + baseline RQA aggregates
- **What it tests**: Does adding baseline nonlinear dynamics help?
- **Example features**: Experimental features + performance aggregates + RQA feature aggregates

#### Model B4: Experimental + All Baseline Aggregates
- **Features**: Experimental linear + all baseline aggregates (performance + linear + RQA)
- **What it tests**: Does the full baseline profile improve classification?
- **Example features**: All experimental features + all baseline aggregates

## Complete Workflow

### Step 1: Generate Baseline Feature Files

```bash
cd Modeling
python prepare_baseline_features.py
```

This creates baseline aggregate files in `Modeling/baseline_features/`:
- `performance_baseline.csv` - Performance aggregates (min, max, range)
- `linear_procrustes_baseline.csv` - Linear pose aggregates (min, max, range)
- `rqa_procrustes_baseline.csv` - RQA/CRQA aggregates (min, max, range)

Each file contains only the baseline aggregate features aligned with experimental windows.

### Step 2: Enable Baseline Experiments

The baseline experiments are already configured in `run_rf_models.py`. Ensure:
```python
"baseline_comparison": {
    "enabled": True,  # Should be True
    ...
}
```

### Step 3: Run All Model Variants

```bash
python run_rf_models.py
```

This will run 8 baseline experiments (Model A already exists from feature_comparison):
- Model B1: 2 experiments (exp + baseline performance)
- Model B2: 2 experiments (exp + baseline perf + linear)
- Model B3: 2 experiments (exp + baseline perf + RQA)
- Model B4: 2 experiments (exp + baseline perf + linear + RQA)

### Step 4: Compare Results

Check `model_output/experiment_log.csv` for balanced accuracy comparison:

```
experiment_name                  split_strategy  n_features  test_bal_acc_mean
linear_procrustes_random         random          120         0.65  ← Model A (baseline)
modelB1_perf_random              random          138         0.68  ← Does perf baseline help?
modelB2_perf_linear_random       random          498         0.70  ← Does adding linear help?
modelB3_perf_rqa_random          random          618         0.69  ← Does adding RQA help?
modelB4_all_baseline_random      random          978         0.71  ← Does full profile help?

linear_procrustes_participant    participant     120         0.58  ← Model A (baseline)
modelB1_perf_participant         participant     138         0.62  ← Improvement on harder split?
...
```

**Key comparisons**:
1. **Does baseline help at all?** Compare `linear_procrustes_*` (Model A) vs Model B1
2. **Which baseline features matter?** Compare B1 vs B2 vs B3 vs B4
3. **Does it generalize?** Check `participant` split (harder test - new people)

## Expected Outcomes

### If baseline aggregates HELP (improvement in balanced accuracy):
- **Model B1 > Model A**: Baseline performance aggregates contain useful information about individual differences
- **Model B2/B3/B4 > Model B1**: Adding baseline pose/RQA aggregates provides additional information
- **Larger improvement in `participant` split**: Baseline aggregates capture individual traits that help generalize to new people

### If baseline aggregates DON'T help:
- **Model A ≈ Model B variants**: No improvement
- **Possible reasons**:
  - Baseline range/variability doesn't predict workload response
  - Within-subjects design already accounts for individual differences
  - Experimental features already capture the relevant individual differences
  - Baseline conditions (L/M/H) too similar to provide informative range

## Analysis Steps

1. Run `python prepare_baseline_features.py` to generate baseline aggregate files
2. Run `python run_rf_models.py` to execute all model variants
3. Compare balanced accuracy across Model A and Model B variants (B1, B2, B3, B4)
4. If improvement found:
   - Check per-class metrics to see which workload conditions benefit most
   - Examine feature importance to identify which baseline aggregates matter
   - Consider whether specific baseline feature types (performance vs pose vs RQA) drive improvements
5. Consider extended analysis:
   - Subgroup analysis: Do high vs low baseline variability participants benefit differently?
   - Interaction effects: Does baseline help more for certain experimental conditions?
   - Cross-validation: Test stability of findings across different random seeds
