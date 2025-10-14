# Facial Pose Analysis Pipeline

Modular pipeline for processing facial pose data from OpenPose landmark detection, extracting features, and analyzing temporal dynamics through linear metrics and recurrence quantification analysis (RQA).

## Overview
This pipeline processes raw OpenPose CSV files through 8 steps:
1. Load raw data → OpenPose CSVs with 70 landmarks (x, y, confidence)
2. Filter landmarks → Extract relevant facial keypoints
3. Mask low confidence → Set landmarks below threshold to NaN
4. Interpolate & filter → Fill short gaps, apply Butterworth low-pass filter
5. Normalize coordinates → Scale to screen dimensions (0-1 range)
6. Build templates → Create global and per-participant reference poses
7. Extract features → Compute windowed features using three normalization approaches:
    - Original: Raw normalized coordinates (no alignment)
    - Procrustes Global: Aligned to global template across all participants
    - Procrustes Participant: Aligned to participant-specific template
8. Compute linear metrics → Calculate velocity, acceleration, and features (RMS, mean, min/max)

The pipeline then supports Recurrence Quantification Analysis (RQA) to examine temporal dynamics of facial movements.

Output: Per-frame time series, windowed features, linear metrics, RQA statistics, and processing summaries.

## Directory Structure

```
Pose/
├── process_pose_linear.py          # Steps 1-8: preprocessing + linear metrics (RUN FIRST)
├── process_pose_recurrence.py      # RQA/CRQA analysis on features (RUN SECOND)
├── linear_pose.ipynb               # Statistical analysis & figures for linear metrics
├── recurrence_pose.ipynb           # Statistical analysis & figures for RQA metrics
│
├── utils/
│   ├── config.py                   # Configuration & processing flags
│   ├── io_utils.py                 # File I/O, filtering, summaries
│   ├── preprocessing_utils.py      # Landmark filtering, confidence masking
│   ├── signal_utils.py             # Interpolation, Butterworth filtering
│   ├── normalize_utils.py          # Screen normalization, interocular distance
│   ├── features_utils.py           # Procrustes alignment, feature extraction
│   ├── window_utils.py             # Sliding window operations
│   ├── rqa_utils.py                # RQA/CRQA computation helpers
│   ├── stats_utils.py              # Statistical tests (LME, post-hoc)
│   ├── nb_utils.py                 # Notebook helpers (table builders, etc.)
│   └── viz_utils.py                # Visualization (interactive viewers, plots)
│
├── data/
│   ├── raw_data/
│   │   ├── experimental_pose/           # Raw CSVs for experimental session from OSF
│   │   └── baseline_pose/               # Raw CSVs for baseline session from OSF
│   │
│   ├── processed_data/
│   │   ├── experimental/           # Processed outputs (see structure below)
│   │   └── baseline/
│   │
│   └── rqa/                        # RQA/CRQA results
│       ├── experimental_original_rqa_crqa.csv
│       ├── experimental_procrustes_global_rqa_crqa.csv
│       └── ...
│
├── figs/                           # Generated figures
│   ├── experimental_original_rqa/
│   ├── experimental_procrustes_global_rqa/
│   └── ...
│
└── tables/                         # LaTeX tables for manuscript
    └── rqa_stats_results/

```


## Outputs

```
data/processed_data/<session>/      # session = experimental or baseline
├── reduced/                        # Step 2: Filtered to relevant landmarks
├── masked/                         # Step 3: Low-confidence points masked
├── interp_filtered/                # Step 4: Interpolated & filtered
├── norm_screen/                    # Step 5: Normalized coordinates
│
├── templates/                      # Step 6: Procrustes templates
│   ├── global_template.csv
│   └── participant_<pid>_template.csv
│
├── features/                       # Step 7: Windowed features
│   ├── original.csv                # No alignment (window-level)
│   ├── procrustes_global.csv       # Global template alignment
│   ├── procrustes_participant.csv  # Participant template alignment
│   │
│   └── per_frame/                  # Frame-level time series
│       ├── original/
│       │   └── <pid>_<cond>_perframe.csv
│       ├── procrustes_global/
│       └── procrustes_participant/
│
├── linear_metrics/                 # Step 8: Displacement, Velocity, Acceleration
│   ├── original_linear.csv
│   ├── procrustes_global_linear.csv
│   └── procrustes_participant_linear.csv
│
└── processing_summary.json         # Config, stats, dropped windows
```


## Data Format
Input files: `data/raw_data/<session>/<participantID>_<condition>.csv`

Example: `472_H.csv` (participant 472, high load condition)
Columns: x1, y1, prob1, ..., x70, y70, prob70 (210 columns total)
Confidence prefix auto-detected: prob*, c*, or confidence*
Sampling rate: 60 Hz (configurable in `config.py`)

Condition codes: L (Low), M (Moderate), H (High)

## Quick Start

### 0. Download Data from OSF

Download the **raw OpenPose CSV files** from the project repository:
[https://osf.io/dzgsv/](https://osf.io/dzgsv/)

1. Extract the downloaded archive (it contains two folders: `experimental_pose` and `baseline_pose`).
2. Place these folders inside your local project directory at:

   ```
   data/raw_data/
   ```

   The structure should look like this:

   ```
   data/
   └── raw_data/
       ├── experimental_pose/
       │   ├── 401_H.csv
       │   ├── 402_M.csv
       │   └── ...
       └── baseline_pose/
           ├── 401_L.csv
           ├── 402_M.csv
           └── ...
   ```

Each `.csv` file corresponds to a single participant-condition trial (e.g., `472_H.csv` = participant 472, high-load condition).


### 1. Configure Paths
By default the config file is set to the experimental (8-min) blocks. To change edit `utils/config.py` to set data directories:

```
RAW_DIR = "data/raw_data/experimental_pose"    # Input directory or "data/raw_data/baseline_pose"
OUT_BASE = "data/processed_data/experimental"  # Output directory or "data/processed_data/baseline"
```

Or use environment variables:

```
export POSE_RAW_DIR="/path/to/raw/data"
export POSE_OUT_BASE="/path/to/output"
```

### 2. Run Linear Processing Pipeline 
This runs steps 1-8 (preprocessing + feature extraction + linear metrics):

`python process_pose_linear.py`

Smart resumption: The pipeline automatically detects if steps 1-5 are complete by checking for normalized files. If found, it loads existing data and runs only steps 6-8. Use `--overwrite` to force reprocessing:

`python process_pose_linear.py --overwrite`

### 3. Run RQA Analysis

After linear processing completes, compute RQA/CRQA metrics:
`python process_pose_recurrence.py`

This analyzes temporal dynamics using recurrence quantification for each normalization type (original, procrustes_global).

### 4. Analyze Results
Open the Jupyter notebooks for statistical analysis and visualization:
`jupyter lab`

`linear_pose.ipynb`: Analyze linear metrics (velocity, acceleration, RMS)
`recurrence_pose.ipynb`: Analyze RQA/CRQA results, generate plots and tables

## Configuration Details
Processing Flags (`utils/config.py`)

### Control which steps execute:

```
# Core steps (Steps 1-6)
RUN_FILTER = True          # Filter to relevant landmarks
RUN_MASK = True            # Mask low confidence
RUN_INTERP_FILTER = True   # Interpolate & Butterworth filter
RUN_NORM = True            # Normalize to screen
RUN_TEMPLATES = True       # Build templates

# Feature extraction (Step 7)
RUN_FEATURES_ORIGINAL = True              # No alignment
RUN_FEATURES_PROCRUSTES_GLOBAL = True     # Global template
RUN_FEATURES_PROCRUSTES_PARTICIPANT = False  # Participant template

# Linear metrics (Step 8)
RUN_LINEAR = True
SCALE_BY_INTEROCULAR = True  # Scale by inter-eye distance
```

### Key Parameters:
```
CONF_THRESH = 0.30         # Min confidence for valid landmarks
MAX_INTERP_RUN = 60        # Max frames to interpolate (1 sec @ 60 Hz)
FILTER_ORDER = 4           # Butterworth filter order
CUTOFF_HZ = 10.0           # Low-pass cutoff frequency
WINDOW_SECONDS = 60        # Window size for features
WINDOW_OVERLAP = 0.5       # 50% overlap between windows
```

## Feature Types

### Normalization Approaches

1. Original: Normalized to screen but no pose alignment
    - Preserves head position/orientation variance
    - Useful for analyzing global movement patterns
2. Procrustes Global: Aligned to global template (all participants)
    - Removes translation, rotation, scaling
    - Isolates shape changes from rigid transformations
3. Procrustes Participant: Aligned to participant-specific template
    - Removes participant-specific anatomical differences
    - Focuses on within-subject movement patterns

### Windowing
- Window size: 60 seconds (3600 frames @ 60 Hz)
- Overlap: 50% (30-second hop)
- NaN handling: Windows containing any NaN for a given metric are dropped
- Per-metric: Each feature drops windows independently

### Output Files

#### Linear Metrics CSV
Each file (e.g., `original_linear.csv`) contains one row per window summarizing displacement-, velocity-, and acceleration-based statistics for each feature.

`<feature>_[min|max|rms|mean]` — displacement statistics

`<feature>_vel_[min|max|rms|mean]` — velocity statistics

`<feature>_acc_[min|max|rms|mean]` — acceleration statistics

Example features: `head_rotation_rad`, `head_tx`, `blink_aperture`, `mouth_aperture`, `pupil_dx`

Each feature yields 12 values (min, max, RMS, mean × displacement, velocity, acceleration).

#### RQA Metrics CSV
Columns: `participant`, `condition`, `column`, `window_index`, `window_start`, `window_end`, then:

`perc_recur`: % recurrence (repetitiveness)
`perc_determ`: % determinism (predictability)
`maxl_found`: Longest diagonal line
`mean_line_length`, std_line_length: Line length statistics
`entropy`: Complexity of recurrence patterns
`laminarity`: % vertical structures (laminar states)
`trapping_time`: Average vertical line length
`vmax`: Longest vertical line
`divergence`: Rate of trajectory separation
`trend_lower_diag`, `trend_upper_diag`: Nonstationarity measures


#### Notebooks
`linear_pose.ipynb`

1. Load linear metrics CSV
2. Run mixed-effects models (LME) for each metric × condition
3. Post-hoc pairwise comparisons (Holm-Bonferroni correction)
4. Generate bar plots with significance markers
5. Export LaTeX tables for manuscript

`recurrence_pose.ipynb`

1. Load RQA/CRQA results
2. Run statistical models for each RQA metric
3. Create 2×2 figure showing key results (customizable)
4. Generate example recurrence plots for specific windows
5. Export tables and figures

## Dependencies
Required:
- numpy
- pandas
- matplotlib
- scipy (for Butterworth filtering)
- tqdm (progress bars)

For statistical analysis (notebooks):

- rpy2 (R interface)
- R with lme4, emmeans packages

For RQA:

RQA package (included in rqa/ submodule) - A Python implementation of univariate and multivariate recurrence quantification analysis methods for analyzing temporal dynamics and nonlinear patterns in time series data.
