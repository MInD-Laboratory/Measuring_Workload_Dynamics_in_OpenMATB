# Performance Analysis

This pipeline extracts and analyzes task performance metrics and subjective workload ratings from OpenMATB experimental sessions.

![combined](https://github.com/user-attachments/assets/900afee7-d457-45b9-9a81-d13feb7a75f4)

## Overview

The performance analysis pipeline processes raw MATB log files to compute per-task and composite accuracy metrics, extracts NASA-TLX subjective workload ratings, and conducts statistical analyses to evaluate performance differences across workload conditions. The pipeline generates publication-ready figures and LaTeX tables for manuscript inclusion.

## Directory Structure

```
performance/
├── data/
│   ├── raw/                      # Raw MATB logs and TLX data
│   └── out/                      # Processed outputs
│       ├── performance_exp.csv   # Experimental session performance
│       ├── performance_bsl.csv   # Baseline session performance
│       └── aggregated_nasatlx.csv # NASA-TLX ratings
│
├── figs/                         # Generated figures (SVG)
├── tables/                       # LaTeX tables for manuscript
│
├── matb_point_accuracy.py        # Compute MATB task accuracy
├── extract_nasa_tlx.py           # Parse NASA-TLX ratings
├── performance_analysis.ipynb    # Statistical analysis notebook
├── performance_utils.py          # Shared utility functions
└── exp4_orders.csv               # Condition order metadata
```

## Quick Start

### 1. Compute Performance Metrics

Process MATB log files to extract per-task accuracy and response times:

```bash
cd performance
python matb_point_accuracy.py
```

**Input**: Raw MATB log files in `data/raw/`

**Output**: `data/out/performance_exp.csv` and `data/out/performance_bsl.csv`

This script computes:
- Per-task point accuracy (tracking, resource management, system monitoring, communications)
- Composite accuracy (weighted average across all tasks)
- Response times for each task
- Metrics are computed per participant per condition per block

### 2. Extract NASA-TLX Ratings

Parse subjective workload ratings from NASA-TLX questionnaires:

```bash
python extract_nasa_tlx.py
```

**Input**: Raw TLX data files in `data/raw/`

**Output**: `data/out/aggregated_nasatlx.csv`

Extracts the six NASA-TLX dimensions:
- Mental Demand
- Physical Demand
- Temporal Demand
- Performance
- Effort
- Frustration

### 3. Run Statistical Analysis and Generate Figures

Open the analysis notebook to conduct statistical tests and create visualizations:

```bash
jupyter lab performance_analysis.ipynb
```

The notebook performs:
- Linear mixed-effects models (LME) comparing performance across conditions
- Post-hoc pairwise comparisons with correction
- Baseline-experimental correlations
- Outlier detection (3 SD criterion)
- Figure generation (heatmaps, scatter plots, bar charts)
- LaTeX table export for manuscript

**Outputs**:
- Figures saved to `figs/` as SVG
- Tables saved to `tables/` as .tex files

## Output Files

### performance_exp.csv / performance_bsl.csv

Per-participant, per-condition, per-block performance metrics:

**Columns**:
- `participant` - Participant ID
- `condition` - Workload level (L, M, H)
- `block` - Block number (1-3 for experimental, 1-3 for baseline)
- `track_point_accuracy` - Tracking task accuracy (0-1)
- `resman_point_accuracy` - Resource management accuracy (0-1)
- `sysmon_point_accuracy` - System monitoring accuracy (0-1)
- `comms_point_accuracy` - Communications accuracy (0-1)
- `composite_point_accuracy` - Weighted average across all tasks
- `*_response_time` - Mean response times per task (ms)

Each row represents one block of data for one participant in one condition.

### aggregated_nasatlx.csv

NASA-TLX ratings per participant per condition:

**Columns**:
- `participant` - Participant ID
- `condition` - Workload level (L, M, H)
- `mental_demand` - Mental demand rating (0-100)
- `physical_demand` - Physical demand rating (0-100)
- `temporal_demand` - Temporal demand rating (0-100)
- `performance` - Perceived performance rating (0-100)
- `effort` - Effort rating (0-100)
- `frustration` - Frustration rating (0-100)

## Scripts

### matb_point_accuracy.py

Parses MATB log files and computes point accuracy metrics for each task.

**Point accuracy calculation**:
- Tracking: Percentage of time within target radius
- Resource management: Percentage of correct pump activations
- System monitoring: Percentage of detected anomalies
- Communications: Percentage of correct responses to prompts

The script handles:
- Multiple participants and conditions
- Both experimental and baseline sessions
- Missing data and incomplete trials
- Composite accuracy weighting across tasks

### extract_nasa_tlx.py

Extracts NASA-TLX ratings from raw questionnaire data files.

Processes:
- Post-task TLX ratings for each condition
- All six NASA-TLX dimensions
- Conversion to standard 0-100 scale
- Missing data handling

### performance_utils.py

Shared utility functions for:
- Data loading and cleaning
- Statistical model fitting (LME via rpy2)
- Correlation computation with outlier filtering
- Figure generation (heatmaps, scatter plots)
- LaTeX table formatting

### performance_analysis.ipynb

Main analysis notebook that:
1. Loads performance and TLX data
2. Fits linear mixed-effects models testing condition effects
3. Performs post-hoc pairwise comparisons
4. Computes baseline-experimental correlations
5. Generates publication figures and tables

## Statistical Analysis

The pipeline uses linear mixed-effects models (LME) via R's `lme4` package through `rpy2`:

**Model structure**: `metric ~ condition + (1|participant)`

**Post-hoc tests**: Pairwise comparisons using `emmeans` with Holm-Bonferroni correction

**Correlation analysis**:
- Pearson correlations between baseline and experimental performance
- Computed with all participants and with 3 SD outlier exclusion
- Separate correlations per task and for composite accuracy

## Dependencies

**Python packages**:
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- rpy2 (for R interface)

**R packages** (for statistical models in notebook):
- lme4 (linear mixed-effects models)
- emmeans (post-hoc comparisons)
- pbkrtest (denominator degrees of freedom)

All Python dependencies are included in the root `requirements.txt`. R packages must be installed separately if running statistical analyses.

## Output Table Descriptions

**corr_pre_main_all.tex**: Baseline-experimental correlations including all participants

**corr_pre_main_3sd.tex**: Baseline-experimental correlations excluding outliers (±3 SD from mean per task)

**performance_stats.tex**: Mixed-effects model results with F-statistics and p-values

**posthoc_comparisons.tex**: Pairwise comparisons between conditions (L vs M, M vs H, L vs H)

All tables are formatted for direct inclusion in LaTeX manuscripts.
