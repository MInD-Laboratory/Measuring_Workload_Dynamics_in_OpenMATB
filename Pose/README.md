# Facial Pose Analysis Pipeline

Modular tools for processing facial pose data from OpenPose landmark detection.

## What it does
1. Load CSVs → filter to relevant landmarks
2. Mask low-confidence points (conf < threshold)
3. Interpolate short gaps + Butterworth low-pass
4. Normalize to screen (default 2560×1440)
5. Build templates (global + per-participant)
6. Extract features:
    - Procrustes vs global template (windowed)
    - Procrustes vs participant template (windowed)
    - Original (no Procrustes) features (windowed)
10. Drop windows containing any NaNs per metric
11. Compute linear metrics (mean |vel|, mean |acc|, RMS) per window
12. Save a JSON summary (config, masking stats, window drops, errors)

Outputs: CSVs under data/processed/* with per-frame features, windowed features, and linear metrics, plus templates and a summary JSON.


## Directory Structure

```
Pose/
├── pose_pipeline.py                # main driver (uses utils/*)
├── Interactive_Pipeline.ipynb  # interactive viewer + bars/stats
├── utils/
│  ├─ __init__.py                   # Config + flags
│  ├─ io_utils.py                   # I/O, filtering, masking, summaries
│  ├─ signal_utils.py               # interpolation, Butterworth filtering
│  ├─ normalize_utils.py            # normalization, interocular series
│  ├─ features_utils.py             # Procrustes/original features, linear metrics
│  ├─ window_utils.py               # windowing + helpers
│  ├─ nb_utils.py                   # notebook helpers (plots/loaders)
│  ├─ stats_utils.py                # omnibus + Holm–Bonferroni
│  └─ viz_utils.py                  # interactive viewer + Procrustes transform
├── data/
│  ├─ raw/                          # put raw CSVs here (or set CFG.RAW_DIR)
│  └─ processed/                    # pipeline writes here (or set CFG.OUT_BASE)
└── _old/                           # legacy code for reference

```

## Data Format
- Files: `data/raw/<participantID>_<condition>.csv` (e.g. 472_H.csv)
- Columns: `x1,y1,prob1,...,x70,y70,prob70` (case-insensitive; confidence prefix auto-detected among prob*, c*, confidence*)
- Default sampling: 60 Hz (configurable)

## Quick start

### Run the pipeline (CLI)
`python pose_pipeline.py`
This processes all CSVs in data/raw (or whatever CFG.RAW_DIR is set to) and writes results under data/processed.

### Run the notebook (interactive)

`jupyer lab`

Open `Interactive_Pipeline.ipynb`. At the top, set:

```
from pose_pipeline import Config
CFG = Config()
CFG.RAW_DIR  = "data/raw"                 # or your path
CFG.OUT_BASE = "data/processed"           # or your path
```

Then run the “Run pipeline” cell. The notebook auto-detects whether to run full pipeline or linear-only (if per-frame CSVs already exist).

The notebook supports:
- Interactive pose viewers (original vs Procrustes alignments)
- Bar charts of metrics by condition
- Stats tables (ANOVA/Kruskal, Holm–Bonferroni pairwise)


## Outputs

```
data/processed/<name>/
├─ reduced/                 # filtered triplets (x, y, prob)
├─ masked/                  # after confidence masking
├─ interp_filtered/         # after interpolation + Butterworth
├─ norm_screen/             # normalized coordinates
├─ templates/
│   ├─ global_template.csv
│   └─ participant_<pid>_template.csv 
├─ features/
│   ├─ original_features.csv # with no alignment
│   ├─ procrustes_global_features.csv
│   ├─ procrustes_participant_features.csv
│   └─ per_frame/           # full time series
│       ├─ original/*.csv
│       ├─ procrustes_global/*.csv
│       └─ procrustes_participant/*.csv
├─ linear_metrics/              # Velocity, acceleration, RMS for each feature
│   ├─ original_linear.csv
│   ├─ procrustes_global_linear.csv
│   └─ procrustes_participant_linear.csv
└─ pipeline_summary.json     # config, masking stats, window drops, etc.
```

