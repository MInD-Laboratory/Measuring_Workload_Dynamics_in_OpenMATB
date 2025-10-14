
<img width="2165" height="1067" alt="matb_figures" src="https://github.com/user-attachments/assets/d87d867b-481a-4840-9004-9bcdcf2f0a12" />

# Overview

This repository contains code and analysis pipelines for a study of cognitive load using a modified Open Multi-Attribute Task Battery (OpenMATB), webcam-based facial pose tracking, Recurrence Quantification Analysis (RQA), and supervised machine learning.

Participants engaged in four simultaneous subtasks—system monitoring, joystick-based tracking, verbal communications, and resource management—designed to simulate multitasking under varying levels of cognitive load. The task was structured into two phases: a baseline phase with three 2-minute blocks, and an experimental phase with three 8-minute blocks. Load was manipulated across low, moderate, and high conditions by adjusting task difficulty parameters such as anomaly frequency, target radius size, prompt rate, and fuel leakage. While performing the task, participants were recorded via webcam, providing data for subsequent behavioral analysis.

Linear and nonlinear analyses were applied to the pose data. Nonlinear dynamics were assessed using Recurrence Quantification Analysis (RQA) via the Recurrence-Quantification-Analysis toolbox. Features from these analyses were used to train Random Forest models to classify workload condition.

## Repo Layout

```
.
├─ Pose/           # Pose processing: preprocessing+linear, then RQA/CRQA, stats/figs
├─ performance/    # MATB performance metrics + NASA-TLX parsing and figures
├─ Modeling/       # Random Forest pipeline, feature selection, learning curves
├─ README.md       # This file
└─ requirements.txt
```

See subsection READMEs for detailed documentation:
- [Pose/README.md](Pose/README.md) - Facial pose processing pipeline
- [performance/README.md](performance/README.md) - Performance metric extraction
- [Modeling/README.md](Modeling/README.md) - Random Forest classification pipeline

**Key scripts**:
- **Pose/process_pose_linear.py** - Preprocessing and linear metrics extraction
- **Pose/process_pose_recurrence.py** - RQA/CRQA analysis on pose features
- **performance/matb_point_accuracy.py** - MATB performance accuracy computation
- **performance/extract_nasa_tlx.py** - NASA-TLX workload rating extraction
- **Modeling/run_rf_models.py** - Random Forest model training and evaluation
- **Modeling/run_learning_curves.py** - Learning curve experiments (optional)
- **Modeling/prepare_baseline_features.py** - Baseline aggregate feature generation (optional)

## 1) Setup

### Clone (with submodules)

```bash
git clone https://github.com/MInD-Laboratory/Measuring_Workload_Dynamics_in_OpenMATB.git
cd Measuring_Workload_Dynamics_in_OpenMATB
git submodule update --init --recursive
```

### Python env

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pip install -e Pose/rqa
```

### Data (OSF)

Download the raw OpenPose CSVs from [https://osf.io/dzgsv/](https://osf.io/dzgsv/) and place them here:

```
Pose/data/raw_data/
├─ experimental_pose/   # 8-min blocks
└─ baseline_pose/       # 2-min blocks
```

Files are 70-keypoint OpenPose outputs (x, y, confidence per keypoint).

**Note**: Processed feature files are included in the repository under `Pose/data/processed_data/`, `Pose/data/rqa/`, and `performance/data/out/`. Users can run Random Forest models directly without reprocessing raw OpenPose data.

## 2) End-to-End Workflow

### Step 1: Pose - Preprocess + Linear Metrics

```bash
cd Pose
python process_pose_linear.py
```

Processes raw OpenPose CSVs through preprocessing, normalization, Procrustes alignment, and linear metric extraction.

Outputs under `Pose/data/processed_data/<experimental|baseline>/`:
- `features/` - Windowed features (original, procrustes_global, procrustes_participant)
- `linear_metrics/` - Velocity, acceleration, RMS statistics
- `templates/` - Procrustes alignment templates
- `processing_summary.json` - Processing metadata

Optional: Edit `utils/config.py` to switch between experimental and baseline paths, or use `--overwrite` to force reprocessing.

### Step 2: Pose - RQA/CRQA Metrics

RQA uses the bundled submodule (see Pose/README for details).

```bash
python process_pose_recurrence.py
```

Computes recurrence quantification metrics on pose time series.

Outputs under `Pose/data/rqa/` and `Pose/figs/`:
- RQA/CRQA metrics per window (recurrence, determinism, entropy, laminarity, etc.)
- Recurrence plots and statistical analysis figures

### Step 3: Performance - MATB Accuracy + NASA-TLX

```bash
cd ../performance
python matb_point_accuracy.py    # Compute task accuracy
python extract_nasa_tlx.py        # Extract NASA-TLX ratings
```

Optional: Open the main analysis notebook for figures and tables:
```bash
jupyter lab performance_analysis.ipynb
```

Outputs to `performance/figs/` and `performance/tables/`:
- Per-task and composite accuracy metrics
- NASA-TLX workload ratings
- Correlation analyses and LaTeX tables

### Step 4: Modeling - Random Forest Classification

Processed feature files are already included in the repository. To run Random Forest experiments:

```bash
cd ../Modeling
python run_rf_models.py
```

This trains Random Forest classifiers on pose features, RQA metrics, and performance data to classify workload conditions (L/M/H).

Results saved to `Modeling/model_output/rf_models/` and `Modeling/model_output/lc_models/`:
- JSON files per experiment with detailed metrics
- `experiment_log.csv` - Summary of experiments (separate logs in each subdirectory)
- Confusion matrices

Optional visualization:
```bash
python visualize_results.py    # Generate publication-ready figures
```

### Step 5: Optional - Baseline Features and Learning Curves

**Generate baseline aggregate features** (to test individual difference models):
```bash
python prepare_baseline_features.py
```

**Run learning curve experiments** (to evaluate temporal prediction):
```bash
python run_learning_curves.py
```

See [Modeling/README.md](Modeling/README.md) for details on baseline features and learning curves.

## 3) What You Get

**Pose outputs**:
- Windowed features (original, procrustes_global, procrustes_participant)
- Linear metrics (displacement, velocity, acceleration statistics)
- RQA/CRQA metrics (recurrence, determinism, entropy, laminarity, etc.)

**Performance outputs**:
- Per-task accuracy (tracking, resource management, system monitoring, communications)
- Composite accuracy across all tasks
- NASA-TLX workload ratings
- Reproducible figures and LaTeX tables

**Modeling outputs**:
- JSON summaries per experiment (balanced accuracy, F1, Cohen's kappa, confusion matrices)
- Consolidated CSV logs (`experiment_log.csv`) for cross-experiment comparison
- Baseline aggregate features capturing individual differences
- Learning curve results showing performance vs. training duration
- Publication-quality SVG figures

## 4) Configuration Notes

**Pose processing**:
- Default config targets experimental (8-min) blocks
- To switch to baseline, edit `Pose/utils/config.py` → `RAW_DIR` and `OUT_BASE`
- Windowing: 60s windows with 50% overlap
- Preprocessing: Low-confidence masking, interpolation, Butterworth low-pass filter
- RQA parameters: Embedding dimension and time delay determined via AMI/FNN

**Random Forest modeling**:
- Default: 15 random seeds, no feature selection, no hyperparameter tuning (configurable)
- Split strategies: Random (80/20 stratified) or leave-participant-out
- Configuration-based filenames prevent overwrites when testing different settings
- Configurable in `Modeling/run_rf_models.py` and `Modeling/run_learning_curves.py`

## 5) Dependencies

- Python ≥ 3.10
- Install from `requirements.txt`
- For statistics in notebooks: `rpy2` + R packages `lme4`, `emmeans` (optional but used in provided analyses)
- RQA submodule dependencies are included (see Pose/README.md for details)

## 6) Reproducibility Tips

- Pose: Use `--overwrite` on `process_pose_linear.py` to refresh outputs after config changes
- Modeling: All experiments use multiple random seeds for stability assessment
- Feature selection: Repeated per seed to evaluate feature stability
- Plots: All figures saved as SVG for publication-ready quality
- Results: Complete configuration and selected features logged with all results

## 7) Citations & Related

**Software**:
- OpenMATB: [https://github.com/juliencegarra/OpenMATB](https://github.com/juliencegarra/OpenMATB)
- OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

**Publications**:
- Thesis (submitted): *Detecting Cognitive Load Through the Structure of Behavior: An Ecological-Dynamical Approach*, Macquarie University (expected 2025)
- Manuscript (submitted, 2025): *Facial Movement Dynamics Reveal Workload During Complex Multitasking*
