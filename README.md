
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
├─ README.md       # (this file)
└─ requirements.txt
```

* **Pose/process_pose_linear.py** → runs **preprocessing and linear metrics** (same script).
* **Pose/process_pose_recurrence.py** → runs **RQA/CRQA** on pose features.
* **performance/** → parses MATB logs, computes accuracy & NASA-TLX, generates figures/tables.
* **Modeling/** → trains/evaluates Random Forest classifiers over pose/performance features.

## 1) Setup

### Clone (with submodules)

```bash
git clone https://github.com/MInD-Laboratory/matb_rqa_workload
cd matb_rqa_workload
git submodule update --init --recursive
```

### Python env

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data (OSF)

Download the raw OpenPose CSVs from **[https://osf.io/dzgsv/](https://osf.io/dzgsv/)** and place them here:

```
Pose/data/raw_data/
├─ experimental_pose/   # 8-min blocks
└─ baseline_pose/       # 2-min blocks
```

> Files are 70-keypoint OpenPose outputs (x, y, confidence per keypoint).

## 2) End-to-End Workflow (TL;DR)

1. **Pose: preprocess + linear metrics**

   ```bash
   cd Pose
   # Optional: edit utils/config.py to switch experimental ↔ baseline paths
   python process_pose_linear.py
   # Use --overwrite to force reprocessing
   # python process_pose_linear.py --overwrite
   ```

   Outputs under `Pose/data/processed_data/<experimental|baseline>/...`:

   * `features/` (original, procrustes_global, procrustes_participant)
   * `linear_metrics/` (velocity, acceleration, RMS)
   * `templates/`, `processing_summary.json`

2. **Pose: RQA/CRQA metrics**
   RQA uses the bundled submodule (see Pose/README for details).

   ```bash
   python process_pose_recurrence.py
   ```

   Outputs under `Pose/data/rqa/` and `Pose/figs/`.

3. **Performance: MATB accuracy + NASA-TLX**

   ```bash
   cd ../performance
   # Parse performance logs and compute accuracy/RT
   python matb_point_accuracy.py
   # Extract NASA-TLX
   python extract_nasa_tlx.py
   # Optional: open the main analysis notebook
   jupyter lab performance_analysis.ipynb
   ```

   Figures/tables saved to `performance/figs/` and `performance/tables/`.

4. **Modeling: Random Forest classification**
   Prepare/point to your feature CSVs (pose linear, RQA, performance). Then:

   ```bash
   cd ../Modeling
   # Configure FEATURE_GROUPS and experiments in run_pipeline.py
   python run_pipeline.py                 # run enabled experiments
   python visualize_results.py            # generate plots
   ```

   Results go to `Modeling/model_output/` and `Modeling/figs/` (SVG).

## 3) What You Get

* **Pose outputs**

  * Windowed features (`original`, `procrustes_global`, `procrustes_participant`)
  * Linear metrics (displacement, velocity, and acceleration)
  * RQA/CRQA metrics (recurrence, determinism, entropy, laminarity, etc.)
* **Performance outputs**

  * Per-task accuracy & composite accuracy
  * NASA-TLX workload ratings
  * Reproducible figures and LaTeX tables
* **Modeling outputs**

  * JSON summaries per experiment (balanced accuracy, F1, confusion matrices)
  * Consolidated CSV logs and publication-quality figures

## 4) Minimal Configuration Notes

* Default **Pose** config targets **experimental** (8-min) blocks. To switch to baseline, edit:

  * `Pose/utils/config.py` → `RAW_DIR` and `OUT_BASE`
* Windowing defaults: 60s windows, 50% overlap; low-confidence masking; Butterworth low-pass filter.
* RQA parameters and embedding (AMI/FNN) handled within the Pose pipeline utils.

## 5) Dependencies

* Python ≥ 3.10; install from `requirements.txt`.
* For statistics in notebooks: `rpy2` + R packages `lme4`, `emmeans` (optional but used in provided analyses).
* RQA submodule dependencies are included; see **Pose/recurrence_pose.ipynb** and **Pose/utils/rqa_utils.py** for any platform-specific notes.

## 6) Reproducibility Tips

* Use `--overwrite` on `process_pose_linear.py` to refresh derived outputs after config changes.
* Modeling supports **feature selection** and multiple **seeds**; heavy runs can be resumed (`--continue`) and summarized via `visualize_results.py`.
* All plots are saved as **SVG** for publication-ready figures.

## 7) Citations & Related

* OpenMATB: [https://github.com/juliencegarra/OpenMATB](https://github.com/juliencegarra/OpenMATB)
* OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Publications

* **Thesis** (submitted): *Detecting Cognitive Load Through the Structure of Behavior: An Ecological-Dynamical Approach*, Macquarie University (expected 2025).
* **Manuscript** (submitted, 2025): *Facial Movement Dynamics Reveal Workload During Complex Multitasking*.