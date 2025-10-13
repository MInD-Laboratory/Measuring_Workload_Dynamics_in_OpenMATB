
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








## 🔧 Setup Instructions

### 0. Clone the repository  
git clone https://github.com/MInD-Laboratory/matb_rqa_workload
⚠️ This repo uses submodules. Run git submodule update --init --recursive after cloning.

cd matb_rqa_workload

### 1. Download data from OSF
Download the raw pose data from https://osf.io/dzgsv/ and place it in the data/ folder inside the matb_rqa_workload directory. The folder contains 70-keypoint OpenPose output. Before running any analysis scripts, preprocess the raw data by running analyze_pose/preprocess_pose.py. This will generate usable pose data and save it to: data/preprocessed_pose/experimental/ for experimental 8-minute blocks and data/preprocessed_pose/baseline/ for baseline 2-minute blocks. All scripts that reference data/preprocessed_pose/ expect the data in this processed form.

### 2. Create a virtual environment and install dependencies 
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

Note: For RQA-related analyses, you must also install the dependencies required by the Recurrence-Quantification-Analysis toolbox. Follow the setup instructions provided in their README to ensure compatibility. https://github.com/xkiwilabs/Recurrence-Quantification-Analysis 

### 3. Perform the following steps for each analysis as detailed below

## MATB Task Performance Data

### Accuracy and Reaction Time
Performance logs from OpenMATB were parsed to extract task-specific accuracy and reaction times for the four subtasks (Monitoring, Tracking, Communications, Resource Management). Script: performance/matb_point_accuracy.py

NASA-TLX questionnares were administed through OpenMATB at the end of each experimental (8-min) block. Those were extracted via performance/extract_nasa_tlx.py

Statistics and figures reported for task performance measures can be found in figures_stats/validation_load.ipynb, which is broken up by topical subheading reported.

### Pose Data
1. First, run the steps in analyze_pose/pose_preprocessing.ipynb. This notebook identifies 60-second windows with low OpenPose confidence (below 30% for more than one second) and saves their indices to analyze_pose/qc_outputs/metric_bad_window_indices.csv. It then preprocesses the raw 70-keypoint OpenPose output (https://github.com/CMU-Perceptual-Computing-Lab/openpose) into meaningful facial and head regions. Only windows flagged by the quality check are removed. The script also generates a drop-report CSV summarizing how many windows were processed, removed, and which metrics were affected. Cleaned, processed data are saved to data/preprocessed_pose/experimental/ for 8-minute blocks and data/preprocessed_pose/baseline/ for 2-minute blocks.

2. Using those set of regions, linear metrics (velocity, acceleration, RMS) were calculated using analyze_pose/linear_pose.py

3. AMI and FNN were calculated for RQA using analyze_pose/ami.py and fnn.py respectively to determine the right optimal embedding dimension and time lag

4. AutoRQA was run using Recurrence-Quantification-Analysis toolbox using the script analyze_pose/rqa_pose.py

5. CrossRQA between head-gaze was also run using Recurrence-Quantification-Analysis crossRQA function using the script under analyze_pose/crqa_pose.py

6. The results (stats/figures) for 4.2 (Pose-Estimated Results) can be found in figures_stats/pose_estimated_results.ipynb, which is broken up by thesis headings

# Machine Learning Models
Data taken from the above was used to train a series of random forest classifiers using scikit. Model selection for each model run is found in machine_learning.ipynb, using a custom function.

The preset settings in each cell for each model are the ones reported, divided by topical headings. 

# Publications
- Thesis: *Detecting Cognitive Load Through the Structure of Behavior: An Ecological-Dynamical Approach* (submitted), Macquarie University, Expected Aug 2025.
- Manuscript: *Facial Movement Dynamics Reveal Workload During Complex Multitasking* — submitted, 2025.
