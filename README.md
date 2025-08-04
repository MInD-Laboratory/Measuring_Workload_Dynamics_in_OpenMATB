
<img width="2165" height="1067" alt="matb_figures" src="https://github.com/user-attachments/assets/d87d867b-481a-4840-9004-9bcdcf2f0a12" />

# Overview
This repository contains scripts and analysis pipelines for a study investigating cognitive load using a multitasking simulation (OpenMATB, https://github.com/juliencegarra/OpenMATB), webcam-based pose tracking, and supervised machine learning. 

Linear and nonlinear analysis is performed on the pose data. Nonlinear analysis (Recurrence-Quantification-Analysis) is performed using https://github.com/xkiwilabs/Recurrence-Quantification-Analysis. 

## 🔧 Setup Instructions

### 0. Clone the repository  
git clone https://github.com/MInD-Laboratory/matb_rqa_workload
cd matb_rqa_workload

### 1. Download data from OSF
Download the folder data from https://osf.io/dzgsv/. Baseline refers to the 2-minute blocks administered prior to the experimental 8-minute blocks. 
Place inside the matb_rqa_workload directory

### 2. Create a virtual environment and install dependencies 
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Perform the following steps for each analysis as detailed below

## MATB Task Performance Data

### Accuracy and Reaction Time
Performance logs from OpenMATB were parsed to extract task-specific accuracy and reaction times for the four subtasks (Monitoring, Tracking, Communications, Resource Management). Script: performance/matb_point_accuracy.py

NASA-TLX questionnares were administed through OpenMATB at the end of each experimental (8-min) block. Those were extracted via performance/extract_nasa_tlx.py

Statistics and figures reported for task performance measures can be found in figures_stats/validation_load.ipynb, which is broken up by topical subheading reported.

### Pose Data
Step 1: Preprocessing

    Raw OpenPose data (70 keypoints) → mapped to body regions.

    Script: analyze_pose/preprocess_pose.py

Step 2: Linear Metrics

    Features: Velocity, Acceleration, Root Mean Square (RMS).

    Script: analyze_pose/linear_pose.py

Step 3: Nonlinear Feature Setup (AMI + FNN)

    Calculate:

        Average Mutual Information (AMI) to estimate time lag.

        False Nearest Neighbors (FNN) to estimate embedding dimension.

    Scripts: analyze_pose/ami.py, analyze_pose/fnn.py

Step 4: Recurrence Quantification Analysis (RQA)

    Script: analyze_pose/rqa_pose.py

    Toolbox: RQA Toolbox

Step 5: Cross-Recurrence (CRQA)

    Gaze-head movement cross-recurrence metrics.

    Script: analyze_pose/crqa_pose.py

Step 6: Statistical Analysis

    Refer to figures_stats/pose_estimated_results.ipynb, organized by thesis subheadings.


1. First, raw pose data (70-keypoints from OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose) was first collapsed into a set of meaningful regions using analyze_pose/preprocess_pose.py. 

2. Using those set of regions, linear metrics (velocity, acceleration, RMS) were calculated using analyze_pose/linear_pose.py

3. AMI and FNN were calculated for RQA using analyze_pose/ami.py and fnn.py respectively to determine the right optimal embedding dimension and time lag

4. AutoRQA was run using Recurrence-Quantification-Analysis toolbox using the script analyze_pose/rqa_pose.py

5. CrossRQA between head-gaze was also run using Recurrence-Quantification-Analysis crossRQA function using the script under analyze_pose/crqa_pose.py

6. The results (stats/figures) for 4.2 (Pose-Estimated Resulst) can be found in figures_stats/pose_estimated_results.ipynb, which is broken up by thesis headings

# Machine Learning Models
Data taken from the above was used to train a series of random forest classifiers using scikit. Model selection for each model run is found in machine_learning.ipynb, using a custom function.
The preset settings in each cell for each model are the ones reported, divided by topical headings. 

# Publications
Available upon request.
