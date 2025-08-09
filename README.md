
<img width="2165" height="1067" alt="matb_figures" src="https://github.com/user-attachments/assets/d87d867b-481a-4840-9004-9bcdcf2f0a12" />

# Overview
This repository contains scripts and analysis pipelines for a study investigating cognitive load using a modified version of the Open Multi-Attribute Task Battery (OpenMATB; https://github.com/juliencegarra/OpenMATB), webcam-based pose tracking, and supervised machine learning.

Participants engaged in four simultaneous subtasks—system monitoring, joystick-based tracking, verbal communications, and resource management—designed to simulate multitasking under varying levels of cognitive load. The task was structured into two phases: a baseline phase with three 2-minute blocks, and an experimental phase with three 8-minute blocks. Load was manipulated across low, moderate, and high conditions by adjusting task difficulty parameters such as anomaly frequency, target radius size, prompt rate, and fuel leakage. While performing the task, participants were recorded via webcam, providing data for subsequent behavioral analysis.

Linear and nonlinear analyses were applied to the pose data. Nonlinear dynamics were assessed using Recurrence Quantification Analysis (RQA) via the Recurrence-Quantification-Analysis toolbox.

## 🔧 Setup Instructions

### 0. Clone the repository  
git clone https://github.com/MInD-Laboratory/matb_rqa_workload

cd matb_rqa_workload

### 1. Download data from OSF
Download the raw pose data from https://osf.io/dzgsv/ and place it in the data/ folder inside the matb_rqa_workload directory. The folder contains 70-keypoint OpenPose output. Before running any analysis scripts, preprocess the raw data by running analyze_pose/preprocess_pose.py. This will generate usable pose data and save it to: data/preprocessed_pose/experimental/ for experimental 8-minute blocks and data/preprocessed_pose/baseline/ for baseline 2-minute blocks. All scripts that reference data/preprocessed_pose/ expect the data in this processed form.

### 2. Create a virtual environment and install dependencies 
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

Note: For RQA-related analyses, you must also install the dependencies required by the Recurrence-Quantification-Analysis toolbox. Follow the setup instructions provided in their README to ensure compatibility.

### 3. Perform the following steps for each analysis as detailed below

## MATB Task Performance Data

### Accuracy and Reaction Time
Performance logs from OpenMATB were parsed to extract task-specific accuracy and reaction times for the four subtasks (Monitoring, Tracking, Communications, Resource Management). Script: performance/matb_point_accuracy.py

NASA-TLX questionnares were administed through OpenMATB at the end of each experimental (8-min) block. Those were extracted via performance/extract_nasa_tlx.py

Statistics and figures reported for task performance measures can be found in figures_stats/validation_load.ipynb, which is broken up by topical subheading reported.

### Pose Data

1. First, raw pose data (70-keypoints from OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose) was first collapsed into a set of meaningful regions using analyze_pose/preprocess_pose.py. 

2. Using those set of regions, linear metrics (velocity, acceleration, RMS) were calculated using analyze_pose/linear_pose.py

3. AMI and FNN were calculated for RQA using analyze_pose/ami.py and fnn.py respectively to determine the right optimal embedding dimension and time lag

4. AutoRQA was run using Recurrence-Quantification-Analysis toolbox using the script analyze_pose/rqa_pose.py

5. CrossRQA between head-gaze was also run using Recurrence-Quantification-Analysis crossRQA function using the script under analyze_pose/crqa_pose.py

6. The results (stats/figures) for 4.2 (Pose-Estimated Results) can be found in figures_stats/pose_estimated_results.ipynb, which is broken up by thesis headings

# Machine Learning Models
Data taken from the above was used to train a series of random forest classifiers using scikit. Model selection for each model run is found in machine_learning.ipynb, using a custom function.

The preset settings in each cell for each model are the ones reported, divided by topical headings. 

# Publications
Available upon request.
