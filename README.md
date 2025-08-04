Baseline refers to the 2-minute blocks administered prior to the experimental 8-minute blocks. 
I’ve attempted to rename all relevant paths from pre and main to baseline and experimental, 
but there may be some remnants I missed. For clarity: baseline = 2-min block, experimental = 8-min block.

# Performance Data
1. Scripts were written to extract meaningful metrics from the csvs outputted by OpenMATB; Specifically, 
to obtain accuracy and reaction time for each subtask. These are found under extras/matb_point_accuracy.py

2. NASA-TLX questionnares were administed through OpenMATB at the end of each experimental (8-min) block.
Those were extracted via extras/extract_nasa_tlx.py

3. The results (stats/figures) for 4.1 (Verification of Load Manipulation) can be found in 
extras/validation_load.ipynb, which is broken up by thesis headings

# Pose Data
1. Raw pose data (70-keypoints) was first collapsed into a set of 
meaningful regions using process_pose/raw_to_metrics.py

2. Using those set of regions, linear metrics were calculated using process_pose/linear_pose.py

3. AMI and FNN were calculated for RQA using process_pose/ami.py and fnn.py respectively

4. AutoRQA was run using Recurrence-Quantification-Analysis toolbox using the script
process_pose/rqa_pose.py

5. CrossRQA between head-gaze was also run using Recurrence-Quantification-Analysis crossRQA
function using the script under process_pose/crqa_pose.py

6. The results (stats/figures) for 4.2 (Pose-Estimated Resulst) can be found in 
extras/pose_estimated_results.ipynb, which is broken up by thesis headings

# Machine Learning Models
Data taken from the above was used to train a series of random forest classifiers using scikit. 
Model selection for each model run is found in machine_learning.ipynb, using a custom function.
The preset settings in each cell for each model are the ones reported in the thesis. 
