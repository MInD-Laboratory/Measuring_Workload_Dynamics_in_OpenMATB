# Facial Pose Analysis Pipeline

Modular tools for processing facial pose data from OpenPose landmark detection.

## Capabilities

- Quality control analysis with automated bad window detection
- Coordinate normalization using Procrustes alignment or eye-corner stabilization
- Feature extraction for facial expressions, head pose, and gaze
- Temporal filtering with Butterworth filters
- Statistical analysis and group comparisons
- Data visualization and report generation


## Directory Structure

```
Pose/
├── pose_analysis.ipynb          # Main analysis notebook
├── utils/                       # Modular utility functions
│   ├── __init__.py
│   ├── landmark_config.py       # Landmark definitions and configurations
│   ├── quality_control.py       # QC analysis and bad window detection
│   ├── coordinate_normalization.py  # Procrustes alignment and stabilization
│   ├── feature_extraction.py    # Individual feature extraction functions
│   ├── temporal_filtering.py    # Temporal smoothing and filtering
│   ├── masking.py              # Quality-based data masking
│   ├── statistical_analysis.py # Statistical analysis and normalization
│   ├── plotting.py             # Visualization functions
│   ├── data_loading.py         # Experimental condition mapping
│   └── pipeline.py             # Complete processing pipeline
├── feature_data/               # Output: Processed feature files
├── output/                     # Output: Analysis results and figures
└── _old/                       # Original code for reference
```

## Usage

Execute `pose_analysis.ipynb` or call pipeline functions directly.

## Features

- Eye aspect ratio (blink detection)
- Mouth opening distance
- Head rotation angle
- Face center coordinates and movement
- Eye region coordinates
- Pupil/gaze position and deviation
- Landmark confidence scores

## Processing

1. Quality control with window-based analysis
2. Coordinate normalization (Procrustes or eye-corner)
3. Feature extraction from landmarks
4. Temporal filtering
5. Statistical analysis

## Functions

**Quality Control**
- `run_quality_control_batch()` - Batch quality analysis
- `analyze_file_quality()` - Single file analysis
- `summarize_quality_control()` - Quality summaries

**Feature Extraction**
- `extract_all_features()` - All facial features
- `extract_eye_aspect_ratio()` - Eye closure
- `extract_mouth_opening()` - Mouth movement
- `extract_head_rotation()` - Head pose
- `extract_pupil_features()` - Gaze tracking

**Coordinate Normalization**
- `compute_procrustes_alignment()` - Procrustes alignment
- `compute_original_alignment()` - Eye-corner stabilization

**Statistical Analysis**
- `calculate_summary_statistics()` - Descriptive statistics
- `compare_groups_statistical()` - Group comparisons
- `perform_feature_analysis()` - Complete analysis
- `apply_z_score_normalization()` - Z-score normalization

**Data Loading**
- `load_pose_data_with_conditions()` - Load with conditions
- `parse_condition_mapping()` - Map conditions
- `prepare_data_for_statistical_analysis()` - Prepare for analysis

**Visualization**
- `plot_qc_summary()` - Quality control plots
- `plot_feature_timeseries()` - Time series plots
- `plot_statistical_bars()` - Statistical plots
- `plot_correlation_matrix()` - Correlation plots

## Outputs

- `feature_data/*.csv` - Extracted features
- `quality_control/*.csv` - Quality assessments
- `reports/*.csv` - Processing logs
- `output/figures/*.png` - Plots and visualizations