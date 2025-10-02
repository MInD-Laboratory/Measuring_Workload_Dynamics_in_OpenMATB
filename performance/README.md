# Performance Analysis

This subfolder contains code, data, tables, and figures used for the analysis of task performance across baseline and experimental sessions.

![combined](https://github.com/user-attachments/assets/900afee7-d457-45b9-9a81-d13feb7a75f4)



## Folder Structure

```
├── data/
│ ├── out/ # Processed data outputs (CSV, cleaned tables)
│ └── raw/ # Raw input data (CSV, logs, etc.)
│
├── figs/ # Figure outputs
├── tables/ # Table outputs
│
├── init.py
├── exp4_orders.csv # Experimental condition order file
├── extract_nasa_tlx.py # Script for parsing NASA-TLX data
├── matb_point_accuracy.py # Script for calculating MATB performance metrics
├── performance_analysis.ipynb # Main Jupyter notebook for generating tables and figures
└── performance_utils.py # Utility functions shared across scripts
```

---

## Key Scripts

- **`matb_point_accuracy.py`**  
  Computes per-task and overall point accuracy for MATB performance data.
- **`extract_nasa_tlx.py`**  
  Extracts and processes NASA-TLX self-report measures.
- **`performance_utils.py`**  
  Utility functions for correlations, model fitting, and figure/table generation.
- **`performance_analysis.ipynb`**  
  Primary analysis notebook. Runs the pipeline, generates statistical results, and exports LaTeX tables and figures.

---

## Workflow

1. **Data ingestion**  
   Raw CSVs (`data/raw`) are cleaned and processed into accuracy metrics and TLX scores.
2. **Performance metrics**  
   - Accuracy scores per task (`tracking`, `resman`, `sysmon`, `comms`)  
   - Overall composite accuracy  
   - NASA-TLX workload ratings  
3. **Statistical analysis**  
   - Linear mixed-effects models (`lme4` via `rpy2`)  
   - Correlations between baseline and experimental sessions  
   - Outlier handling (3 SD cutoff)
4. **Outputs**  
   - **Figures** (in `figs/`): heatmaps, performance correlations.  
   - **Tables** (in `tables/`): formatted LaTeX tables for manuscript.  
---
## Reproducibility

- All analysis scripts assume Python ≥3.10 with `pandas`, `numpy`, `scipy`, `matplotlib`, and `rpy2` installed.  
- R dependencies: `lme4`, `emmeans`, `pbkrtest`.  
- Outputs (figures/tables) are automatically written to the `figs/` and `tables/` directories.
---

## Notes
- `corr_pre_main_all.tex` includes correlations with **all participants**.  
- `corr_pre_main_3sd.tex` excludes participants ±3 SD from mean (per task).  
- Figures are saved as `.svg` for publication-ready scaling.  
- Tables are in `.tex` for direct inclusion in LaTeX manuscripts.
