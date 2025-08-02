import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def ami(timeseries, min_lag, max_lag):
    # Ensure the input is a NumPy array
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        timeseries = timeseries.values.flatten()  # Convert to 1D NumPy array if it's a Series or DataFrame
    elif not isinstance(timeseries, np.ndarray):
        raise ValueError("Input timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = timeseries
    length = len(x)

    # Create Lag Vector
    if max_lag <= (length // 2 - 1):
        if min_lag < max_lag:
            lag = np.arange(min_lag, max_lag + 1)
        else:
            print('error - maximum lag not greater than minimum lag')
            print('default lag vector used (0 - 50)')
            lag = np.arange(0, 51)
    else:
        print('error - maximum lag exceeds recommendation')
        print('maximum lag set to n/2-1')
        lag = np.arange(0, length // 2)

    # Normalize the data
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    ami_values = np.zeros(len(lag))

    # Compute Average Mutual Information (AMI)
    for i in tqdm(range(len(lag)), desc='Processing AMI'):
        k = int(np.floor(1 + np.log2(length - lag[i]) + 0.5))

        if np.var(x, ddof=1) == 0:
            ami_values[i] = 0
        else:
            ami_sum = 0
            for k1 in range(1, k + 1):
                for k2 in range(1, k + 1):
                    cond1 = (k1 - 1) / k < x[:length - lag[i]]
                    cond2 = x[:length - lag[i]] <= k1 / k
                    cond3 = (k2 - 1) / k < x[lag[i]:]
                    cond4 = x[lag[i]:] <= k2 / k

                    ppp = np.sum(cond1 & cond2 & cond3 & cond4)
                    px1 = np.sum(cond1 & cond2)
                    px2 = np.sum(cond3 & cond4)

                    if ppp > 0:
                        ppp = ppp / (length - lag[i])
                        px1 = px1 / (length - lag[i])
                        px2 = px2 / (length - lag[i])
                        ami_sum += ppp * np.log2(ppp / (px1 * px2))

            ami_values[i] = ami_sum

    # Create the AMI result array
    ami_result = np.column_stack((lag, ami_values))
    
    return ami_result

# ── Parameters ─────────────────────────────────────────────────────
ROOT_DIR = "data/pose/experimental_pose"
TARGET_COLUMNS = [
    "avg_pupil_x"
]
N_FILES_TO_USE = 20
MIN_LAG = 1
MAX_LAG = 100

# ── Run AMI across all signals ─────────────────────────────────────
ami_results = {col: [] for col in TARGET_COLUMNS}
csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))
csv_files = csv_files[:N_FILES_TO_USE]

for file in tqdm(csv_files, desc="Files"):
    df = pd.read_csv(os.path.join(ROOT_DIR, file))
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna().values
        if len(series) < 2 * MAX_LAG:
            continue
        ami_result = ami(series, MIN_LAG, MAX_LAG)
        ami_values = ami_result[:, 1]  
        ami_results[col].append(ami_values)

# ── Average AMI per signal ─────────────────────────────────────────
ami_averaged = {
    col: np.nanmean(ami_results[col], axis=0)
    for col in TARGET_COLUMNS if ami_results[col]
}
lags = ami_result[:, 0]  


for col in ami_averaged:
    plt.figure()
    plt.plot(lags, ami_averaged[col], marker='o')
    plt.title(f"AMI Curve (Averaged)\n{col}")
    plt.xlabel("Time Lag")
    plt.ylabel("Average Mutual Information")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
