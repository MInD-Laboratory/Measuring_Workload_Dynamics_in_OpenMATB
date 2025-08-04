import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def ami(timeseries, min_lag, max_lag):
    # This function calculates the Average Mutual Information (AMI) for a given time series.
    # AMI helps to find patterns and dependencies in time-based data.

    # Convert input data to a NumPy array if it's not already
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        timeseries = timeseries.values.flatten()  # Make sure it's a 1D array
    elif not isinstance(timeseries, np.ndarray):
        raise ValueError("Input timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = timeseries
    length = len(x)

    # Create a vector of lag values (how far apart to compare data points)
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

    # Normalize the data so all values are between 0 and 1
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    ami_values = np.zeros(len(lag))  # Prepare an array to store AMI results

    # Calculate AMI for each lag value
    for i in tqdm(range(len(lag)), desc='Processing AMI'):
        k = int(np.floor(1 + np.log2(length - lag[i]) + 0.5))  # Number of bins for calculation

        if np.var(x, ddof=1) == 0:  # If data doesn't vary, AMI is zero
            ami_values[i] = 0
        else:
            ami_sum = 0
            # Loop through all possible bin combinations
            for k1 in range(1, k + 1):
                for k2 in range(1, k + 1):
                    # Check which data points fall into each bin
                    cond1 = (k1 - 1) / k < x[:length - lag[i]]
                    cond2 = x[:length - lag[i]] <= k1 / k
                    cond3 = (k2 - 1) / k < x[lag[i]:]
                    cond4 = x[lag[i]:] <= k2 / k

                    # Count how many data points meet all conditions
                    ppp = np.sum(cond1 & cond2 & cond3 & cond4)
                    px1 = np.sum(cond1 & cond2)
                    px2 = np.sum(cond3 & cond4)

                    # If there are matching points, calculate AMI contribution
                    if ppp > 0:
                        ppp = ppp / (length - lag[i])
                        px1 = px1 / (length - lag[i])
                        px2 = px2 / (length - lag[i])
                        ami_sum += ppp * np.log2(ppp / (px1 * px2))

            ami_values[i] = ami_sum  # Store AMI value for this lag

    # Combine lag values and AMI results into a single array
    ami_result = np.column_stack((lag, ami_values))
    
    return ami_result  # Return the results

# ── Parameters ─────────────────────────────────────────────────────
ROOT_DIR = "data/pose/experimental_pose"  # Folder containing data files
TARGET_COLUMNS = [
    "avg_pupil_x"  # Which column in the data to analyze
]
N_FILES_TO_USE = 100  # How many files to process
MIN_LAG = 1          # Smallest lag to check
MAX_LAG = 100        # Largest lag to check

# ── Run AMI across all signals ─────────────────────────────────────
ami_results = {col: [] for col in TARGET_COLUMNS}  # Prepare to store results
csv_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".csv"))  # List all CSV files
csv_files = csv_files[:N_FILES_TO_USE]  # Only use the first N files

for file in tqdm(csv_files, desc="Files"):
    df = pd.read_csv(os.path.join(ROOT_DIR, file))  # Read each file into a table
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            continue  # Skip if the column isn't in the file
        series = df[col].dropna().values  # Get the data, remove missing values
        if len(series) < 2 * MAX_LAG:
            continue  # Skip if not enough data points
        ami_result = ami(series, MIN_LAG, MAX_LAG)  # Calculate AMI
        ami_values = ami_result[:, 1]  # Get just the AMI values
        ami_results[col].append(ami_values)  # Store results

# ── Average AMI per signal ─────────────────────────────────────────
ami_averaged = {
    col: np.nanmean(ami_results[col], axis=0)  # Average AMI across all files
    for col in TARGET_COLUMNS if ami_results[col]
}
lags = ami_result[:, 0]  # Get the lag values (same for all files)

# ── Plot the results ───────────────────────────────────────────────
for col in ami_averaged:
    plt.figure()
    plt.plot(lags, ami_averaged[col], marker='o')  # Draw the AMI curve
    plt.title(f"AMI Curve (Averaged)\n{col}")      # Add a title
    plt.xlabel("Time Lag")                         # Label the x-axis
    plt.ylabel("Average Mutual Information")       # Label the y-axis
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # Display the plot