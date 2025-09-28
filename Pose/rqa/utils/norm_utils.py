import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Helper function: Data normalization
def normalize_data(data, norm):
    if norm == 1:
        return (data - np.min(data)) / (np.max(data) - np.min(data))  # Unit interval
    elif norm == 2:
        return (data - np.mean(data)) / np.std(data)  # Z-score
    elif norm == 3:
        return data - np.mean(data)  # Center around mean
    else:
        return data  # No normalization