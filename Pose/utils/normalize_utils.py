from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from .io_utils import find_real_colname

def normalize_to_screen(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    out = df.copy()
    x_cols = [c for c in out.columns if c.lower().startswith('x')]
    y_cols = [c for c in out.columns if c.lower().startswith('y')]
    out[x_cols] = out[x_cols] / float(width)
    out[y_cols] = out[y_cols] / float(height)
    return out

def interocular_series(df: pd.DataFrame, conf_prefix: Optional[str] = None) -> pd.Series:
    cols = list(df.columns)
    x37_col = find_real_colname("x", 37, cols)
    y37_col = find_real_colname("y", 37, cols)
    x46_col = find_real_colname("x", 46, cols)
    y46_col = find_real_colname("y", 46, cols)
    if not (x37_col and y37_col and x46_col and y46_col):
        return pd.Series(np.nan, index=df.index, dtype=float)
    x37 = pd.to_numeric(df[x37_col], errors="coerce").astype(float)
    y37 = pd.to_numeric(df[y37_col], errors="coerce").astype(float)
    x46 = pd.to_numeric(df[x46_col], errors="coerce").astype(float)
    y46 = pd.to_numeric(df[y46_col], errors="coerce").astype(float)
    return np.sqrt((x46 - x37) ** 2 + (y46 - y37) ** 2)
