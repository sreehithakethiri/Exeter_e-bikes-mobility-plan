# src/utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_csv(path, parse_dates=None):
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)

def train_test_split_time_series(df, date_col='datetime', test_days=30):
    # Sort and split by time: last `test_days` days as test
    df = df.sort_values(date_col)
    last_date = df[date_col].max()
    cutoff = last_date - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    return train, test

def safe_mean(x):
    return np.nan if len(x)==0 else np.mean(x)
