# src/model.py
"""
Train a simple linear regression model to forecast hourly demand.
Features: hour, weekday, month, is_weekend, trips_lag_1, trips_rolling_24h_mean
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils import train_test_split_time_series
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='processed data CSV (hourly)')
    p.add_argument('--out', default='models/linreg.joblib', help='output model path')
    p.add_argument('--test-days', type=int, default=30, help='how many last days to hold out for test')
    return p.parse_args()

def prepare_features(df):
    # Ensure features exist
    f_cols = ['hour','weekday','month','is_weekend','trips_lag_1','trips_rolling_24h_mean']
    for c in f_cols:
        if c not in df.columns:
            df[c] = 0
    # drop na
    df = df.dropna(subset=['trips_total'])
    df = df.dropna(subset=['trips_lag_1'])  # require lag
    X = df[f_cols].copy()
    y = df['trips_total'].astype(float)
    return X, y

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae':mae, 'rmse':rmse, 'r2':r2}

def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=['datetime'])
    df = df.sort_values('datetime')
    train, test = train_test_split_time_series(df, date_col='datetime', test_days=args.test_days)
    X_train, y_train = prepare_features(train)
    X_test, y_test = prepare_features(test)
    # pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores = evaluate(y_test, y_pred)
    print("Evaluation on test set:", scores)
    # Save model and simple metrics
    joblib.dump(pipe, args.out)
    metrics_path = Path(args.out).with_suffix('.metrics.json')
    import json
    with open(metrics_path,'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Saved model to {args.out} and metrics to {metrics_path}")

if __name__ == '__main__':
    main()
