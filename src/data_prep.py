import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess London bike hire CSV to hourly aggregated CSV")
    p.add_argument('--data', required=True, help='path to input CSV (trip-level or aggregated)')
    p.add_argument('--out', default='data/processed_hourly.csv', help='path to write processed hourly CSV')
    return p.parse_args()

def load_any(csv_path):
    """Load CSV with pandas (low_memory=False to avoid dtype warnings)."""
    return pd.read_csv(csv_path, low_memory=False)

def detect_datetime_column(df):
    """Return the best candidate datetime column name or raise ValueError."""
    candidates = [c for c in df.columns if any(tok in c.lower() for tok in ['date','time','start','timestamp'])]
    if not candidates:
        # to find common names
        for alt in ['datetime','start_date','start_time','started_at']:
            if alt in df.columns:
                return alt
        raise ValueError("No datetime-like column found. Please ensure your CSV has a datetime column (name containing 'date'/'time'/'start').")
    return candidates[0]

def parse_datetime_series(series):
    
    fmt = '%d/%m/%Y %H:%M'
    try:
        parsed = pd.to_datetime(series, format=fmt, errors='coerce', dayfirst=True)
        # if too many failures, fallback
        if parsed.isna().sum() > len(parsed) * 0.1:
            parsed = pd.to_datetime(series, errors='coerce', dayfirst=True, infer_datetime_format=True)
    except Exception:
        parsed = pd.to_datetime(series, errors='coerce', dayfirst=True, infer_datetime_format=True)
    return parsed

def clean_and_engineer(df):
    # detects datetime-like column
    dtc = detect_datetime_column(df)
    # parse datetimes robustly
    df[dtc] = parse_datetime_series(df[dtc])

    # looks for trip-duration or station columns typically present in trip-level datasets
    trip_level_indicators = [
        'tripduration', 'duration', 'end_station_id', 'start_station_id',
        'start_station_name', 'end_station_name', 'bikeid', 'bike_id'
    ]
    is_trip_level = any(col for col in trip_level_indicators if col in (c.lower() for c in df.columns))

    if is_trip_level:
        # aggregates trip-level rows into hourly counts
        df['datetime_hour'] = df[dtc].dt.floor('h')
        # count rows per hour
        agg = df.groupby('datetime_hour').agg(
            trips_total = ('datetime_hour', 'size')
        ).reset_index().rename(columns={'datetime_hour':'datetime'})

        # if tripduration present, compute mean_duration
        if 'tripduration' in (c.lower() for c in df.columns):
            # find actual column name (case-insensitive)
            td_col = next(c for c in df.columns if c.lower() == 'tripduration')
            durations = df.groupby(df[dtc].dt.floor('h'))[td_col].mean().reset_index(drop=True)
            # safe: only add if lengths match
            if len(durations) == len(agg):
                agg['mean_duration'] = durations
    else:
        # Assume already-aggregated: find a count-like column
        # Heuristic: any column with 'trip'/'rental'/'count'/'num'/'total'
        count_cols = [c for c in df.columns if any(k in c.lower() for k in ['count','trip','rental','num','total','hire'])]
        if not count_cols:
            # fallback: take second column if first is datetime-like
            # ensure we rename properly
            # find actual datetime column (dtc)
            other_cols = [c for c in df.columns if c != dtc]
            if not other_cols:
                raise ValueError("No count-like column found in aggregated data. Provide a 'trips_total' column or use trip-level CSV.")
            count_col = other_cols[0]
        else:
            count_col = count_cols[0]
        # standardize names
        df = df.rename(columns={dtc:'datetime', count_col:'trips_total'})
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', dayfirst=True, infer_datetime_format=True)
        agg = df[['datetime','trips_total']].copy()

    # Feature engineering
    agg = agg.sort_values('datetime').reset_index(drop=True)
    agg['hour'] = agg['datetime'].dt.hour
    agg['weekday'] = agg['datetime'].dt.weekday
    agg['month'] = agg['datetime'].dt.month
    agg['is_weekend'] = agg['weekday'].isin([5,6]).astype(int)
    # lags and rolling means
    agg['trips_lag_1'] = agg['trips_total'].shift(1)
    agg['trips_rolling_24h_mean'] = agg['trips_total'].rolling(window=24, min_periods=1).mean()

    # drop rows missing trips_total
    agg = agg.dropna(subset=['trips_total']).reset_index(drop=True)

    return agg

def main():
    args = parse_args()
    in_path = Path(args.data)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    df = load_any(in_path)
    processed = clean_and_engineer(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")

if __name__ == '__main__':
    main()
