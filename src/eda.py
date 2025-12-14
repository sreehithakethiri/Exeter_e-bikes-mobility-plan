
"""
Exploratory analysis: compute summary stats for three key metrics and save simple plots.

Key metrics:
1) trips_total (demand)
2) mean_duration (if available) OR trips_rolling_24h_mean
3) hourly pattern (avg trips by hour)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='processed data CSV (hourly)')
    p.add_argument('--out', default='outputs', help='output dir')
    return p.parse_args()

def summary_stats(df, col):
    s = df[col].describe()[['mean','50%','std','min','max']].rename({'50%':'median'})
    return s

def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=['datetime'])
    # ensure numeric
    df['trips_total'] = pd.to_numeric(df['trips_total'], errors='coerce')
    df = df.dropna(subset=['trips_total'])
    metrics = {}
    metrics['trips_total'] = summary_stats(df, 'trips_total').to_dict()
    # Rolling mean metric
    if 'mean_duration' in df.columns:
        metrics['mean_duration'] = summary_stats(df, 'mean_duration').to_dict()
    else:
        metrics['trips_rolling_24h_mean'] = summary_stats(df, 'trips_rolling_24h_mean').to_dict()
    # Save metrics
    with open(os.path.join(args.out,'summary_metrics.txt'),'w') as f:
        for k,v in metrics.items():
            f.write(f"Metric: {k}\n")
            for stat,val in v.items():
                f.write(f"  {stat}: {val}\n")
            f.write("\n")
    print("Saved metrics to", args.out+'/summary_metrics.txt')

    # plots
    # 1. time series plot (sample)
    plt.figure(figsize=(12,4))
    df.set_index('datetime')['trips_total'].rolling(window=24).mean().plot()
    plt.title('24-hour Rolling Mean of Trips')
    plt.ylabel('trips (rolling mean)')
    plt.savefig(os.path.join(args.out,'timeseries_rolling24.png'), bbox_inches='tight')

    # 2. average trips by hour (diurnal pattern)
    plt.figure(figsize=(8,4))
    hourly = df.groupby('hour')['trips_total'].mean()
    sns.barplot(x=hourly.index, y=hourly.values)
    plt.title('Average trips by hour of day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Average trips')
    plt.savefig(os.path.join(args.out,'avg_by_hour.png'), bbox_inches='tight')

    # 3. weekday vs weekend
    plt.figure(figsize=(8,4))
    wd = df.groupby('weekday')['trips_total'].mean()
    sns.lineplot(x=wd.index, y=wd.values, marker='o')
    plt.title('Average trips by weekday (0=Mon)')
    plt.savefig(os.path.join(args.out,'avg_by_weekday.png'), bbox_inches='tight')

    print("Saved plots to", args.out)

if __name__ == '__main__':
    main()
