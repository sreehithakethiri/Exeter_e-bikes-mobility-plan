import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "london_bike_hire.csv")
output_dir = os.path.join(script_dir, "..", "outputs")
plots_dir = os.path.join(output_dir, "station_plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# Loading data
df = pd.read_csv(data_path, low_memory=False)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])

#cleaning colunms to get required information.
df['start_station_name'] = df['start_station_name'].astype(str).str.strip()
df['end_station_name'] = df['end_station_name'].astype(str).str.strip()
df = df[(df['start_station_name'] != '') & (df['end_station_name'] != '')]
df = df[(df['start_station_name'] != '0') & (df['end_station_name'] != '0')]

#grouping the start and end staion data.
start_station_summary = df.groupby('start_station_name').size().reset_index(name='num_trips_start')
end_station_summary = df.groupby('end_station_name').size().reset_index(name='num_trips_end')

combined_summary = pd.merge(
    start_station_summary,
    end_station_summary,
    left_on='start_station_name',
    right_on='end_station_name',
    how='outer'
).fillna(0)

combined_summary['station'] = combined_summary['start_station_name'].combine_first(combined_summary['end_station_name'])
combined_summary = combined_summary[['station', 'num_trips_start', 'num_trips_end']]
combined_summary['total_trips'] = combined_summary['num_trips_start'] + combined_summary['num_trips_end']

#top 10 stations are listed below.
top_stations = combined_summary.sort_values(by='total_trips', ascending=False).head(10)['station'].tolist()
print("Top 10 stations:", top_stations)

#forcasting
daily_forecasts = []
monthly_forecasts = []

for station in top_stations:
    station_df = df[(df['start_station_name'] == station) | (df['end_station_name'] == station)]
    ts = station_df.groupby('date').size()
    ts = ts.asfreq('D', fill_value=0)

    if len(ts) < 14:
        print(f"Skipping station '{station}' due to insufficient data")
        continue

    try:
        model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,7))
        results = model.fit(disp=False, maxiter=50)

        forecast = results.get_forecast(steps=30)
        forecast_df = forecast.summary_frame()[['mean']].reset_index()
        forecast_df.rename(columns={'mean': 'forecast_trips', 'index': 'date'}, inplace=True)

        
         #Fixing negative or near-zero forecasts
       
        forecast_df['forecast_trips'] = forecast_df['forecast_trips'].clip(lower=0).round().astype(int)
        forecast_df['station'] = station
        daily_forecasts.append(forecast_df)

        # Monthly aggregation
        forecast_df['month'] = forecast_df['date'].dt.to_period('M')
        monthly_forecast = forecast_df.groupby('month')['forecast_trips'].sum().reset_index()
        monthly_forecast['station'] = station
        monthly_forecasts.append(monthly_forecast)

        # Plot
        plt.figure(figsize=(10,5))
        plt.plot(ts.index, ts.values, label='Actual Trips', marker='o')
        plt.plot(forecast_df['date'], forecast_df['forecast_trips'], label='Forecast Trips', marker='x')
        plt.title(f'Station: {station}')
        plt.xlabel('Date')
        plt.ylabel('Number of Trips')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{station}_forecast.png'))
        plt.close()

        print(f"Forecast & plot generated for station: {station}")

    except Exception as e:
        print(f"Error forecasting for station '{station}': {e}")

# Combine all forecasts and save Excel dashboard

if daily_forecasts:
    all_daily = pd.concat(daily_forecasts, ignore_index=True)
    all_monthly = pd.concat(monthly_forecasts, ignore_index=True)

    dashboard_path = os.path.join(output_dir, "station_dashboard_top10.xlsx")
    with pd.ExcelWriter(dashboard_path, engine='xlsxwriter') as writer:
        combined_summary[combined_summary['station'].isin(top_stations)].to_excel(writer, sheet_name="Top 10 Summary", index=False)
        all_daily.to_excel(writer, sheet_name="Daily Forecasts", index=False)
        all_monthly.to_excel(writer, sheet_name="Monthly Forecasts", index=False)

    print(f"Dashboard saved successfully at {dashboard_path}")
else:
    print("No forecasts generated. Dashboard not created.")
