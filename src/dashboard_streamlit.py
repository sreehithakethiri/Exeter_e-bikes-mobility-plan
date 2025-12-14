import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(layout='wide', page_title="Bike Hire Dashboard - Exeter case")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=['datetime'])

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def main():
    st.title("London Cycle Hire — Insights for Exeter City Council")
    st.markdown(
        "Interactive dashboard to explore demand patterns and model forecasts. "
        "SARIMAX forecasts and heatmaps included. "
        "**Forecasts are indicative for Exeter based on TfL data.**"
    )

    st.sidebar.header("Data & Model")
    data_path = st.sidebar.text_input("Processed CSV path", value="data/processed_hourly.csv")
    model_path = st.sidebar.text_input("Trained model path", value="models/linreg.joblib")

    # Load data
    if not Path(data_path).exists():
        st.error(f"Data file not found at {data_path}. Please update path in sidebar.")
        return
    df = load_data(data_path)
    st.sidebar.success("Data loaded.")

    st.write("Sample of loaded data:")
    st.dataframe(df.head())

    # Overview of metrics
    total_period = df['datetime'].min().date(), df['datetime'].max().date()
    st.metric("Data range", f"{total_period[0]} → {total_period[1]}")
    st.metric("Total observations (hourly)", int(len(df)))
    st.metric("Average hourly trips", round(df['trips_total'].mean(), 2))

    # -------------------------
    # Time series plot
    # -------------------------
    st.subheader("Time series (sample slice)")
    min_date = df['datetime'].dt.date.min()
    max_date = df['datetime'].dt.date.max()
    date_range = st.date_input(
        "Select date range",
        value=[min_date, max_date],  # Show full dataset by default
        min_value=min_date,
        max_value=max_date
    )

    df_slice = df.copy()
    if len(date_range) == 2:
        start, end = date_range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
        filtered = df[(df['datetime'] >= start_ts) & (df['datetime'] < end_ts)]
        if not filtered.empty:
            df_slice = filtered
        else:
            st.warning("No data for selected range. Showing full dataset instead.")

    fig = px.line(df_slice, x='datetime', y='trips_total', title='Trips over selected period')
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Diurnal & weekly patterns
    # -------------------------
    st.subheader("Diurnal & weekly patterns")
    col1, col2 = st.columns(2)
    with col1:
        hourly = df_slice.groupby('hour')['trips_total'].mean().reset_index()
        fig2 = px.bar(hourly, x='hour', y='trips_total', title='Avg trips by hour')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        weekday = df_slice.groupby('weekday')['trips_total'].mean().reset_index()
        fig3 = px.line(
            weekday, x='weekday', y='trips_total', markers=True,
            title='Avg trips by weekday (0=Mon)'
        )
        st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # Heatmap of hourly × weekday demand
    # -------------------------
    st.subheader("Heatmap: Hour vs Weekday Demand")
    heatmap_data = df_slice.groupby(['weekday', 'hour'])['trips_total'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='hour', columns='weekday', values='trips_total')
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Weekday (0=Mon)", y="Hour", color="Avg Trips"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        aspect="auto",
        title="Hourly Demand Heatmap"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    
    st.subheader("Scenario explorer (hourly & weekday)")
    hour = st.slider("Hour of day", 0, 23, 8)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, 1)
    month = st.slider("Month", 1, 12, 7)
    is_weekend = 1 if weekday in [5,6] else 0

    avg_lag = float(df_slice['trips_total'].shift(1).mean())
    avg_roll = float(df_slice['trips_total'].rolling(24).mean().mean())

    scenario = pd.DataFrame([{
        'hour': hour, 'weekday': weekday, 'month': month, 'is_weekend': is_weekend,
        'trips_lag_1': avg_lag, 'trips_rolling_24h_mean': avg_roll
    }])

    if Path(model_path).exists():
        model = load_model(model_path)
        if all(col in model.feature_names_in_ for col in scenario.columns):
            pred_s = model.predict(scenario)[0]
            st.metric("Predicted trips (scenario)", f"{pred_s:.1f}")
        else:
            st.warning("Model does not include all required features for scenario.")
    else:
        st.write("Train model to enable scenario predictions.")

   
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=90, value=7, step=1)

   
  
    st.subheader(f"Forecast: Next {forecast_days} Days (Hourly)")
    try:
        df_small = df_slice[df_slice['datetime'] >= df_slice['datetime'].max() - pd.Timedelta(days=7)]
        sarimax_model = SARIMAX(
            df_small['trips_total'],
            order=(1,1,1),
            seasonal_order=(1,1,1,24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarimax_results = sarimax_model.fit(disp=False)

        future_steps = forecast_days * 24
        forecast = sarimax_results.get_forecast(steps=future_steps)
        predicted_trips = forecast.predicted_mean

        last_datetime = df_small['datetime'].max()
        future_dates = pd.date_range(start=last_datetime + pd.Timedelta(hours=1),
                                     periods=future_steps, freq='H')

        future_df = pd.DataFrame({'datetime': future_dates, 'predicted_trips': predicted_trips})
        fig_forecast = px.line(future_df, x='datetime', y='predicted_trips', title=f"Next {forecast_days} Days Forecast (Hourly)")
        st.plotly_chart(fig_forecast, use_container_width=True)

    except Exception as e:
        st.error(f"SARIMAX hourly forecast could not be computed: {e}")

    # SARIMAX Daily Total Forecast
    
    st.subheader(f"Forecast: Daily Total Trips (Next {forecast_days} Days)")
    try:
        df_daily = df_slice.resample('D', on='datetime')['trips_total'].sum().reset_index()
        df_daily_small = df_daily[df_daily['datetime'] >= df_daily['datetime'].max() - pd.Timedelta(days=14)]

        sarimax_daily_model = SARIMAX(
            df_daily_small['trips_total'],
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarimax_daily_results = sarimax_daily_model.fit(disp=False)

        future_steps_daily = forecast_days
        forecast_daily = sarimax_daily_results.get_forecast(steps=future_steps_daily)
        predicted_daily_trips = forecast_daily.predicted_mean

        last_date = df_daily_small['datetime'].max()
        future_dates_daily = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=future_steps_daily, freq='D')

        future_daily_df = pd.DataFrame({'date': future_dates_daily, 'predicted_trips': predicted_daily_trips})
        fig_daily = px.bar(future_daily_df, x='date', y='predicted_trips', title=f"Next {forecast_days} Days Forecast (Daily Total Trips)")
        st.plotly_chart(fig_daily, use_container_width=True)

    except Exception as e:
        st.error(f"SARIMAX daily forecast could not be computed: {e}")

    st.markdown("---")
    st.caption(
        "Notes: Dashboard uses TfL hourly bike hire data. "
        "SARIMAX provides hourly and daily forecasts for the selected horizon. "
        "Heatmap shows typical hourly demand patterns. Forecasts are indicative for Exeter."
    )

if __name__ == '__main__':
    main()
