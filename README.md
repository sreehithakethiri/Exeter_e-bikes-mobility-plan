# Exeter_e-bikes-mobility-plan
The model is trained with TFL data i.e csv files from 14th april to 14 may2025 of daily usage of Ebikes in london.The csv files contain the station number,start station,end station,start station id ,end station id,trip duration , date and time of the trip made .THe data does not include any senstive information.
Objectives

The objective of this project is to study bike-sharing demand using real-world data. It aims to identify peak usage periods, analyze demand at start and end stations, and generate short-term demand forecasts to support operational and strategic planning.

Methodology

The dataset is cleaned and preprocessed using Python libraries such as Pandas and NumPy. Exploratory data analysis is performed to identify trends and seasonality in demand. A SARIMAX time-series model is then applied to forecast future demand. Results are visualized and exported for reporting purposes.

Project Structure

The repository is organized into clearly defined folders for data, source code, dashboards, and outputs. Separate scripts handle preprocessing, station-level analysis, forecasting, and result export, ensuring a modular and maintainable structure.

Technologies Used

The project is implemented using Python. Key libraries include Pandas and NumPy for data handling, Statsmodels for SARIMAX forecasting, Plotly for data visualization, Streamlit for the interactive dashboard, and Joblib for model management.
Results

The project generates station-level demand statistics, identifies peak weeks and months, and provides demand forecasts for future periods. Outputs include visualizations and Excel reports that support further analysis and interpretation.

How to Run

The repository can be cloned from GitHub and dependencies installed using the requirements.txt file. Individual analysis scripts can be executed from the source folder, and an optional Streamlit dashboard can be launched to view interactive results.
