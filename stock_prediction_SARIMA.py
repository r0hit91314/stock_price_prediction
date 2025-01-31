import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv(r"C:\Users\Aparna Anand\Downloads\archive (7)\ITC.csv")
print("First few rows of the dataset:")
print(data.head())  # Debug: Check the first few rows of the dataset

# Strip any leading/trailing spaces from column names
data.rename(columns=lambda x: x.strip(), inplace=True)

# Ensure the 'Date' column is present and formatted correctly
if 'Date' not in data.columns:
    raise KeyError("The 'Date' column is missing from the dataset.")

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Handle any invalid date formats
data.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates
data = data.sort_values('Date')  # Sort data by date
print("Date column properly formatted.")

# Check for duplicate dates and remove them
duplicates = data.duplicated(subset=['Date']).sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate dates. Removing them.")
    data = data.drop_duplicates(subset=['Date'])

# Ensure the 'Close' column is present
if 'Close' not in data.columns:
    raise KeyError("The 'Close' column is missing from the dataset.")

# Filter out necessary columns for time series forecasting
time_series_data = data[['Date', 'Close']].set_index('Date')

# Train-Test Split
train_data = time_series_data[time_series_data.index.year <= 2020]
test_data = time_series_data[time_series_data.index.year == 2021]
print(f"Training Data Shape: {train_data.shape}, Test Data Shape: {test_data.shape}")

# SARIMAX training with grid search for best parameters
best_mape = float('inf')
best_params = None
best_sarimax_model = None

for p in range(2):  # Reduced parameter space for faster testing
    for d in range(3):
        for q in range(2):
            for P in range(2):
                for D in range(3):
                    for Q in range(2):
                        try:
                            model = SARIMAX(
                                train_data['Close'],
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 7)  # Weekly seasonality
                            )
                            results = model.fit(disp=False)
                            predictions = results.fittedvalues
                            mape = np.mean(np.abs((train_data['Close'] - predictions) / train_data['Close'])) * 100
                            if mape < best_mape:
                                best_mape = mape
                                best_params = (p, d, q, P, D, Q)
                                best_sarimax_model = results
                        except Exception as e:
                            print(f"Error with parameters {(p, d, q, P, D, Q)}: {e}")
                            continue

print(f"Best SARIMAX Parameters: {best_params}")
print(f"Training MAPE: {best_mape:.2f}%")

# Forecast for 2021
forecast_steps = len(test_data)
forecast = best_sarimax_model.forecast(steps=forecast_steps)

# Combine actual and forecasted values
forecast_results = pd.DataFrame({
    'Date': test_data.index,
    'Actual': test_data['Close'].values,
    'Forecast': forecast
})
forecast_results['Error'] = abs(forecast_results['Actual'] - forecast_results['Forecast'])
forecast_results['Percentage_Error'] = (forecast_results['Error'] / forecast_results['Actual']) * 100
test_mape = forecast_results['Percentage_Error'].mean()

print(f"Test MAPE for 2021: {test_mape:.2f}%")

# Plot the forecast results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Close'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['Close'], label='Actual Data (2021)', color='green')
plt.plot(forecast_results['Date'], forecast_results['Forecast'], label='Forecast (2021)', color='orange')
plt.title('SARIMAX Forecast vs Actual (Daily)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

# Save forecast results to CSV
forecast_results.to_csv('SARIMAX_Forecast_Results_2021_Daily.csv', index=False)
print("Forecast results saved to 'SARIMAX_Forecast_Results_2021_Daily.csv'")

