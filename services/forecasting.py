import pandas as pd
import numpy as np
from datetime import timedelta

# Placeholder for more complex forecasting models
# You might integrate models like ARIMA, Prophet, XGBoost, etc. here


def fit_simple_forecast(df, forecast_horizon=30):
    """Fits a very simple forecast (e.g., naive forecast or moving average)."""
    if df.empty or "date" not in df.columns or "demand" not in df.columns:
        return pd.DataFrame()  # Return empty if data is invalid

    df = df.sort_values("date")
    last_date = df["date"].iloc[-1]
    last_demand = df["demand"].iloc[-1]

    # Naive forecast: Use the last known value
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D"
    )
    forecast_values = [last_demand] * forecast_horizon

    forecast_df = pd.DataFrame({"date": forecast_dates, "forecast": forecast_values})

    # Optional: Add more sophisticated placeholder logic, e.g., moving average
    # window_size = 7
    # if len(df) >= window_size:
    #     moving_avg = df['demand'].rolling(window=window_size).mean().iloc[-1]
    #     forecast_values = [moving_avg] * forecast_horizon
    #     forecast_df['forecast'] = forecast_values

    return forecast_df


# --- Example of how you might structure other forecast models ---
# def fit_arima_forecast(df, order=(5,1,0), forecast_horizon=30):
#     """Placeholder for ARIMA model fitting."""
#     # Import statsmodels here if using
#     # try:
#     #     from statsmodels.tsa.arima.model import ARIMA
#     #     model = ARIMA(df['demand'], order=order)
#     #     model_fit = model.fit()
#     #     forecast = model_fit.predict(start=len(df), end=len(df) + forecast_horizon - 1)
#     #     forecast_dates = pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
#     #     return pd.DataFrame({'date': forecast_dates, 'forecast': forecast})
#     # except Exception as e:
#     #     print(f"ARIMA Error: {e}")
#     #     return pd.DataFrame() # Return empty on error
#     pass
