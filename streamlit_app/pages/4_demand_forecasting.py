import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")


def load_historical_data():
    """Load historical sales data from Supabase"""
    from ..config import supabase
    
    try:
        response = supabase.table('historical_sales').select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"Error loading data from Supabase: {str(e)}")
        return None

    # Convert date column to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def prepare_time_series(df, aggregation_level, target_store=None, target_product=None):
    """Prepare time series data for forecasting"""
    # Filter data if needed
    if target_store and target_store != "All Stores":
        df = df[df["store_id"] == target_store]

    if target_product and target_product != "All Products":
        df = df[df["product_id"] == target_product]

    # Aggregate data according to selected level
    if aggregation_level == "Daily":
        ts = df.groupby("date")["sales_quantity"].sum().reset_index()
        ts.set_index("date", inplace=True)
    elif aggregation_level == "Weekly":
        df["week"] = df["date"].dt.to_period("W").dt.start_time
        ts = df.groupby("week")["sales_quantity"].sum().reset_index()
        ts.set_index("week", inplace=True)
    elif aggregation_level == "Monthly":
        df["month"] = df["date"].dt.to_period("M").dt.start_time
        ts = df.groupby("month")["sales_quantity"].sum().reset_index()
        ts.set_index("month", inplace=True)

    return ts


def test_stationarity(ts):
    """Test time series for stationarity using ADF test"""
    result = adfuller(ts.values)

    # Format the results
    output = {
        "Test Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4],
    }

    is_stationary = result[1] <= 0.05

    return output, is_stationary


def apply_differencing(ts, d=1):
    """Apply differencing to time series"""
    return ts.diff(d).dropna()


def identify_arima_params(ts, max_p=5, max_d=2, max_q=5):
    """Identify optimal ARIMA parameters using AIC"""
    best_aic = float("inf")
    best_params = None

    # If series is not stationary, start with d=1
    _, is_stationary = test_stationarity(ts)
    min_d = 0 if is_stationary else 1

    # Limit the search space for performance
    for d in range(min_d, min(2, max_d) + 1):
        for p in range(min(3, max_p) + 1):
            for q in range(min(3, max_q) + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                except:
                    continue

    return best_params


def forecast_arima(ts, forecast_horizon, order=None):
    """Forecast using ARIMA model"""
    # Identify parameters if not provided
    if order is None:
        order = identify_arima_params(ts)

    # Fit model
    model = ARIMA(ts, order=order)
    results = model.fit()

    # Make forecast
    forecast = results.forecast(steps=forecast_horizon)

    # Calculate confidence intervals
    pred_interval = results.get_forecast(steps=forecast_horizon).conf_int()
    lower_bound = pred_interval.iloc[:, 0]
    upper_bound = pred_interval.iloc[:, 1]

    return forecast, lower_bound, upper_bound, results


def forecast_exponential_smoothing(ts, forecast_horizon):
    """Forecast using Exponential Smoothing (Holt-Winters)"""
    # Determine seasonality
    if len(ts) >= 12:
        # Use seasonal model if we have enough data points
        model = ExponentialSmoothing(
            ts,
            seasonal_periods=7 if ts.index.freqstr == "D" else 12,
            trend="add",
            seasonal="add",
        )
    else:
        # Otherwise use non-seasonal model
        model = ExponentialSmoothing(ts, trend="add")

    # Fit model
    results = model.fit()

    # Make forecast
    forecast = results.forecast(forecast_horizon)

    # Get confidence intervals
    # Note: statsmodels ExponentialSmoothing doesn't provide confidence intervals directly,
    # so we'll approximate them based on residual errors
    residuals = results.resid
    error_std = residuals.std()

    # Create confidence intervals (± 1.96 standard errors for 95% intervals)
    lower_bound = forecast - 1.96 * error_std
    upper_bound = forecast + 1.96 * error_std

    return forecast, lower_bound, upper_bound, results


def plot_forecast(ts, forecast, lower_bound, upper_bound, title="Forecast"):
    """Plot the original time series and forecast with confidence intervals"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot original data
    ts.plot(ax=ax, label="Historical Data")

    # Plot forecast
    forecast.plot(ax=ax, label="Forecast", color="red")

    # Plot confidence intervals
    ax.fill_between(
        lower_bound.index,
        lower_bound,
        upper_bound,
        color="pink",
        alpha=0.3,
        label="95% Confidence Interval",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Quantity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_diagnostics(model_results):
    """Plot diagnostic plots for the fitted model"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Plot residuals over time
    model_results.resid.plot(ax=axes[0])
    axes[0].set_title("Residuals")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)

    # Plot residual histogram
    model_results.resid.plot(kind="hist", ax=axes[1], bins=20)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    # Plot ACF of residuals
    plot_acf(model_results.resid, ax=axes[2], lags=20)
    axes[2].set_title("ACF of Residuals")
    axes[2].grid(True, alpha=0.3)

    # Plot PACF of residuals
    plot_pacf(model_results.resid, ax=axes[3], lags=20)
    axes[3].set_title("PACF of Residuals")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_forecast(forecast_df, filename="forecast_results"):
    """Save forecast results to file"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    filepath = f"data/{filename}.csv"
    forecast_df.to_csv(filepath)

    return filepath


def main():
    st.title("Demand Forecasting")

    st.markdown(
        """
    This tool helps you forecast future demand based on historical sales data.
    You can select different forecasting methods and customize parameters for your specific needs.
    """
    )

    # Check for available data files
    data_files = []
    if os.path.exists("data"):
        for file in os.listdir("data"):
            if file.endswith((".csv", ".parquet", ".json")) and not file.endswith(
                "_metadata.json"
            ):
                data_files.append(file)

    # Sidebar for data selection and forecasting options
    with st.sidebar:
        st.header("Data Selection")

        if data_files:
            selected_file = st.selectbox("Select Sales Data File", data_files)
            data_path = os.path.join("data", selected_file)
        else:
            st.warning(
                "No data files found. Please generate historical sales data first."
            )
            data_path = None

        st.header("Forecasting Options")

        aggregation = st.selectbox(
            "Time Aggregation", ["Daily", "Weekly", "Monthly"], index=1
        )

        forecast_horizon = st.slider("Forecast Horizon (periods)", 1, 52, 12)

        forecast_method = st.selectbox(
            "Forecasting Method", ["ARIMA", "Exponential Smoothing", "Compare Both"]
        )

        with st.expander("Advanced Settings"):
            if forecast_method == "ARIMA" or forecast_method == "Compare Both":
                st.subheader("ARIMA Parameters")
                auto_arima = st.checkbox("Auto-select parameters", value=True)

                if not auto_arima:
                    p = st.slider("p (AR order)", 0, 5, 1)
                    d = st.slider("d (Differencing)", 0, 2, 1)
                    q = st.slider("q (MA order)", 0, 5, 1)
                    arima_order = (p, d, q)
                else:
                    arima_order = None

            if (
                forecast_method == "Exponential Smoothing"
                or forecast_method == "Compare Both"
            ):
                st.subheader("Exponential Smoothing")
                st.markdown(
                    "Using Holt-Winters seasonal method with automatic parameter selection"
                )

    # Main content area
    if data_path and os.path.exists(data_path):
        # Load data
        with st.spinner("Loading data..."):
            df = load_historical_data(data_path)

            if df is not None:
                st.success(f"Loaded {len(df)} records from {selected_file}")

                # Display data summary
                st.subheader("Data Summary")

                col1, col2 = st.columns(2)

                with col1:
                    # Date range
                    min_date = df["date"].min()
                    max_date = df["date"].max()
                    st.write(f"**Date Range:** {min_date.date()} to {max_date.date()}")
                    st.write(f"**Total Days:** {(max_date - min_date).days + 1}")

                with col2:
                    # Products and stores
                    num_products = df["product_id"].nunique()
                    num_stores = df["store_id"].nunique()
                    st.write(f"**Products:** {num_products}")
                    st.write(f"**Stores:** {num_stores}")

                # Data filtering
                st.subheader("Data Filtering")

                col1, col2 = st.columns(2)

                with col1:
                    store_options = ["All Stores"] + sorted(
                        df["store_id"].unique().tolist()
                    )
                    selected_store = st.selectbox("Select Store", store_options)

                with col2:
                    product_options = ["All Products"] + sorted(
                        df["product_id"].unique().tolist()
                    )
                    selected_product = st.selectbox("Select Product", product_options)

                # Prepare time series
                with st.spinner("Preparing time series data..."):
                    ts = prepare_time_series(
                        df, aggregation, selected_store, selected_product
                    )

                # Display time series
                st.subheader("Historical Sales Data")
                fig, ax = plt.subplots(figsize=(12, 6))
                ts.plot(ax=ax)
                ax.set_title(f"Sales Over Time ({aggregation})")
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales Quantity")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Run forecasting
                if st.button("Generate Forecast", type="primary"):
                    with st.spinner("Generating forecast..."):
                        # Create tabs for different forecasting methods
                        if forecast_method == "Compare Both":
                            tab1, tab2 = st.tabs(["ARIMA", "Exponential Smoothing"])

                            with tab1:
                                # ARIMA forecast
                                st.subheader("ARIMA Forecast")

                                # Test stationarity
                                adf_results, is_stationary = test_stationarity(ts)

                                st.write("**Stationarity Test Results:**")
                                for key, value in adf_results.items():
                                    if key != "Critical Values":
                                        st.write(f"• {key}: {value:.4f}")

                                if is_stationary:
                                    st.success("The time series is stationary")
                                else:
                                    st.warning(
                                        "The time series is not stationary. Differencing may be needed."
                                    )

                                # Identify ARIMA parameters if auto-selection is enabled
                                if auto_arima:
                                    with st.spinner(
                                        "Identifying optimal ARIMA parameters..."
                                    ):
                                        arima_order = identify_arima_params(ts)
                                        st.write(
                                            f"**Identified optimal ARIMA parameters:** {arima_order}"
                                        )

                                # Run ARIMA forecast
                                forecast, lower, upper, model = forecast_arima(
                                    ts, forecast_horizon, arima_order
                                )

                                # Plot forecast
                                st.pyplot(
                                    plot_forecast(
                                        ts,
                                        forecast,
                                        lower,
                                        upper,
                                        f"ARIMA({arima_order[0]},{arima_order[1]},{arima_order[2]}) Forecast",
                                    )
                                )

                                # Diagnostic plots
                                st.subheader("Model Diagnostics")
                                st.pyplot(plot_diagnostics(model))

                                # Create downloadable forecast
                                forecast_df = pd.DataFrame(
                                    {
                                        "date": forecast.index,
                                        "forecast": forecast.values,
                                        "lower_bound": lower.values,
                                        "upper_bound": upper.values,
                                    }
                                )
                                forecast_df.set_index("date", inplace=True)

                                csv = forecast_df.to_csv()
                                st.download_button(
                                    label="Download ARIMA Forecast",
                                    data=csv,
                                    file_name=f"arima_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                )

                            with tab2:
                                # Exponential Smoothing forecast
                                st.subheader("Exponential Smoothing Forecast")

                                # Run Exponential Smoothing forecast
                                forecast, lower, upper, model = (
                                    forecast_exponential_smoothing(ts, forecast_horizon)
                                )

                                # Plot forecast
                                st.pyplot(
                                    plot_forecast(
                                        ts,
                                        forecast,
                                        lower,
                                        upper,
                                        "Holt-Winters Exponential Smoothing Forecast",
                                    )
                                )

                                # Diagnostic plots
                                st.subheader("Model Diagnostics")
                                st.pyplot(plot_diagnostics(model))

                                # Create downloadable forecast
                                forecast_df = pd.DataFrame(
                                    {
                                        "date": forecast.index,
                                        "forecast": forecast.values,
                                        "lower_bound": lower.values,
                                        "upper_bound": upper.values,
                                    }
                                )
                                forecast_df.set_index("date", inplace=True)

                                csv = forecast_df.to_csv()
                                st.download_button(
                                    label="Download Exponential Smoothing Forecast",
                                    data=csv,
                                    file_name=f"exp_smoothing_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                )

                        elif forecast_method == "ARIMA":
                            # ARIMA forecast
                            st.subheader("ARIMA Forecast")

                            # Test stationarity
                            adf_results, is_stationary = test_stationarity(ts)

                            st.write("**Stationarity Test Results:**")
                            for key, value in adf_results.items():
                                if key != "Critical Values":
                                    st.write(f"• {key}: {value:.4f}")

                            if is_stationary:
                                st.success("The time series is stationary")
                            else:
                                st.warning(
                                    "The time series is not stationary. Differencing may be needed."
                                )

                            # Identify ARIMA parameters if auto-selection is enabled
                            if auto_arima:
                                with st.spinner(
                                    "Identifying optimal ARIMA parameters..."
                                ):
                                    arima_order = identify_arima_params(ts)
                                    st.write(
                                        f"**Identified optimal ARIMA parameters:** {arima_order}"
                                    )

                            # Run ARIMA forecast
                            forecast, lower, upper, model = forecast_arima(
                                ts, forecast_horizon, arima_order
                            )

                            # Plot forecast
                            st.pyplot(
                                plot_forecast(
                                    ts,
                                    forecast,
                                    lower,
                                    upper,
                                    f"ARIMA({arima_order[0]},{arima_order[1]},{arima_order[2]}) Forecast",
                                )
                            )

                            # Diagnostic plots
                            st.subheader("Model Diagnostics")
                            st.pyplot(plot_diagnostics(model))

                            # Create downloadable forecast
                            forecast_df = pd.DataFrame(
                                {
                                    "date": forecast.index,
                                    "forecast": forecast.values,
                                    "lower_bound": lower.values,
                                    "upper_bound": upper.values,
                                }
                            )
                            forecast_df.set_index("date", inplace=True)

                            csv = forecast_df.to_csv()
                            st.download_button(
                                label="Download ARIMA Forecast",
                                data=csv,
                                file_name=f"arima_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )

                        else:  # Exponential Smoothing
                            # Exponential Smoothing forecast
                            st.subheader("Exponential Smoothing Forecast")

                            # Run Exponential Smoothing forecast
                            forecast, lower, upper, model = (
                                forecast_exponential_smoothing(ts, forecast_horizon)
                            )

                            # Plot forecast
                            st.pyplot(
                                plot_forecast(
                                    ts,
                                    forecast,
                                    lower,
                                    upper,
                                    "Holt-Winters Exponential Smoothing Forecast",
                                )
                            )

                            # Diagnostic plots
                            st.subheader("Model Diagnostics")
                            st.pyplot(plot_diagnostics(model))

                            # Create downloadable forecast
                            forecast_df = pd.DataFrame(
                                {
                                    "date": forecast.index,
                                    "forecast": forecast.values,
                                    "lower_bound": lower.values,
                                    "upper_bound": upper.values,
                                }
                            )
                            forecast_df.set_index("date", inplace=True)

                            csv = forecast_df.to_csv()
                            st.download_button(
                                label="Download Exponential Smoothing Forecast",
                                data=csv,
                                file_name=f"exp_smoothing_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )
            else:
                st.error(f"Could not load data from {selected_file}")
    else:
        st.info("Please select a valid data file to continue")

        # Option to generate sample data
        st.subheader("No Data Available?")
        st.markdown(
            """
        If you don't have historical sales data yet, go to the **Historical Sales Generator** 
        page to create synthetic data for testing.
        """
        )


if __name__ == "__main__":
    main()
