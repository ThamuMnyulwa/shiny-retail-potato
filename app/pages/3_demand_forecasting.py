import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
from datetime import datetime, timedelta
import time

# Try to import XGBoost, but provide a fallback if it fails
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.toast("XGBoost not available. Using Gradient Boosting instead.", icon="⚠️")

st.set_page_config(
    page_title="Sales & Demand Forecasting", page_icon="📈", layout="wide"
)


def load_historical_data(file_path):
    """Load historical sales data from file"""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            st.toast(f"CSV file loaded successfully", icon="📊")
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
            st.toast(f"Parquet file loaded successfully", icon="📊")
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
            st.toast(f"JSON file loaded successfully", icon="📊")
        else:
            st.error(f"Unsupported file format: {file_path}")
            st.toast(f"Unsupported file format: {file_path}", icon="❌")
            return None

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Rename product_id to sku_id if it exists
        if "product_id" in df.columns and "sku_id" not in df.columns:
            df = df.rename(columns={"product_id": "sku_id"})
            st.toast("Renamed 'product_id' to 'sku_id'", icon="🔄")

        # If stock_on_hand not in columns, add it with random values
        if "stock_on_hand" not in df.columns:
            st.warning(
                "Stock on hand data not found in dataset. Using simulated values."
            )
            st.toast("Simulating stock levels (20-80%)", icon="⚠️")
            # Generate random stock levels between 0.2 and 0.8 (lower than before to simulate constrained historical stock)
            df["stock_on_hand"] = np.random.uniform(0.2, 0.8, size=len(df))
        else:
            # If stock_on_hand exists but we want to lower it for this simulation
            # Scale it down to be between 0.2 and 0.8 of its original value
            max_stock = df["stock_on_hand"].max()
            if max_stock > 0:
                # Scale to keep the relative differences but lower the overall values
                df["stock_on_hand"] = 0.2 + (df["stock_on_hand"] / max_stock) * 0.6
                st.toast("Adjusted stock levels to 20-80% range", icon="📉")

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.toast(f"Failed to load data: {str(e)}", icon="❌")
        return None


def prepare_time_series(
    df, aggregation_level="Weekly", target_store=None, target_sku=None
):
    """Prepare time series data for forecasting"""
    # Filter data if needed
    if target_store and target_store != "All Stores":
        df = df[df["store_id"] == target_store]

    if target_sku and target_sku != "All SKUs":
        df = df[df["sku_id"] == target_sku]

    # Create week column if using weekly aggregation
    if aggregation_level == "Weekly" and "week" not in df.columns:
        df["week"] = df["date"].dt.to_period("W").dt.start_time

    # Return the filtered dataframe for demand modeling
    return df


def create_demand_forecast_dataset(df, aggregation_level="Weekly", forecast_horizon=12):
    """Create a dataset with store-SKU-week level data for demand forecasting"""
    # Show a toast notification when starting the dataset creation
    st.toast("Preparing forecast dataset...", icon="⏳")

    # Ensure we have week column
    if "week" not in df.columns and aggregation_level == "Weekly":
        df["week"] = df["date"].dt.to_period("W").dt.start_time

    # Aggregate to store-sku-week level
    if aggregation_level == "Weekly":
        grouped = (
            df.groupby(["store_id", "sku_id", "week"])
            .agg(
                {
                    "sales_quantity": "sum",
                    "stock_on_hand": "mean",  # Average stock level for the week
                }
            )
            .reset_index()
        )
    elif aggregation_level == "Daily":
        grouped = (
            df.groupby(["store_id", "sku_id", "date"])
            .agg({"sales_quantity": "sum", "stock_on_hand": "mean"})
            .reset_index()
        )
        grouped = grouped.rename(columns={"date": "time_period"})
    elif aggregation_level == "Monthly":
        df["month"] = df["date"].dt.to_period("M").dt.start_time
        grouped = (
            df.groupby(["store_id", "sku_id", "month"])
            .agg({"sales_quantity": "sum", "stock_on_hand": "mean"})
            .reset_index()
        )
        grouped = grouped.rename(columns={"month": "time_period"})

    if aggregation_level == "Weekly":
        grouped = grouped.rename(columns={"week": "time_period"})

    # No longer calculating estimated_demand using linear relationship
    # Instead, we'll use sales_quantity directly as our target and use stock_on_hand as a feature

    # Extract time features
    grouped["time_period"] = pd.to_datetime(grouped["time_period"])
    grouped["year"] = grouped["time_period"].dt.year
    grouped["month"] = grouped["time_period"].dt.month
    grouped["quarter"] = grouped["time_period"].dt.quarter
    grouped["week_of_year"] = grouped["time_period"].dt.isocalendar().week
    grouped["day_of_year"] = grouped["time_period"].dt.dayofyear
    grouped["days_since_start"] = (
        grouped["time_period"] - grouped["time_period"].min()
    ).dt.days

    # Add Fourier terms for seasonality - weekly and yearly
    grouped["week_sin"] = np.sin(2 * np.pi * grouped["week_of_year"] / 52)
    grouped["week_cos"] = np.cos(2 * np.pi * grouped["week_of_year"] / 52)
    grouped["month_sin"] = np.sin(2 * np.pi * grouped["month"] / 12)
    grouped["month_cos"] = np.cos(2 * np.pi * grouped["month"] / 12)

    # Add lag features
    store_sku_groups = grouped.groupby(["store_id", "sku_id"])

    # First determine how much history we have
    max_weeks = (grouped["time_period"].max() - grouped["time_period"].min()).days // 7
    st.info(f"Dataset spans approximately {max_weeks} weeks")

    # Dynamic lags based on forecast horizon
    # Short-term: 1x, 2x, 3x horizon
    # Medium-term: 0.25x, 0.5x, 0.75x horizon
    # Long-term: 1.0x horizon (if data permits)

    # Define dynamic lags based on forecast horizon
    lag_fractions = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    dynamic_lags = sorted(
        list(
            set(
                [
                    max(1, round(fraction * forecast_horizon))
                    for fraction in lag_fractions
                ]
            )
        )
    )

    # Add dynamic lags
    for lag in dynamic_lags:
        if lag <= max_weeks:  # Only add lags if we have enough data
            grouped[f"sales_lag_{lag}"] = store_sku_groups["sales_quantity"].shift(lag)

    # Always include essential short lags regardless of horizon
    essential_lags = [1, 2]
    for lag in essential_lags:
        if lag not in dynamic_lags and lag <= max_weeks:
            grouped[f"sales_lag_{lag}"] = store_sku_groups["sales_quantity"].shift(lag)

    # Add seasonal lags (52 weeks ago, 51 weeks ago, 53 weeks ago) for annual patterns
    # Only add these if we have at least 53 weeks of data
    if max_weeks >= 53:
        for seasonal_lag in [52, 51, 53]:
            grouped[f"sales_lag_{seasonal_lag}"] = store_sku_groups[
                "sales_quantity"
            ].shift(seasonal_lag)

    # Add rolling windows based on data availability
    available_windows = []
    if max_weeks >= 2:
        available_windows.append(2)
    if max_weeks >= 4:
        available_windows.append(4)
    if max_weeks >= 8:
        available_windows.append(8)
    if max_weeks >= 12:
        available_windows.append(12)

    # Add dynamic rolling windows based on forecast horizon
    horizon_windows = [max(2, round(forecast_horizon * 0.5)), forecast_horizon]
    for window in horizon_windows:
        if window <= max_weeks and window not in available_windows:
            available_windows.append(window)

    available_windows = sorted(list(set(available_windows)))

    for window in available_windows:
        grouped[f"sales_rolling_{window}"] = (
            store_sku_groups["sales_quantity"].rolling(window).mean().values
        )

    # Add larger windows only if we have enough data
    if max_weeks >= 26:
        grouped["sales_rolling_26"] = (
            store_sku_groups["sales_quantity"].rolling(26).mean().values
        )

    # Add expanding averages (all history)
    grouped["sales_expanding_mean"] = (
        store_sku_groups["sales_quantity"].expanding().mean().values
    )

    # Add volatility measures (standard deviation) - only for sufficient data
    volatility_windows = []
    if max_weeks >= 4:
        volatility_windows.append(4)
    if max_weeks >= 12:
        volatility_windows.append(12)

    # Add dynamic volatility window based on forecast horizon
    horizon_volatility = max(4, round(forecast_horizon * 0.5))
    if horizon_volatility <= max_weeks and horizon_volatility not in volatility_windows:
        volatility_windows.append(horizon_volatility)

    for window in volatility_windows:
        grouped[f"sales_volatility_{window}"] = (
            store_sku_groups["sales_quantity"].rolling(window).std().values
        )

    # Add temporal trend features
    # Normalize days_since_start for better model convergence
    grouped["time_trend"] = grouped["days_since_start"] / 365.0

    # Add diff features to capture rate of change - only if we have at least 2 periods
    if max_weeks >= 2:
        grouped["sales_diff"] = store_sku_groups["sales_quantity"].diff()

    # Fill NaN values in lag features with 0 for the first few periods
    # This allows us to keep more rows in the dataset, especially
    # important for short time series
    columns_to_fill = [
        col
        for col in grouped.columns
        if "lag_" in col or "rolling_" in col or "volatility_" in col or "diff" in col
    ]

    # Keep track of how many rows we're filling
    nan_rows_before = grouped.isna().any(axis=1).sum()
    grouped[columns_to_fill] = grouped[columns_to_fill].fillna(0)
    nan_rows_after = grouped.isna().any(axis=1).sum()

    filled_rows = nan_rows_before - nan_rows_after
    if filled_rows > 0:
        st.info(f"Filled {filled_rows} rows with zeros instead of dropping them")
        st.toast(f"Filled {filled_rows} rows with zeros", icon="✅")

    # Drop any remaining rows with NaN values
    # This should be much fewer now that we've filled most NaNs
    initial_rows = len(grouped)
    grouped = grouped.dropna()
    dropped_rows = initial_rows - len(grouped)

    if dropped_rows > 0:
        st.warning(
            f"Still dropped {dropped_rows} rows with missing values that couldn't be filled"
        )
        st.toast(f"Dropped {dropped_rows} rows with missing values", icon="⚠️")
    else:
        st.toast("Dataset prepared successfully", icon="✅")

    return grouped


def generate_future_stock_levels(
    historical_stock, forecast_horizon, expected_future_stock=None
):
    """Generate future stock level forecasts for use in sales forecasting"""
    # If no expected future stock is provided, use a simple moving average of historical levels
    if expected_future_stock is None:
        # Calculate average of last 12 periods (increase from 4)
        recent_avg = (
            historical_stock.iloc[-12:].mean()
            if len(historical_stock) >= 12
            else historical_stock.mean()
        )

        # Create future stock levels with slight random variation around the recent average
        # Set random seed for reproducibility
        np.random.seed(42)
        future_stock = np.clip(
            recent_avg + np.random.normal(0, 0.05, forecast_horizon), 0.1, 1.0
        )
    else:
        # Use the provided expected stock level
        future_stock = np.full(forecast_horizon, expected_future_stock)

    return future_stock


def prepare_features_for_forecast(
    df, store_id, sku_id, forecast_horizon, stock_levels=None
):
    """Prepare features for forecasting future periods"""
    # Get historical data for this store-SKU
    hist_data = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].sort_values(
        "time_period"
    )

    if len(hist_data) < 4:  # Changed from 12 to 4 to be more permissive
        return None

    # Get the last date in the historical data
    last_date = hist_data["time_period"].iloc[-1]

    # Generate future dates (in weekly increments)
    future_dates = [
        last_date + timedelta(weeks=i) for i in range(1, forecast_horizon + 1)
    ]

    # Create dataframe for future periods
    future_df = pd.DataFrame(
        {
            "time_period": future_dates,
            "store_id": store_id,
            "sku_id": sku_id,
        }
    )

    # Add time features
    future_df["year"] = future_df["time_period"].dt.year
    future_df["month"] = future_df["time_period"].dt.month
    future_df["quarter"] = future_df["time_period"].dt.quarter
    future_df["week_of_year"] = future_df["time_period"].dt.isocalendar().week
    future_df["day_of_year"] = future_df["time_period"].dt.dayofyear

    # Calculate days_since_start (keeping same reference point)
    start_date = hist_data["time_period"].min()
    future_df["days_since_start"] = (future_df["time_period"] - start_date).dt.days

    # Normalize time trend
    future_df["time_trend"] = future_df["days_since_start"] / 365.0

    # Add Fourier terms for seasonality
    future_df["week_sin"] = np.sin(2 * np.pi * future_df["week_of_year"] / 52)
    future_df["week_cos"] = np.cos(2 * np.pi * future_df["week_of_year"] / 52)
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)

    # Add stock levels if provided (only for demand forecast)
    if stock_levels is not None:
        future_df["stock_on_hand"] = stock_levels

    # Get historical values for lagged features
    hist_sales = hist_data["sales_quantity"].values

    # Define dynamic lags based on forecast horizon - same as in create_demand_forecast_dataset
    lag_fractions = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    dynamic_lags = sorted(
        list(
            set(
                [
                    max(1, round(fraction * forecast_horizon))
                    for fraction in lag_fractions
                ]
            )
        )
    )

    # Always include essential lags
    essential_lags = [1, 2]
    all_lags = sorted(list(set(dynamic_lags + essential_lags)))

    # Add all relevant lags that exist in the training data
    for lag in all_lags:
        lag_col_name = f"sales_lag_{lag}"
        if lag_col_name in df.columns:
            if len(hist_sales) >= lag:
                future_df[lag_col_name] = hist_sales[-lag]
            else:
                # If not enough history, use the oldest available or mean
                future_df[lag_col_name] = (
                    hist_sales.mean() if len(hist_sales) > 0 else 0
                )

    # Add seasonal lag features - only if we have enough data and columns exist
    max_weeks = (
        hist_data["time_period"].max() - hist_data["time_period"].min()
    ).days // 7
    for lag in [52, 51, 53]:
        lag_col_name = f"sales_lag_{lag}"
        if lag_col_name in df.columns:
            if len(hist_sales) >= lag and max_weeks >= lag:
                future_df[lag_col_name] = hist_sales[-lag]
            else:
                # If column exists in training data but we don't have enough history
                future_df[lag_col_name] = (
                    hist_sales.mean() if len(hist_sales) > 0 else 0
                )

    # Determine which rolling windows might exist in the training data
    potential_windows = [2, 4, 8, 12, 26]
    # Add dynamic windows based on forecast horizon
    potential_windows.extend([max(2, round(forecast_horizon * 0.5)), forecast_horizon])
    potential_windows = sorted(list(set(potential_windows)))

    # Add rolling window features that exist in the training data
    for window in potential_windows:
        window_col_name = f"sales_rolling_{window}"
        if window_col_name in df.columns:
            window_size = min(window, len(hist_sales))
            if window_size > 0:
                future_df[window_col_name] = hist_sales[-window_size:].mean()
            else:
                future_df[window_col_name] = 0

    # Add expanding average (full history)
    if "sales_expanding_mean" in df.columns:
        future_df["sales_expanding_mean"] = (
            hist_sales.mean() if len(hist_sales) > 0 else 0
        )

    # Determine which volatility windows might exist in the training data
    potential_volatility = [4, 12]
    # Add dynamic volatility window based on forecast horizon
    potential_volatility.append(max(4, round(forecast_horizon * 0.5)))
    potential_volatility = sorted(list(set(potential_volatility)))

    # Add volatility measures that exist in the training data
    for window in potential_volatility:
        volatility_col_name = f"sales_volatility_{window}"
        if volatility_col_name in df.columns:
            window_size = min(window, len(hist_sales))
            if window_size > 1:  # Need at least 2 points for std
                future_df[volatility_col_name] = hist_sales[-window_size:].std()
            else:
                future_df[volatility_col_name] = 0

    # Diff features
    if "sales_diff" in df.columns:
        if len(hist_sales) > 0:
            future_df["sales_diff"] = (
                hist_sales[-1] - hist_sales[-2] if len(hist_sales) > 1 else 0
            )
        else:
            future_df["sales_diff"] = 0

    return future_df


def train_ml_model(
    df, model_type="xgboost", target="sales_quantity", include_stock=True
):
    """
    Train a machine learning model on the dataset with hyperparameter tuning

    Parameters:
    - df: DataFrame with historical data
    - model_type: Type of ML model to use
    - target: Target variable to predict
    - include_stock: Whether to include stock_on_hand as a feature
    """
    # Define features and target
    features_to_drop = ["time_period", target]

    # If we don't want to include stock as a feature (for pure sales forecasting)
    if not include_stock and "stock_on_hand" not in features_to_drop:
        features_to_drop.append("stock_on_hand")

    X = df.drop(features_to_drop, axis=1)
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Select the appropriate model with improved hyperparameters
    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            random_state=42,
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
        )
    elif model_type == "gradient_boosting" or (
        model_type == "xgboost" and not XGBOOST_AVAILABLE
    ):
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features="sqrt",
            random_state=42,
        )
    else:  # linear_regression
        model = LinearRegression()

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    # Return the trained model, metrics, and processed features
    return pipeline, metrics, X.columns.tolist()


def forecast_by_store_sku(
    df,
    forecast_horizon,
    model_type="xgboost",
    expected_future_stock=None,
    demand_stock_multiplier=1.2,
):
    """Generate both sales and demand forecasts for each store-SKU combination"""
    # Show a toast when starting the forecasting process
    st.toast(
        f"Starting forecasts for {len(df[['store_id', 'sku_id']].drop_duplicates())} store-SKU combinations",
        icon="🔮",
    )

    # Show the stock multiplier being used
    st.info(f"Using demand stock multiplier: {demand_stock_multiplier}")

    # Get unique store-SKU combinations
    store_sku_combinations = df[["store_id", "sku_id"]].drop_duplicates()
    total_combinations = len(store_sku_combinations)

    forecasts = []
    debug_info = {
        "total_combinations": total_combinations,
        "filtered_out": 0,
        "failed": 0,
        "successful": 0,
    }

    # Add a debugging checkbox
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

    show_debug = st.checkbox(
        "Show model diagnostics", value=st.session_state.show_debug
    )
    st.session_state.show_debug = show_debug

    # Create containers for diagnostic information
    if show_debug:
        debug_expander = st.expander("Model Diagnostics", expanded=True)
        with debug_expander:
            st.write("This section shows diagnostic information about the models")
            stock_impact_container = st.container()
            feature_container = st.container()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Track model types and feature importance
    feature_importance_data = []
    sample_predictions = []

    # For each store-SKU combination
    for i, (_, row) in enumerate(store_sku_combinations.iterrows()):
        store_id = row["store_id"]
        sku_id = row["sku_id"]

        # Update progress using counter i instead of DataFrame index
        progress = min(1.0, i / max(1, total_combinations - 1))
        progress_bar.progress(progress)
        status_text.text(f"Processing: {i+1}/{total_combinations} combinations")

        # Show toast every 25% progress
        if i > 0 and i % max(1, total_combinations // 4) == 0:
            st.toast(f"Forecasting: {int(progress*100)}% complete", icon="🔄")

        # Filter data for this store-SKU
        subset = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)]

        if len(subset) <= 4:  # Need sufficient data points
            debug_info["filtered_out"] += 1
            continue

        try:
            # Train sales model WITHOUT stock_on_hand as a feature - pure time series forecast
            sales_model, sales_metrics, sales_features = train_ml_model(
                subset,
                model_type=model_type,
                target="sales_quantity",
                include_stock=False,  # Don't include stock for sales forecast
            )

            # Train demand model WITH stock_on_hand as a feature - learn stock-sales relationship
            demand_model, demand_metrics, demand_features = train_ml_model(
                subset,
                model_type=model_type,
                target="sales_quantity",  # Still predict sales_quantity
                include_stock=True,  # Include stock for demand forecasting
            )

            # Extract feature importance if using tree-based models
            if (
                model_type in ["xgboost", "random_forest", "gradient_boosting"]
                and show_debug
                and i < 5
            ):
                try:
                    # Get feature importance from the demand model
                    if hasattr(
                        demand_model.named_steps["model"], "feature_importances_"
                    ):
                        # For XGBoost/RF/GB
                        feat_imp = demand_model.named_steps[
                            "model"
                        ].feature_importances_
                        # Need to map to the original feature names
                        processed_features = demand_model.named_steps[
                            "preprocessor"
                        ].get_feature_names_out()
                        if len(processed_features) == len(feat_imp):
                            for feat, imp in zip(processed_features, feat_imp):
                                # Filter to only look at stock_on_hand importance
                                if "stock_on_hand" in feat:
                                    feature_importance_data.append(
                                        {
                                            "store_id": store_id,
                                            "sku_id": sku_id,
                                            "feature": feat,
                                            "importance": imp,
                                        }
                                    )
                except Exception as e:
                    st.warning(f"Could not extract feature importance: {str(e)}")

            # Get historical stock levels
            historical_stock = subset["stock_on_hand"]

            # Generate future stock levels for sales forecast
            sales_stock_levels = generate_future_stock_levels(
                historical_stock, forecast_horizon, expected_future_stock
            )

            # Generate future stock levels for demand forecast - always use the multiplier
            # This represents "what if we had abundant stock?"
            demand_stock_levels = np.full(forecast_horizon, demand_stock_multiplier)

            # Prepare features for forecasting - sales model (no stock feature)
            sales_future_df = prepare_features_for_forecast(
                df,
                store_id,
                sku_id,
                forecast_horizon,
                None,  # No stock needed for sales model
            )

            # Prepare features for forecasting - demand model (with stock feature)
            demand_future_df = prepare_features_for_forecast(
                df, store_id, sku_id, forecast_horizon, demand_stock_levels
            )

            if sales_future_df is None or demand_future_df is None:
                debug_info["filtered_out"] += 1
                continue

            # Make sales forecast using the sales model (no stock feature)
            X_sales = sales_future_df.drop(["time_period"], axis=1)
            sales_forecast = sales_model.predict(X_sales)

            # Make demand forecast using the demand model (with stock feature)
            X_demand = demand_future_df.drop(["time_period"], axis=1)
            demand_forecast = demand_model.predict(X_demand)

            # Store sample predictions for diagnostics
            if show_debug and i < 3:  # Store just a few samples
                sample_predictions.append(
                    {
                        "store_id": store_id,
                        "sku_id": sku_id,
                        "sales_prediction": sales_forecast[0],
                        "demand_prediction": demand_forecast[0],
                        "historical_avg_stock": historical_stock.mean(),
                        "historical_max_stock": historical_stock.max(),
                        "demand_stock_level": demand_stock_multiplier,
                    }
                )

            # Check if any demand prediction is less than its corresponding sales prediction
            # This shouldn't happen often with the correct approach, but let's ensure it
            demand_adjustments = 0
            for j in range(forecast_horizon):
                if demand_forecast[j] < sales_forecast[j]:
                    demand_forecast[j] = sales_forecast[j]
                    demand_adjustments += 1

            if demand_adjustments > 0 and show_debug:
                st.warning(
                    f"Adjusted {demand_adjustments} demand values to match sales for store {store_id}, SKU {sku_id}"
                )

            # Combine results
            for j in range(forecast_horizon):
                forecasts.append(
                    {
                        "store_id": store_id,
                        "sku_id": sku_id,
                        "time_period": sales_future_df["time_period"].iloc[j],
                        "forecasted_sales": sales_forecast[j],
                        "forecasted_demand": demand_forecast[j],
                        "stock_on_hand": sales_stock_levels[j],
                    }
                )

            debug_info["successful"] += 1

        except Exception as e:
            debug_info["failed"] += 1
            st.warning(f"Error forecasting store {store_id}, SKU {sku_id}: {str(e)}")
            # Add a toast for significant failures
            if debug_info["failed"] % 5 == 0:  # Show toast every 5 failures
                st.toast(
                    f"Encountered {debug_info['failed']} forecast errors", icon="⚠️"
                )

    # Set progress to 100% when done
    progress_bar.progress(1.0)
    status_text.text(
        f"Completed: {debug_info['successful']} successful, {debug_info['filtered_out']} filtered, {debug_info['failed']} failed"
    )

    # Display diagnostic information if requested
    if show_debug and feature_importance_data:
        with debug_expander:
            with stock_impact_container:
                st.subheader("Stock Impact on Demand Model")
                if feature_importance_data:
                    importance_df = pd.DataFrame(feature_importance_data)
                    st.write("Feature importance of stock_on_hand in demand models:")
                    st.dataframe(importance_df)

                    # Calculate average importance
                    avg_importance = importance_df["importance"].mean()
                    st.metric(
                        "Average stock_on_hand importance", f"{avg_importance:.4f}"
                    )

                    if avg_importance < 0.05:
                        st.error(
                            "⚠️ Stock level has very little impact on the demand model! This explains why demand ≈ sales."
                        )
                    elif avg_importance < 0.1:
                        st.warning(
                            "⚠️ Stock level has limited impact on the demand model."
                        )
                    else:
                        st.success(
                            "✅ Stock level has significant impact on the demand model."
                        )
                else:
                    st.write("No feature importance data available")

            with feature_container:
                st.subheader("Sample Predictions")
                if sample_predictions:
                    sample_df = pd.DataFrame(sample_predictions)
                    st.dataframe(sample_df)

                    # Calculate average difference
                    sample_df["difference"] = (
                        sample_df["demand_prediction"] - sample_df["sales_prediction"]
                    )
                    avg_diff = sample_df["difference"].mean()
                    if avg_diff < 0.1:
                        st.error(
                            f"⚠️ Tiny difference between demand and sales: {avg_diff:.2f}"
                        )
                    elif avg_diff < 1.0:
                        st.warning(
                            f"⚠️ Small difference between demand and sales: {avg_diff:.2f}"
                        )
                    else:
                        st.success(
                            f"✅ Good difference between demand and sales: {avg_diff:.2f}"
                        )
                else:
                    st.write("No sample predictions available")

    # Final toast with summary
    if debug_info["successful"] > 0:
        st.toast(
            f"Successfully generated {debug_info['successful']} forecasts", icon="✅"
        )
    else:
        st.toast("No forecasts could be generated", icon="❌")

    # Convert to dataframe
    if not forecasts:
        st.error(f"No forecasts could be generated. Debug info: {debug_info}")
        return pd.DataFrame()

    return pd.DataFrame(forecasts)


def plot_forecast_comparison(df, forecast_df, store_id, sku_id):
    """Plot comparison of historical data and forecasts"""
    # Filter data
    hist_data = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].sort_values(
        "time_period"
    )
    forecast_data = forecast_df[
        (forecast_df["store_id"] == store_id) & (forecast_df["sku_id"] == sku_id)
    ]

    if len(hist_data) == 0 or len(forecast_data) == 0:
        st.warning(f"No data available for Store {store_id}, SKU {sku_id}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical sales
    ax.plot(
        hist_data["time_period"],
        hist_data["sales_quantity"],
        marker="o",
        color="blue",
        label="Historical Sales",
    )

    # We no longer have estimated_demand in the historical data
    # so we'll remove that plot line

    # Plot forecast
    ax.plot(
        forecast_data["time_period"],
        forecast_data["forecasted_sales"],
        marker="o",
        color="red",
        label="Sales Forecast",
    )

    ax.plot(
        forecast_data["time_period"],
        forecast_data["forecasted_demand"],
        marker="x",
        color="purple",
        linestyle="--",
        label="Demand Forecast",
    )

    # Add stock level on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(
        hist_data["time_period"],
        hist_data["stock_on_hand"],
        color="gray",
        alpha=0.3,
        label="Historical Stock Level",
    )
    ax2.plot(
        forecast_data["time_period"],
        forecast_data["stock_on_hand"],
        color="gray",
        alpha=0.7,
        label="Projected Stock Level",
    )
    ax2.set_ylabel("Stock Level (0-1)")
    ax2.set_ylim(0, 1.1)

    # Format the plot
    ax.set_title(f"Sales & Demand Forecast - Store {store_id}, SKU {sku_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Quantity")

    # Format dates for better readability
    fig.autofmt_xdate()

    # Add legend for both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Show the plot
    st.pyplot(fig, use_container_width=True)


def calculate_lost_sales(forecast_df):
    """Calculate lost sales as the difference between potential demand and forecasted sales"""
    forecast_df["lost_sales"] = (
        forecast_df["forecasted_demand"] - forecast_df["forecasted_sales"]
    )
    forecast_df["lost_sales_percentage"] = (
        (forecast_df["lost_sales"] / forecast_df["forecasted_demand"] * 100)
        .fillna(0)
        .clip(lower=0)
    )

    return forecast_df


def find_latest_historical_data():
    """Find the most recent historical sales file generated"""
    if not os.path.exists("data"):
        return None

    data_files = []
    for file in os.listdir("data"):
        if file.endswith((".csv", ".parquet", ".json")) and not file.endswith(
            "_metadata.json"
        ):
            # Get file creation time
            file_path = os.path.join("data", file)
            creation_time = os.path.getctime(file_path)
            data_files.append((file, creation_time))

    if not data_files:
        return None

    # Sort by creation time (newest first)
    data_files.sort(key=lambda x: x[1], reverse=True)

    # Return the newest file
    return data_files[0][0]


def main():
    st.title("Sales & Demand Forecasting")

    # Application startup toast
    st.toast("Welcome to Sales & Demand Forecasting", icon="📈")

    st.markdown(
        """
    This tool forecasts both future sales and potential demand based on historical data.
    
    **Key Concepts:**
    - **Sales Forecast**: Expected future sales based purely on historical sales patterns
    - **Demand Forecast**: What could sell with optimal stock availability 
    - **Lost Sales**: The gap between potential demand and actual sales
    
    Forecasts are provided at the store-SKU-week level of granularity.
    
    ### How it Works:
    1. The **Historical Sales Generator** creates synthetic sales data with constrained stock levels
    2. This tool uses that data to forecast future sales and potential demand
    3. You can adjust stock level assumptions to see their impact on sales
    
    This simulation helps you understand the impact of increased inventory levels on sales performance.
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

    # Try to get the most recent file
    latest_file = find_latest_historical_data()

    # Sidebar for data selection and forecasting options
    with st.sidebar:
        st.header("Data Selection")

        if data_files:
            # Use the latest file as default if available
            default_index = (
                data_files.index(latest_file) if latest_file in data_files else 0
            )

            selected_file = st.selectbox(
                "Select Historical Sales Data",
                data_files,
                index=default_index,
                help="Choose a file generated by the Historical Sales Generator",
            )
            data_path = os.path.join("data", selected_file)

            if st.button("Go to Historical Sales Generator"):
                # Use the correct rerun method based on Streamlit version
                st.switch_page("pages/1_historical_sales_generator.py")
        else:
            st.warning(
                "No data files found. Please generate historical sales data first using the Historical Sales Generator."
            )
            if st.button("Go to Historical Sales Generator"):
                # Use the correct rerun method based on Streamlit version
                st.switch_page("pages/1_historical_sales_generator.py")
            data_path = None

        st.header("Forecasting Options")

        # Always use weekly aggregation as per requirements
        aggregation = "Weekly"
        st.info("Using weekly aggregation as per requirements")

        forecast_horizon = st.slider(
            "Forecast Horizon (weeks)",
            1,
            52,
            12,
            help="Number of weeks to forecast into the future",
        )

        # Adjust model options based on XGBoost availability
        if XGBOOST_AVAILABLE:
            model_options = ["xgboost", "random_forest", "linear_regression"]
            display_names = {
                "xgboost": "XGBoost",
                "random_forest": "Random Forest",
                "linear_regression": "Multivariate Regression",
            }
        else:
            model_options = ["gradient_boosting", "random_forest", "linear_regression"]
            display_names = {
                "gradient_boosting": "Gradient Boosting",
                "random_forest": "Random Forest",
                "linear_regression": "Multivariate Regression",
            }

        model_type = st.selectbox(
            "Forecasting Model",
            model_options,
            index=0,
            format_func=lambda x: display_names.get(x, x),
        )

        with st.expander("Stock Level Assumptions"):
            st.info("Historical stock levels are constrained (20-80% availability)")

            use_expected_stock = st.checkbox(
                "Use expected future stock level for sales forecast",
                value=True,
                help="Set a specific future stock level instead of projecting from historical data",
            )
            if use_expected_stock:
                expected_stock = st.slider(
                    "Expected future stock level (0-1)",
                    0.0,
                    1.0,
                    1.0,  # Default to 1.0 (fully stocked)
                    0.05,
                    help="0 = No stock, 1 = Fully stocked",
                )
                st.info("This affects only the sales forecast.")
            else:
                expected_stock = None
                st.info(
                    "Sales forecast will use projected stock levels based on historical patterns"
                )

            # Add demand stock multiplier option - using a higher default
            demand_stock_multiplier = st.slider(
                "Demand stock level multiplier",
                1.0,
                1.5,  # Range now 1.0-1.5
                1.2,  # Changed default to 1.2 for more pronounced effect
                0.1,
                help="1.0 = Fully stocked; >1.0 = Over-stocked (potential higher demand)",
            )

            st.info(
                "Demand forecast uses this stock level multiplier to estimate potential demand with optimal or increased stock."
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
                    st.write(
                        f"**Total Weeks:** {((max_date - min_date).days // 7) + 1}"
                    )

                with col2:
                    # SKUs and stores
                    num_skus = df["sku_id"].nunique()
                    num_stores = df["store_id"].nunique()
                    st.write(f"**SKUs:** {num_skus}")
                    st.write(f"**Stores:** {num_stores}")
                    st.write(
                        f"**Store-SKU Combinations:** {df.groupby(['store_id', 'sku_id']).ngroups}"
                    )

                # Data filtering
                st.subheader("Data Filtering (for Analysis)")
                st.info(
                    "Note: Forecasts will be generated for all store-SKU combinations"
                )

                col1, col2 = st.columns(2)

                with col1:
                    store_options = ["All Stores"] + sorted(
                        df["store_id"].unique().tolist()
                    )
                    selected_store = st.selectbox(
                        "Select Store for Analysis", store_options
                    )

                with col2:
                    sku_options = ["All SKUs"] + sorted(df["sku_id"].unique().tolist())
                    selected_sku = st.selectbox("Select SKU for Analysis", sku_options)

                # Prepare data for forecasting
                st.subheader("Data Preparation")

                with st.spinner("Preparing time series data..."):
                    # Filter data based on selections
                    filtered_df = prepare_time_series(
                        df, aggregation, selected_store, selected_sku
                    )

                    # Pass the forecast horizon to create dataset for proper lag features
                    forecast_data = create_demand_forecast_dataset(
                        filtered_df, aggregation, forecast_horizon
                    )

                    st.write(f"Prepared {len(forecast_data)} records for forecasting")

                    # Show which lag features were created
                    lag_cols = [col for col in forecast_data.columns if "lag_" in col]
                    if lag_cols:
                        st.info(
                            f"Created {len(lag_cols)} lag features including: {', '.join(sorted(lag_cols)[:10])}"
                            + (
                                f" and {len(lag_cols)-10} more..."
                                if len(lag_cols) > 10
                                else ""
                            )
                        )

                # Generate forecasts
                st.subheader("Generate Forecasts")

                if st.button("Generate Forecasts"):
                    with st.spinner(
                        "Generating forecasts... This may take a few minutes."
                    ):
                        start_time = time.time()

                        # Get stock level for forecast
                        future_stock = expected_stock if use_expected_stock else None

                        # Generate forecasts
                        forecast_df = forecast_by_store_sku(
                            forecast_data,
                            forecast_horizon,
                            model_type,
                            future_stock,
                            demand_stock_multiplier,
                        )

                        if len(forecast_df) > 0:
                            # Calculate lost sales
                            forecast_df = calculate_lost_sales(forecast_df)

                            # Store in session state
                            st.session_state["forecast_data"] = forecast_data
                            st.session_state["forecasts"] = forecast_df
                            st.session_state["demand_stock_multiplier"] = (
                                demand_stock_multiplier
                            )

                            elapsed_time = time.time() - start_time
                            st.success(
                                f"Generated forecasts for {forecast_df['store_id'].nunique()} stores and {forecast_df['sku_id'].nunique()} SKUs"
                            )
                            st.toast(
                                f"Forecasts complete in {elapsed_time:.1f} seconds",
                                icon="🎉",
                            )

                            # Display store-SKU summary
                            st.subheader("Store-SKU Forecast Summary")

                            # Group by store and SKU
                            summary = (
                                forecast_df.groupby(["store_id", "sku_id"])
                                .agg(
                                    {
                                        "forecasted_sales": "sum",
                                        "forecasted_demand": "sum",
                                        "lost_sales": "sum",
                                    }
                                )
                                .reset_index()
                            )

                            # Calculate percentages
                            summary["lost_sales_percentage"] = (
                                (
                                    summary["lost_sales"]
                                    / summary["forecasted_demand"]
                                    * 100
                                )
                                .fillna(0)
                                .clip(lower=0)
                            )

                            # Display summary
                            st.dataframe(
                                summary.sort_values(
                                    "lost_sales_percentage", ascending=False
                                )
                            )

                            # Rerun to show the additional UI elements
                            st.rerun()
                        else:
                            st.error(
                                "Failed to generate forecasts. Please try a different dataset."
                            )
                            st.toast("Failed to generate forecasts", icon="❌")

                # Plot forecasts if they exist in session state
                if (
                    "forecasts" in st.session_state
                    and "forecast_data" in st.session_state
                ):
                    st.subheader("Forecast Visualization")

                    # Copy data from session state
                    forecast_df = st.session_state["forecasts"]
                    forecast_data = st.session_state["forecast_data"]
                    demand_stock_multiplier = st.session_state.get(
                        "demand_stock_multiplier", 1.0
                    )

                    # Allow user to select specific store-SKU to view
                    store_sku_combos = [
                        (s, p)
                        for s, p in forecast_df[["store_id", "sku_id"]]
                        .drop_duplicates()
                        .values
                    ]

                    col1, col2 = st.columns(2)

                    with col1:
                        store_to_plot = st.selectbox(
                            "Select Store",
                            sorted(forecast_df["store_id"].unique().tolist()),
                        )

                    with col2:
                        sku_to_plot = st.selectbox(
                            "Select SKU",
                            sorted(
                                forecast_df[forecast_df["store_id"] == store_to_plot][
                                    "sku_id"
                                ]
                                .unique()
                                .tolist()
                            ),
                        )

                    # Plot the selected store-SKU
                    plot_forecast_comparison(
                        forecast_data, forecast_df, store_to_plot, sku_to_plot
                    )

                    # Download forecasts
                    st.subheader("Download Forecasts")

                    csv = forecast_df.to_csv(index=False)
                    if st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name="demand_forecast.csv",
                        mime="text/csv",
                    ):
                        st.toast("Forecast data downloaded", icon="📁")

                    # Educational section
                    st.subheader("Understanding the Results")
                    st.info(
                        f"""
                    **Key Insights:**
                    
                    - **Forecasted Sales**: Expected sales based purely on historical sales patterns
                    - **Forecasted Demand**: Potential sales if fully stocked (stock_on_hand = {demand_stock_multiplier:.1f})
                    - **Lost Sales**: The gap between demand and actual sales (demand - sales)
                    
                    **Conceptual Approach:**
                    
                    This forecast uses two separate models with a crucial difference in how they handle stock levels:
                    
                    1. **Sales Model**: Trained on historical sales patterns WITHOUT stock level as a feature
                       - Assumes sales will follow historical patterns
                       - This is a "business as usual" forecast if you were to maintain current stock practices
                       
                    2. **Demand Model**: Trained on historical sales WITH stock level as a feature
                       - Learns how stock availability impacts sales from limited historical data (20-80% stock)
                       - Then for forecasting, we set stock level to {demand_stock_multiplier:.1f}
                       - This shows what could be sold with abundant or excessive stock
                    
                    **Impact of Stock Level Changes:**
                    
                    - Historical stock levels were constrained (typically 20-80%)
                    - Setting the demand multiplier to values above 1.0 models what might happen if you had *more than perfect* stock
                    - This approach helps identify SKUs that are most sensitive to stock improvements
                    
                    **Why This Matters:**
                    - Limited stock levels have likely been constraining your sales
                    - Understanding true demand helps with better inventory planning
                    - The gap between sales and demand indicates opportunity for growth
                    
                    **Actions You Can Take:**
                    1. Identify SKUs with high lost sales percentage
                    2. Experiment with higher stock levels for those SKUs
                    3. Re-run the forecast to see the impact on sales
                    """
                    )

                    # Educational section about the dynamic lag features
                    with st.expander("Advanced Feature Engineering Details"):
                        st.markdown(
                            """
                        ### Dynamic Lag Features
                        
                        This forecasting model creates lag features that adapt based on your forecast horizon. This means that the model will automatically:
                        
                        - Use **short-term lags** (1-2 weeks) for immediate patterns
                        - Create **medium-term lags** at fractions of your forecast horizon (25%, 50%, 75%)
                        - Include **long-term lags** equal to your forecast horizon
                        - Add **longer-term lags** at multiples of your horizon (2x, 3x) when sufficient data exists
                        
                        For example, with a 12-week forecast horizon, the model will use lags of approximately 3, 6, 9, 12, 24, and 36 weeks (if enough historical data is available).
                        
                        With a 52-week horizon, it will include lags of approximately 13, 26, 39, 52, 104, and 156 weeks.
                        
                        ### Two-Model Approach
                        
                        Our forecasting system uses two specialized models:
                        
                        1. **Pure Time Series Model for Sales**
                           - Ignores stock levels entirely
                           - Captures seasonality, trends, and patterns in sales history
                           - Predicts what will sell given historical sales patterns
                           
                        2. **Stock-Aware Model for Demand**
                           - Uses stock levels as a key feature
                           - Understands how stock availability affects sales
                           - When given optimal stock levels, predicts true potential demand
                        
                        This dual-model approach properly handles the causal relationship between stock levels and sales, avoiding the logical inconsistency of having demand less than sales.
                        """
                        )

                        # Show which lag features were used for this forecast
                        lag_cols = [
                            col
                            for col in forecast_data.columns
                            if "lag_" in col and col.startswith("sales_lag_")
                        ]
                        if lag_cols:
                            lag_values = [int(col.split("_")[-1]) for col in lag_cols]
                            lag_values.sort()
                            st.markdown(
                                f"**Lag features used in this forecast**: {', '.join([str(lag) for lag in lag_values])} weeks"
                            )

                            # Create a visual representation of the lags
                            fig, ax = plt.subplots(figsize=(10, 2))
                            ax.scatter(
                                lag_values, [1] * len(lag_values), s=100, color="blue"
                            )
                            ax.axvline(
                                x=forecast_horizon,
                                linestyle="--",
                                color="red",
                                label=f"Forecast Horizon ({forecast_horizon} weeks)",
                            )
                            ax.set_xlabel("Weeks Ago")
                            ax.set_yticks([])
                            ax.grid(axis="x", linestyle="--", alpha=0.5)
                            ax.legend()
                            ax.set_title("Lag Features Used in Forecast")
                            st.pyplot(fig)

            else:
                st.error("Failed to load data. Please check the file format.")
    else:
        st.warning("Please select a data file or generate historical sales data first.")
        st.toast("Please select data or generate historical sales first", icon="ℹ️")


if __name__ == "__main__":
    main()
