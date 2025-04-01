import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json

st.set_page_config(
    page_title="Historical Sales Generator", page_icon="📊", layout="wide"
)


def generate_sales_data(
    start_date,
    end_date,
    num_skus,
    num_stores,
    base_demand=100,
    seasonality=True,
    trend=0,
    noise_level=0.3,
    weekend_effect=0.2,
    promo_frequency=0.1,
    promo_effect=0.5,
):
    """
    Generate synthetic historical sales data for retail simulation.

    Parameters:
    -----------
    start_date : datetime
        Start date for the simulation
    end_date : datetime
        End date for the simulation
    num_skus : int
        Number of different SKUs to simulate
    num_stores : int
        Number of retail stores to simulate
    base_demand : float
        Base level of daily demand
    seasonality : bool
        Whether to include seasonality effect
    trend : float
        Trend coefficient (-1 to 1) for decreasing/increasing sales over time
    noise_level : float
        Level of randomness in the data
    weekend_effect : float
        Effect of weekends on sales (higher value = stronger effect)
    promo_frequency : float
        Frequency of promotional events (0-1)
    promo_effect : float
        Effect of promotions on sales (multiplier)

    Returns:
    --------
    DataFrame with sales data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    num_days = len(date_range)

    # Create SKU and store IDs
    sku_ids = [f"S{i:03d}" for i in range(1, num_skus + 1)]
    store_ids = [f"S{i:03d}" for i in range(1, num_stores + 1)]

    # Pre-calculate time-based effects
    time_effects = np.ones(num_days)

    # Add trend effect
    if trend != 0:
        time_index = np.linspace(0, 1, num_days)
        time_effects *= 1 + trend * time_index

    # Add seasonality (yearly cycle)
    if seasonality:
        day_of_year = np.array([d.dayofyear for d in date_range])
        seasonality_effect = 0.2 * np.sin(2 * np.pi * day_of_year / 365) + 0.1 * np.sin(
            4 * np.pi * day_of_year / 365
        )
        time_effects *= 1 + seasonality_effect

    # Add weekend effect
    day_of_week = np.array([d.dayofweek for d in date_range])
    weekend_mask = day_of_week >= 5  # 5 = Saturday, 6 = Sunday
    time_effects[weekend_mask] *= 1 + weekend_effect

    # Create empty list to store records
    records = []

    # Generate sales for each SKU, store, and day
    for sku_id in sku_ids:
        # SKU-specific base demand (some SKUs sell better than others)
        sku_popularity = np.random.uniform(0.7, 1.3)

        for store_id in store_ids:
            # Store-specific base demand (some stores sell more than others)
            store_size = np.random.uniform(0.8, 1.2)

            # Generate promotional events
            promo_days = np.random.random(num_days) < promo_frequency

            # Calculate daily sales
            for i, date in enumerate(date_range):
                base_sales = base_demand * sku_popularity * store_size * time_effects[i]

                # Add promotion effect
                if promo_days[i]:
                    base_sales *= 1 + promo_effect

                # Add random noise
                noise = np.random.normal(1, noise_level)
                sales = max(0, round(base_sales * noise))

                # Generate random stock on hand (between 0.5 and 1.0)
                stock_on_hand = np.random.uniform(0.5, 1.0)

                records.append(
                    {
                        "date": date,
                        "sku_id": sku_id,
                        "store_id": store_id,
                        "sales_quantity": sales,
                        "promotion": 1 if promo_days[i] else 0,
                        "stock_on_hand": stock_on_hand,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Add week column for easier weekly aggregation
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time

    return df


def save_data(df, filename, format="csv"):
    """Save generated data to file"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    filepath = f"data/{filename}.{format}"

    if format == "csv":
        df.to_csv(filepath, index=False)
    elif format == "parquet":
        df.to_parquet(filepath, index=False)
    elif format == "json":
        df.to_json(filepath, orient="records")

    # Save metadata
    metadata = {
        "filename": f"{filename}.{format}",
        "format": format,
        "rows": len(df),
        "unique_stores": df["store_id"].nunique(),
        "unique_skus": df["sku_id"].nunique(),
        "date_range": [
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        ],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(f"data/{filename}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return filepath


def plot_total_sales(df):
    """Plot total sales over time"""
    daily_sales = df.groupby("date")["sales_quantity"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_sales["date"], daily_sales["sales_quantity"])
    ax.set_title("Total Daily Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Quantity")
    ax.grid(True, alpha=0.3)

    return fig


def plot_sku_sales(df):
    """Plot sales by SKU"""
    sku_sales = (
        df.groupby("sku_id")["sales_quantity"].sum().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sku_sales.plot(kind="bar", ax=ax)
    ax.set_title("Total Sales by SKU")
    ax.set_xlabel("SKU ID")
    ax.set_ylabel("Sales Quantity")
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def plot_store_sales(df):
    """Plot sales by store"""
    store_sales = (
        df.groupby("store_id")["sales_quantity"].sum().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    store_sales.plot(kind="bar", ax=ax)
    ax.set_title("Total Sales by Store")
    ax.set_xlabel("Store ID")
    ax.set_ylabel("Sales Quantity")
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def plot_weekly_sales(df):
    """Plot sales by week"""
    weekly_sales = df.groupby("week")["sales_quantity"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weekly_sales["week"], weekly_sales["sales_quantity"])
    ax.set_title("Total Weekly Sales")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales Quantity")
    ax.grid(True, alpha=0.3)

    return fig


def main():
    st.title("Historical Sales Data Generator")

    st.markdown(
        """
    Use this tool to generate synthetic historical sales data for your retail simulation.
    Customize the parameters below to create realistic sales patterns with seasonality, trends, and promotions.
    
    The generated data will include:
    - **Weekly time periods**: Data is aggregated by week for demand forecasting
    - **SKUs**: Individual products identified by SKU ID
    - **Stock levels**: Random stock-on-hand values between 0.5 and 1.0
    """
    )

    with st.sidebar:
        st.header("Configuration Parameters")

        st.subheader("Simulation Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input("End Date", datetime.now())

        st.subheader("Network Size")
        num_skus = st.slider("Number of SKUs", 1, 50, 10)
        num_stores = st.slider("Number of Stores", 1, 30, 5)

        st.subheader("Sales Patterns")
        base_demand = st.slider("Base Daily Demand", 10, 500, 100)

        with st.expander("Advanced Parameters"):
            seasonality = st.checkbox("Include Seasonality", True)
            trend = st.slider(
                "Sales Trend",
                -0.5,
                0.5,
                0.1,
                0.05,
                help="Negative values for decreasing trend, positive for increasing",
            )
            noise_level = st.slider(
                "Noise Level",
                0.1,
                1.0,
                0.3,
                0.05,
                help="Higher values create more random variations",
            )
            weekend_effect = st.slider(
                "Weekend Effect", 0.0, 1.0, 0.2, 0.05, help="Sales increase on weekends"
            )
            promo_frequency = st.slider(
                "Promotion Frequency",
                0.0,
                0.3,
                0.1,
                0.01,
                help="How often promotions occur",
            )
            promo_effect = st.slider(
                "Promotion Effect",
                0.1,
                2.0,
                0.5,
                0.1,
                help="How much promotions boost sales",
            )

        st.subheader("Output Format")
        filename = st.text_input("Filename (without extension)", "sales_data")
        file_format = st.selectbox("File Format", ["csv", "parquet", "json"])

    # Calculate number of days
    days_diff = (end_date - start_date).days + 1
    num_records = days_diff * num_skus * num_stores

    st.write(
        f"This will generate approximately **{num_records:,}** records of sales data over **{days_diff}** days."
    )
    st.write(
        f"Time period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    if st.button("Generate Sales Data"):
        with st.spinner("Generating sales data..."):
            # Convert dates to datetime objects
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())

            # Generate data
            df = generate_sales_data(
                start_dt,
                end_dt,
                num_skus,
                num_stores,
                base_demand,
                seasonality,
                trend,
                noise_level,
                weekend_effect,
                promo_frequency,
                promo_effect,
            )

            # Save to file
            filepath = save_data(df, filename, file_format)
            st.success(f"Data generated and saved to {filepath}")

            # Display info
            st.session_state.generated_data = df
            st.session_state.data_path = filepath

            # Rerun to show visualization options
            st.rerun()

    # Visualization section (only if data exists)
    if "generated_data" in st.session_state:
        df = st.session_state.generated_data
        filepath = st.session_state.data_path

        st.header("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Data summary
        st.header("Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Sales", f"{df['sales_quantity'].sum():,}")
        with col3:
            st.metric(
                "Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}"
            )

        # Visualizations
        st.header("Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Daily Sales", "Weekly Sales", "By SKU", "By Store"]
        )

        with tab1:
            st.pyplot(plot_total_sales(df))

        with tab2:
            st.pyplot(plot_weekly_sales(df))

        with tab3:
            st.pyplot(plot_sku_sales(df))

        with tab4:
            st.pyplot(plot_store_sales(df))

        # Download options
        st.header("Download Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Generated Data",
                df.to_csv(index=False).encode("utf-8"),
                f"generated_sales_data.csv",
                "text/csv",
            )

        with col2:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    if file_format == "csv":
                        mime = "text/csv"
                    elif file_format == "parquet":
                        mime = "application/octet-stream"
                    else:  # json
                        mime = "application/json"

                    st.download_button(
                        f"Download {file_format.upper()} File",
                        f,
                        os.path.basename(filepath),
                        mime,
                    )


if __name__ == "__main__":
    main()
