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
    num_products,
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
    num_products : int
        Number of different products to simulate
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

    # Create product and store IDs
    product_ids = [f"P{i:03d}" for i in range(1, num_products + 1)]
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

    # Generate sales for each product, store, and day
    for product_id in product_ids:
        # Product-specific base demand (some products sell better than others)
        product_popularity = np.random.uniform(0.7, 1.3)

        for store_id in store_ids:
            # Store-specific base demand (some stores sell more than others)
            store_size = np.random.uniform(0.8, 1.2)

            # Generate promotional events
            promo_days = np.random.random(num_days) < promo_frequency

            # Calculate daily sales
            for i, date in enumerate(date_range):
                base_sales = (
                    base_demand * product_popularity * store_size * time_effects[i]
                )

                # Add promotion effect
                if promo_days[i]:
                    base_sales *= 1 + promo_effect

                # Add random noise
                noise = np.random.normal(1, noise_level)
                sales = max(0, round(base_sales * noise))

                records.append(
                    {
                        "date": date,
                        "product_id": product_id,
                        "store_id": store_id,
                        "sales_quantity": sales,
                        "promotion": 1 if promo_days[i] else 0,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(records)
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


def plot_product_sales(df):
    """Plot sales by product"""
    product_sales = (
        df.groupby("product_id")["sales_quantity"].sum().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    product_sales.plot(kind="bar", ax=ax)
    ax.set_title("Total Sales by Product")
    ax.set_xlabel("Product ID")
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


def main():
    st.title("Historical Sales Data Generator")

    st.markdown(
        """
    Use this tool to generate synthetic historical sales data for your retail simulation.
    Customize the parameters below to create realistic sales patterns with seasonality, trends, and promotions.
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
        num_products = st.slider("Number of Products", 1, 50, 10)
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
                help="Percentage of days with promotions",
            )
            promo_effect = st.slider(
                "Promotion Effect",
                0.1,
                2.0,
                0.5,
                0.1,
                help="Sales increase during promotions",
            )

    # Main content area
    tab1, tab2 = st.tabs(["Generate Data", "Visualize Sample"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            output_format = st.selectbox(
                "Output Format", ["csv", "parquet", "json"], index=0
            )
            filename = st.text_input("Filename (without extension)", "historical_sales")

        with col2:
            st.write("Summary:")
            st.write(f"• Time Period: {(end_date - start_date).days + 1} days")
            st.write(
                f"• Total Records: {num_products * num_stores * ((end_date - start_date).days + 1):,}"
            )
            st.write(f"• Output: data/{filename}.{output_format}")

        if st.button("Generate Sales Data", type="primary"):
            with st.spinner("Generating sales data..."):
                # Convert datetime.date to datetime.datetime for pandas date_range
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.min.time())

                # Generate data
                df = generate_sales_data(
                    start_datetime,
                    end_datetime,
                    num_products,
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
                filepath = save_data(df, filename, output_format)

                # Display success message and sample
                st.success(
                    f"Successfully generated {len(df):,} records and saved to {filepath}"
                )

                # Save metadata about the generation parameters
                metadata = {
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "parameters": {
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "num_products": num_products,
                        "num_stores": num_stores,
                        "base_demand": base_demand,
                        "seasonality": seasonality,
                        "trend": trend,
                        "noise_level": noise_level,
                        "weekend_effect": weekend_effect,
                        "promo_frequency": promo_frequency,
                        "promo_effect": promo_effect,
                    },
                }

                with open(f"data/{filename}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Store in session state for visualization
                st.session_state.generated_data = df
                st.session_state.has_data = True

                # Show sample of the data
                st.subheader("Sample Data (First 10 rows)")
                st.dataframe(df.head(10))

    with tab2:
        # Check if data exists in session state
        if not st.session_state.get("has_data", False):
            st.info("Generate data first or upload existing data to visualize it.")

            # Option to generate sample data just for visualization
            if st.button("Generate Sample Data for Preview"):
                with st.spinner("Generating sample data..."):
                    # Generate a smaller sample for quick visualization
                    sample_start = datetime.now() - timedelta(days=180)
                    sample_end = datetime.now()
                    sample_df = generate_sales_data(
                        sample_start,
                        sample_end,
                        5,
                        3,
                        base_demand,
                        seasonality,
                        trend,
                        noise_level,
                        weekend_effect,
                        promo_frequency,
                        promo_effect,
                    )
                    st.session_state.generated_data = sample_df
                    st.session_state.has_data = True
                    st.rerun()
        else:
            df = st.session_state.generated_data

            # Display visualizations
            st.subheader("Sales Overview")

            # Total sales over time
            st.pyplot(plot_total_sales(df))

            col1, col2 = st.columns(2)

            with col1:
                # Sales by product
                st.pyplot(plot_product_sales(df))

            with col2:
                # Sales by store
                st.pyplot(plot_store_sales(df))

            # Advanced analysis
            with st.expander("Additional Analysis"):
                # Day of week pattern
                df["day_of_week"] = pd.to_datetime(df["date"]).dt.day_name()
                day_order = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                daily_avg = (
                    df.groupby("day_of_week")["sales_quantity"]
                    .mean()
                    .reindex(day_order)
                )

                st.subheader("Average Sales by Day of Week")
                st.bar_chart(daily_avg)

                # Promotion effect
                promo_effect = df.groupby("promotion")["sales_quantity"].mean()

                st.subheader("Average Sales With/Without Promotion")
                st.bar_chart(promo_effect)


if __name__ == "__main__":
    main()
