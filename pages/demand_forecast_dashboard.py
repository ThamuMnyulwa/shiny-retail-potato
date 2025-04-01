import streamlit as st
import pandas as pd
import altair as alt
from services.data_loader import load_data
from services.forecasting import fit_simple_forecast

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")

st.title("📊 Demand vs. Sales & Forecast Dashboard")
st.markdown(
    "Visualizing potential demand, actual sales, lost sales, and future demand forecast."
)

# --- Load Data ---
df = load_data()  # Cached data loading (now includes demand and sales)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

all_branches_values = sorted(df["branch"].unique())
branch_options = ["All"] + all_branches_values
selected_branches = st.sidebar.multiselect(
    "Select Branch(es):",
    options=branch_options,
    default=["All"],
)

all_skus_values = sorted(df["super_sku"].unique())
sku_options = ["All"] + all_skus_values
selected_skus = st.sidebar.multiselect(
    "Select Super SKU(s):",
    options=sku_options,
    default=["All"],
)

# --- Determine actual filters based on selection ---
if "All" in selected_branches:
    branches_to_filter = all_branches_values
else:
    branches_to_filter = selected_branches

if "All" in selected_skus:
    skus_to_filter = all_skus_values
else:
    skus_to_filter = selected_skus

# --- Filter Data ---
if not branches_to_filter or not skus_to_filter:
    st.warning("Please select at least one Branch/option and one Super SKU/option.")
    st.stop()

filtered_df = df[
    df["branch"].isin(branches_to_filter) & df["super_sku"].isin(skus_to_filter)
]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# --- Aggregate Data for Visualization/Forecasting ---
# Aggregate BOTH demand and sales across selected filters by WEEK
agg_df = (
    filtered_df.set_index("date")[["demand", "sales"]]
    .resample("W-MON")  # Resample to weekly frequency, starting Mondays
    .sum()
    .reset_index()  # Get 'date' column back (represents week starting date)
)
agg_df = agg_df.sort_values("date")

# Calculate Lost Sales (weekly)
agg_df["lost_sales"] = agg_df["demand"] - agg_df["sales"]
total_lost_sales = agg_df["lost_sales"].sum()

st.subheader("Weekly Historical Demand vs. Actual Sales")
st.metric("Total Lost Sales (Selected Period)", f"{total_lost_sales:,}")

# --- Plot Historical Demand vs Sales using Altair (weekly) ---
# Melt the dataframe for easier Altair plotting
historical_plot_df = pd.melt(
    agg_df,
    id_vars=["date"],
    value_vars=["demand", "sales"],
    var_name="metric_type",
    value_name="value",
)

historical_chart = (
    alt.Chart(historical_plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Week Starting"),  # Update axis title
        y=alt.Y("value:Q", title="Aggregated Weekly Units"),  # Update axis title
        color="metric_type:N",  # Color lines by Demand vs Sales
        tooltip=["date:T", "value:Q", "metric_type:N"],
    )
    .properties(title="Aggregated Weekly Historical Demand vs. Sales")  # Update title
    .interactive()
)

st.altair_chart(historical_chart, use_container_width=True)


# --- Forecasting Section (using weekly data) ---
st.subheader("Future Weekly Demand Forecast")
st.markdown("Forecast is based on *historical weekly potential demand*.")

# NOTE: The simple forecast function might not be ideal for weekly data.
# A real implementation would need a model suited for the chosen frequency.
forecast_horizon_weeks = st.slider(
    "Select Forecast Horizon (weeks):",
    min_value=4,
    max_value=26,
    value=12,  # Adjust slider for weeks
)

# Run the placeholder forecast on the aggregated weekly *demand* data
forecast_input_df = agg_df[["date", "demand"]].copy()
# Pass the horizon in weeks to the forecast function (assuming it can handle it or we adjust it)
# For the simple naive forecast, the horizon unit doesn't strictly matter, just the number of periods
forecast_df = fit_simple_forecast(
    forecast_input_df, forecast_horizon=forecast_horizon_weeks
)

# --- Combine Historical and Forecast Data for Plotting (weekly) ---
if not forecast_df.empty:
    # Adjust date index for forecast to represent weeks if needed by the forecast function output
    # The naive forecast output dates might need shifting depending on how fit_simple_forecast is interpreted.
    # Assuming fit_simple_forecast returns dates correctly spaced weekly:
    pass  # Placeholder for potential date adjustments if forecast function output needs it

    # Prepare historical data for combined plot
    hist_demand_plot = agg_df[["date", "demand"]].copy()
    hist_demand_plot["type"] = "Historical Demand (Weekly)"

    hist_sales_plot = agg_df[["date", "sales"]].copy()
    hist_sales_plot["type"] = "Historical Sales (Weekly)"
    hist_sales_plot = hist_sales_plot.rename(columns={"sales": "value"})
    hist_demand_plot = hist_demand_plot.rename(columns={"demand": "value"})

    # Prepare forecast data for combined plot
    forecast_plot = forecast_df.copy()
    forecast_plot["type"] = "Demand Forecast (Weekly)"
    forecast_plot = forecast_plot.rename(columns={"forecast": "value"})

    # Combine all for plotting
    combined_plot_df = pd.concat(
        [hist_demand_plot, hist_sales_plot, forecast_plot[["date", "value", "type"]]],
        ignore_index=True,
    )

    # --- Plot Historical + Forecast using Altair (weekly) ---
    combined_chart = (
        alt.Chart(combined_plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Week Starting"),  # Update axis title
            y=alt.Y("value:Q", title="Aggregated Weekly Units"),  # Update axis title
            color="type:N",
            tooltip=["date:T", "value:Q", "type:N"],
        )
        .properties(
            title="Weekly Historical Performance and Demand Forecast"
        )  # Update title
        .interactive()
    )

    st.altair_chart(combined_chart, use_container_width=True)

    st.write("Weekly Forecast Data (Demand):")
    st.dataframe(forecast_df)
else:
    st.warning("Could not generate forecast for the selected data.")

# --- Optional: Show Raw Filtered Data ---
with st.expander("Show Filtered Raw Data (Demand & Sales)"):
    st.dataframe(filtered_df)  # Now includes both demand and sales columns
