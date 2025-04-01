import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@st.cache_data  # Cache the data loading
def load_data():
    """Loads sample demand and sales data, simulating lost sales."""
    # Create a sample DataFrame
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    branches = ["Branch A", "Branch B", "Branch C"]
    skus = ["SKU001", "SKU002", "SKU003", "SKU004"]

    data = []
    for date in dates:
        for branch in branches:
            for sku in skus:
                # Simulate POTENTIAL DEMAND (more positive)
                base_demand = (
                    70  # Increased base demand
                    + 20
                    * np.sin(date.dayofyear / 365 * 2 * np.pi)  # Increased seasonality
                    + date.dayofyear * 0.1  # Increased trend
                )
                # Add branch/sku specific factors and noise
                branch_factor = (
                    1.0
                    if branch == "Branch A"
                    else (0.8 if branch == "Branch B" else 1.2)
                )
                sku_factor = 1.0 if sku in ["SKU001", "SKU003"] else 1.5
                noise = np.random.normal(0, 20)  # Slightly increased noise
                potential_demand = max(
                    0, base_demand * branch_factor * sku_factor + noise
                )
                potential_demand = int(potential_demand)

                # Simulate ACTUAL SALES based on potential demand and simulated constraints
                # Introduce a probability of stockout or constraint
                stockout_probability = 0.15  # 15% chance of some constraint
                if np.random.rand() < stockout_probability:
                    # If constraint occurs, sales are a fraction of demand
                    constraint_factor = np.random.uniform(
                        0.5, 0.9
                    )  # Sell 50-90% of demand
                    actual_sales = int(potential_demand * constraint_factor)
                else:
                    # No constraint, sales meet demand
                    actual_sales = potential_demand

                # Ensure sales are not negative (though demand is already capped at 0)
                actual_sales = max(0, actual_sales)

                data.append(
                    {
                        "date": date,
                        "branch": branch,
                        "super_sku": sku,
                        "demand": potential_demand,  # Potential Demand
                        "sales": actual_sales,  # Actual Sales
                    }
                )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df
