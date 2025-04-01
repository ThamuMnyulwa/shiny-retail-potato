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
                        0.4, 0.9
                    )  # Sell 40-90% of demand
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


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import json

# Set up logging
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading service for retail data."""

    @staticmethod
    @st.cache_data  # Cache the data loading
    def load_historical_data(file_path=None):
        """
        Load historical sales data from file.

        If no file is provided or file doesn't exist, generates sample data.

        Args:
            file_path (str, optional): Path to the data file. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with sales data
        """
        if file_path and os.path.exists(file_path):
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded CSV file: {file_path}")
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded Parquet file: {file_path}")
                elif file_path.endswith(".json"):
                    df = pd.read_json(file_path)
                    logger.info(f"Loaded JSON file: {file_path}")
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    return DataLoader.generate_sample_data()

                # Ensure necessary columns exist
                required_columns = ["date", "branch", "super_sku", "sales"]
                if not all(col in df.columns for col in required_columns):
                    logger.warning(
                        f"Missing required columns in {file_path}. Using sample data."
                    )
                    return DataLoader.generate_sample_data()

                # Convert date column to datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])

                # Rename columns if needed for compatibility
                column_mapping = {
                    "branch": "store_id",
                    "super_sku": "sku_id",
                    "product_id": "sku_id",
                }
                df = df.rename(
                    columns={k: v for k, v in column_mapping.items() if k in df.columns}
                )

                # Add stock_on_hand if missing
                if "stock_on_hand" not in df.columns:
                    logger.info("Adding simulated stock_on_hand column")
                    df["stock_on_hand"] = np.random.uniform(0.2, 0.8, size=len(df))

                # Add demand column if missing (estimated from sales and stock)
                if "demand" not in df.columns:
                    logger.info("Adding estimated demand column")
                    # Simple model: lower stock levels likely indicate constrained sales
                    # Higher stock = sales closer to true demand
                    stock_factor = (
                        1 + (1 - df["stock_on_hand"]) * 0.5
                    )  # Scale factor between 1-1.5
                    df["demand"] = df["sales"] * stock_factor

                return df

            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {str(e)}")
                return DataLoader.generate_sample_data()
        else:
            logger.info(f"No valid file path provided. Using sample data.")
            return DataLoader.generate_sample_data()

    @staticmethod
    def generate_sample_data():
        """Generate sample demand and sales data if no file is available."""
        logger.info("Generating sample data")

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
                        * np.sin(
                            date.dayofyear / 365 * 2 * np.pi
                        )  # Increased seasonality
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
                            0.4, 0.9
                        )  # Sell 40-90% of demand
                        actual_sales = int(potential_demand * constraint_factor)
                    else:
                        # No constraint, sales meet demand
                        actual_sales = potential_demand

                    # Ensure sales are not negative (though demand is already capped at 0)
                    actual_sales = max(0, actual_sales)

                    # Calculate stock level based on sales vs. demand
                    stock_on_hand = 1.0
                    if potential_demand > 0:
                        stock_on_hand = actual_sales / potential_demand

                        # Add some noise to stock level
                        stock_on_hand = min(
                            1.0, max(0.1, stock_on_hand + np.random.normal(0, 0.05))
                        )

                    data.append(
                        {
                            "date": date,
                            "store_id": branch,
                            "sku_id": sku,
                            "demand": potential_demand,  # Potential Demand
                            "sales": actual_sales,  # Actual Sales
                            "stock_on_hand": stock_on_hand,  # Stock level
                        }
                    )

        df = pd.DataFrame(data)

        # Add week for easier aggregation
        df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time

        return df

    @staticmethod
    def find_latest_data_file(data_dir="data"):
        """Find the most recent data file in the data directory.

        Args:
            data_dir (str): Directory to search for data files

        Returns:
            str or None: Path to the latest data file, or None if no files found
        """
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist")
            return None

        # Get all data files with supported extensions
        data_files = []
        for file in os.listdir(data_dir):
            if file.endswith((".csv", ".parquet", ".json")) and not file.endswith(
                "_metadata.json"
            ):
                file_path = os.path.join(data_dir, file)
                data_files.append((file_path, os.path.getmtime(file_path)))

        if not data_files:
            logger.info(f"No data files found in {data_dir}")
            return None

        # Sort by modification time (newest first)
        data_files.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Latest data file: {data_files[0][0]}")
        return data_files[0][0]

    @staticmethod
    def save_data(df, filename, format="csv", data_dir="data"):
        """Save data to file with metadata.

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Base filename without extension
            format (str): File format ('csv', 'parquet', or 'json')
            data_dir (str): Directory to save the data

        Returns:
            str: Path to the saved file
        """
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        filepath = os.path.join(data_dir, f"{filename}.{format}")

        try:
            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "parquet":
                df.to_parquet(filepath, index=False)
            elif format == "json":
                df.to_json(filepath, orient="records")
            else:
                logger.error(f"Unsupported format: {format}")
                return None

            # Save metadata
            metadata = {
                "filename": f"{filename}.{format}",
                "format": format,
                "rows": len(df),
                "columns": df.columns.tolist(),
                "date_range": [
                    (
                        df["date"].min().strftime("%Y-%m-%d")
                        if not pd.isna(df["date"].min())
                        else None
                    ),
                    (
                        df["date"].max().strftime("%Y-%m-%d")
                        if not pd.isna(df["date"].max())
                        else None
                    ),
                ],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            metadata_path = os.path.join(data_dir, f"{filename}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved data to {filepath} with metadata")
            return filepath

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None
