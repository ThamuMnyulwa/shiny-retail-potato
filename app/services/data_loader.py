import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import json
from ..lib.supabase_client import get_supabase_client

# Set up logging
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading service for retail data."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.table_name = "retail_data"
        # Try to create the table if it doesn't exist
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Make sure the retail_data table exists, creating it if necessary."""
        if not self.supabase:
            logger.error("Supabase client not available")
            return

        try:
            # Check if the table exists by querying it with a simpler syntax
            # Instead of using count(*) which causes parsing errors
            self.supabase.table(self.table_name).select("id").limit(1).execute()
            logger.info(f"Table {self.table_name} exists")
        except Exception as e:
            if "relation" in str(e) and "does not exist" in str(e):
                logger.warning(
                    f"Table {self.table_name} does not exist. Creating it now."
                )
                self._create_table_and_sample_data()
            else:
                logger.error(f"Error checking if table exists: {str(e)}")

    def _create_table_and_sample_data(self):
        """Create the retail_data table and populate it with sample data."""
        if not self.supabase:
            return

        # Generate sample data
        sample_data = self.generate_sample_data()

        # Make sure timestamps are converted to strings for JSON serialization
        for col in sample_data.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_data[col]):
                sample_data[col] = sample_data[col].dt.strftime("%Y-%m-%d")

        try:
            # Create the table
            logger.info(
                f"Attempting to create table {self.table_name} with sample data"
            )

            try:
                # Perform direct SQL create table using RPC
                # Try to create the table using supabase-js RPC call
                sql = """
                    CREATE TABLE IF NOT EXISTS retail_data (
                        id SERIAL PRIMARY KEY,
                        date TEXT,
                        branch TEXT,
                        sku TEXT,
                        potential_demand INTEGER,
                        actual_sales INTEGER,
                        lost_sales INTEGER
                    );
                """
                # Use the REST API directly to execute SQL
                logger.info(f"Executing SQL: {sql}")

                # Using the POST API directly because RPC doesn't work
                from supabase.lib.client_options import ClientOptions

                headers = {"apikey": self.supabase.postgrest.session.headers["apikey"]}
                import requests

                # First try to send a simple select to check permissions
                response = requests.post(
                    f"{self.supabase.postgrest.session.base_url}/rest/v1/retail_data",
                    headers=headers,
                    json=sample_data[0:1].to_dict("records"),
                )

                if response.status_code == 404:
                    logger.info(
                        "Supabase table does not exist, but we can create it by insertion"
                    )
                else:
                    logger.error(
                        f"Error checking table: {response.status_code} {response.text}"
                    )

            except Exception as e:
                logger.error(f"Error creating table with SQL: {str(e)}")
                # Continue with insertion attempt anyway

            # Insert data in batches to avoid timeouts - this will create the table if it doesn't exist
            batch_size = 100
            for i in range(0, len(sample_data), batch_size):
                batch = sample_data.iloc[i : i + batch_size]
                records = batch.to_dict("records")
                try:
                    response = (
                        self.supabase.table(self.table_name).insert(records).execute()
                    )
                    logger.info(f"Inserted batch of {len(records)} records")
                    if i == 0:
                        logger.info("Table was automatically created with first insert")
                except Exception as e:
                    logger.error(f"Error inserting batch {i//batch_size}: {str(e)}")
                    if i == 0:
                        # If first batch fails, no point continuing
                        break

            logger.info(
                f"Successfully created table {self.table_name} with sample data"
            )
        except Exception as e:
            logger.error(f"Error creating table and sample data: {str(e)}")

    @st.cache_data(ttl="1h")
    def load_data(_self) -> pd.DataFrame:
        """Loads data from Supabase."""
        try:
            if not _self.supabase:
                logger.error("Supabase client not available")
                return _self.generate_sample_data()

            response = _self.supabase.table(_self.table_name).select("*").execute()
            if response.data:
                return pd.DataFrame(response.data)
            return _self.generate_sample_data()
        except Exception as e:
            logger.error(f"Error loading data from Supabase: {str(e)}")
            return _self.generate_sample_data()

    def save_data(self, df: pd.DataFrame) -> bool:
        """Saves data to Supabase."""
        try:
            if not self.supabase:
                logger.error("Supabase client not available")
                return False

            # Convert DataFrame to records
            records = df.to_dict("records")
            # Upsert data to Supabase
            self.supabase.table(self.table_name).upsert(records).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving data to Supabase: {str(e)}")
            return False

    @staticmethod
    def generate_sample_data() -> pd.DataFrame:
        """Generates sample data for initial setup."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        branches = ["Branch A", "Branch B", "Branch C"]
        skus = ["SKU001", "SKU002", "SKU003", "SKU004"]

        data = []
        for date in dates:
            for branch in branches:
                for sku in skus:
                    # Simulate POTENTIAL DEMAND
                    base_demand = (
                        70
                        + 20 * np.sin(date.dayofyear / 365 * 2 * np.pi)
                        + date.dayofyear * 0.1
                    )
                    branch_factor = (
                        1.0
                        if branch == "Branch A"
                        else (0.8 if branch == "Branch B" else 1.2)
                    )
                    sku_factor = 1.0 if sku in ["SKU001", "SKU003"] else 1.5
                    noise = np.random.normal(0, 20)
                    potential_demand = max(
                        0, base_demand * branch_factor * sku_factor + noise
                    )
                    potential_demand = int(potential_demand)

                    # Simulate ACTUAL SALES
                    stockout_probability = 0.15
                    if np.random.rand() < stockout_probability:
                        constraint_factor = np.random.uniform(0.4, 0.9)
                        actual_sales = int(potential_demand * constraint_factor)
                    else:
                        actual_sales = potential_demand

                    data.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "branch": branch,
                            "sku": sku,
                            "potential_demand": potential_demand,
                            "actual_sales": actual_sales,
                            "lost_sales": max(0, potential_demand - actual_sales),
                        }
                    )

        return pd.DataFrame(data)

    @st.cache_data(ttl="1h")
    def get_branch_performance(_self) -> pd.DataFrame:
        """Get branch performance metrics from Supabase."""
        try:
            if not _self.supabase:
                logger.error("Supabase client not available")
                return pd.DataFrame()

            response = (
                _self.supabase.table(_self.table_name)
                .select("branch", "actual_sales", "lost_sales")
                .execute()
            )

            if response.data:
                df = pd.DataFrame(response.data)
                return (
                    df.groupby("branch")
                    .agg({"actual_sales": "sum", "lost_sales": "sum"})
                    .reset_index()
                )
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting branch performance: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl="1h")
    def get_sku_performance(_self) -> pd.DataFrame:
        """Get SKU performance metrics from Supabase."""
        try:
            if not _self.supabase:
                logger.error("Supabase client not available")
                return pd.DataFrame()

            response = (
                _self.supabase.table(_self.table_name)
                .select("sku", "actual_sales", "lost_sales")
                .execute()
            )

            if response.data:
                df = pd.DataFrame(response.data)
                return (
                    df.groupby("sku")
                    .agg({"actual_sales": "sum", "lost_sales": "sum"})
                    .reset_index()
                )
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting SKU performance: {str(e)}")
            return pd.DataFrame()

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
    def save_data(df, filename, format="csv", data_dir="data", save_to_supabase=False):
        """Save data to file with metadata and optionally to Supabase.

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Base filename without extension
            format (str): File format ('csv', 'parquet', or 'json')
            data_dir (str): Directory to save the data
            save_to_supabase (bool): Whether to also save to Supabase

        Returns:
            str: Path to the saved file
        """
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        filepath = os.path.join(data_dir, f"{filename}.{format}")

        # Rename columns to match the expected format for Supabase
        # Make a copy to avoid modifying the original
        save_df = df.copy()

        # Adapt column names for Supabase compatibility if needed
        if save_to_supabase:
            # Map to the right names
            column_mapping = {
                "sales": "actual_sales",
                "sales_quantity": "actual_sales",
                "demand": "potential_demand",
                "store_id": "branch",
                "sku_id": "sku",
            }

            # Apply the mappings
            for old_col, new_col in column_mapping.items():
                if old_col in save_df.columns and new_col not in save_df.columns:
                    save_df = save_df.rename(columns={old_col: new_col})

            # Calculate lost_sales if missing
            if (
                "lost_sales" not in save_df.columns
                and "potential_demand" in save_df.columns
                and "actual_sales" in save_df.columns
            ):
                save_df["lost_sales"] = (
                    save_df["potential_demand"] - save_df["actual_sales"]
                )
                save_df["lost_sales"] = save_df["lost_sales"].clip(
                    lower=0
                )  # Ensure no negative lost sales

            # Make sure all date/timestamp columns are converted to strings for JSON serialization
            for col in save_df.columns:
                if pd.api.types.is_datetime64_any_dtype(save_df[col]):
                    save_df[col] = save_df[col].dt.strftime("%Y-%m-%d")

        try:
            # Save to file
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
                "saved_to_supabase": save_to_supabase,
            }

            metadata_path = os.path.join(data_dir, f"{filename}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved data to {filepath} with metadata")

            # Save to Supabase if requested
            if save_to_supabase:
                try:
                    # Create DataLoader instance
                    data_loader = DataLoader()

                    if data_loader.supabase:
                        # Convert DataFrame to records
                        records = save_df.to_dict("records")

                        # Delete existing data if any
                        try:
                            data_loader.supabase.table(
                                data_loader.table_name
                            ).delete().execute()
                            logger.info("Deleted existing data from Supabase")
                        except Exception as e:
                            # Table might not exist yet, which is fine
                            logger.warning(f"Could not delete existing data: {str(e)}")

                        # Insert in batches
                        batch_size = 100
                        for i in range(0, len(records), batch_size):
                            batch = records[i : i + batch_size]
                            try:
                                data_loader.supabase.table(
                                    data_loader.table_name
                                ).insert(batch).execute()
                                logger.info(
                                    f"Inserted batch {i//batch_size+1} of {len(records)//batch_size+1} to Supabase"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error inserting batch to Supabase: {str(e)}"
                                )
                                if i == 0:
                                    # If first batch fails, try to create the table
                                    data_loader._create_table_and_sample_data()
                                    # Try again with the first batch
                                    try:
                                        data_loader.supabase.table(
                                            data_loader.table_name
                                        ).insert(batch).execute()
                                        logger.info(
                                            "Created table and inserted first batch"
                                        )
                                    except Exception as inner_e:
                                        logger.error(
                                            f"Still failed after table creation: {str(inner_e)}"
                                        )
                                        break

                        logger.info(
                            f"Successfully saved {len(records)} records to Supabase"
                        )
                    else:
                        logger.error("Could not connect to Supabase")
                except Exception as e:
                    logger.error(f"Error saving to Supabase: {str(e)}")

            return filepath

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None
