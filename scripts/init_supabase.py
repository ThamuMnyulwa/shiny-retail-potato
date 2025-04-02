"""
Supabase database initialization script.

This script creates the necessary tables and initializes sample data in Supabase.
"""

import streamlit as st
import pandas as pd
import sys
import os
import logging

# Add the parent directory to the Python path to import app modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from app.lib.supabase_client import get_supabase_client
from app.services.data_loader import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_retail_data_table(supabase):
    """
    Create the retail_data table in Supabase.

    Args:
        supabase: The Supabase client

    Returns:
        bool: True if the table was created successfully, False otherwise
    """
    try:
        # SQL to create the retail_data table - using rpc method
        supabase.rpc(
            "exec",
            {
                "query": """
                CREATE TABLE IF NOT EXISTS retail_data (
                    id SERIAL PRIMARY KEY,
                    date TEXT,
                    branch TEXT,
                    sku TEXT,
                    potential_demand INTEGER,
                    actual_sales INTEGER,
                    lost_sales INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_retail_data_branch ON retail_data(branch);
                CREATE INDEX IF NOT EXISTS idx_retail_data_sku ON retail_data(sku);
                CREATE INDEX IF NOT EXISTS idx_retail_data_date ON retail_data(date);
                """
            },
        ).execute()

        logger.info("Successfully created retail_data table")
        return True
    except Exception as e:
        logger.error(f"Error creating retail_data table: {str(e)}")
        return False


def initialize_sample_data(supabase):
    """
    Initialize the retail_data table with sample data.

    Args:
        supabase: The Supabase client

    Returns:
        bool: True if data was initialized successfully, False otherwise
    """
    try:
        # Check if the table already has data
        response = supabase.table("retail_data").select("id").limit(1).execute()
        if response.data:
            logger.info("retail_data table already has data, skipping initialization")
            return True

        # Generate sample data
        df = DataLoader.generate_sample_data()

        # Insert data in batches
        batch_size = 1000
        total_records = len(df)

        for i in range(0, total_records, batch_size):
            batch = df.iloc[i : i + batch_size]
            records = batch.to_dict("records")
            supabase.table("retail_data").insert(records).execute()
            logger.info(f"Inserted records {i} to {min(i+batch_size, total_records)}")

        logger.info(
            f"Successfully initialized retail_data table with {total_records} records"
        )
        return True
    except Exception as e:
        logger.error(f"Error initializing sample data: {str(e)}")
        return False


def main():
    """Main function to initialize the Supabase database."""
    try:
        logger.info("Connecting to Supabase...")
        supabase = get_supabase_client()

        if not supabase:
            logger.error("Failed to connect to Supabase")
            return

        logger.info("Creating retail_data table...")
        if create_retail_data_table(supabase):
            logger.info("Initializing sample data...")
            initialize_sample_data(supabase)
    except Exception as e:
        logger.error(f"Error initializing Supabase: {str(e)}")


if __name__ == "__main__":
    main()
