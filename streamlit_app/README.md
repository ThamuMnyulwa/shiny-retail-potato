# Retail Pulse Streamlit App

This directory contains the Streamlit application for visualizing and interacting with the retail supply chain simulation.

## Features

- **Historical Sales Generator:** Create synthetic historical sales data with customizable parameters
- **Supply Chain Network:** Visualize the retail supply chain network and simulate product flow
- **Simulation:** Run discrete event simulations of the retail supply chain
- **Demand Forecasting:** Generate forecasts based on historical sales data

## How to Run

From the project root directory:

```bash
# Using the setup script
./setup.sh

# OR manually
streamlit run streamlit-app/main.py
```

## Structure

- `main.py` - The main entry point for the Streamlit app
- `pages/` - Individual pages of the multi-page application:
  - `1_historical_sales_generator.py` - Generate synthetic sales data
  - `2_supply_chain_network.py` - Visualize supply chain network
  - `3_simulation.py` - Run SimPy-based discrete event simulations
  - `4_demand_forecasting.py` - Apply forecasting models to sales data

## Data Flow

1. Generate historical sales data and save to the `data/` directory
2. Use the generated data for forecasting and simulation
3. Export results for further analysis 