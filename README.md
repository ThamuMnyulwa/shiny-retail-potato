# retail-pulse

Repo on some retail stuff to try to fine tune an LLM to teach someone about retail going.

## Documentation: simulation_design.md, all documents in `documents` folder

```markdown
# Retail Supply Chain Simulation Design

## Overview

This project simulates a retail supply chain network that moves products from a source to various distribution centers and ultimately to retail stores. The simulation is designed to support demand forecasting and inventory management across multiple layers of the supply chain. The simulation components are modular, allowing adjustments to parameters like transit times, reorder points, and demand patterns.

## Key Components

### 1. Supply Chain Network

- **Source:** The production or procurement origin where goods are first acquired.
- **Durban D.C. (DDC):** A regional distribution center that consolidates shipments.
- **CDC (Country Distribution Center):** A national hub for distributing products to various regions.
- **Branches/Stores:** Final points of sale where the customer demand is realized.

### 2. Simulation Logic

The simulation is implemented as a discrete event simulation (DES) using a time-stepped or event-driven approach. Key processes include:

- **Demand Generation:** Simulated customer purchases using either historical data or synthetic patterns (e.g., Poisson or normal distributions).
- **Inventory Management:** Each node maintains inventory levels, and orders are generated when stock falls below a predefined threshold.
- **Order Processing:** Orders are routed upstream through the network with associated transit times and delays.
- **Event Scheduling:** The simulation engine manages events like order placements, shipments, and sales.

### 3. Demand Forecasting

- **Forecast Models:** Time series models (e.g., ARIMA, exponential smoothing) are used to forecast demand based on past sales data.
- **Feedback Loop:** Forecasts inform replenishment policies and inventory management decisions.

### 4. Logistics and Transportation

- **Transit Simulation:** Models the movement of goods between nodes with configurable transit times and possible delays.
- **Order Fulfillment:** Tracks the processing of orders from initiation to delivery at branch stores.

## Technology Stack

- **Language:** Python
- **Libraries:**
  - **SimPy:** For discrete event simulation.
  - **Pandas:** For data manipulation and logging.
  - **Statsmodels/Scikit-learn:** For implementing forecasting models.
  - **Matplotlib:** For visualization of simulation results.

## Configuration

Parameters such as transit times, reorder points, and demand variability are stored in external configuration files (e.g., JSON). This design allows you to experiment with different settings without modifying the core code.

## How to Run

1. **Install Dependencies:**  
   Run `pip install -r requirements.txt` to install necessary packages.

2. **Configure Parameters:**  
   Adjust configuration files in the `data/` folder as needed.

3. **Start Simulation:**  
   Execute the main simulation script found in the `app/simulation` directory.

4. **Analyze Output:**  
   Results are logged and can be analyzed using provided Jupyter notebooks in the `notebooks/` folder or custom scripts in the `app/utils` folder.

## Future Enhancements

- Integration of dynamic pricing strategies.
- More granular event logging and real-time dashboards.
- Expanded forecasting models including machine learning techniques.
```

---

## Streamlit Application

The project includes a Streamlit web application for interactive data generation, visualization, and simulation. The application provides the following features:

- **Historical Sales Generation**: Create synthetic sales data with customizable parameters such as seasonality, trends, and product counts.
- **Supply Chain Network Visualization**: Visualize the supply chain network and simulate product flow between nodes.
- **Discrete Event Simulation**: Run SimPy-based simulations of the retail supply chain with configurable parameters.
- **Demand Forecasting**: Apply ARIMA and Exponential Smoothing models to forecast future demand based on historical data.

### Running the Streamlit App

1. Install the required dependencies using uv:
   ```bash
   uv sync
   ```

2. Run the Streamlit app:
   ```bash
   python run_app.py
   ```
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit-app/main.py
   ```

3. The app will open in your default web browser at `http://localhost:8501`.

### Usage

1. **Generate Historical Sales Data**:
   - Navigate to the "Historical Sales Generator" page
   - Configure your parameters (number of products, stores, date range, etc.)
   - Generate and save the data to the `data/` directory

2. **Visualize Supply Chain Network**:
   - Navigate to the "Supply Chain Network" page
   - Explore the network structure and simulate product flow

3. **Run Supply Chain Simulation**:
   - Navigate to the "Simulation" page
   - Configure simulation parameters
   - Run the simulation and analyze the results

4. **Forecast Demand**:
   - Navigate to the "Demand Forecasting" page
   - Select previously generated sales data
   - Choose forecasting method and parameters
   - Generate and download forecasts

---

## Repository Structure with an `app` Folder

```
Retail_Simulation/
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Python dependencies (e.g., SimPy, pandas, statsmodels)
├── setup.py                    # Optional packaging setup
├── docs/
│   └── simulation_design.md    # Detailed design and component interactions (see above)
├── data/                       # Configuration and historical data files
│   ├── historical_sales.csv
│   └── config.json
├── app/                        # Core application logic
│   ├── __init__.py
│   ├── simulation/             # Simulation engine and supply chain network logic
│   │   ├── __init__.py
│   │   ├── nodes.py          # Classes for supply chain nodes (source, DDC, CDC, stores)
│   │   ├── simulation.py     # Main simulation engine using SimPy or event-driven architecture
│   │   └── events.py         # Event scheduling and handling (orders, shipments, sales)
│   ├── forecasting/            # Demand forecasting components
│   │   ├── __init__.py
│   │   ├── forecast.py       # Interface to integrate forecasting into simulation
│   │   └── models.py         # Forecasting model implementations (ARIMA, etc.)
│   ├── logistics/              # Modules handling transportation and logistics details
│   │   ├── __init__.py
│   │   └── transport.py      # Simulates transit times and logistics delays between nodes
│   └── utils/                  # Utility functions and helper modules
│       ├── __init__.py
│       ├── logger.py         # Logging and error handling utilities
│       └── helper.py         # Miscellaneous functions (e.g., config loader, data parsers)
├── tests/                      # Unit and integration tests for various components
│   ├── __init__.py
│   ├── test_simulation.py      # Tests for simulation logic and event handling
│   └── test_forecasting.py     # Tests for forecasting models and demand predictions
└── notebooks/                  # Jupyter notebooks for analysis and visualization of simulation output
    └── analysis.ipynb          # Data visualization and analysis of simulation results
```

---

### Explanation

- **`docs/simulation_design.md`:** Contains your design document with an overview, key components, technology stack, and usage instructions.
- **`app/`:** Houses the core simulation logic.
  - **`simulation/`:** Contains modules that define the supply chain nodes, main simulation engine, and event management.
  - **`forecasting/`:** Includes code for demand forecasting and various forecasting models.
  - **`logistics/`:** Focuses on transportation and logistics, simulating transit delays and order fulfillment.
  - **`utils/`:** Provides utility functions such as logging and configuration loading.
- **`data/`:** Holds external data and configuration files.
- **`tests/`:** Contains unit and integration tests to ensure each module functions correctly.
- **`notebooks/`:** For exploratory analysis and visualization of simulation results.
- **`streamlit-app/`:** Contains the interactive web application.
  - **`main.py`:** Main entry point for the Streamlit app.
  - **`pages/`:** Individual pages of the multi-page Streamlit application.

This structure keeps your project organized, making it easy to develop, test, and extend the simulation while keeping documentation and configuration separate from the core logic.