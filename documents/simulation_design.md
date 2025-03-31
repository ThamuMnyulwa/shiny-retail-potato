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
