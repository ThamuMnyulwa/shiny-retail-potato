Below is an outline to help you design and implement your retail supply chain simulation. This plan covers both the supply chain network (from the source to branch stores), the upstream supply chain and the integration of a demand forecasting model.

---

### 1. Define Your Supply Chain Structure

**Key Nodes:**
- **Source:** The production or procurement facility where goods originate.
- **Durban D.C. / DDC:** Regional distribution center to consolidate and route goods.
- **CDC (Country Distribution Center):** National hub that manages distribution to different areas.
- **Branch/Store:** Final retail outlets where products are sold.

Each node can be modeled as an independent entity with properties such as inventory levels, processing times, and handling capacities.

---

### 2. Outline Simulation Components

**Stores and Sales:**
- **Sales Generation:** Use historical sales data or generate synthetic demand patterns. You might model sales as stochastic processes (e.g., Poisson or Normal distributions) that capture daily or seasonal variability.
- **Inventory Tracking:** Simulate inventory movements at each node, including order placements, stock replenishment, and stockouts.

**Logistics Flow:**
- **Movement Between Nodes:** Define transit times, shipping costs, and delays. For instance, simulate a delay from the source to the Durban D.C., then to the CDC, and finally to the branch.
- **Order Management:** Implement rules for triggering orders (e.g., reorder points, economic order quantities) and simulate how orders propagate through the network.

---

### 3. Develop the Simulation Flow

**Event-Driven Simulation:**
- **Discrete Event Simulation (DES):** Use frameworks like SimPy (in Python) to create events such as order arrivals, shipments, and sales occurrences.
- **Time Steps:** Decide whether your simulation will work in discrete time steps (daily, hourly) or continuously with event triggers.

**Key Processes:**
- **Demand Realization:** Simulate customer purchases at branches.
- **Replenishment Orders:** When inventory drops below a threshold, trigger an order that moves upstream in the network.
- **Transit and Processing Delays:** Add realistic delays (shipping, processing, and handling times) at each stage.

---

### 4. Integrate Demand Forecasting

**Forecasting Models:**
- **Time Series Analysis:** Implement models like ARIMA, Exponential Smoothing, or machine learning models to forecast future demand based on simulated historical data.
- **Feedback Loop:** Use the forecast to adjust reorder points and quantities in the simulation, making the system more adaptive to changes in demand.

**Data Collection & Analysis:**
- **Simulation Output:** Generate detailed logs of sales, inventory levels, and order histories.
- **Forecast Evaluation:** Compare forecasted demand against actual simulated demand to refine your forecasting model.

---

### 5. Technology & Tools

**Programming Environment:**
- **Python:** Leverage libraries such as:
  - **SimPy:** For creating the discrete event simulation.
  - **Pandas:** To handle time series data and simulation logs.
  - **Statsmodels / Scikit-learn:** For implementing forecasting models.

**Visualization & Reporting:**
- Plot inventory levels, sales data, and forecast vs. actual demand over time using libraries like Matplotlib.
- Create dashboards to interactively monitor simulation performance and key metrics.

---

### 6. Customization and Scalability

**Flexibility:**
- **Parameterization:** Allow simulation parameters (e.g., transit times, reorder points, demand variability) to be easily adjusted. This lets you test different scenarios within your retail environment.
- **Modular Design:** Structure the simulation so that nodes (DCs, stores) can be added, removed, or modified without overhauling the entire model.

**Scalability:**
- **Multiple Scenarios:** Run multiple simulation scenarios in parallel to evaluate the impact of different policies.
- **Iterative Improvement:** Use simulation results to iteratively refine both the supply chain model and the demand forecasting model.

---

### 7. Example Workflow

1. **Initialize Nodes:** Set up objects for the source, DDC, CDC, and branches with initial inventories.
2. **Generate Demand:** Simulate sales at each branch.
3. **Process Orders:** Trigger replenishment orders based on inventory levels.
4. **Simulate Logistics:** Move orders through the supply chain with appropriate delays.
5. **Collect Data:** Log events such as sales, shipments, and inventory changes.
6. **Forecast Demand:** Periodically run the forecasting model using accumulated data and adjust ordering policies accordingly.
7. **Analyze and Adjust:** Review simulation outputs and adjust parameters to improve supply chain efficiency and forecast accuracy.

---

This outline provides a framework for building a flexible and comprehensive simulation that captures the complexities of a retail supply chain. As you build your simulation, you can start simple and gradually add more detailed processes (like dynamic pricing, seasonal promotions, or multi-modal logistics) as needed.