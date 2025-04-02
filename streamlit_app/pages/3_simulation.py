import streamlit as st
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Supply Chain Simulation", page_icon="⏱️", layout="wide")


class SupplyChainNode:
    """Base class for all supply chain nodes"""

    def __init__(
        self, env, name, initial_inventory=100, reorder_point=20, order_quantity=50
    ):
        self.env = env
        self.name = name
        self.inventory = initial_inventory
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
        self.orders_placed = 0
        self.orders_received = 0
        self.stockouts = 0
        self.inventory_history = []

        # Start processes
        self.process = env.process(self.run())

    def run(self):
        """Override in subclasses"""
        pass

    def place_order(self, supplier):
        """Place an order to upstream supplier"""
        self.orders_placed += 1
        return supplier.process_order(self, self.order_quantity)

    def receive_inventory(self, quantity):
        """Receive inventory from an order"""
        self.inventory += quantity
        self.orders_received += 1
        self.inventory_history.append((self.env.now, self.inventory))

    def fulfill_order(self, quantity):
        """Fulfill an order from downstream node"""
        # Check if we have enough inventory
        if self.inventory >= quantity:
            self.inventory -= quantity
            self.inventory_history.append((self.env.now, self.inventory))
            return quantity
        else:
            # Record stockout
            self.stockouts += 1
            available = self.inventory
            self.inventory = 0
            self.inventory_history.append((self.env.now, self.inventory))
            return available


class Source(SupplyChainNode):
    """Source node (manufacturer/supplier)"""

    def __init__(self, env, name, production_rate=100, initial_inventory=1000):
        super().__init__(env, name, initial_inventory)
        self.production_rate = production_rate

    def run(self):
        """Source produces goods at a fixed rate"""
        while True:
            yield self.env.timeout(1)  # Production cycle
            self.inventory += self.production_rate
            self.inventory_history.append((self.env.now, self.inventory))

    def process_order(self, requester, quantity):
        """Process an order from a downstream node"""
        # No delay for the source, assume infinite capacity
        fulfilled = self.fulfill_order(quantity)
        yield self.env.timeout(0)  # immediate fulfillment
        requester.receive_inventory(fulfilled)
        # We must yield at least one event in SimPy process functions
        yield self.env.timeout(0)


class DistributionCenter(SupplyChainNode):
    """Distribution center node"""

    def __init__(
        self,
        env,
        name,
        supplier,
        initial_inventory=500,
        reorder_point=100,
        order_quantity=200,
        processing_time=1,
        transit_time=2,
    ):
        super().__init__(env, name, initial_inventory, reorder_point, order_quantity)
        self.supplier = supplier
        self.processing_time = processing_time
        self.transit_time = transit_time

    def run(self):
        """Check inventory levels and place orders as needed"""
        while True:
            if self.inventory <= self.reorder_point:
                order_process = self.place_order(self.supplier)
                yield self.env.process(order_process)

            yield self.env.timeout(1)  # Check inventory daily
            self.inventory_history.append((self.env.now, self.inventory))

    def process_order(self, requester, quantity):
        """Process an order from a downstream node"""
        # Processing delay
        yield self.env.timeout(self.processing_time)

        # Fulfill order
        fulfilled = self.fulfill_order(quantity)

        # Transit delay
        yield self.env.timeout(self.transit_time)

        # Deliver to requester
        requester.receive_inventory(fulfilled)


class Store(SupplyChainNode):
    """Retail store node"""

    def __init__(
        self,
        env,
        name,
        supplier,
        demand_mean=10,
        demand_sd=3,
        initial_inventory=100,
        reorder_point=30,
        order_quantity=50,
    ):
        super().__init__(env, name, initial_inventory, reorder_point, order_quantity)
        self.supplier = supplier
        self.demand_mean = demand_mean
        self.demand_sd = demand_sd
        self.sales = 0
        self.lost_sales = 0
        self.sales_history = []

    def run(self):
        """Handle daily customer demand and reordering"""
        while True:
            # Generate daily customer demand
            daily_demand = max(
                0, int(np.random.normal(self.demand_mean, self.demand_sd))
            )

            # Fulfill customer demand
            if daily_demand <= self.inventory:
                self.inventory -= daily_demand
                self.sales += daily_demand
                self.sales_history.append((self.env.now, daily_demand))
            else:
                self.sales += self.inventory
                self.lost_sales += daily_demand - self.inventory
                self.sales_history.append((self.env.now, self.inventory))
                self.inventory = 0
                self.stockouts += 1

            # Record inventory level
            self.inventory_history.append((self.env.now, self.inventory))

            # Check if reorder is needed
            if self.inventory <= self.reorder_point:
                order_process = self.place_order(self.supplier)
                yield self.env.process(order_process)

            # Wait for next day
            yield self.env.timeout(1)


def run_simulation(config, progress_bar=None):
    """Run a complete supply chain simulation with the given configuration"""
    # Create SimPy environment
    env = simpy.Environment()

    # Create supply chain nodes
    source = Source(
        env,
        "Source",
        production_rate=config["source_production_rate"],
        initial_inventory=config["source_initial_inventory"],
    )

    ddc = DistributionCenter(
        env,
        "DDC",
        source,
        initial_inventory=config["ddc_initial_inventory"],
        reorder_point=config["ddc_reorder_point"],
        order_quantity=config["ddc_order_quantity"],
        processing_time=config["ddc_processing_time"],
        transit_time=config["source_to_ddc_transit"],
    )

    cdc = DistributionCenter(
        env,
        "CDC",
        ddc,
        initial_inventory=config["cdc_initial_inventory"],
        reorder_point=config["cdc_reorder_point"],
        order_quantity=config["cdc_order_quantity"],
        processing_time=config["cdc_processing_time"],
        transit_time=config["ddc_to_cdc_transit"],
    )

    # Create stores
    stores = []
    for i in range(1, config["num_stores"] + 1):
        store = Store(
            env,
            f"Store {i}",
            cdc,
            demand_mean=config["store_demand_mean"],
            demand_sd=config["store_demand_sd"],
            initial_inventory=config["store_initial_inventory"],
            reorder_point=config["store_reorder_point"],
            order_quantity=config["store_order_quantity"],
        )
        stores.append(store)

    # Run simulation with progress updates
    total_days = config["simulation_days"]

    if progress_bar:
        for day in range(total_days):
            env.run(until=day + 1)
            progress_bar.progress((day + 1) / total_days)
    else:
        env.run(until=total_days)

    # Collect and return results
    results = {"source": source, "ddc": ddc, "cdc": cdc, "stores": stores}

    return results


def plot_inventory_levels(results, days):
    """Plot inventory levels over time for all nodes"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Source inventory
    df_source = pd.DataFrame(
        results["source"].inventory_history, columns=["day", "inventory"]
    )
    axes[0].plot(df_source["day"], df_source["inventory"])
    axes[0].set_title(f"Source Inventory")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Inventory")
    axes[0].grid(True, alpha=0.3)

    # DDC inventory
    df_ddc = pd.DataFrame(
        results["ddc"].inventory_history, columns=["day", "inventory"]
    )
    axes[1].plot(df_ddc["day"], df_ddc["inventory"])
    axes[1].set_title(f"DDC Inventory")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Inventory")
    axes[1].grid(True, alpha=0.3)

    # CDC inventory
    df_cdc = pd.DataFrame(
        results["cdc"].inventory_history, columns=["day", "inventory"]
    )
    axes[2].plot(df_cdc["day"], df_cdc["inventory"])
    axes[2].set_title(f"CDC Inventory")
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("Inventory")
    axes[2].grid(True, alpha=0.3)

    # Store inventory (average across all stores)
    store_inventories = []
    for store in results["stores"]:
        df_store = pd.DataFrame(store.inventory_history, columns=["day", "inventory"])
        store_inventories.append(df_store)

    # Combine all store data
    if store_inventories:
        df_stores = pd.concat(store_inventories)
        df_stores_avg = df_stores.groupby("day").mean().reset_index()

        axes[3].plot(df_stores_avg["day"], df_stores_avg["inventory"])
        axes[3].set_title(f"Average Store Inventory")
        axes[3].set_xlabel("Day")
        axes[3].set_ylabel("Inventory")
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_store_sales(results):
    """Plot store sales and stockouts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Collect sales data from all stores
    sales_data = []
    total_sales = 0
    total_lost_sales = 0
    total_stockouts = 0

    for i, store in enumerate(results["stores"]):
        sales_data.append(
            {
                "Store": f"Store {i+1}",
                "Sales": store.sales,
                "Lost Sales": store.lost_sales,
                "Stockouts": store.stockouts,
            }
        )
        total_sales += store.sales
        total_lost_sales += store.lost_sales
        total_stockouts += store.stockouts

    # Create DataFrame
    df_sales = pd.DataFrame(sales_data)

    # Plot sales by store
    df_sales.plot(kind="bar", x="Store", y=["Sales", "Lost Sales"], ax=ax1)
    ax1.set_title("Sales Performance by Store")
    ax1.set_ylabel("Units")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot summary pie chart
    ax2.pie(
        [total_sales, total_lost_sales],
        labels=["Fulfilled", "Lost Sales"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        explode=(0, 0.1),
    )
    ax2.set_title(f"Sales Fulfillment (Total: {total_sales + total_lost_sales} units)")

    plt.tight_layout()
    return fig


def create_default_config():
    """Create default simulation configuration"""
    return {
        # Simulation parameters
        "simulation_days": 60,
        "num_stores": 5,
        # Source parameters
        "source_production_rate": 100,
        "source_initial_inventory": 1000,
        # DDC parameters
        "ddc_initial_inventory": 500,
        "ddc_reorder_point": 100,
        "ddc_order_quantity": 300,
        "ddc_processing_time": 1,
        "source_to_ddc_transit": 2,
        # CDC parameters
        "cdc_initial_inventory": 800,
        "cdc_reorder_point": 200,
        "cdc_order_quantity": 400,
        "cdc_processing_time": 1,
        "ddc_to_cdc_transit": 1,
        # Store parameters
        "store_initial_inventory": 100,
        "store_reorder_point": 30,
        "store_order_quantity": 80,
        "store_demand_mean": 15,
        "store_demand_sd": 5,
        "cdc_to_store_transit": 1,
    }


def main():
    st.title("Supply Chain Simulation")

    st.markdown(
        """
    Run a discrete event simulation of the retail supply chain network using SimPy.
    
    This simulation models:
    - Product flow from Source → DDC → CDC → Stores
    - Inventory management with reorder points
    - Customer demand at stores
    - Order processing delays and transit times
    """
    )

    # Create default configuration
    if "simulation_config" not in st.session_state:
        st.session_state.simulation_config = create_default_config()

    # Sidebar configuration
    with st.sidebar:
        st.header("Simulation Parameters")

        # General simulation parameters
        st.subheader("General")
        days = st.slider(
            "Simulation Days",
            10,
            365,
            st.session_state.simulation_config["simulation_days"],
        )
        num_stores = st.slider(
            "Number of Stores", 1, 10, st.session_state.simulation_config["num_stores"]
        )

        # Create expandable sections for detailed parameters
        with st.expander("Source Parameters"):
            source_prod_rate = st.slider(
                "Production Rate (units/day)",
                50,
                500,
                st.session_state.simulation_config["source_production_rate"],
            )
            source_inventory = st.slider(
                "Initial Inventory",
                500,
                5000,
                st.session_state.simulation_config["source_initial_inventory"],
            )

        with st.expander("DDC Parameters"):
            ddc_inventory = st.slider(
                "DDC Initial Inventory",
                100,
                2000,
                st.session_state.simulation_config["ddc_initial_inventory"],
            )
            ddc_reorder = st.slider(
                "DDC Reorder Point",
                50,
                500,
                st.session_state.simulation_config["ddc_reorder_point"],
            )
            ddc_order_qty = st.slider(
                "DDC Order Quantity",
                100,
                1000,
                st.session_state.simulation_config["ddc_order_quantity"],
            )
            source_ddc_transit = st.slider(
                "Source to DDC Transit (days)",
                1,
                10,
                st.session_state.simulation_config["source_to_ddc_transit"],
            )

        with st.expander("CDC Parameters"):
            cdc_inventory = st.slider(
                "CDC Initial Inventory",
                100,
                2000,
                st.session_state.simulation_config["cdc_initial_inventory"],
            )
            cdc_reorder = st.slider(
                "CDC Reorder Point",
                50,
                500,
                st.session_state.simulation_config["cdc_reorder_point"],
            )
            cdc_order_qty = st.slider(
                "CDC Order Quantity",
                100,
                1000,
                st.session_state.simulation_config["cdc_order_quantity"],
            )
            ddc_cdc_transit = st.slider(
                "DDC to CDC Transit (days)",
                1,
                10,
                st.session_state.simulation_config["ddc_to_cdc_transit"],
            )

        with st.expander("Store Parameters"):
            store_inventory = st.slider(
                "Store Initial Inventory",
                50,
                500,
                st.session_state.simulation_config["store_initial_inventory"],
            )
            store_reorder = st.slider(
                "Store Reorder Point",
                10,
                100,
                st.session_state.simulation_config["store_reorder_point"],
            )
            store_order_qty = st.slider(
                "Store Order Quantity",
                20,
                200,
                st.session_state.simulation_config["store_order_quantity"],
            )
            store_demand = st.slider(
                "Store Average Daily Demand",
                5,
                50,
                st.session_state.simulation_config["store_demand_mean"],
            )
            store_demand_sd = st.slider(
                "Store Demand Standard Deviation",
                1,
                20,
                st.session_state.simulation_config["store_demand_sd"],
            )
            cdc_store_transit = st.slider(
                "CDC to Store Transit (days)",
                1,
                10,
                st.session_state.simulation_config["cdc_to_store_transit"],
            )

        # Update configuration
        config = {
            "simulation_days": days,
            "num_stores": num_stores,
            "source_production_rate": source_prod_rate,
            "source_initial_inventory": source_inventory,
            "ddc_initial_inventory": ddc_inventory,
            "ddc_reorder_point": ddc_reorder,
            "ddc_order_quantity": ddc_order_qty,
            "ddc_processing_time": 1,  # Fixed for simplicity
            "source_to_ddc_transit": source_ddc_transit,
            "cdc_initial_inventory": cdc_inventory,
            "cdc_reorder_point": cdc_reorder,
            "cdc_order_quantity": cdc_order_qty,
            "cdc_processing_time": 1,  # Fixed for simplicity
            "ddc_to_cdc_transit": ddc_cdc_transit,
            "store_initial_inventory": store_inventory,
            "store_reorder_point": store_reorder,
            "store_order_quantity": store_order_qty,
            "store_demand_mean": store_demand,
            "store_demand_sd": store_demand_sd,
            "cdc_to_store_transit": cdc_store_transit,
        }

        # Save configuration to session state
        st.session_state.simulation_config = config

        # Run simulation button
        run_button = st.button("Run Simulation", type="primary")

    # Main content area - show tabs for setup and results
    tab1, tab2 = st.tabs(["Setup", "Results"])

    with tab1:
        st.subheader("Simulation Setup Summary")

        # Display a summary of the configuration
        col1, col2 = st.columns(2)

        with col1:
            st.write("**General Settings**")
            st.write(f"• Simulation Period: {days} days")
            st.write(f"• Number of Stores: {num_stores}")

            st.write("**Source Settings**")
            st.write(f"• Production Rate: {source_prod_rate} units/day")
            st.write(f"• Initial Inventory: {source_inventory} units")

            st.write("**DDC Settings**")
            st.write(f"• Initial Inventory: {ddc_inventory} units")
            st.write(f"• Reorder Point: {ddc_reorder} units")
            st.write(f"• Order Quantity: {ddc_order_qty} units")
            st.write(f"• Source → DDC Transit: {source_ddc_transit} days")

        with col2:
            st.write("**CDC Settings**")
            st.write(f"• Initial Inventory: {cdc_inventory} units")
            st.write(f"• Reorder Point: {cdc_reorder} units")
            st.write(f"• Order Quantity: {cdc_order_qty} units")
            st.write(f"• DDC → CDC Transit: {ddc_cdc_transit} days")

            st.write("**Store Settings**")
            st.write(f"• Initial Inventory: {store_inventory} units")
            st.write(f"• Reorder Point: {store_reorder} units")
            st.write(f"• Order Quantity: {store_order_qty} units")
            st.write(f"• Daily Demand: {store_demand} ± {store_demand_sd} units")
            st.write(f"• CDC → Store Transit: {cdc_store_transit} days")

    with tab2:
        if run_button or ("simulation_results" in st.session_state):
            if run_button:
                # Show progress bar
                st.subheader("Running Simulation...")
                progress_bar = st.progress(0)

                # Run simulation
                results = run_simulation(
                    st.session_state.simulation_config, progress_bar
                )

                # Store results in session state
                st.session_state.simulation_results = results
                st.session_state.simulation_timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Simulate a delay for better UX
                time.sleep(0.5)

                # Reset session state
                st.session_state.progress = 0
                st.session_state.status = "Not started"
                st.rerun()

            # Display results
            results = st.session_state.simulation_results
            config = st.session_state.simulation_config

            st.subheader("Simulation Results")
            st.info(
                f"Simulation completed at {st.session_state.simulation_timestamp} for {config['simulation_days']} days"
            )

            # Display inventory charts
            st.pyplot(plot_inventory_levels(results, config["simulation_days"]))

            # Display sales performance
            st.subheader("Sales Performance")
            st.pyplot(plot_store_sales(results))

            # Display summary metrics
            st.subheader("Performance Metrics")

            # Calculate metrics
            total_sales = sum(store.sales for store in results["stores"])
            total_lost_sales = sum(store.lost_sales for store in results["stores"])
            fill_rate = (
                total_sales / (total_sales + total_lost_sales)
                if (total_sales + total_lost_sales) > 0
                else 0
            )
            total_stockouts = sum(store.stockouts for store in results["stores"])

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sales", f"{total_sales} units")
            col2.metric("Fill Rate", f"{fill_rate:.1%}")
            col3.metric("Total Stockouts", total_stockouts)

            # Add option to download results
            if st.button("Export Results"):
                # Create a combined dataframe with inventory histories
                inventory_data = []

                # Source inventory
                for day, inv in results["source"].inventory_history:
                    inventory_data.append(
                        {"day": day, "node": "Source", "inventory": inv}
                    )

                # DDC inventory
                for day, inv in results["ddc"].inventory_history:
                    inventory_data.append({"day": day, "node": "DDC", "inventory": inv})

                # CDC inventory
                for day, inv in results["cdc"].inventory_history:
                    inventory_data.append({"day": day, "node": "CDC", "inventory": inv})

                # Store inventory
                for i, store in enumerate(results["stores"]):
                    for day, inv in store.inventory_history:
                        inventory_data.append(
                            {"day": day, "node": f"Store {i+1}", "inventory": inv}
                        )

                # Convert to DataFrame
                df_inventory = pd.DataFrame(inventory_data)

                # Create a download link
                csv = df_inventory.to_csv(index=False)
                st.download_button(
                    label="Download Inventory Data",
                    data=csv,
                    file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("Run a simulation to see results here")


if __name__ == "__main__":
    main()
