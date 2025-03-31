import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64
from PIL import Image

st.set_page_config(page_title="Supply Chain Network", page_icon="🔄", layout="wide")


def create_supply_chain_network(num_cdcs=1, num_stores=5):
    """Create a basic supply chain network structure"""
    G = nx.DiGraph()

    # Add base nodes
    G.add_node("Source", pos=(0, 0), type="source", inventory=1000)
    G.add_node("DDC", pos=(1, 0), type="distribution", inventory=500)

    # Add CDC nodes
    cdc_positions = []
    if num_cdcs == 1:
        # Single CDC centered
        G.add_node("CDC", pos=(2, 0), type="distribution", inventory=800)
        cdc_positions.append(("CDC", (2, 0)))
    else:
        # Multiple CDCs arranged vertically
        cdc_spacing = 1 if num_cdcs <= 3 else 0.5
        start_y = -((num_cdcs - 1) * cdc_spacing) / 2

        for i in range(1, num_cdcs + 1):
            cdc_name = f"CDC {i}"
            y_pos = start_y + (i - 1) * cdc_spacing
            G.add_node(cdc_name, pos=(2, y_pos), type="distribution", inventory=800)
            cdc_positions.append((cdc_name, (2, y_pos)))

    # Add stores
    store_positions = []
    stores_per_cdc = num_stores // num_cdcs if num_cdcs > 0 else num_stores
    remaining_stores = num_stores % num_cdcs if num_cdcs > 0 else 0

    store_count = 1
    for cdc_idx, (cdc_name, cdc_pos) in enumerate(cdc_positions):
        # Calculate number of stores for this CDC
        this_cdc_stores = stores_per_cdc + (1 if cdc_idx < remaining_stores else 0)

        # Calculate store positions
        store_spacing = 0.5
        start_y = cdc_pos[1] - ((this_cdc_stores - 1) * store_spacing) / 2

        for i in range(this_cdc_stores):
            store_name = f"Store {store_count}"
            y_pos = start_y + i * store_spacing
            G.add_node(store_name, pos=(3, y_pos), type="store", inventory=100)
            store_positions.append((store_name, (3, y_pos)))
            store_count += 1

    # Add edges
    G.add_edge("Source", "DDC", transit_time=2)

    if num_cdcs == 1:
        G.add_edge("DDC", "CDC", transit_time=1)
        for store_name, _ in store_positions:
            G.add_edge("CDC", store_name, transit_time=1)
    else:
        # Connect DDC to each CDC
        for cdc_name, _ in cdc_positions:
            G.add_edge("DDC", cdc_name, transit_time=1)

        # Connect each CDC to its stores
        store_idx = 0
        for cdc_idx, (cdc_name, _) in enumerate(cdc_positions):
            this_cdc_stores = stores_per_cdc + (1 if cdc_idx < remaining_stores else 0)

            for i in range(this_cdc_stores):
                if store_idx < len(store_positions):
                    store_name = store_positions[store_idx][0]
                    G.add_edge(cdc_name, store_name, transit_time=1)
                    store_idx += 1

    return G


def visualize_network(G, highlight_path=None, show_inventory=False):
    """Generate a network visualization of the supply chain"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # Define node colors based on type
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]["type"] == "source":
            node_colors.append("green")
        elif G.nodes[node]["type"] == "distribution":
            node_colors.append("orange")
        else:  # store
            node_colors.append("blue")

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=700, alpha=0.8, ax=ax
    )

    # Draw all edges
    nx.draw_networkx_edges(
        G, pos, width=1.5, alpha=0.5, arrows=True, arrowstyle="->", arrowsize=15, ax=ax
    )

    # Highlight specific path if provided
    if highlight_path:
        edge_list = [
            (highlight_path[i], highlight_path[i + 1])
            for i in range(len(highlight_path) - 1)
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_list,
            width=3,
            alpha=1,
            edge_color="red",
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            ax=ax,
        )

    # Add labels
    labels = {}
    for node in G.nodes():
        if show_inventory:
            labels[node] = f"{node}\nInv: {G.nodes[node]['inventory']}"
        else:
            labels[node] = node

    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold", ax=ax)

    # Add edge labels (transit times)
    edge_labels = {
        (u, v): f"{d['transit_time']} days" for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # Add a legend
    node_types = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10)
        for c in ["green", "orange", "blue"]
    ]
    ax.legend(node_types, ["Source", "Distribution Center", "Store"], loc="upper left")

    plt.title("Retail Supply Chain Network")
    plt.axis("off")
    plt.tight_layout()

    return fig


def animate_product_flow(G, path):
    """Create an animation of product flow along a specified path"""
    pos = nx.get_node_attributes(G, "pos")

    # Create a path of positions for animation
    edge_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    path_pos = []

    for u, v in edge_list:
        start_pos = pos[u]
        end_pos = pos[v]

        # Create intermediate positions (10 frames per edge)
        for i in range(11):
            t = i / 10
            x = start_pos[0] * (1 - t) + end_pos[0] * t
            y = start_pos[1] * (1 - t) + end_pos[1] * t
            path_pos.append((x, y))

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw network
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]["type"] == "source":
            node_colors.append("green")
        elif G.nodes[node]["type"] == "distribution":
            node_colors.append("orange")
        else:  # store
            node_colors.append("blue")

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=700, alpha=0.8, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, width=1.5, alpha=0.5, arrows=True, arrowstyle="->", arrowsize=15, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # Product marker
    (product,) = ax.plot([], [], "ro", markersize=15)

    # Set axis properties
    plt.title("Product Flow Animation")
    plt.axis("off")
    plt.tight_layout()

    # Save animation to a temporary buffer
    buffer = io.BytesIO()

    frames = []
    for i in range(len(path_pos)):
        # Update the plot
        if i < len(path_pos):
            x, y = path_pos[i]
            product.set_data([x], [y])

        # Convert to image - using buffer_rgba instead of tostring_rgb
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        # Convert to a NumPy array
        image = np.asarray(buf)
        frames.append(image)

    # Convert RGBA to RGB for GIF compatibility
    rgb_frames = [Image.fromarray(frame[:, :, :3]) for frame in frames]

    # Save as GIF
    rgb_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=rgb_frames[1:],
        optimize=False,
        duration=100,
        loop=0,
    )
    buffer.seek(0)

    plt.close(fig)  # Close the figure to free memory
    return buffer


def main():
    st.title("Supply Chain Network Visualization")

    st.markdown(
        """
    This page visualizes the retail supply chain network with its key components:
    - **Source**: The production or procurement origin where goods are first acquired
    - **DDC (Durban Distribution Center)**: A regional distribution center
    - **CDC (Country Distribution Center)**: A national hub for distribution
    - **Stores**: Final points of sale where customer demand is realized
    """
    )

    # Sidebar options for network structure
    with st.sidebar:
        st.header("Network Structure")
        num_cdcs = st.slider("Number of CDCs", 1, 5, 1)
        num_stores = st.slider("Number of Stores", 1, 10, 5)

        # Create network based on selected parameters
        network = create_supply_chain_network(num_cdcs=num_cdcs, num_stores=num_stores)

        st.header("Visualization Options")
        show_inventory = st.checkbox("Show Inventory Levels", True)

        # Generate path options based on network structure
        st.subheader("Simulate Product Flow")

        path_options = []
        # Build path options based on actual network structure
        if num_cdcs == 1:
            for i in range(1, num_stores + 1):
                path_options.append(f"Source → DDC → CDC → Store {i}")
        else:
            for cdc_i in range(1, num_cdcs + 1):
                # Find stores connected to this CDC
                connected_stores = []
                for u, v in network.edges():
                    if u == f"CDC {cdc_i}" and v.startswith("Store"):
                        connected_stores.append(v)

                for store in connected_stores:
                    path_options.append(f"Source → DDC → CDC {cdc_i} → {store}")

        # Default to first path if available
        default_index = 0 if path_options else None
        selected_path = (
            st.selectbox("Select Path", path_options, index=default_index)
            if path_options
            else None
        )

        animate_button = st.button("Animate Product Flow", type="primary")

    # Main content area
    tab1, tab2 = st.tabs(["Network Visualization", "Simulation Parameters"])

    with tab1:
        # Convert selected path to list of nodes
        path = None
        if selected_path:
            path = selected_path.split(" → ")

        # Display static network visualization
        st.pyplot(
            visualize_network(
                network, highlight_path=path, show_inventory=show_inventory
            )
        )

        # Display animation if requested
        if animate_button and path:
            with st.spinner("Creating animation..."):
                buffer = animate_product_flow(network, path)

                # Display animation as GIF
                st.subheader("Product Flow Animation")
                st.image(buffer, caption="Product Flow Animation")

    with tab2:
        st.subheader("Simulation Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Transit Times (Days)**")
            st.slider("Source to DDC", 1, 10, 2, key="transit_source_ddc")
            st.slider("DDC to CDC", 1, 10, 1, key="transit_ddc_cdc")
            st.slider("CDC to Stores", 1, 10, 1, key="transit_cdc_stores")

        with col2:
            st.write("**Initial Inventory Levels**")
            st.slider("Source", 0, 2000, 1000, 100, key="inventory_source")
            st.slider("DDC", 0, 1000, 500, 50, key="inventory_ddc")
            st.slider("CDC", 0, 1000, 800, 50, key="inventory_cdc")
            st.slider("Stores", 0, 500, 100, 10, key="inventory_stores")

        # Update network with new parameters
        if st.button("Update Network Parameters"):
            # Update transit times
            if "Source" in network.nodes and "DDC" in network.nodes:
                network.edges["Source", "DDC"][
                    "transit_time"
                ] = st.session_state.transit_source_ddc

            # Update CDC transit times
            if num_cdcs == 1 and "DDC" in network.nodes and "CDC" in network.nodes:
                network.edges["DDC", "CDC"][
                    "transit_time"
                ] = st.session_state.transit_ddc_cdc
            else:
                for i in range(1, num_cdcs + 1):
                    cdc_name = f"CDC {i}"
                    if "DDC" in network.nodes and cdc_name in network.nodes:
                        network.edges["DDC", cdc_name][
                            "transit_time"
                        ] = st.session_state.transit_ddc_cdc

            # Update stores transit times
            for edge in list(network.edges()):
                # For any edge connecting a CDC to a Store
                if (edge[0] == "CDC" or edge[0].startswith("CDC ")) and edge[
                    1
                ].startswith("Store"):
                    network.edges[edge][
                        "transit_time"
                    ] = st.session_state.transit_cdc_stores

            # Update inventory levels
            if "Source" in network.nodes:
                network.nodes["Source"]["inventory"] = st.session_state.inventory_source

            if "DDC" in network.nodes:
                network.nodes["DDC"]["inventory"] = st.session_state.inventory_ddc

            # Update CDC inventory
            if num_cdcs == 1 and "CDC" in network.nodes:
                network.nodes["CDC"]["inventory"] = st.session_state.inventory_cdc
            else:
                for i in range(1, num_cdcs + 1):
                    cdc_name = f"CDC {i}"
                    if cdc_name in network.nodes:
                        network.nodes[cdc_name][
                            "inventory"
                        ] = st.session_state.inventory_cdc

            # Update stores inventory
            for node in network.nodes():
                if node.startswith("Store"):
                    network.nodes[node]["inventory"] = st.session_state.inventory_stores

            st.success("Network parameters updated successfully!")

            # Re-render the visualization tab
            st.rerun()


if __name__ == "__main__":
    main()
