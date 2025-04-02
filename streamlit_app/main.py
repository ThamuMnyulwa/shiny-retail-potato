import streamlit as st

st.set_page_config(
    page_title="Retail Pulse",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Retail Pulse: Supply Chain Simulation")

    st.markdown(
        """
    ## Welcome to Retail Pulse
    
    This application helps you simulate and analyze retail supply chain dynamics.
    
    ### Features:
    - Generate historical sales data with custom parameters
    - Visualize supply chain network and inventory levels
    - Run simulations with different configurations
    - Analyze results through interactive dashboards
    
    👈 Use the sidebar to navigate through different pages.
    """
    )

    st.image(
        "https://img.freepik.com/free-vector/supply-chain-management-abstract-concept-illustration_335657-3987.jpg",
        caption="Supply Chain Visualization",
        use_column_width=True,
    )


if __name__ == "__main__":
    main()
