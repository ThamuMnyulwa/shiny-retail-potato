import streamlit as st
from app.components.auth import auth_component
from app.services.data_loader import DataLoader

st.set_page_config(
    page_title="Retail Pulse",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Check authentication using existing auth component
    if not auth_component():
        return

    # Initialize data loader
    data_loader = DataLoader()  # TODO: not sure if this is needed or why it's here

    # Main app content
    st.title("Retail Pulse: Supply Chain Simulation")

    # Load data
    df = data_loader.load_data()

    # Display welcome message
    st.markdown(
        """
    ## Welcome to Retail Pulse
    
    This application helps you simulate and analyze retail supply chain dynamics.
    
    ### Features:
    - View real-time sales and inventory data
    - Analyze branch and SKU performance
    - Track lost sales and potential revenue
    - Monitor supply chain metrics
    
    👈 Use the sidebar to navigate through different pages.
    """
    )

    st.image(
        "https://img.freepik.com/free-vector/supply-chain-management-abstract-concept-illustration_335657-3987.jpg",
        caption="Supply Chain Visualization",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
