import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecast App",
    page_icon="📊",
)

st.title("Retail Demand Forecast App")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Welcome to the Retail Demand Forecast App!
    
    This application demonstrates demand forecasting techniques. 
    
    Use the sidebar to navigate to different sections, such as the 
    **Demand Forecast Dashboard**.
    
    **👈 Select a page from the sidebar** to get started.
    """
)

# Note: Streamlit automatically finds files in the 'pages/' directory
# and adds them to the navigation.
