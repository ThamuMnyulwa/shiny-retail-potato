import streamlit as st
from app.components.auth import auth_component

st.set_page_config(
    page_title="Retail Demand Forecast App",
    page_icon="📊",
)

# Add authentication
auth_component()

# Only show the main content if authenticated
if st.session_state.get("authenticated", False):
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
