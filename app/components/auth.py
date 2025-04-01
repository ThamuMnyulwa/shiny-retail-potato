import streamlit as st
import sys
import os
from urllib.parse import parse_qs, urlparse
import logging
from ..lib.supabase_client import get_supabase_client

# Set up logging
logger = logging.getLogger(__name__)


def handle_auth_callback():
    """Handle the authentication callback from Supabase."""
    query_params = st.query_params.to_dict()

    if "access_token" in query_params:
        access_token = query_params["access_token"]
        try:
            supabase = get_supabase_client()
            if supabase:
                user = supabase.auth.get_user(access_token)
                st.session_state["user"] = user
                st.session_state["authenticated"] = True
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Database connection not available. Please try again later.")
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            st.error(f"Error authenticating user. Please try again.")


def show_login_ui():
    """Show the login UI."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            st.error("Database connection not available. Please try again later.")
            return

        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                    return

                try:
                    response = supabase.auth.sign_in_with_password(
                        {"email": email, "password": password}
                    )
                    st.session_state["user"] = response.user
                    st.session_state["authenticated"] = True
                    st.rerun()
                except Exception as e:
                    logger.error(f"Login failed: {str(e)}")
                    st.error(f"Login failed. Please check your credentials.")

        with st.expander("Create Account"):
            with st.form("signup_form"):
                new_email = st.text_input("New Email")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup = st.form_submit_button("Sign Up")

                if signup:
                    if not new_email or not new_password:
                        st.error("Please enter both email and password")
                        return

                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                        return

                    try:
                        response = supabase.auth.sign_up(
                            {
                                "email": new_email,
                                "password": new_password,
                                "options": {
                                    "email_redirect_to": f"{st.get_option('server.baseUrlPath')}/auth/callback"
                                },
                            }
                        )
                        st.success("Please check your email to confirm your account")
                    except Exception as e:
                        logger.error(f"Sign up failed: {str(e)}")
                        st.error(f"Sign up failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error showing login UI: {str(e)}")
        st.error("An error occurred. Please try again later.")


def auth_component():
    """Main authentication component."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if "user" not in st.session_state:
        st.session_state["user"] = None

    # Check for auth callback
    if st.query_params.get("access_token"):
        handle_auth_callback()

    if not st.session_state["authenticated"]:
        st.title("Retail Pulse - Login")
        st.markdown("Please login to access the Retail Pulse application.")
        show_login_ui()
        return False
    else:
        # Show user info in sidebar
        with st.sidebar:
            st.write(f"👤 Logged in as: {st.session_state['user'].email}")
            if st.button("Logout", key="logout_button"):
                try:
                    supabase = get_supabase_client()
                    if supabase:
                        supabase.auth.sign_out()
                    st.session_state["authenticated"] = False
                    st.session_state["user"] = None
                    st.rerun()
                except Exception as e:
                    logger.error(f"Logout failed: {str(e)}")
                    st.error("Logout failed. Please try again.")

        return True
