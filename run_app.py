#!/usr/bin/env python3
"""
Helper script to run the Retail Pulse Streamlit app.
"""
import os
import subprocess
import sys
from pathlib import Path


def ensure_directories():
    """Ensure necessary directories exist"""
    os.makedirs("data", exist_ok=True)


def run_app():
    """Run the Streamlit app"""
    streamlit_app_dir = Path("streamlit_app")

    if not streamlit_app_dir.exists():
        print("Error: streamlit-app directory not found.")
        sys.exit(1)

    # Ensure the data directory exists
    ensure_directories()

    # Run the Streamlit app
    try:
        print("Starting Retail Pulse Streamlit app...")
        subprocess.run(
            ["streamlit", "run", str(streamlit_app_dir / "main.py")], check=True
        )
    except KeyboardInterrupt:
        print("\nStopping Streamlit app...")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_app()
