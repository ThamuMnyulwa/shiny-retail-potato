#!/bin/bash

# Run script for Retail Pulse application

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one now..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing required packages..."
    python -m pip install -r requirements.txt
fi

# Run the application
echo "Starting Retail Pulse application..."
python -m streamlit run streamlit-app/main.py

# Deactivate virtual environment on exit
trap "deactivate" EXIT 