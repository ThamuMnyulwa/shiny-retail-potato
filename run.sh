#!/bin/bash

# Run script for Retail Pulse application

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Run the application
echo "Starting Retail Pulse application..."
streamlit run streamlit-app/main.py

# Deactivate virtual environment on exit
deactivate 