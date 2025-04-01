#!/bin/bash

# Setup script for Retail Pulse application using UV for Python package management

echo "Setting up Retail Pulse application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Print Python version
python_version=$(python3 --version)
echo "Using $python_version"

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
    # Add UV to PATH for the current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtualenv using UV (if it doesn't exist)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies using UV
echo "Installing dependencies with UV..."
uv pip install -r requirements.txt

# Create data directory
mkdir -p data

echo "Setup complete! You can now run the application with: streamlit run streamlit-app/main.py" 