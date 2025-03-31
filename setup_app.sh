#!/bin/bash

# Setup and run script for Retail Pulse app

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Retail Pulse environment...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Please install uv package manager:${NC}"
    echo "pip install uv"
    exit 1
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo -e "${GREEN}Creating data directory...${NC}"
    mkdir -p data
fi

# Install dependencies using uv pip
echo -e "${GREEN}Installing dependencies...${NC}"
uv pip install -r requirements.txt

# Check if the installation was successful
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to install dependencies. Please check the error messages above.${NC}"
    exit 1
fi

# Run the Streamlit app
echo -e "${GREEN}Starting Streamlit app...${NC}"
python run_app.py

# Script end
exit 0 