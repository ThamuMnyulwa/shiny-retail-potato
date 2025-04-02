#!/bin/bash

# Setup script for Retail Pulse application

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install required packages
echo "Installing required packages..."
python -m pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p app/data
mkdir -p app/logs

# Set up environment variables
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file from template. Please update with your credentials."
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create config.toml if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    cat > .streamlit/config.toml << EOL
[theme]
primaryColor="#FF4B4B"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

[server]
enableCORS=false
enableXsrfProtection=true

[browser]
gatherUsageStats=false
EOL
fi

# Create secrets.toml if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    cat > .streamlit/secrets.toml << EOL
[supabase]
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-key"
EOL
    echo "Created .streamlit/secrets.toml. Please update with your Supabase credentials."
fi

echo "Setup complete! You can now run the application with: streamlit run main.py" 