# Retail Pulse

A Streamlit-based retail analytics dashboard for educational purposes.

## Project Structure

```
retail-pulse/
├── app/                    # Main application package
│   ├── components/        # Reusable UI components
│   ├── pages/            # Streamlit pages
│   ├── services/         # Business logic and data services
│   ├── lib/              # Utility functions and configurations
│   ├── data/             # Data storage directory
│   └── logs/             # Application logs
├── .streamlit/           # Streamlit configuration
├── documents/            # Documentation and guides
├── notebooks/            # Jupyter notebooks for analysis
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata and build configuration
├── setup.sh            # Setup script
└── run.sh              # Application runner script
```

## Features

- Interactive retail analytics dashboard
- Real-time data visualization
- Branch performance metrics
- SKU performance analysis
- User authentication with Supabase
- Responsive and modern UI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/retail-pulse.git
cd retail-pulse
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Update your Supabase credentials in `.streamlit/secrets.toml`

4. Run the application:
```bash
./run.sh
```

## Development

The application is built using:
- Streamlit for the web interface
- Supabase for authentication and data storage
- Pandas for data manipulation
- Plotly for interactive visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
