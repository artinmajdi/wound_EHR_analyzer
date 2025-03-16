#!/bin/bash

# Define colors for better user experience
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}=== Wound EHR Analyzer - Dashboard Runner ===${NC}"
echo "This script helps you run the wound analysis dashboard."
echo ""

# Check if .env file exists
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "Warning: .env file not found. Some features may not work properly."
    echo "Consider running './scripts/setup_env.sh' first to set up your environment."
    echo ""
fi

# Prompt user to choose between Docker and Streamlit
echo "Please choose how you would like to run the dashboard:"
echo "1) Run using Docker (recommended for production use)"
echo "2) Run directly with Streamlit (recommended for development)"
echo ""

read -p "Enter your choice (1/2): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting Docker container...${NC}"
        "$SCRIPT_DIR/run_docker.sh" start
        ;;
    2)
        echo -e "${GREEN}Starting Streamlit directly...${NC}"

        # Check if virtual environment exists and activate if found
        if [ -d "$ROOT_DIR/.venv" ]; then
            echo "Using virtual environment..."
            source "$ROOT_DIR/.venv/bin/activate" || source "$ROOT_DIR/.venv/Scripts/activate"
        else
            echo "Warning: Virtual environment not found, using system Python."
            echo "Consider running './scripts/activate_venv.sh' first."
        fi

        # Run Streamlit
        cd "$ROOT_DIR"
        streamlit run wound_analysis/dashboard.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
