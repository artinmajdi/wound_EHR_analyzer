#!/bin/bash
# Script to set up conda environment from config/environment.yml

# Navigate to the project root directory
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "$ROOT_DIR/config/environment.yml" ]; then
    echo "Error: config/environment.yml not found"
    exit 1
fi

# Create conda environment from environment.yml
echo "Creating conda environment from config/environment.yml..."
conda env create -f "$ROOT_DIR/config/environment.yml"

# Setup .env file if it doesn't exist
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "Setting up .env file..."
    "$ROOT_DIR/scripts/setup_env.sh"
fi

echo ""
echo "Conda environment 'wound_analysis' has been created."
echo "To activate the environment, run:"
echo "    conda activate wound_analysis"
echo ""
echo "After activating, you can run the application with:"
echo "    streamlit run wound_analysis/dashboard.py"
