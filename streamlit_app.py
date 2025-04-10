"""
Streamlit app entry point for Wound EHR Analyzer.
This file serves as the main entry point for Streamlit Cloud deployment.
"""
import streamlit as st
import os
import sys
import pathlib

# Add the project root directory to the Python path
root_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(root_path))

# Import the dashboard directly
from wound_analysis.dashboard import Dashboard

def main():
    """Main entry point for the Streamlit application."""
    # Create and run the dashboard
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
