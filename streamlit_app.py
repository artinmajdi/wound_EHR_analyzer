"""
Streamlit app entry point for Wound EHR Analyzer.
This file serves as the main entry point for Streamlit Cloud deployment.
"""
import streamlit as st
import os
import sys

# Add the project root directory to the Python path
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

# Now import the dashboard module
from wound_analysis.dashboard import main

if __name__ == "__main__":
    main()
