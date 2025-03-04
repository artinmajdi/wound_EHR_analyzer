#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit application wrapper for the wound analysis dashboard.
This file is used by the wound-dashboard entry point.
"""

import sys
import os
import importlib.util
from pathlib import Path

def run_dashboard():
    """
    Run the Streamlit dashboard.
    This function is called when the user runs the wound-dashboard command.
    It uses Streamlit's CLI to run the dashboard.py file.
    """
    import streamlit.web.cli as stcli
    
    # Get the path to the dashboard.py file
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    # Use streamlit CLI to run the dashboard
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_dashboard()
