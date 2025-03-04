#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for the wound analysis dashboard.
This file provides the entry points for the wound-dashboard command.
"""

import sys
import os
from pathlib import Path

def run_dashboard():
    """
    Run the Streamlit dashboard.
    This function is called when the user runs the wound-dashboard command.
    It uses Streamlit to run the dashboard.py file.
    """
    # Add the parent directory to sys.path to ensure imports work correctly
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import streamlit after setting up the path
    import streamlit.web.cli as stcli
    
    # Get the path to the dashboard.py file
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    # Use streamlit CLI to run the dashboard
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run_dashboard()
