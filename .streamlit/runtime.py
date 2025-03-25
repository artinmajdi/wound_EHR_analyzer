"""
Streamlit runtime configuration for Python path.
This file is automatically loaded by Streamlit when the app starts.
"""
import os
import sys
import pathlib

# Add the project root to the Python path
root_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_path))

# Print for debugging
print(f"Added {root_path} to Python path")
print(f"Current Python path: {sys.path}")
