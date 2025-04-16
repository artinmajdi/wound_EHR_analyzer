"""
Configuration file for pytest.
This file is automatically loaded by pytest and can contain fixtures, hooks, and path adjustments.
"""
import os
import sys
import pytest

# Add the repository root to sys.path to ensure imports work correctly
# This helps resolve the issue where the repository name differs from the package name
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# You can add shared fixtures here that will be available to all test modules
# For example:
# @pytest.fixture
# def global_test_data():
#     return {...}
