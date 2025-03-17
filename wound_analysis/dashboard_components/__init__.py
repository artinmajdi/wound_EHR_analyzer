"""
Dashboard components package for the wound analysis application.

This package contains modular components used in the main dashboard application,
including settings, visualizations, and other UI elements.
"""

from .settings import DashboardSettings
from .visualizer import Visualizer

__all__ = ['DashboardSettings', 'Visualizer']
