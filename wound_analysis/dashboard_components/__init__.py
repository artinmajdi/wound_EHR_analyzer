"""
Dashboard components package for the wound analysis application.

This package contains modular components used in the main dashboard application,
including settings, visualizations, and other UI elements.
"""

from .settings import DashboardSettings
from .visualizer import Visualizer
from .exudate_tab import ExudateTab
from .impedance_tab import ImpedanceTab
from .impedance_tab_original import ImpedanceTab as ImpedanceTabOriginal
from .llm_analysis_tab import LLMAnalysisTab
from .risk_factors_tab import RiskFactorsTab
from .oxygenation_tab import OxygenationTab
from .temperature_tab import TemperatureTab
from .overview_tab import OverviewTab

__all__ = [ 'DashboardSettings',
            'Visualizer',
            'ExudateTab',
            'ImpedanceTab',
            'ImpedanceTabOriginal',
            'LLMAnalysisTab',
            'RiskFactorsTab',
            'OxygenationTab',
            'TemperatureTab',
            'OverviewTab']
