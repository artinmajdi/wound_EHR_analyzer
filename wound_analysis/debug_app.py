#!/usr/bin/env python3
"""
Professional Streamlit App Debugger

This script allows you to debug a Streamlit application as if it were a normal Python script.
It creates an isolated testing environment that simulates Streamlit functionality while allowing
you to use standard debugging tools like pdb, logging, and print statements.

Usage:
    python debug_app.py

Features:
    - Loads and runs app code without starting Streamlit server
    - Allows standard Python debugging tools (pdb, breakpoints)
    - Provides data simulation for testing
    - Displays debugging information about function calls and data flow
"""

import os
import sys
import inspect
import pandas as pd
import numpy as np
import pathlib
import importlib
import pdb
import traceback
import logging
import importlib.util
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("streamlit_debugger")

# Add the parent directory to the path so we can import modules correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


class StreamlitMock:
    """
    Mock implementation of Streamlit API for debugging purposes.
    This class simulates Streamlit's behavior while allowing normal debugging.
    """
    def __init__(self):
        self.session_state = {}
        self.container_stack = []
        self.current_tab_index = 0
        self.current_tab_name = "default"
        self.current_column = 0
        self.metrics = {}
        self.selectbox_values = {}
        self.number_input_values = {}
        self.plots = []
        self.logger = logging.getLogger("streamlit_mock")
        self.logger.info("StreamlitMock initialized")

    def set_page_config(self, page_title=None, page_icon=None, layout=None, **kwargs):
        self.logger.info(f"Setting page config: title={page_title}, icon={page_icon}, layout={layout}")
        return None

    def title(self, text):
        self.logger.info(f"TITLE: {text}")
        return None

    def header(self, text):
        self.logger.info(f"HEADER: {text}")
        return None

    def subheader(self, text):
        self.logger.info(f"SUBHEADER: {text}")
        return None

    def markdown(self, text, unsafe_allow_html=False):
        self.logger.debug(f"MARKDOWN: {text[:50]}..." if len(text) > 50 else f"MARKDOWN: {text}")
        return None

    def write(self, text):
        self.logger.debug(f"WRITE: {str(text)[:50]}..." if len(str(text)) > 50 else f"WRITE: {text}")
        return None

    def info(self, text):
        self.logger.info(f"INFO: {text}")
        return None

    def warning(self, text):
        self.logger.warning(f"WARNING: {text}")
        return None

    def error(self, text):
        self.logger.error(f"ERROR: {text}")
        return None

    def success(self, text):
        self.logger.info(f"SUCCESS: {text}")
        return None

    def metric(self, label, value, delta=None):
        self.metrics[label] = (value, delta)
        self.logger.info(f"METRIC: {label}={value} (delta={delta})")
        return None

    def _get_default_value(self, key, default_options, default_index=0):
        if key in self.selectbox_values:
            return self.selectbox_values[key]
        if isinstance(default_options, list) and len(default_options) > default_index:
            return default_options[default_index]
        return default_options[0] if default_options else None

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        selected = self._get_default_value(key or label, options, index)
        self.logger.info(f"SELECTBOX: {label} = {selected}")
        return selected

    def multiselect(self, label, options, default=None, key=None, **kwargs):
        selected = default if default else []
        self.logger.info(f"MULTISELECT: {label} = {selected}")
        return selected

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **kwargs):
        if label in self.number_input_values:
            value = self.number_input_values[label]
        self.logger.info(f"NUMBER_INPUT: {label} = {value}")
        return value

    def text_input(self, label, value="", **kwargs):
        self.logger.info(f"TEXT_INPUT: {label} = {value}")
        return value

    def file_uploader(self, label, type=None, accept_multiple_files=False, **kwargs):
        self.logger.info(f"FILE_UPLOADER: {label} (types={type})")
        # Return a mock file that will be used in testing
        return MockFile()

    def button(self, label, key=None, **kwargs):
        self.logger.info(f"BUTTON: {label}")
        # Always return true to simulate button press
        return True

    def checkbox(self, label, value=False, key=None, **kwargs):
        self.logger.info(f"CHECKBOX: {label} = {value}")
        return value

    def tabs(self, labels):
        self.logger.info(f"TABS: {labels}")
        return [TabMock(self, label, i) for i, label in enumerate(labels)]

    def columns(self, spec=None):
        if spec is None:
            spec = [1]
        self.logger.info(f"COLUMNS: {spec}")
        return [ColumnMock(self, i) for i in range(len(spec))]

    def container(self):
        self.logger.debug("CONTAINER created")
        return ContainerMock(self)

    def expander(self, label, expanded=False):
        self.logger.info(f"EXPANDER: {label} (expanded={expanded})")
        return ContainerMock(self)

    def plotly_chart(self, fig, use_container_width=False, **kwargs):
        self.logger.info("PLOTLY_CHART displayed")
        self.plots.append(fig)
        return None

    def pyplot(self, fig=None, **kwargs):
        self.logger.info("PYPLOT displayed")
        self.plots.append(fig or plt.gcf())
        return None

    def line_chart(self, data, **kwargs):
        self.logger.info(f"LINE_CHART with {len(data)} data points")
        return None

    def bar_chart(self, data, **kwargs):
        self.logger.info(f"BAR_CHART with {len(data)} data points")
        return None

    def area_chart(self, data, **kwargs):
        self.logger.info(f"AREA_CHART with {len(data)} data points")
        return None

    def cache_data(self, func):
        """Mock for @st.cache_data decorator - just returns the original function"""
        self.logger.debug(f"CACHE_DATA: Decorating {func.__name__}")
        return func

    def set_option(self, key, value):
        self.logger.info(f"SET_OPTION: {key}={value}")
        return None

    def sidebar(self):
        self.logger.debug("SIDEBAR accessed")
        return SidebarMock(self)


class TabMock:
    """Mock for st.tabs tab objects"""
    def __init__(self, parent, name, index):
        self.parent = parent
        self.name = name
        self.index = index
        self.parent.logger.debug(f"Created tab '{name}' with index {index}")

    def __enter__(self):
        self.parent.current_tab_index = self.index
        self.parent.current_tab_name = self.name
        self.parent.logger.debug(f"Entering tab '{self.name}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent.logger.debug(f"Exiting tab '{self.name}'")
        return False

    def __getattr__(self, name):
        return getattr(self.parent, name)


class ColumnMock:
    """Mock for st.columns column objects"""
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index
        self.parent.logger.debug(f"Created column with index {index}")

    def __enter__(self):
        self.parent.current_column = self.index
        self.parent.logger.debug(f"Entering column {self.index}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent.logger.debug(f"Exiting column {self.index}")
        return False

    def __getattr__(self, name):
        return getattr(self.parent, name)


class ContainerMock:
    """Mock for st.container objects"""
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        self.parent.container_stack.append(id(self))
        self.parent.logger.debug(f"Entering container {id(self)}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent.container_stack.pop()
        self.parent.logger.debug(f"Exiting container {id(self)}")
        return False

    def __getattr__(self, name):
        return getattr(self.parent, name)


class SidebarMock:
    """Mock for st.sidebar object"""
    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, name):
        attr = getattr(self.parent, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                self.parent.logger.debug(f"SIDEBAR.{name} called")
                return attr(*args, **kwargs)
            return wrapper
        return attr


class MockFile:
    """Mock file object that simulates uploaded files in Streamlit"""
    def __init__(self, test_data_path=None):
        # Use the specified path or find the default dataset
        if test_data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_dir = os.path.join(script_dir, "dataset")
            # Try to find the first CSV file in the dataset directory
            test_data_path = None
            if os.path.exists(dataset_dir):
                for file in os.listdir(dataset_dir):
                    if file.endswith('.csv'):
                        test_data_path = os.path.join(dataset_dir, file)
                        break

        self.name = os.path.basename(test_data_path) if test_data_path else "mock_data.csv"
        self.size = os.path.getsize(test_data_path) if test_data_path else 0
        self.type = "text/csv"
        self._test_data_path = test_data_path

        logger.info(f"MockFile initialized with data from: {test_data_path}")

    def read(self):
        if self._test_data_path and os.path.exists(self._test_data_path):
            with open(self._test_data_path, 'rb') as f:
                return f.read()
        return b''


class StreamlitAppDebugger:
    """Class to help debug a Streamlit application"""
    def __init__(self, app_path=None, test_data_path=None):
        self.app_path = app_path or os.path.join(script_dir, "app2_refactored.py")
        self.test_data_path = test_data_path
        self.st_mock = StreamlitMock()
        self.app_module = None

        # Set up mock values for testing
        self.st_mock.selectbox_values = {
            "Select Patient": "Patient 1"  # Default to testing with Patient 1
        }
        self.st_mock.number_input_values = {
            "Impedance Outlier Threshold": 0.2,
            "Temperature Outlier Threshold": 0.0,
            "Oxygenation Outlier Threshold": 0.2
        }

        logger.info(f"Initializing StreamlitAppDebugger with app: {self.app_path}")
        logger.info(f"Test data path: {self.test_data_path}")

    def load_app(self):
        """Load the Streamlit application module"""
        try:
            logger.info(f"Loading app from {self.app_path}")
            # Load the module by path
            spec = importlib.util.spec_from_file_location("app2_refactored", self.app_path)
            self.app_module = importlib.util.module_from_spec(spec)
            with patch('streamlit', self.st_mock):
                spec.loader.exec_module(self.app_module)
            logger.info("App module loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading app module: {str(e)}")
            traceback.print_exc()
            return False

    def run_dashboard(self):
        """Create and run the Dashboard instance from the loaded app"""
        if self.app_module is None:
            logger.error("App module not loaded yet, call load_app() first")
            return False

        try:
            logger.info("Creating Dashboard instance")
            with patch('streamlit', self.st_mock):
                dashboard = self.app_module.Dashboard()

                # Set up the test file uploader
                dashboard.uploaded_file = MockFile(self.test_data_path)

                # Allow configuring the platform and model
                dashboard.llm_platform = "ai-verde"
                dashboard.llm_model = "llama-3.3-70b-fp8"

                # Set up the LLM module
                if not hasattr(dashboard, 'data_processor'):
                    dashboard.data_processor = None

                # Run the dashboard
                logger.info("Running dashboard")
                dashboard.run()
                logger.info("Dashboard execution completed")
            return dashboard
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            traceback.print_exc()
            return None

    def debug_impedance_tab(self):
        """Specifically debug the impedance analysis tab implementation"""
        if self.app_module is None:
            logger.error("App module not loaded yet, call load_app() first")
            return False

        try:
            logger.info("Starting impedance tab debugging")

            # Create the dashboard instance
            with patch('streamlit', self.st_mock):
                dashboard = self.app_module.Dashboard()
                dashboard.uploaded_file = MockFile(self.test_data_path)

                # Load data
                logger.info("Loading data for impedance analysis")
                df = dashboard.load_data(dashboard.uploaded_file)

                if df is None:
                    logger.error("Failed to load data")
                    return False

                # Get a test patient ID
                patient_ids = df['Record ID'].unique()
                if len(patient_ids) > 0:
                    test_patient_id = int(patient_ids[0])
                    logger.info(f"Testing with patient ID: {test_patient_id}")

                    # Test patient data
                    patient_data = dashboard.data_processor.get_patient_visits(record_id=test_patient_id)
                    visits = patient_data['visits']

                    # Create a new ImpedanceAnalyzer instance if it exists
                    if hasattr(self.app_module, 'ImpedanceAnalyzer'):
                        logger.info("Testing ImpedanceAnalyzer functions")
                        impedance_analyzer = self.app_module.ImpedanceAnalyzer()

                        # Insert breakpoint for debugging ImpedanceAnalyzer
                        print("\n======== IMPEDANCE ANALYZER DEBUGGING ========")
                        print(f"Patient ID: {test_patient_id}")
                        print(f"Number of visits: {len(visits)}")
                        print("Available functions:")
                        for name, func in inspect.getmembers(impedance_analyzer, predicate=inspect.ismethod):
                            if not name.startswith('_'):
                                print(f"  - {name}{inspect.signature(func)}")
                        print("\nSet breakpoint to debug individual functions")

                        # Set breakpoint for interactive debugging
                        pdb.set_trace()

                        # Test impedance analysis functions
                        if len(visits) >= 2:
                            logger.info("Testing tissue health calculation")
                            tissue_health = impedance_analyzer.calculate_tissue_health_index(visits[-1])
                            logger.info(f"Tissue health result: {tissue_health}")

                            logger.info("Testing healing trajectory analysis")
                            trajectory = impedance_analyzer.analyze_healing_trajectory(visits)
                            logger.info(f"Trajectory result: {trajectory}")

                            logger.info("Testing frequency response analysis")
                            freq_response = impedance_analyzer.analyze_frequency_response(visits[-1])
                            logger.info(f"Frequency response: {freq_response}")

                            logger.info("Testing infection risk assessment")
                            risk = impedance_analyzer.assess_infection_risk(visits[-1], visits[-2])
                            logger.info(f"Infection risk: {risk}")

                            # Generate comprehensive analysis
                            analyses = {
                                'tissue_health': tissue_health,
                                'healing_trajectory': trajectory,
                                'frequency_response': freq_response,
                                'infection_risk': risk
                            }

                            logger.info("Testing clinical insights generation")
                            insights = impedance_analyzer.generate_clinical_insights(analyses)
                            logger.info(f"Clinical insights: {insights}")
                        else:
                            logger.warning(f"Not enough visits ({len(visits)}) to test all functions")
                    else:
                        logger.warning("ImpedanceAnalyzer class not found in app module")

                    # Try to debug the impedance tab directly
                    logger.info("Debugging impedance tab rendering")
                    with patch('streamlit', self.st_mock):
                        print("\n======== IMPEDANCE TAB DEBUGGING ========")
                        print(f"Testing _impedance_tab with patient: {test_patient_id}")
                        print("Set breakpoint to debug tab rendering")
                        pdb.set_trace()

                        # Call the impedance tab function directly
                        dashboard._impedance_tab(df, f"Patient {test_patient_id}")

                    logger.info("Impedance tab debugging complete")
                else:
                    logger.error("No patient IDs found in data")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error during impedance tab debugging: {str(e)}")
            traceback.print_exc()
            return False

def main():
    """Main entry point for the debugger"""
    logger.info("Starting Streamlit debugger")

    # Get the paths
    app_path = os.path.join(script_dir, "app2_refactored.py")
    test_data_path = os.path.join(script_dir, "dataset", "SmartBandage-Data_for_llm.csv")

    if not os.path.exists(app_path):
        logger.error(f"App file not found at {app_path}")
        return

    if not os.path.exists(test_data_path):
        logger.warning(f"Test data not found at {test_data_path}")
        logger.warning("Will attempt to find a CSV file in the dataset directory")

    debugger = StreamlitAppDebugger(app_path, test_data_path)
    success = debugger.load_app()

    if not success:
        logger.error("Failed to load app, exiting")
        return

    print("\n=== Streamlit App Debug Menu ===")
    print("1: Debug full dashboard")
    print("2: Debug impedance tab only")
    print("3: Exit")

    choice = input("\nEnter your choice (1-3): ")

    if choice == '1':
        print("\n======== FULL DASHBOARD DEBUGGING ========")
        print("Setting breakpoint before dashboard execution")
        print("Use 'n' to step to next line, 's' to step into functions")
        print("'c' to continue to next breakpoint, and 'q' to quit.")
        pdb.set_trace()
        dashboard = debugger.run_dashboard()

    elif choice == '2':
        debugger.debug_impedance_tab()

    else:
        print("Exiting")

    logger.info("Debugger session completed")

if __name__ == "__main__":
    main()
