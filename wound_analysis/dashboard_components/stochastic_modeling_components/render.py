"""
Main tab for stochastic modeling analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any

from wound_analysis.dashboard_components.stochastic_modeling_tab.parameter_selector import ParameterSelector
from wound_analysis.dashboard_components.stochastic_modeling_tab.distribution_viewer import DistributionViewer
from wound_analysis.dashboard_components.stochastic_modeling_tab.model_viewer import ModelViewer
from wound_analysis.dashboard_components.stochastic_modeling_tab.uncertainty_viewer import UncertaintyViewer
from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.stochastic_modeling import (
    DistributionAnalyzer,
    PolynomialModeler,
    UncertaintyQuantifier,
    DataPreprocessor,
    AdvancedStatistics
)


class StochasticModelingTab:
    """
    Main class for the stochastic modeling tab.
    Integrates all components for analysis and visualization.
    """

    def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
        """
        Initialize the stochastic modeling tab.

        Parameters:
        ----------
        selected_patient : str
            The currently selected patient from the sidebar dropdown
        wound_data_processor : WoundDataProcessor
            The data processor instance containing the filtered DataFrame
        """
        # Store input parameters
        self.wound_data_processor = wound_data_processor
        self.patient_id           = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
        self.df                   = wound_data_processor.df
        self.CN                   = DColumns(df=self.df)

        # Initialize components
        self.parameter_selector  = ParameterSelector(CN=self.CN)
        self.distribution_viewer = DistributionViewer()
        self.model_viewer        = ModelViewer()
        self.uncertainty_viewer  = UncertaintyViewer()

        # Initialize analysis components
        self.data_preprocessor      = DataPreprocessor(column_names=self.CN)
        self.distribution_analyzer  = DistributionAnalyzer()
        self.polynomial_modeler     = PolynomialModeler()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.advanced_statistics    = AdvancedStatistics()

        # Initialize state
        if 'stochastic_results' not in st.session_state:
            st.session_state.stochastic_results = None

        # Cache for storing computed results
        self.cache: Dict[str, Any] = {}

    def render(self):
        """Render the stochastic modeling tab."""
        st.title("Stochastic Modeling Analysis")

        # Introduction
        self._render_introduction()

        # Parameter selection
        selected_params = self._render_parameter_selection(self.df)
        if not selected_params:
            return

        # Get filtered data
        filtered_df = self._get_filtered_data(self.df, selected_params)
        if filtered_df.empty:
            st.warning("No data available after filtering.")
            return

        # Extract variables
        x, y = self._extract_variables(filtered_df, selected_params)
        if x is None or y is None:
            return

        # Analysis sections
        with st.expander("Distribution Analysis", expanded=True):
            dist_results = self._render_distribution_analysis(
                x, y,
                selected_params['x_name'],
                selected_params['y_name']
            )

        with st.expander("Model Analysis", expanded=True):
            model_results = self._render_model_analysis(
                x, y,
                selected_params['x_name'],
                selected_params['y_name']
            )

        if model_results:
            with st.expander("Uncertainty Analysis", expanded=True):
                uncertainty_results = self._render_uncertainty_analysis(
                    model_results,
                    x, y,
                    selected_params['x_name'],
                    selected_params['y_name']
                )

            # Store results in session state
            st.session_state.stochastic_results = {
                'distribution': dist_results,
                'model': model_results,
                'uncertainty': uncertainty_results
            }

    def _render_introduction(self):
        """Render the introduction section."""
        with st.expander("Introduction to Probabilistic Modeling", expanded=False):
            st.markdown("""
            ## Introduction to Probabilistic Modeling

            Traditional deterministic models of wound healing assume that measurements represent fixed,
            exact values. However, biological systems have inherent variability and uncertainty that
            deterministic models cannot fully capture.

            ### Deterministic vs. Probabilistic Approaches

            **Deterministic Approach:**
            - Treats measurements as fixed values
            - Produces single-valued predictions
            - Cannot account for natural biological variability
            - Limited ability to quantify uncertainty

            **Probabilistic Approach:**
            - Treats measurements as random variables with probability distributions
            - Produces probability distributions rather than point estimates
            - Accounts for natural biological variability
            - Provides robust uncertainty quantification
            - Enables risk assessment and decision-making under uncertainty

            This analysis implements a two-component model approach:
            1. A deterministic component representing the expected trend
            2. A random component representing the variability around that trend
            """)

    def _render_parameter_selection(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Render parameter selection interface.

        Parameters:
        ----------
        df : pd.DataFrame
            Input data

        Returns:
        -------
        Optional[Dict]
            Selected parameters or None if selection is incomplete
        """
        st.header("Parameter Selection")

        # Create variable selectors
        dep_var, indep_var = self.parameter_selector.create_variable_selectors(df=df)
        if not (dep_var and indep_var):
            return None

        # Create filter controls
        filter_params = self.parameter_selector.create_filter_controls(df=df)
        if filter_params.empty:
            return None

        # Create analysis controls
        analysis_params = self.parameter_selector.create_analysis_controls()
        if not analysis_params:
            return None

        return {
            'dependent_var': dep_var,
            'independent_var': indep_var,
            'y_name': dep_var,
            'x_name': indep_var,
            **filter_params,
            **analysis_params
        }

    def _get_filtered_data(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Filter data based on selected parameters.

        Parameters:
        ----------
        df : pd.DataFrame
            Input data
        params : Dict
            Filter parameters

        Returns:
        -------
        pd.DataFrame
            Filtered data
        """
        filtered_df = df.copy()

        # Apply date range filter if specified
        if 'date_range' in params:
            start_date, end_date = params['date_range']
            filtered_df = filtered_df[
                (filtered_df[self.CN.VISIT_DATE] >= start_date) &
                (filtered_df[self.CN.VISIT_DATE] <= end_date)
            ]

        # Apply patient filter if specified
        if 'patient_ids' in params and params['patient_ids']:
            filtered_df = filtered_df[
                filtered_df[self.CN.RECORD_ID].isin(params['patient_ids'])
            ]

        # Apply minimum observations filter if specified
        if 'min_observations' in params:
            min_obs = params['min_observations']
            patient_counts = filtered_df[self.CN.RECORD_ID].value_counts()
            valid_patients = patient_counts[patient_counts >= min_obs].index
            filtered_df = filtered_df[
                filtered_df[self.CN.RECORD_ID].isin(valid_patients)
            ]

        return filtered_df

    def _extract_variables(self, df: pd.DataFrame, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract variables for analysis.

        Parameters:
        ----------
        df : pd.DataFrame
            Input data
        params : Dict
            Selected parameters

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            Independent and dependent variables
        """
        x = df[params['independent_var']].values
        y = df[params['dependent_var']].values

        # Remove rows with NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 5:
            st.error("Not enough valid data points for analysis (minimum 5 required).")
            return None, None

        return x, y

    def _render_distribution_analysis(self,
                                   x: np.ndarray,
                                   y: np.ndarray,
                                   x_name: str,
                                   y_name: str) -> Dict:
        """
        Render distribution analysis section.

        Parameters:
        ----------
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        x_name : str
            Name of independent variable
        y_name : str
            Name of dependent variable

        Returns:
        -------
        Dict
            Distribution analysis results
        """
        st.subheader("Distribution Analysis")

        # Create tabs for each variable
        tab1, tab2 = st.tabs([f"{y_name} Distribution", f"{x_name} Distribution"])

        with tab1:
            y_dist_results = self.distribution_viewer.display_distribution_analysis(
                y, y_name
            )

        with tab2:
            x_dist_results = self.distribution_viewer.display_distribution_analysis(
                x, x_name
            )

        return {
            'dependent_var': y_dist_results,
            'independent_var': x_dist_results
        }

    def _render_model_analysis(self,
                            x: np.ndarray,
                            y: np.ndarray,
                            x_name: str,
                            y_name: str) -> Dict:
        """
        Render model analysis section.

        Parameters:
        ----------
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        x_name : str
            Name of independent variable
        y_name : str
            Name of dependent variable

        Returns:
        -------
        Dict
            Model analysis results
        """
        st.subheader("Model Analysis")

        # Create polynomial model
        model_results = self.model_viewer.display_model_analysis(
            x, y, x_name, y_name
        )

        # Display model metrics
        self.model_viewer.display_model_metrics(
            model_results,
            st.container()
        )

        # Display model equation
        self.model_viewer.display_equation(
            model_results['best_model'],
            x_name,
            st.container()
        )

        return model_results

    def _render_uncertainty_analysis(self,
                                  model_results: Dict,
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  x_name: str,
                                  y_name: str) -> Dict:
        """
        Render uncertainty analysis section.

        Parameters:
        ----------
        model_results : Dict
            Model analysis results
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        x_name : str
            Name of independent variable
        y_name : str
            Name of dependent variable

        Returns:
        -------
        Dict
            Uncertainty analysis results
        """
        st.subheader("Uncertainty Analysis")

        # Display uncertainty analysis
        uncertainty_results = self.uncertainty_viewer.display_uncertainty_analysis(
            model_results,
            x, y,
            x_name, y_name
        )

        # Display prediction controls
        x_value, conf_level = self.model_viewer.display_prediction_controls(
            (min(x), max(x)),
            x_name,
            st.container()
        )

        # Display prediction results
        if x_value is not None:
            self.uncertainty_viewer.display_prediction_results(
                x_value,
                model_results['best_model']['prediction'],
                model_results['best_model']['residual_std'],
                conf_level,
                x_name, y_name,
                st.container()
            )

        return uncertainty_results
