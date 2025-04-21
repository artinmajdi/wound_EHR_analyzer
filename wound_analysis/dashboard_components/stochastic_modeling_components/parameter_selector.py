"""
Parameter selection UI components for stochastic modeling.
"""

import streamlit as st
import pandas as pd
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
from wound_analysis.utils.column_schema import DColumns

class ParameterSelector:
    """
    Handles parameter selection UI components for stochastic modeling.
    """

    def __init__(self, key_prefix: str = "", CN: DColumns = None):
        """
        Initialize the parameter selector.

        Parameters:
        ----------
        key_prefix : str, optional
            Prefix for Streamlit widget keys to avoid conflicts
        CN : DColumns, optional
            Column names object for data processing
        """
        self.key_prefix = key_prefix
        self.CN = CN


    def create_variable_selectors(self, df: pd.DataFrame,
                                default_dependent: str = "impedance",
                                default_independent: str = "wound_size") -> Tuple[str, str]:
        """
        Create dropdown selectors for dependent and independent variables.

        Parameters:
        ----------
        df : pd.DataFrame
            Input dataframe containing the variables
        default_dependent : str, optional
            Default dependent variable
        default_independent : str, optional
            Default independent variable

        Returns:
        -------
        Tuple[str, str]
            Selected dependent and independent variables
        """
        st.subheader("Variable Selection")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            dependent_var = st.selectbox(
                "Select Dependent Variable",
                options=numeric_cols,
                index=numeric_cols.index(default_dependent) if default_dependent in numeric_cols else 0,
                key=f"{self.key_prefix}dependent_var",
                help="The variable to be predicted (e.g., impedance)"
            )

        with col2:
            # Remove dependent variable from options for independent variable
            independent_options = [col for col in numeric_cols if col != dependent_var]
            independent_var = st.selectbox(
                "Select Independent Variable",
                options=independent_options,
                index=independent_options.index(default_independent) if default_independent in independent_options else 0,
                key=f"{self.key_prefix}independent_var",
                help="The predictor variable (e.g., wound size)"
            )

        return dependent_var, independent_var

    def create_filter_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create controls for filtering the dataset.

        Parameters:
        ----------
        df : pd.DataFrame
            Input dataframe to filter

        Returns:
        -------
        pd.DataFrame
            Filtered dataframe
        """
        st.subheader("Data Filtering")

        filtered_df = df.copy()

        with st.expander("Filter Options"):
            # Date range filter if date column exists
            if 'date' in df.columns:
                min_date = pd.to_datetime(df['date'].min())
                max_date = pd.to_datetime(df['date'].max())

                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    key=f"{self.key_prefix}date_range"
                )

                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['date']).dt.date >= start_date) &
                        (pd.to_datetime(filtered_df['date']).dt.date <= end_date)
                    ]

            # Patient ID filter if patient_id column exists
            if self.CN.RECORD_ID in df.columns:
                patient_ids = sorted(df[self.CN.RECORD_ID].unique())
                selected_patients = st.multiselect(
                    "Select Patient IDs",
                    options=patient_ids,
                    default=patient_ids,
                    key=f"{self.key_prefix}patient_ids"
                )

                if selected_patients:
                    filtered_df = filtered_df[filtered_df[self.CN.RECORD_ID].isin(selected_patients)]

            # Add minimum number of observations filter
            min_observations = st.number_input(
                "Minimum Number of Observations",
                min_value=1,
                value=5,
                key=f"{self.key_prefix}min_observations",
                help="Filter out groups with fewer than this many observations"
            )

            if min_observations > 1:
                # Apply minimum observations filter
                group_counts = filtered_df.groupby(self.CN.RECORD_ID).size()
                valid_groups = group_counts[group_counts >= min_observations].index
                filtered_df = filtered_df[filtered_df[self.CN.RECORD_ID].isin(valid_groups)]

        # Show current filter status
        st.info(f"Currently showing {len(filtered_df)} observations from {filtered_df[self.CN.RECORD_ID].nunique()} patients")

        return filtered_df

    def create_analysis_controls(self) -> Tuple[bool, dict]:
        """
        Create controls for analysis parameters.

        Returns:
        -------
        Tuple[bool, dict]
            Boolean indicating if analysis should run and dictionary of parameters
        """
        st.subheader("Analysis Parameters")

        col1, col2 = st.columns(2)

        with col1:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key=f"{self.key_prefix}confidence_level",
                help="Confidence level for prediction intervals"
            )

            max_poly_degree = st.slider(
                "Maximum Polynomial Degree",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key=f"{self.key_prefix}max_poly_degree",
                help="Maximum degree for polynomial modeling"
            )

        with col2:
            model_selection = st.radio(
                "Model Selection Criterion",
                options=['AIC', 'BIC'],
                key=f"{self.key_prefix}model_selection",
                help="Criterion for selecting the best model"
            )

            n_bootstrap = st.number_input(
                "Bootstrap Samples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key=f"{self.key_prefix}n_bootstrap",
                help="Number of bootstrap samples for uncertainty estimation"
            )

        run_analysis = st.button(
            "Run Analysis",
            key=f"{self.key_prefix} run_analysis",
            help="Click to run the stochastic modeling analysis"
        )

        parameters = {
            'confidence_level': confidence_level,
            'max_poly_degree': max_poly_degree,
            'model_selection': model_selection.lower(),
            'n_bootstrap': n_bootstrap
        }

        return run_analysis, parameters
