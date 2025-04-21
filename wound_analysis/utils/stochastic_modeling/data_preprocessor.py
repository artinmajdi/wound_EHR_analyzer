"""Data preprocessing utilities for stochastic modeling."""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st


class DataPreprocessor:
    """Class for preprocessing data for stochastic modeling analysis."""

    def __init__(self, column_names):
        """
        Initialize DataPreprocessor.

        Parameters:
        ----------
        column_names : object
            Object containing column name constants
        """
        self.CN = column_names

    def filter_data(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Filter data based on specified parameters.

        Parameters:
        ----------
        df : pd.DataFrame
            Input data
        params : Dict
            Filter parameters including:
            - start_date : str
            - end_date : str
            - patient_ids : List[str]
            - min_observations : int

        Returns:
        -------
        pd.DataFrame
            Filtered data
        """
        filtered_df = df.copy()

        # Apply date range filter if specified
        if 'start_date' in params and 'end_date' in params:
            filtered_df = filtered_df[
                (filtered_df[self.CN.DATE] >= params['start_date']) &
                (filtered_df[self.CN.DATE] <= params['end_date'])
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

    def extract_variables(self, df: pd.DataFrame, params: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract variables for analysis.

        Parameters:
        ----------
        df : pd.DataFrame
            Input data
        params : Dict
            Selected parameters including:
            - independent_var : str
            - dependent_var : str

        Returns:
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            Independent and dependent variables, or (None, None) if insufficient data
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
