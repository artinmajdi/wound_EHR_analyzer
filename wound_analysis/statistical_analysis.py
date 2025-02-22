"""
Statistical analysis module for wound healing data.

This module provides statistical analysis functionality for wound healing data,
including overall statistics and patient-specific analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalAnalysis:
    """Handles statistical analysis of wound healing data."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with wound healing dataset.

        Args:
            df: DataFrame containing wound healing data
        """
        self.df = df.copy()
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """Preprocess data to ensure correct types for calculations."""
        try:
            # Ensure numeric columns are float
            numeric_cols = [
                'Calculated Wound Area',
                'Healing Rate (%)',
                'Total Temp Gradient',
                'Skin Impedance (kOhms) - Z'
            ]
            
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Sort data by Record ID and Visit Number
            if 'Visit Number' in self.df.columns:
                self.df['Visit Number'] = pd.to_numeric(self.df['Visit Number'], errors='coerce')
                self.df = self.df.sort_values(['Record ID', 'Visit Number'])

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")

    def _safe_mean(self, series: pd.Series) -> float:
        """Calculate mean safely handling NaN values."""
        try:
            if series.empty:
                return 0.0
            valid_values = series.dropna()
            return float(valid_values.mean()) if not valid_values.empty else 0.0
        except Exception as e:
            logger.warning(f"Error calculating mean: {str(e)}")
            return 0.0

    def _safe_float(self, value: Union[float, str, int]) -> float:
        """Safely convert value to float."""
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def get_overall_statistics(self) -> Dict:
        """Calculate overall statistics for the dataset.

        Returns:
            Dict containing overall statistics
        """
        try:
            stats = {}
            
            # Count statistics
            unique_patients = self.df['Record ID'].dropna().unique()
            stats["Total Patients"] = len(unique_patients)
            
            # Calculate means safely
            if 'Calculated Wound Area' in self.df.columns:
                valid_areas = self.df['Calculated Wound Area'].dropna()
                stats["Average Wound Area (cm²)"] = self._safe_mean(valid_areas)
            
            if 'Healing Rate (%)' in self.df.columns:
                valid_rates = self.df[
                    (self.df['Healing Rate (%)'].notna()) & 
                    (self.df['Healing Rate (%)'] > 0)
                ]['Healing Rate (%)']
                stats["Average Healing Rate (%)"] = self._safe_mean(valid_rates)
            
            if 'Total Temp Gradient' in self.df.columns:
                valid_temps = self.df['Total Temp Gradient'].dropna()
                stats["Average Temperature Gradient (°F)"] = self._safe_mean(valid_temps)
            
            if 'Skin Impedance (kOhms) - Z' in self.df.columns:
                valid_impedance = self.df['Skin Impedance (kOhms) - Z'].dropna()
                stats["Average Impedance (kOhms)"] = self._safe_mean(valid_impedance)

            # Add diabetes breakdown if available
            if 'Diabetes?' in self.df.columns:
                valid_diabetes = self.df.dropna(subset=['Diabetes?', 'Record ID'])
                diabetic_count = len(valid_diabetes[valid_diabetes['Diabetes?'] == 'Yes']['Record ID'].unique())
                non_diabetic_count = len(valid_diabetes[valid_diabetes['Diabetes?'] == 'No']['Record ID'].unique())
                stats["Diabetic Patients"] = diabetic_count
                stats["Non-diabetic Patients"] = non_diabetic_count

            # Format numeric values
            stats = {
                k: f"{v:.2f}" if isinstance(v, float) else v 
                for k, v in stats.items()
            }
            
            # Remove any remaining nan values
            stats = {k: v for k, v in stats.items() if str(v).lower() != "nan"}

            return stats

        except Exception as e:
            logger.error(f"Error calculating overall statistics: {str(e)}")
            return {}

    def get_patient_statistics(self, patient_id: int) -> Dict:
        """Calculate statistics for a specific patient.

        Args:
            patient_id: The patient's ID number

        Returns:
            Dict containing patient-specific statistics
        """
        try:
            patient_data = self.df[self.df['Record ID'] == patient_id].copy()
            if patient_data.empty:
                logger.warning(f"No data found for patient {patient_id}")
                return {}

            stats = {}
            
            # Basic counts
            stats["Total Visits"] = len(patient_data)
            
            # Area measurements
            if 'Calculated Wound Area' in patient_data.columns and len(patient_data) > 0:
                stats["Initial Wound Area (cm²)"] = self._safe_float(patient_data.iloc[0]['Calculated Wound Area'])
                stats["Latest Wound Area (cm²)"] = self._safe_float(patient_data.iloc[-1]['Calculated Wound Area'])
            
            # Calculate means safely
            if 'Healing Rate (%)' in patient_data.columns:
                valid_rates = patient_data[
                    (patient_data['Healing Rate (%)'].notna()) & 
                    (patient_data['Healing Rate (%)'] > 0)
                ]['Healing Rate (%)']
                stats["Average Healing Rate (%)"] = self._safe_mean(valid_rates)
            
            if 'Total Temp Gradient' in patient_data.columns:
                valid_temps = patient_data['Total Temp Gradient'].dropna()
                stats["Average Temperature Gradient (°F)"] = self._safe_mean(valid_temps)
            
            if 'Skin Impedance (kOhms) - Z' in patient_data.columns:
                valid_impedance = patient_data['Skin Impedance (kOhms) - Z'].dropna()
                stats["Average Impedance (kOhms)"] = self._safe_mean(valid_impedance)

            # Calculate total healing progress
            if len(patient_data) >= 2 and 'Calculated Wound Area' in patient_data.columns:
                initial_area = self._safe_float(patient_data.iloc[0]['Calculated Wound Area'])
                final_area = self._safe_float(patient_data.iloc[-1]['Calculated Wound Area'])
                if initial_area > 0:
                    total_healing = ((initial_area - final_area) / initial_area) * 100
                    stats["Total Healing Progress (%)"] = total_healing

            # Format numeric values
            stats = {
                k: f"{v:.2f}" if isinstance(v, float) else v 
                for k, v in stats.items()
            }
            
            # Remove any remaining nan values
            stats = {k: v for k, v in stats.items() if str(v).lower() != "nan"}

            return stats

        except Exception as e:
            logger.error(f"Error calculating patient statistics: {str(e)}")
            return {}
