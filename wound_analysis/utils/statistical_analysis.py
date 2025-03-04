import pandas as pd
from typing import Dict, Union, Self
import logging
from scipy import stats

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
            stats_data = {}

            # Count statistics
            unique_patients = self.df['Record ID'].dropna().unique()
            stats_data["Total Patients"] = len(unique_patients)

            # Calculate means safely
            if 'Calculated Wound Area' in self.df.columns:
                valid_areas = self.df['Calculated Wound Area'].dropna()
                stats_data["Average Wound Area (cm²)"] = self._safe_mean(valid_areas)

            if 'Healing Rate (%)' in self.df.columns:
                valid_rates = self.df[
                    (self.df['Healing Rate (%)'].notna()) &
                    (self.df['Healing Rate (%)'] > 0)
                ]['Healing Rate (%)']
                stats_data["Average Healing Rate (%)"] = self._safe_mean(valid_rates)

            if 'Total Temp Gradient' in self.df.columns:
                valid_temps = self.df['Total Temp Gradient'].dropna()
                stats_data["Average Temperature Gradient (°F)"] = self._safe_mean(valid_temps)

            if 'Skin Impedance (kOhms) - Z' in self.df.columns:
                valid_impedance = self.df['Skin Impedance (kOhms) - Z'].dropna()
                stats_data["Average Impedance (kOhms)"] = self._safe_mean(valid_impedance)

            # Add diabetes breakdown if available
            if 'Diabetes?' in self.df.columns:
                valid_diabetes = self.df.dropna(subset=['Diabetes?', 'Record ID'])
                diabetic_count = len(valid_diabetes[valid_diabetes['Diabetes?'] == 'Yes']['Record ID'].unique())
                non_diabetic_count = len(valid_diabetes[valid_diabetes['Diabetes?'] == 'No']['Record ID'].unique())
                stats_data["Diabetic Patients"] = diabetic_count
                stats_data["Non-diabetic Patients"] = non_diabetic_count

            # Format numeric values
            stats_data = {
                k: f"{v:.2f}" if isinstance(v, float) else v
                for k, v in stats_data.items()
            }

            # Remove any remaining nan values
            stats_data = {k: v for k, v in stats_data.items() if str(v).lower() != "nan"}

            return stats_data

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

            stats_data = {}

            # Basic counts
            stats_data["Total Visits"] = len(patient_data)

            # Area measurements
            if 'Calculated Wound Area' in patient_data.columns and len(patient_data) > 0:
                stats_data["Initial Wound Area (cm²)"] = self._safe_float(patient_data.iloc[0]['Calculated Wound Area'])
                stats_data["Latest Wound Area (cm²)"] = self._safe_float(patient_data.iloc[-1]['Calculated Wound Area'])

            # Calculate means safely
            if 'Healing Rate (%)' in patient_data.columns:
                valid_rates = patient_data[
                    (patient_data['Healing Rate (%)'].notna()) &
                    (patient_data['Healing Rate (%)'] > 0)
                ]['Healing Rate (%)']
                stats_data["Average Healing Rate (%)"] = self._safe_mean(valid_rates)

            if 'Total Temp Gradient' in patient_data.columns:
                valid_temps = patient_data['Total Temp Gradient'].dropna()
                stats_data["Average Temperature Gradient (°F)"] = self._safe_mean(valid_temps)

            if 'Skin Impedance (kOhms) - Z' in patient_data.columns:
                valid_impedance = patient_data['Skin Impedance (kOhms) - Z'].dropna()
                stats_data["Average Impedance (kOhms)"] = self._safe_mean(valid_impedance)

            # Calculate total healing progress
            if len(patient_data) >= 2 and 'Calculated Wound Area' in patient_data.columns:
                initial_area = self._safe_float(patient_data.iloc[0]['Calculated Wound Area'])
                final_area = self._safe_float(patient_data.iloc[-1]['Calculated Wound Area'])
                if initial_area > 0:
                    total_healing = ((initial_area - final_area) / initial_area) * 100
                    stats_data["Total Healing Progress (%)"] = total_healing

            # Format numeric values
            stats_data = {
                k: f"{v:.2f}" if isinstance(v, float) else v
                for k, v in stats_data.items()
            }

            # Remove any remaining nan values
            stats_data = {k: v for k, v in stats_data.items() if str(v).lower() != "nan"}

            return stats_data

        except Exception as e:
            logger.error(f"Error calculating patient statistics: {str(e)}")
            return {}


class CorrelationAnalysis:

    def __init__(self, data: pd.DataFrame, x_col: str, y_col: str, outlier_threshold: float = 0.2, REMOVE_OUTLIERS: bool = True):
        self.data  = data
        self.x_col = x_col
        self.y_col = y_col
        self.r, self.p = None, None
        self.outlier_threshold = outlier_threshold
        self.REMOVE_OUTLIERS = REMOVE_OUTLIERS

    def _calculate_correlation(self) -> Self:
        """
        Calculate Pearson correlation between x and y columns of the data.

        This method:
        1. Drops rows with NaN values in either the x or y columns
        2. Calculates the Pearson correlation coefficient (r) and p-value
        3. Stores the results in self.r and self.p

        Returns:
            Self: The current instance for method chaining

        Raises:
            Exception: Logs any error that occurs during correlation calculation
        """
        try:
            valid_data = self.data.dropna(subset=[self.x_col, self.y_col])

            if len(valid_data) > 1:
                self.r, self.p = stats.pearsonr(valid_data[self.x_col], valid_data[self.y_col])

        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")

        return self

    @property
    def format_p_value(self) -> str:
        """
        Format the p-value for display.

        Returns:
            str: A string representation of the p-value. Returns "N/A" if the p-value is None,
                    "< 0.001" if the p-value is less than 0.001, or the p-value rounded to three
                    decimal places with an equals sign prefix otherwise.
        """
        if self.p is None:
            return "N/A"
        return "< 0.001" if self.p < 0.001 else f"= {self.p:.3f}"

    def get_correlation_text(self, text: str="Statistical correlation") -> str:
        """
        Generate a formatted text string describing a statistical correlation.

        This method creates a string that includes the correlation coefficient (r) and p-value
        in a standardized format.

        Parameters
        ----------
        text : str, optional
            The descriptive text to prepend to the correlation statistics.
            Default is "Statistical correlation".

        Returns
        -------
        str
            A formatted string containing the correlation coefficient and p-value.
            Returns "N/A" if either r or p values are None.

        Examples
        --------
        >>> obj.r, obj.p = 0.75, 0.03
        >>> obj.get_correlation_text("Pearson correlation")
        'Pearson correlation: r = 0.75 (p < 0.05)'
        """
        if self.r is None or self.p is None:
            return "N/A"
        return f"{text}: r = {self.r:.2f} (p {self.format_p_value})"

    def calculate_correlation(self):
        """
        Calculate the correlation between impedance data points after removing outliers.

        This method first calls `_remove_outliers()` to clean the data set and then calculates
        the correlation using `_calculate_correlation()`.

        Returns:
            tuple: A tuple containing three elements:
                - data (pd.DataFrame): The processed data after outlier removal
                - r (float or None): Correlation coefficient if calculation succeeds, None otherwise
                - p (float or None): P-value of the correlation if calculation succeeds, None otherwise

        Raises:
            Exception: The method catches any exceptions that occur during calculation and
                        logs the error but returns partial results (data and None values).
        """

        try:
            self._remove_outliers()
            self._calculate_correlation()

            return self.data, self.r, self.p

        except Exception as e:
            logger.error(f"Error calculating impedance correlation: {str(e)}")
            return self.data, None, None

    def _remove_outliers(self) -> 'CorrelationAnalysis':
        """
        Removes outliers from the data based on quantile thresholds.

        This method filters out data points whose x_col values fall outside the range
        defined by the lower and upper quantile bounds. The bounds are calculated using
        the outlier_threshold attribute.

        The method will only remove outliers if:
        - REMOVE_OUTLIERS flag is True
        - outlier_threshold is greater than 0
        - data is not empty
        - x_col exists in the data columns

        Returns:
            CorrelationAnalysis: The current instance with outliers removed

        Raises:
            Exceptions are caught and logged but not propagated.
        """

        try:
            if self.REMOVE_OUTLIERS and (self.outlier_threshold > 0) and (not self.data.empty) and (self.x_col in self.data.columns):

                # Calculate lower and upper bounds
                lower_bound = self.data[self.x_col].quantile(self.outlier_threshold)
                upper_bound = self.data[self.x_col].quantile(1 - self.outlier_threshold)

                # Filter out outliers
                mask = (self.data[self.x_col] >= lower_bound) & (self.data[self.x_col] <= upper_bound)
                self.data = self.data[mask]

        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")

        return self
