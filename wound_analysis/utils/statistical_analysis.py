import pandas as pd
from typing import Dict, Union, Self
import logging
from scipy import stats

from wound_analysis.utils.column_schema import DColumns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
