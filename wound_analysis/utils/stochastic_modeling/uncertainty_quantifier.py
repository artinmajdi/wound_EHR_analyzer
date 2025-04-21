"""
Uncertainty quantification utilities for wound healing analysis.
"""

from typing import Dict, Tuple, Optional, List, Union
import numpy as np
from scipy import stats
import pandas as pd


class UncertaintyQuantifier:
    """
    A class for quantifying uncertainty in wound healing predictions and performing risk assessments.

    This class provides methods for:
    - Calculating prediction intervals
    - Assessing risk probabilities
    - Generating confidence bands
    - Computing risk scores and categories
    """

    def __init__(self):
        """Initialize the UncertaintyQuantifier with default risk thresholds."""
        self.risk_thresholds = {
            'low': 0.25,
            'moderate': 0.50,
            'high': 0.75
        }

    def calculate_prediction_interval(
        self,
        model: object,
        x_new: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for new observations using bootstrap resampling.

        Args:
            model: Fitted model object with predict method
            x_new: New predictor values
            confidence_level: Confidence level for intervals (default: 0.95)
            n_bootstrap: Number of bootstrap samples (default: 1000)

        Returns:
            Tuple containing:
            - Mean predictions
            - Lower prediction bounds
            - Upper prediction bounds
        """
        predictions = []

        # Ensure x_new is 2D
        x_new = np.atleast_2d(x_new)

        # Generate bootstrap predictions
        for _ in range(n_bootstrap):
            y_pred = model.predict(x_new)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        # Calculate intervals
        mean_pred = np.mean(predictions, axis=0)
        lower_bound = np.percentile(predictions, (1 - confidence_level) * 100 / 2, axis=0)
        upper_bound = np.percentile(predictions, (1 + confidence_level) * 100 / 2, axis=0)

        return mean_pred, lower_bound, upper_bound

    def assess_risk(
        self,
        predicted_value: float,
        threshold: float,
        distribution: stats.rv_continuous,
        direction: str = 'above'
    ) -> Tuple[float, str]:
        """
        Assess risk based on probability of exceeding/falling below a threshold.

        Args:
            predicted_value: Predicted value from the model
            threshold: Clinical threshold value
            distribution: Fitted probability distribution
            direction: Risk direction ('above' or 'below')

        Returns:
            Tuple containing:
            - Risk probability
            - Risk category (Low, Moderate, High, Very High)
        """
        if direction == 'above':
            risk_prob = 1 - distribution.cdf(threshold)
        else:
            risk_prob = distribution.cdf(threshold)

        # Determine risk category
        if risk_prob < self.risk_thresholds['low']:
            category = 'Low'
        elif risk_prob < self.risk_thresholds['moderate']:
            category = 'Moderate'
        elif risk_prob < self.risk_thresholds['high']:
            category = 'High'
        else:
            category = 'Very High'

        return risk_prob, category

    def calculate_confidence_bands(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model: object,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence bands for the regression line.

        Args:
            x: Independent variable values
            y: Dependent variable values
            model: Fitted model object
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Tuple containing:
            - Lower confidence band
            - Upper confidence band
        """
        # Get predictions
        y_pred = model.predict(x)

        # Calculate standard error of prediction
        n = len(y)
        mse = np.sum((y - y_pred) ** 2) / (n - 2)
        std_error = np.sqrt(mse * (1 + 1/n))

        # Calculate confidence bands
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 2)
        margin = t_value * std_error

        lower_band = y_pred - margin
        upper_band = y_pred + margin

        return lower_band, upper_band

    def compute_risk_score(
        self,
        predicted_value: float,
        uncertainty: float,
        threshold: float,
        weight_prediction: float = 0.7,
        weight_uncertainty: float = 0.3
    ) -> float:
        """
        Compute a weighted risk score based on predicted value and uncertainty.

        Args:
            predicted_value: Predicted value from the model
            uncertainty: Measure of uncertainty (e.g., prediction interval width)
            threshold: Clinical threshold value
            weight_prediction: Weight for prediction component (default: 0.7)
            weight_uncertainty: Weight for uncertainty component (default: 0.3)

        Returns:
            Composite risk score between 0 and 1
        """
        # Normalize prediction component
        pred_component = abs(predicted_value - threshold) / threshold
        pred_component = min(pred_component, 1.0)

        # Normalize uncertainty component
        uncert_component = uncertainty / threshold
        uncert_component = min(uncert_component, 1.0)

        # Calculate weighted score
        risk_score = (weight_prediction * pred_component +
                     weight_uncertainty * uncert_component)

        return min(risk_score, 1.0)

    def set_risk_thresholds(
        self,
        low: float = 0.25,
        moderate: float = 0.50,
        high: float = 0.75
    ) -> None:
        """
        Update risk category thresholds.

        Args:
            low: Threshold for low risk (default: 0.25)
            moderate: Threshold for moderate risk (default: 0.50)
            high: Threshold for high risk (default: 0.75)
        """
        self.risk_thresholds = {
            'low': low,
            'moderate': moderate,
            'high': high
        }
