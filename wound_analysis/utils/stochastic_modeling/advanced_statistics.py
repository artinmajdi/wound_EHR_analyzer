"""
Advanced statistical methods for wound healing analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import warnings


class AdvancedStatistics:
    """
    Implements advanced statistical methods for wound healing analysis.

    This class provides methods for:
    - Polynomial Chaos Expansion (PCE)
    - Bayesian Hierarchical Modeling
    - Mixed-Effects Modeling
    - Variance Function Modeling
    - Conditional Distribution Analysis
    """

    def __init__(self):
        """Initialize the AdvancedStatistics class."""
        self.pce_basis = None
        self.pce_coefficients = None
        self.variance_function = None
        self.mixed_effects_results = None

    def polynomial_chaos_expansion(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        distribution: str = 'normal'
    ) -> Dict:
        """
        Perform Polynomial Chaos Expansion analysis.

        Args:
            x: Independent variable values
            y: Dependent variable values
            degree: Maximum polynomial degree (default: 3)
            distribution: Underlying distribution ('normal' or 'uniform')

        Returns:
            Dictionary containing:
            - PCE coefficients
            - Basis functions
            - Error metrics
            - Sensitivity indices
        """
        # Ensure arrays are 2D
        x = np.atleast_2d(x).T if x.ndim == 1 else x
        y = np.atleast_1d(y)

        # Generate orthogonal polynomial basis
        if distribution == 'normal':
            # Hermite polynomials for normal distribution
            basis = self._hermite_basis(x, degree)
        else:
            # Legendre polynomials for uniform distribution
            basis = self._legendre_basis(x, degree)

        # Fit PCE coefficients using least squares
        coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]

        # Calculate error metrics
        y_pred = basis @ coefficients
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Calculate Sobol sensitivity indices
        sensitivity = self._calculate_sobol_indices(coefficients, degree)

        self.pce_basis = basis
        self.pce_coefficients = coefficients

        return {
            'coefficients': coefficients,
            'basis': basis,
            'mse': mse,
            'r2': r2,
            'sensitivity': sensitivity
        }

    def variance_function_modeling(
        self,
        x: np.ndarray,
        residuals: np.ndarray,
        model_type: str = 'power'
    ) -> Dict:
        """
        Model how variance changes with predictor values.

        Args:
            x: Independent variable values
            residuals: Model residuals
            model_type: Type of variance function ('power' or 'exponential')

        Returns:
            Dictionary containing:
            - Fitted variance function parameters
            - Diagnostic plots data
            - Goodness of fit metrics
        """
        # Square residuals to get variance estimates
        squared_residuals = residuals ** 2

        if model_type == 'power':
            # Fit power law: σ²(x) = a * x^b
            log_x = np.log(x + 1e-10)  # Add small constant to avoid log(0)
            log_var = np.log(squared_residuals + 1e-10)

            # Linear regression on log-transformed data
            coeffs = np.polyfit(log_x, log_var, 1)
            a = np.exp(coeffs[1])
            b = coeffs[0]

            self.variance_function = lambda x: a * x ** b

        else:  # exponential
            # Fit exponential: σ²(x) = a * exp(b*x)
            coeffs = np.polyfit(x, np.log(squared_residuals + 1e-10), 1)
            a = np.exp(coeffs[1])
            b = coeffs[0]

            self.variance_function = lambda x: a * np.exp(b * x)

        # Calculate fitted values
        fitted_variance = self.variance_function(x)

        # Goodness of fit
        r2 = 1 - np.sum((squared_residuals - fitted_variance) ** 2) / \
            np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)

        return {
            'parameters': {'a': a, 'b': b},
            'fitted_values': fitted_variance,
            'r2': r2,
            'model_type': model_type
        }

    def conditional_distribution_analysis(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Analyze how the distribution of y changes with x.

        Args:
            x: Independent variable values
            y: Dependent variable values
            n_bins: Number of bins for x-axis discretization

        Returns:
            Dictionary containing:
            - Conditional statistics for each bin
            - Distribution parameters
            - Test results for distribution changes
        """
        # Create bins for x
        bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
        bin_indices = np.digitize(x, bins)

        results = []
        for i in range(1, n_bins + 1):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 1:  # Need at least 2 points
                y_bin = y[bin_mask]

                # Calculate statistics
                stats_dict = {
                    'bin_center': (bins[i-1] + bins[i]) / 2,
                    'mean': np.mean(y_bin),
                    'std': np.std(y_bin),
                    'skewness': stats.skew(y_bin),
                    'kurtosis': stats.kurtosis(y_bin),
                    'n_points': len(y_bin)
                }

                # Test for normality
                if len(y_bin) >= 3:  # Minimum required for normality test
                    _, p_value = stats.normaltest(y_bin)
                    stats_dict['normality_p_value'] = p_value

                results.append(stats_dict)

        # Test for homogeneity of variances
        _, levene_p = stats.levene(*[y[bin_indices == i]
                                   for i in range(1, n_bins + 1)
                                   if np.sum(bin_indices == i) > 1])

        return {
            'bin_statistics': results,
            'levene_p_value': levene_p,
            'n_bins': n_bins,
            'bin_edges': bins
        }

    def _hermite_basis(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Generate Hermite polynomial basis functions."""
        basis = []
        for i in range(degree + 1):
            hermite = stats.hermite(i)
            basis.append(hermite(x))
        return np.column_stack(basis)

    def _legendre_basis(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Generate Legendre polynomial basis functions."""
        # Scale x to [-1, 1]
        x_scaled = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

        basis = []
        for i in range(degree + 1):
            legendre = np.polynomial.legendre.Legendre.basis(i)
            basis.append(legendre(x_scaled))
        return np.column_stack(basis)

    def _calculate_sobol_indices(
        self,
        coefficients: np.ndarray,
        degree: int
    ) -> Dict[str, float]:
        """Calculate Sobol sensitivity indices from PCE coefficients."""
        total_variance = np.sum(coefficients[1:] ** 2)  # Exclude constant term

        # First-order indices
        first_order = {}
        for i in range(1, degree + 1):
            first_order[f'S{i}'] = coefficients[i] ** 2 / total_variance

        return {
            'first_order': first_order,
            'total': 1.0  # For single variable case
        }

    def cross_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        degree: int = 3
    ) -> Dict:
        """
        Perform cross-validation for PCE model.

        Args:
            x: Independent variable values
            y: Dependent variable values
            n_folds: Number of CV folds
            degree: PCE polynomial degree

        Returns:
            Dictionary containing:
            - Cross-validation scores
            - Mean and std of scores
            - Fold predictions
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        predictions = []

        for train_idx, test_idx in kf.split(x):
            # Fit PCE on training data
            pce_results = self.polynomial_chaos_expansion(
                x[train_idx], y[train_idx], degree=degree
            )

            # Predict on test data
            if x.ndim == 1:
                x_test = x[test_idx].reshape(-1, 1)
            else:
                x_test = x[test_idx]

            if degree == 3:
                basis_test = self._hermite_basis(x_test, degree)
            else:
                basis_test = self._legendre_basis(x_test, degree)

            y_pred = basis_test @ pce_results['coefficients']

            # Calculate R² score
            r2 = 1 - np.sum((y[test_idx] - y_pred) ** 2) / \
                np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)

            scores.append(r2)
            predictions.append((test_idx, y_pred))

        return {
            'cv_scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'predictions': predictions
        }
