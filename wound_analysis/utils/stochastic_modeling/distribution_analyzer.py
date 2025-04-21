"""
Distribution analysis utilities for wound healing data.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


class DistributionAnalyzer:
    """
    Handles distribution fitting and analysis for wound healing data.
    """

    def __init__(self):
        # Available probability distributions for fitting
        self.available_distributions = {
            'Normal': stats.norm,
            'Log-Normal': stats.lognorm,
            'Gamma': stats.gamma,
            'Weibull': stats.weibull_min,
            'Exponential': stats.expon
        }

    def fit_distributions(self, data: np.ndarray) -> Dict:
        """
        Fit multiple probability distributions to the data.

        Parameters:
        ----------
        data : np.ndarray
            Data to fit distributions to

        Returns:
        -------
        Dict
            Dictionary with distribution objects and their parameters
        """
        # Remove NaN values
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return {}

        results = {}

        for dist_name, distribution in self.available_distributions.items():
            try:
                # Fit the distribution to the data
                if dist_name == 'Log-Normal':
                    # Handle special case for lognormal
                    shape, loc, scale = distribution.fit(data, floc=0)
                    params = {'s': shape, 'loc': loc, 'scale': scale}
                else:
                    params = distribution.fit(data)

                # Calculate goodness of fit
                ks_stat, p_value = stats.kstest(data, distribution.name, params)

                # Store results
                results[dist_name] = {
                    'distribution': distribution,
                    'params': params,
                    'ks_stat': ks_stat,
                    'p_value': p_value,
                    'aic': self._calculate_aic(distribution, params, data)
                }
            except Exception as e:
                print(f"Could not fit {dist_name} distribution: {str(e)}")

        return results

    def _calculate_aic(self, distribution, params, data: np.ndarray) -> float:
        """
        Calculate Akaike Information Criterion for the distribution fit.

        Parameters:
        ----------
        distribution : scipy.stats distribution
            The distribution object
        params : tuple or dict
            Distribution parameters
        data : np.ndarray
            Data used for fitting

        Returns:
        -------
        float
            AIC value
        """
        if isinstance(params, dict):
            params = tuple(params.values())
        k = len(params)
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        return 2 * k - 2 * log_likelihood

    def get_best_distribution(self, results: Dict) -> Tuple[str, Dict]:
        """
        Get the best fitting distribution based on AIC.

        Parameters:
        ----------
        results : Dict
            Dictionary of distribution fitting results

        Returns:
        -------
        Tuple[str, Dict]
            Name of best distribution and its results
        """
        if not results:
            return None, None

        # Sort by AIC
        sorted_results = sorted(results.items(), key=lambda x: x[1]['aic'])
        return sorted_results[0][0], sorted_results[0][1]

    def calculate_basic_stats(self, data: np.ndarray) -> Dict:
        """
        Calculate basic statistical measures for the data.

        Parameters:
        ----------
        data : np.ndarray
            Data to analyze

        Returns:
        -------
        Dict
            Dictionary of basic statistics
        """
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return {}

        return {
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data),
            'n': len(valid_data),
            'shapiro_test': stats.shapiro(valid_data)
        }

    def calculate_conditional_stats(self, data: np.ndarray, conditions: np.ndarray,
                                  n_bins: int = 5) -> Dict:
        """
        Calculate statistics conditional on another variable.

        Parameters:
        ----------
        data : np.ndarray
            Data to analyze
        conditions : np.ndarray
            Conditioning variable values
        n_bins : int, optional
            Number of bins for conditioning, default 5

        Returns:
        -------
        Dict
            Dictionary of conditional statistics
        """
        # Remove NaN values
        mask = ~(np.isnan(data) | np.isnan(conditions))
        data = data[mask]
        conditions = conditions[mask]

        if len(data) == 0:
            return {}

        # Create bins
        bins = np.linspace(min(conditions), max(conditions), n_bins + 1)
        bin_stats = []

        for i in range(n_bins):
            # Get data in this bin
            mask = (conditions >= bins[i]) & (conditions < bins[i+1])
            bin_data = data[mask]

            if len(bin_data) >= 5:  # Ensure enough points for statistics
                stats_dict = {
                    'range': [bins[i], bins[i+1]],
                    'count': len(bin_data),
                    'mean': np.mean(bin_data),
                    'std': np.std(bin_data),
                    'median': np.median(bin_data),
                    'skewness': stats.skew(bin_data),
                    'kurtosis': stats.kurtosis(bin_data)
                }
                bin_stats.append(stats_dict)

        return {
            'bin_stats': bin_stats,
            'anova': stats.f_oneway(*[data[conditions >= bins[i] & conditions < bins[i+1]]
                                    for i in range(n_bins)]) if len(bin_stats) >= 2 else None,
            'bartlett': stats.bartlett(*[data[conditions >= bins[i] & conditions < bins[i+1]]
                                       for i in range(n_bins)]) if len(bin_stats) >= 2 else None
        }
