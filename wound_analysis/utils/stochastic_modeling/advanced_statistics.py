"""
Advanced statistical methods for wound healing analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import hermite, eval_hermite
from sklearn.model_selection import KFold
import warnings
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


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


    def polynomial_chaos_expansion( self, x: np.ndarray, y: np.ndarray, degree: int = 3, distribution: str = 'normal' ) -> Dict:
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


    def variance_function_modeling( self, x: np.ndarray, residuals: np.ndarray, model_type: str = 'power' ) -> Dict:
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


    def conditional_distribution_analysis( self, x: np.ndarray, y: np.ndarray, n_bins: int = 10 ) -> Dict:
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
        """
        Generate Hermite polynomial basis functions using NumPy's polynomial module.
        This implementation is more numerically stable than using scipy.special.hermite directly.
        """
        from numpy.polynomial.hermite import hermval

        # Ensure x is properly shaped
        x = x.flatten()

        # Normalize input to improve numerical stability
        x_scaled = (x - np.mean(x)) / (np.std(x) + 1e-10)

        # Create basis matrix using NumPy's hermval function
        basis = np.zeros((len(x_scaled), degree + 1))

        for i in range(degree + 1):
            # Create coefficient array for polynomial of degree i
            # (1 at position i, 0 elsewhere)
            coef = np.zeros(i + 1)
            coef[i] = 1.0

            # Evaluate Hermite polynomial at the scaled input points
            try:
                # More robust approach with error handling
                basis[:, i] = hermval(x_scaled, coef)
            except Exception:
                # Fallback for numerical stability if hermval fails
                warnings.warn(f"Hermite polynomial evaluation failed for degree {i}. Using fallback.")
                # Use simple powers as fallback (less accurate but more stable)
                basis[:, i] = x_scaled ** i if i > 0 else np.ones_like(x_scaled)

        return basis


    def _legendre_basis(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Generate Legendre polynomial basis functions."""
        # Scale x to [-1, 1]
        x_scaled = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

        basis = []
        for i in range(degree + 1):
            legendre = np.polynomial.legendre.Legendre.basis(i)
            basis.append(legendre(x_scaled))
        return np.column_stack(basis)


    def _calculate_sobol_indices( self, coefficients: np.ndarray, degree: int ) -> Dict[str, float]:
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


    def cross_validate( self, x: np.ndarray, y: np.ndarray, n_folds: int = 5, degree: int = 3 ) -> Dict:
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


    def bayesian_hierarchical_modeling( self, x: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None, n_samples: int = 2000, n_tune: int = 1000, random_seed: int = 42 ) -> Dict:
        """
        Perform Bayesian Hierarchical Modeling analysis.

        Args:
            x: Independent variable values
            y: Dependent variable values
            groups: Optional group labels for hierarchical structure
            n_samples: Number of MCMC samples (default: 2000)
            n_tune: Number of tuning steps (default: 1000)
            random_seed: Random seed for reproducibility (default: 42)

        Returns:
            Dictionary containing:
            - Trace of MCMC samples
            - Model summary statistics
            - Posterior predictions
            - Model diagnostics
        """
        # Standardize variables
        x_standardized = (x - np.mean(x)) / np.std(x)
        y_standardized = (y - np.mean(y)) / np.std(y)

        try:
            import pymc3
            import arviz

            with pymc3.Model() as model:
                # Priors for unknown model parameters
                alpha = pymc3.Normal('alpha', mu=0, sd=10)
                beta  = pymc3.Normal('beta', mu=0, sd=10)
                sigma = pymc3.HalfNormal('sigma', sd=1)

                # Expected value of outcome
                mu = alpha + beta * x_standardized

                # Add hierarchical structure if groups are provided
                if groups is not None:
                    # Random intercepts for groups
                    sigma_group   = pymc3.HalfNormal('sigma_group', sd=1)
                    group_idx     = pd.Categorical(groups).codes
                    group_effects = pymc3.Normal('group_effects', mu=0, sd=sigma_group, shape=len(np.unique(groups)))
                    mu            = mu + group_effects[group_idx]

                # Likelihood (sampling distribution) of observations
                Y_obs = pymc3.Normal('Y_obs', mu=mu, sd=sigma, observed=y_standardized)

                # Inference
                trace = pymc3.sample(n_samples, tune=n_tune, random_seed=random_seed, return_inferencedata=True)

            # Generate posterior predictive samples
            with model:
                ppc = pymc3.sample_posterior_predictive(trace, samples=1000)

            # Calculate summary statistics
            summary = arviz.summary(trace, var_names=['alpha', 'beta', 'sigma'])
            if groups is not None:
                summary_groups = arviz.summary(trace, var_names=['group_effects', 'sigma_group'])
                summary = pd.concat([summary, summary_groups])

            # Calculate model diagnostics
            diagnostics = {
                'r2_score': 1 - np.sum((y_standardized - ppc['Y_obs'].mean(axis=0))**2) /
                           np.sum((y_standardized - np.mean(y_standardized))**2),
                'waic': arviz.waic(trace),
                'loo' : arviz.loo(trace)
            }

            return {
                'trace': trace,
                'summary': summary,
                'posterior_predictive': ppc,
                'diagnostics': diagnostics,
                'model': model
            }

        except Exception as e:
            warnings.warn(f"Error in Bayesian analysis: {str(e)}")
            return None


    def mixed_effects_modeling( self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, random_effects: Optional[List[str]] = None, fixed_effects: Optional[List[str]] = None ) -> Dict:
        """
        Perform Mixed-Effects Modeling analysis.

        Args:
            x: Independent variable values (can be 2D for multiple predictors)
            y: Dependent variable values
            groups: Group labels for random effects
            random_effects: List of column names for random effects (default: None)
            fixed_effects: List of column names for fixed effects (default: None)

        Returns:
            Dictionary containing:
            - Model fit results
            - Random effects estimates
            - Model diagnostics
            - Variance components
        """
        try:
            # Ensure x is 2D array
            X = np.atleast_2d(x).T if x.ndim == 1 else x

            # Add constant term for intercept
            X = sm.add_constant(X)

            # Create random effects design matrix
            Z = np.ones_like(X) if random_effects is None else X[:, [0] + [i+1 for i, col in enumerate(random_effects)]]

            # Fit mixed-effects model
            model = MixedLM(y, X, groups, Z)
            result = model.fit()

            # Extract random effects
            random_effects_pred = result.random_effects
            random_effects_df   = pd.concat(random_effects_pred, axis=0)

            # Calculate variance components
            variance_components = {
                'random_effects': result.cov_re.values.diagonal(),
                'residual': result.scale
            }

            # Calculate model diagnostics
            y_pred = result.predict()
            residuals = y - y_pred
            r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

            # Perform likelihood ratio test for random effects
            null_model = sm.OLS(y, X).fit()
            lr_stat = -2 * (null_model.llf - result.llf)
            lr_pvalue = stats.chi2.sf(lr_stat, df=len(result.cov_re.values.diagonal()))

            diagnostics = {
                'r2_score': r2,
                'aic': result.aic,
                'bic': result.bic,
                'log_likelihood': result.llf,
                'lr_test': {
                    'statistic': lr_stat,
                    'p_value': lr_pvalue
                }
            }

            return {
                'model': result,
                'random_effects': random_effects_df,
                'variance_components': variance_components,
                'diagnostics': diagnostics,
                'predictions': y_pred,
                'residuals': residuals
            }

        except Exception as e:
            warnings.warn(f"Error in mixed-effects analysis: {str(e)}")
            return None
