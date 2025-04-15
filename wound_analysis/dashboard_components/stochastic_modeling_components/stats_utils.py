from io import BytesIO
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats
from scipy.special import hermite, eval_hermite
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class StatsUtils:

    @staticmethod
    def plot_distribution_fit(data, results, var_name):
        """
        Create a plot of the data histogram with fitted distributions.

        Parameters:
        ----------
        data : np.ndarray
            Data to plot
        results : Dict
            Dictionary with distribution fitting results
        var_name : str
            Name of the variable being plotted

        Returns:
        -------
        BytesIO
            Plot as a BytesIO object for Streamlit
        """
        # Remove NaN values
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        hist, bin_edges, _ = ax.hist(data, bins=20, density=True, alpha=0.6, color='gray', label='Data')

        # Sort distributions by goodness of fit (AIC)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['aic'])

        # Select top 3 distributions to plot
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        x = np.linspace(min(data), max(data), 1000)

        # Plot the fitted distributions
        for i, (dist_name, result) in enumerate(sorted_results[:3]):
            if i >= len(colors):
                break

            distribution = result['distribution']
            params = result['params']

            # Calculate PDF
            try:
                if dist_name == 'Log-Normal':
                    pdf = distribution.pdf(x, result['params']['s'], result['params']['loc'], result['params']['scale'])
                else:
                    pdf = distribution.pdf(x, *params)

                ax.plot(x, pdf, color=colors[i], linewidth=2,
                        label=f'{dist_name} (AIC: {result["aic"]:.2f}, p-value: {result["p_value"]:.3f})')
            except Exception as e:
                st.warning(f"Error plotting {dist_name} distribution: {str(e)}")

        ax.set_title(f'Distribution Analysis for {var_name}')
        ax.set_xlabel(var_name)
        ax.set_ylabel('Density')
        ax.legend()

        # Save plot to BytesIO
        buf = BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return buf


    @staticmethod
    def fit_distributions(data: np.ndarray, available_distributions: Dict) -> Dict:
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

        for dist_name, distribution in available_distributions.items():
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
                    'aic': StatsUtils.calculate_aic(distribution, params, data)
                }
            except Exception as e:
                st.warning(f"Could not fit {dist_name} distribution: {str(e)}")

        return results


    @staticmethod
    def calculate_aic(distribution, params, data):
        """Calculate Akaike Information Criterion for the distribution fit."""
        k = len(params)
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        return 2 * k - 2 * log_likelihood


    @staticmethod
    def fit_polynomial_models(X: np.ndarray, y: np.ndarray, max_degree: int = 5) -> Dict[int, Dict]:
        """
        Fit polynomial models of different degrees to the data.

        Parameters:
        ----------
        X : np.ndarray
            Independent variable data, shape (n_samples, 1)
        y : np.ndarray
            Dependent variable data, shape (n_samples,)
        max_degree : int
            Maximum polynomial degree to fit

        Returns:
        -------
        Dict[int, Dict]
            Dictionary with results for each polynomial degree
        """
        # Ensure X is 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}

        # Fit polynomial models of different degrees
        for degree in range(1, max_degree + 1):
            # Create polynomial features
            poly         = PolynomialFeatures(degree=degree)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test  = poly.transform(X_test)

            # Fit the model
            model = LinearRegression()
            model.fit(X_poly_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_poly_train)
            y_pred_test  = model.predict(X_poly_test)

            # Calculate metrics
            r2_train  = r2_score(y_train, y_pred_train)
            r2_test   = r2_score(y_test, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test  = mean_squared_error(y_test, y_pred_test)

            # Calculate AIC and BIC
            n_train = len(y_train)
            k       = degree + 1  # number of parameters
            aic     = n_train * np.log(mse_train) + 2 * k
            bic     = n_train * np.log(mse_train) + k * np.log(n_train)

            # Store results
            results[degree] = {
                'model': model,
                'poly': poly,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'aic': aic,
                'bic': bic,
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }

        return results


    @staticmethod
    def fit_hermite_polynomial_models(X: np.ndarray, y: np.ndarray, max_degree: int = 5) -> Dict[int, Dict]:
        """
        Fit Hermite polynomial models of different degrees to the data.

        Hermite polynomials are a set of orthogonal polynomials that are particularly useful
        for modeling random processes with Gaussian characteristics.

        Parameters:
        ----------
        X : np.ndarray
            Independent variable data, shape (n_samples, 1)
        y : np.ndarray
            Dependent variable data, shape (n_samples,)
        max_degree : int
            Maximum polynomial degree to fit

        Returns:
        -------
        Dict[int, Dict]
            Dictionary with results for each polynomial degree
        """
        # Ensure X is 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Standardize X to improve numerical stability (Hermite polynomials are defined on (-∞, ∞))
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_standardized = (X - X_mean) / X_std

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

        results = {}

        # Fit Hermite polynomial models of different degrees
        for degree in range(1, max_degree + 1):
            # Create Hermite polynomial features
            X_hermite_train = np.zeros((X_train.shape[0], degree + 1))
            X_hermite_test = np.zeros((X_test.shape[0], degree + 1))

            # Compute Hermite polynomials for each degree
            for d in range(degree + 1):
                X_hermite_train[:, d] = np.array([eval_hermite(d, x[0]) for x in X_train])
                X_hermite_test[:, d] = np.array([eval_hermite(d, x[0]) for x in X_test])

            # Fit the model
            model = LinearRegression(fit_intercept=False)  # Hermite polynomials include a constant term
            model.fit(X_hermite_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_hermite_train)
            y_pred_test = model.predict(X_hermite_test)

            # Calculate metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)

            # Calculate AIC and BIC
            n_train = len(y_train)
            k = degree + 1  # number of parameters
            aic = n_train * np.log(mse_train) + 2 * k
            bic = n_train * np.log(mse_train) + k * np.log(n_train)

            # Store results
            results[degree] = {
                'model': model,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'aic': aic,
                'bic': bic,
                'coefficients': model.coef_,
                'intercept': 0.0,  # Since we're using fit_intercept=False
                'X_mean': X_mean,
                'X_std': X_std,
                'is_hermite': True
            }

        return results


    @staticmethod
    def plot_polynomial_fit(X: np.ndarray, y: np.ndarray, model_results: Dict, degree: int,
                            x_label: str, y_label: str, with_confidence_intervals: bool = False) -> BytesIO:
        """
        Create a plot of the data with the fitted polynomial curve.

        Parameters:
        ----------
        X : np.ndarray
            Independent variable data, shape (n_samples, 1)
        y : np.ndarray
            Dependent variable data, shape (n_samples,)
        model_results : Dict
            Dictionary with model results
        degree : int
            Polynomial degree to plot
        x_label : str
            Label for the x-axis
        y_label : str
            Label for the y-axis
        with_confidence_intervals : bool
            Whether to plot confidence intervals (default: False)

        Returns:
        -------
        BytesIO
            Plot as a BytesIO object for Streamlit
        """
        # Ensure X is 2D array for scatter plot
        X_flat = X.flatten() if X.ndim > 1 else X

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.scatter(X_flat, y, color='blue', alpha=0.6, label='Data')

        # Sort X for smooth curve
        X_sorted = np.sort(X_flat)
        X_sorted_2d = X_sorted.reshape(-1, 1)

        # Create polynomial features for the sorted X values
        model_info = model_results[degree]
        X_poly = model_info['poly'].transform(X_sorted_2d)

        # Make predictions
        y_poly_pred = model_info['model'].predict(X_poly)

        # Plot the fitted curve
        ax.plot(X_sorted, y_poly_pred, color='red', linewidth=2,
                label=f'Polynomial (degree {degree})')

        # Add confidence intervals if requested
        if with_confidence_intervals:
            try:
                # Get the residual standard error from the training data
                y_train_pred = model_info['model'].predict(model_info['poly'].transform(X))
                residuals = y - y_train_pred
                n = len(X)
                p = degree + 1  # Number of parameters
                residual_std = np.sqrt(np.sum(residuals**2) / (n - p))

                # Calculate confidence intervals (95%)
                t_value = stats.t.ppf(0.975, n - p)  # Two-tailed 95% CI

                # Calculate standard error of prediction for each point
                # This is a simplified approach - for exact CI you'd need the full covariance matrix
                pred_std = residual_std * np.sqrt(1 + 1/n)

                # Calculate upper and lower bounds
                upper_bound = y_poly_pred + t_value * pred_std
                lower_bound = y_poly_pred - t_value * pred_std

                # Plot confidence intervals
                ax.fill_between(X_sorted, lower_bound, upper_bound, color='red', alpha=0.2,
                                label='95% Confidence Interval')

            except Exception as e:
                # If confidence intervals fail, just continue without them
                pass

        # Add equation to the plot
        coef = model_info['coefficients']
        intercept = model_info['intercept']

        equation = f"y = {intercept:.3f}"
        for i, c in enumerate(coef[1:]):  # Skip the first coefficient (it's just 1)
            if c >= 0:
                equation += f" + {c:.3f}x^{i+1}"
            else:
                equation += f" - {abs(c):.3f}x^{i+1}"

        # Add R² to the plot
        r2_train = model_info['r2_train']
        r2_test = model_info['r2_test']

        ax.set_title(f'Polynomial Fit (Degree {degree})')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.text(0.05, 0.95, f"Equation: {equation}\nR² (train): {r2_train:.3f}\nR² (test): {r2_test:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend()

        # Save plot to BytesIO
        buf = BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return buf


    @staticmethod
    def plot_hermite_polynomial_fit(X: np.ndarray, y: np.ndarray, model_results: Dict, degree: int,
                                   x_label: str, y_label: str, with_confidence_intervals: bool = False) -> BytesIO:
        """
        Create a plot of the data with the fitted Hermite polynomial curve.

        Parameters:
        ----------
        X : np.ndarray
            Independent variable data, shape (n_samples, 1)
        y : np.ndarray
            Dependent variable data, shape (n_samples,)
        model_results : Dict
            Dictionary with model results
        degree : int
            Polynomial degree to plot
        x_label : str
            Label for the x-axis
        y_label : str
            Label for the y-axis
        with_confidence_intervals : bool
            Whether to plot confidence intervals (default: False)

        Returns:
        -------
        BytesIO
            Plot as a BytesIO object for Streamlit
        """
        # Ensure X is 2D array for scatter plot
        X_flat = X.flatten() if X.ndim > 1 else X

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.scatter(X_flat, y, color='blue', alpha=0.6, label='Data')

        # Sort X for smooth curve
        X_sorted = np.sort(X_flat)
        X_sorted_2d = X_sorted.reshape(-1, 1)

        # Get model info
        model_info = model_results[degree]
        X_mean = model_info['X_mean']
        X_std = model_info['X_std']

        # Standardize the sorted X
        X_standardized = (X_sorted_2d - X_mean) / X_std

        # Create Hermite polynomial features using numpy.polynomial.hermite for better stability
        try:
            from numpy.polynomial.hermite import hermval

            X_hermite = np.zeros((X_standardized.shape[0], degree + 1))
            for d in range(degree + 1):
                # Create coefficient array for polynomial of degree d
                coef = np.zeros(d + 1)
                coef[d] = 1.0
                X_hermite[:, d] = hermval(X_standardized.flatten(), coef)
        except ImportError:
            # Fall back to scipy's eval_hermite if numpy's hermval is not available
            X_hermite = np.zeros((X_standardized.shape[0], degree + 1))
            for d in range(degree + 1):
                X_hermite[:, d] = np.array([eval_hermite(d, x[0]) for x in X_standardized])

        # Make predictions
        y_hermite_pred = model_info['model'].predict(X_hermite)

        # Plot the fitted curve
        ax.plot(X_sorted, y_hermite_pred, color='red', linewidth=2,
                label=f'Hermite polynomial (degree {degree})')

        # Add confidence intervals if requested
        if with_confidence_intervals:
            try:
                # Get the residual standard error
                # First, get predictions for the original data points
                X_orig_standardized = (X - X_mean) / X_std
                X_orig_hermite = np.zeros((X_orig_standardized.shape[0], degree + 1))

                try:
                    # Use the more stable numpy implementation if possible
                    for d in range(degree + 1):
                        coef = np.zeros(d + 1)
                        coef[d] = 1.0
                        X_orig_hermite[:, d] = hermval(X_orig_standardized.flatten(), coef)
                except NameError:
                    # Fall back to scipy
                    for d in range(degree + 1):
                        X_orig_hermite[:, d] = np.array([eval_hermite(d, x[0]) for x in X_orig_standardized])

                y_train_pred = model_info['model'].predict(X_orig_hermite)
                residuals = y - y_train_pred
                n = len(X)
                p = degree + 1  # Number of parameters
                residual_std = np.sqrt(np.sum(residuals**2) / (n - p))

                # Calculate confidence intervals (95%)
                t_value = stats.t.ppf(0.975, n - p)

                # Calculate standard error of prediction for each point
                pred_std = residual_std * np.sqrt(1 + 1/n)

                # Calculate upper and lower bounds
                upper_bound = y_hermite_pred + t_value * pred_std
                lower_bound = y_hermite_pred - t_value * pred_std

                # Plot confidence intervals
                ax.fill_between(X_sorted, lower_bound, upper_bound, color='red', alpha=0.2,
                                label='95% Confidence Interval')

            except Exception as e:
                # If confidence intervals fail, just continue without them
                pass

        # Add equation to the plot
        coefs = model_info['coefficients']

        equation = f"y = {coefs[0]:.3f}H₀(x)"
        for i in range(1, len(coefs)):
            if coefs[i] >= 0:
                equation += f" + {coefs[i]:.3f}H₍{i}₎(x)"
            else:
                equation += f" - {abs(coefs[i]):.3f}H₍{i}₎(x)"

        # Add standardization info
        equation += f"\nwhere x = (X - {X_mean:.3f}) / {X_std:.3f}"

        # Add R² to the plot
        r2_train = model_info['r2_train']
        r2_test = model_info['r2_test']

        ax.set_title(f'Hermite Polynomial Fit (Degree {degree})')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.text(0.05, 0.95, f"Equation: {equation}\nR² (train): {r2_train:.3f}\nR² (test): {r2_test:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend()

        # Save plot to BytesIO
        buf = BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return buf


    @staticmethod
    def plot_residual_diagnostics(residuals: np.ndarray, predicted: np.ndarray, independent_var: np.ndarray) -> BytesIO:
        """
        Create diagnostic plots for regression residuals.

        Parameters:
        ----------
        residuals : np.ndarray
            Residual values (actual - predicted)
        predicted : np.ndarray
            Predicted values from the model
        independent_var : np.ndarray
            Independent variable values

        Returns:
        -------
        BytesIO
            Plot as a BytesIO object for Streamlit
        """
        # Create a 2x2 grid of diagnostic plots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Residuals vs Fitted values plot
        axs[0, 0].scatter(predicted, residuals, alpha=0.6)
        axs[0, 0].axhline(y=0, color='r', linestyle='-')
        axs[0, 0].set_xlabel('Fitted values')
        axs[0, 0].set_ylabel('Residuals')
        axs[0, 0].set_title('Residuals vs Fitted')

        # Add a lowess smooth line if scipy.stats has it
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            lowess_y = lowess(residuals, predicted, frac=0.6, it=3, return_sorted=False)
            axs[0, 0].plot(predicted, lowess_y, color='red', linewidth=2)
        except ImportError:
            pass

        # 2. Normal Q-Q plot
        from scipy import stats
        sorted_residuals = np.sort(residuals)
        quantiles = np.arange(1, len(residuals) + 1) / (len(residuals) + 1)
        theoretical_quantiles = stats.norm.ppf(quantiles)

        axs[0, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)

        # Add a reference line
        min_q, max_q = theoretical_quantiles.min(), theoretical_quantiles.max()
        min_r, max_r = sorted_residuals.min(), sorted_residuals.max()
        slope = (max_r - min_r) / (max_q - min_q)
        intercept = min_r - slope * min_q

        ref_line = slope * theoretical_quantiles + intercept
        axs[0, 1].plot(theoretical_quantiles, ref_line, 'r-')
        axs[0, 1].set_xlabel('Theoretical Quantiles')
        axs[0, 1].set_ylabel('Sample Quantiles')
        axs[0, 1].set_title('Normal Q-Q Plot')

        # 3. Scale-Location plot (sqrt of standardized residuals vs fitted values)
        std_resid = residuals / np.std(residuals)
        sqrt_std_resid = np.sqrt(np.abs(std_resid))

        axs[1, 0].scatter(predicted, sqrt_std_resid, alpha=0.6)
        axs[1, 0].set_xlabel('Fitted values')
        axs[1, 0].set_ylabel('√|Standardized residuals|')
        axs[1, 0].set_title('Scale-Location')

        # Add a lowess smooth line if scipy.stats has it
        try:
            lowess_y = lowess(sqrt_std_resid, predicted, frac=0.6, it=3, return_sorted=False)
            axs[1, 0].plot(predicted, lowess_y, color='red', linewidth=2)
        except NameError:
            pass

        # 4. Residuals vs Independent Variable
        axs[1, 1].scatter(independent_var, residuals, alpha=0.6)
        axs[1, 1].axhline(y=0, color='r', linestyle='-')
        axs[1, 1].set_xlabel('Independent Variable')
        axs[1, 1].set_ylabel('Residuals')
        axs[1, 1].set_title('Residuals vs Independent Variable')

        # Add a lowess smooth line if scipy.stats has it
        try:
            lowess_y = lowess(residuals, independent_var, frac=0.6, it=3, return_sorted=False)
            axs[1, 1].plot(independent_var, lowess_y, color='red', linewidth=2)
        except NameError:
            pass

        # Adjust layout
        fig.tight_layout()

        # Save plot to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return buf


