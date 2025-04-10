from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import stats
from sklearn.base import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st


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
    def plot_polynomial_fit(X: np.ndarray, y: np.ndarray, model_results: Dict, degree: int,
                            x_label: str, y_label: str) -> BytesIO:
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


