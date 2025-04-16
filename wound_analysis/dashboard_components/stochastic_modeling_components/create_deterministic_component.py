import base64
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional, List, Union
import warnings
from dataclasses import dataclass

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermval
from scipy import stats

from wound_analysis.dashboard_components.stochastic_modeling_components.stats_utils import StatsUtils


@dataclass
class DeterministicComponentResult:
    """Data class to store the results of deterministic component analysis."""
    deterministic_model: Any
    residuals          : Optional[np.ndarray]
    polynomial_degree  : int
    deterministic_coefs: Optional[Tuple]
    polynomial_type    : str


class CreateDeterministicComponent:
    """
    Handles creation and visualization of deterministic components for stochastic modeling.

    This class allows users to fit polynomial models (regular or Hermite) to data,
    visualize the fits, and analyze the results.
    """

    def __init__(self, df: pd.DataFrame, parent: 'StochasticModelingTab'):
        """
        Initialize the component with data and parent references.

        Args:
            df: The dataframe containing wound data
            parent: Reference to the parent tab for accessing variables
        """
        self.df = df
        self.CN = parent.CN

        # User defined variables
        self.independent_var = parent.independent_var
        self.dependent_var = parent.dependent_var
        self.independent_var_name = parent.independent_var_name
        self.dependent_var_name = parent.dependent_var_name

        # Instance variables
        self.deterministic_model = None
        self.residuals = None
        self.polynomial_degree = None
        self.deterministic_coefs = None
        self.polynomial_type = None  # Variable to track polynomial type


    def render(self) -> Dict[str, Any]:
        """
        Create and display the deterministic component analysis.

        Returns:
            Dict containing the deterministic model, residuals, polynomial degree,
            coefficients, and polynomial type.
        """
        st.subheader("Deterministic Component Analysis")

        with st.container():
            self._setup_page_style()
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)

            st.write(f"Analyzing the relationship between {self.dependent_var_name} and {self.independent_var_name}")

            # Show explanations and setup polynomial type
            self._show_polynomial_explanations()

            # Get data, preprocess, and validate
            X, y = self._prepare_data()
            if X is None or y is None:
                return self._empty_result()

            # Fit polynomial models
            model_results = self._fit_polynomial_models(X, y)
            if not model_results:
                return self._empty_result()

            # Display degree selection and metrics
            self._display_degree_selection_and_metrics(model_results, X, y)

            # Plot and display the model
            self._plot_and_display_model(X, y, model_results)

            # Calculate and display residuals
            self._calculate_and_display_residuals(X, y, model_results)

            # Add download options
            self._add_download_options()

            st.markdown('</div>', unsafe_allow_html=True)

        return {
            'deterministic_model': self.deterministic_model,
            'residuals'          : self.residuals,
            'polynomial_degree'  : self.polynomial_degree,
            'deterministic_coefs': self.deterministic_coefs,
            'polynomial_type'    : self.polynomial_type
        }


    def _empty_result(self) -> Dict[str, Any]:
        """
        Return an empty result when analysis cannot be completed.

        Returns:
            Dictionary with None/default values
        """
        return {
            'deterministic_model': None,
            'residuals'          : None,
            'polynomial_degree'  : None,
            'deterministic_coefs': None,
            'polynomial_type'    : 'regular' if self.polynomial_type == "Regular Polynomial" else 'hermite'
        }

    def _setup_page_style(self) -> None:
        """Set up CSS styles for the analysis section."""
        st.markdown("""
        <style>
        .analysis-section {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            background-color: #f8f9fa;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    def _show_polynomial_explanations(self) -> None:
        """Display explanations about polynomial types and allow user selection."""
        # Add an explanation about polynomial types
        with st.expander("About Polynomial Types", expanded=False):
            st.markdown("""
            ### Regular vs. Hermite Polynomials

            **Regular Polynomials**:
            - Standard polynomials in the form: $g(X) = β₀ + β₁X + β₂X² + ... + βₙXⁿ$
            - Good for general trend modeling
            - Simple to interpret

            **Hermite Polynomials**:
            - Orthogonal polynomials defined by: $H_n(x) = (-1)^n e^{x^2} (d^n / dx^n)(e^{-x^2})$
            - More suitable for modeling data with Gaussian characteristics
            - Better for capturing oscillatory behavior in the data
            - Particularly useful in stochastic modeling as they relate to derivatives of the normal probability density function
            - Naturally emerge in quantum mechanics and random process modeling
            """)

        # Add polynomial type selection
        self.polynomial_type = st.radio(
            "Select Polynomial Type",
            options=["Regular Polynomial", "Hermite Polynomial"],
            index=0 if st.session_state.get('polynomial_type') != 'hermite' else 1,
            horizontal=True,
            help="Hermite polynomials are particularly useful for modeling data with Gaussian characteristics"
        )

        # Store the polynomial type in session state
        st.session_state['polynomial_type'] = 'regular' if self.polynomial_type == "Regular Polynomial" else 'hermite'

        # Display the appropriate polynomial description
        if self.polynomial_type == "Regular Polynomial":
            st.markdown("### Deterministic Component")
            with st.expander("About Regular Polynomials", expanded=True):
                st.markdown("""
                The deterministic component represents the expected trend in the data without considering
                the random variability. It is modeled as a polynomial function:

                $g(X) = β₀ + β₁X + β₂X² + ... + βₙXⁿ$

                where:
                - $g(X)$ is the predicted value of the dependent variable
                - $X$ is the independent variable
                - $β₀, β₁, ..., βₙ$ are the polynomial coefficients
                - $n$ is the degree of the polynomial
                """)
        else:
            st.markdown("### Deterministic Component with Hermite Polynomials")
            with st.expander("About Hermite Polynomials", expanded=True):
                st.markdown("""
                The deterministic component is modeled using Hermite polynomials:

                $g(X) = c₀H₀(x) + c₁H₁(x) + c₂H₂(x) + ... + cₙHₙ(x)$

                where:
                - $g(X)$ is the predicted value of the dependent variable
                - $x$ is the standardized independent variable: $x = (X - μ_X) / σ_X$
                - $H_n(x)$ is the $n$th Hermite polynomial
                - $c₀, c₁, ..., cₙ$ are the coefficients
                - $n$ is the degree of the polynomial

                The Hermite polynomials are defined recursively as:
                - $H_0(x) = 1$
                - $H_1(x) = 2x$
                - $H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)$

                Hermite polynomials are particularly useful for modeling stochastic processes with Gaussian characteristics.
                """)

    def _prepare_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare the data for analysis by extracting and preprocessing.

        Returns:
            Tuple of (X, y) arrays, or (None, None) if data preparation fails
        """
        # Get data for variables
        X = self.df[self.independent_var].values
        y = self.df[self.dependent_var].values

        # Remove rows with NaN values
        mask = ~(np.isnan(X) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Check for sufficient data points
        if len(X) < 5:
            st.error(f"Not enough valid data points to perform polynomial fitting (minimum 5 required, got {len(X)})")
            return None, None

        # Check for extreme values that might cause numerical issues
        if np.abs(X).max() > 1e5 or np.abs(y).max() > 1e5:
            st.warning("""
            ⚠️ Warning: Your data contains extreme values which may cause numerical instability.
            Consider preprocessing your data (e.g., scaling or log-transformation) before analysis.
            """)

        # Convert to appropriate shape for sklearn
        X = X.reshape(-1, 1)
        return X, y

    def _fit_polynomial_models(self, X: np.ndarray, y: np.ndarray, max_degree: int = 5) -> Optional[Dict]:
        """
        Fit polynomial models to the data.

        Args:
            X: Independent variable data
            y: Dependent variable data
            max_degree: Maximum polynomial degree to fit

        Returns:
            Dictionary of model results or None if fitting fails
        """
        try:
            with st.spinner("Fitting polynomial models..."):
                # Check for data suitability before fitting
                if self.polynomial_type == "Hermite Polynomial":
                    # Check if data might be suitable for Hermite polynomials
                    # Hermite polynomials work best with data that has Gaussian-like characteristics
                    shapiro_stat, shapiro_p = stats.shapiro(y) if len(y) <= 5000 else (None, None)
                    if shapiro_p is not None and shapiro_p < 0.01:
                        st.info("Note: Your data may not follow a normal distribution. Hermite polynomials work best with data that has Gaussian characteristics.")

                    # Check for extreme values that might cause numerical instability
                    if np.std(X) > 0 and np.max(np.abs(X - np.mean(X)) / np.std(X)) > 5:
                        st.info("Your data contains extreme outliers which may affect the stability of Hermite polynomials. Consider removing outliers first.")

                    return StatsUtils.fit_hermite_polynomial_models(X=X, y=y, max_degree=max_degree)
                else:
                    return StatsUtils.fit_polynomial_models(X=X, y=y, max_degree=max_degree)

        except Exception as e:
            st.error(f"Error fitting polynomial models: {str(e)}")
            st.info("Try using a different polynomial type or inspect your data for irregularities.")
            return None

    def _create_metrics_dataframe(self, model_results: Dict, max_degree: int = 5) -> pd.DataFrame:
        """
        Create a dataframe with model metrics for different polynomial degrees.

        Args:
            model_results: Dictionary with model results
            max_degree: Maximum polynomial degree

        Returns:
            DataFrame with model metrics
        """
        metrics_df = pd.DataFrame({
            'Degree'     : list(range(1, max_degree + 1)),
            'R² (Train)' : [model_results[d]['r2_train'] for d in range(1, max_degree + 1)],
            'R² (Test)'  : [model_results[d]['r2_test'] for d in range(1, max_degree + 1)],
            'MSE (Train)': [model_results[d]['mse_train'] for d in range(1, max_degree + 1)],
            'MSE (Test)' : [model_results[d]['mse_test'] for d in range(1, max_degree + 1)],
            'AIC'        : [model_results[d]['aic'] for d in range(1, max_degree + 1)],
            'BIC'        : [model_results[d]['bic'] for d in range(1, max_degree + 1)]
        })

        # Add overfitting metric
        metrics_df['Overfitting'] = metrics_df['R² (Train)'] - metrics_df['R² (Test)']
        return metrics_df

    def _find_optimal_degree(self, metrics_df: pd.DataFrame, overfitting_threshold: float = 0.1) -> int:
        """
        Find the optimal polynomial degree balancing fit quality and overfitting.

        Args:
            metrics_df: DataFrame with model metrics
            overfitting_threshold: Threshold for considering a model as overfitting

        Returns:
            Optimal polynomial degree
        """
        potential_degrees = metrics_df[metrics_df['Overfitting'] < overfitting_threshold]

        if not potential_degrees.empty:
            return potential_degrees['AIC'].idxmin() + 1
        else:
            return metrics_df['AIC'].idxmin() + 1

    def _display_degree_selection_and_metrics(self, model_results: Dict, X: np.ndarray, y: np.ndarray) -> None:
        """
        Display polynomial degree selection UI and model metrics.

        Args:
            model_results: Dictionary with model results
            X: Independent variable data
            y: Dependent variable data
        """
        # Create metrics dataframe
        metrics_df = self._create_metrics_dataframe(model_results)

        # Display polynomial degree selection
        col1, col2 = st.columns([1, 1])

        with col1:
            selected_degree = st.slider("Select Polynomial Degree",
                                       min_value=1, max_value=5,
                                       value=st.session_state.polynomial_degree if st.session_state.polynomial_degree is not None else 2,
                                       step=1,
                                       help="Higher degrees can fit the data better but may overfit",
                                       key="polynomial_degree_slider")

        # Store the selected model and parameters in session state
        st.session_state.deterministic_model = model_results[selected_degree]
        st.session_state.polynomial_degree = selected_degree

        # Update instance variables
        self.deterministic_model = model_results[selected_degree]
        self.polynomial_degree = st.session_state.polynomial_degree
        self.polynomial_type = "hermite" if self.polynomial_type == "Hermite Polynomial" else "regular"

        # Display model metrics
        with col2:
            st.markdown("### Model Selection Metrics")
            st.write("Lower AIC and BIC values indicate better models, balancing fit and complexity.")
            best_aic = metrics_df['AIC'].idxmin() + 1
            best_bic = metrics_df['BIC'].idxmin() + 1

            # Find optimal degree balancing fit and complexity
            optimal_degree = self._find_optimal_degree(metrics_df)

            st.write(f"Best degree by AIC: {best_aic}")
            st.write(f"Best degree by BIC: {best_bic}")
            st.write(f"Recommended degree (balancing fit and overfitting): {optimal_degree}")

            # Add recommendation highlighting if different from selected
            if optimal_degree != selected_degree:
                st.info(f"💡 Consider using degree {optimal_degree} to balance fit quality and model complexity.")

        self._show_metrics_interpretation()

        # Highlight multiple metrics in the dataframe
        st.dataframe(metrics_df.style
                    .highlight_min(subset=['AIC', 'BIC'], color='lightgreen')
                    .highlight_max(subset=['R² (Test)'], color='lightblue')
                    .highlight_between(subset=['Overfitting'], left=0, right=0.1, color='lightyellow'))

    def _show_metrics_interpretation(self) -> None:
        """Display explanation of model metrics."""
        with st.expander("How to Interpret Model Metrics"):
            st.markdown("""
            - **R² (R-squared)**: Measures the proportion of variance in the dependent variable that's explained by the model.
              - Values range from 0 to 1, where 1 indicates perfect fit
              - Higher values generally indicate better fit, but beware of overfitting
              - Compare train vs test R² to check for overfitting

            - **MSE (Mean Squared Error)**: Measures the average squared difference between predicted and actual values
              - Lower values indicate better fit
              - More sensitive to outliers than R²
              - Compare train vs test MSE to check for overfitting

            - **AIC (Akaike Information Criterion)**: Measures model quality while penalizing complexity
              - Lower values indicate better model
              - Useful for comparing models with different complexities
              - Prefers simpler models that fit well

            - **BIC (Bayesian Information Criterion)**: Similar to AIC but with stronger penalty for complexity
              - Lower values indicate better model
              - More conservative than AIC in selecting complex models
              - Prefers simpler models more strongly than AIC

            - **Overfitting**: Difference between training and test R²
              - Higher values indicate the model fits training data much better than test data
              - Values > 0.1 suggest overfitting
              - Ideally, choose the highest degree with overfitting < 0.1
            """)

    def _plot_and_display_model(self, X: np.ndarray, y: np.ndarray, model_results: Dict) -> None:
        """
        Plot and display the polynomial model.

        Args:
            X: Independent variable data
            y: Dependent variable data
            model_results: Dictionary with model results
        """
        selected_degree = self.polynomial_degree

        # Display the polynomial plot
        st.markdown("### Polynomial Fit Visualization")

        try:
            if self.polynomial_type == "hermite":
                buf = StatsUtils.plot_hermite_polynomial_fit(
                    X=X, y=y, model_results=model_results, degree=selected_degree,
                    x_label=self.independent_var_name, y_label=self.dependent_var_name,
                    with_confidence_intervals=True
                )
            else:
                buf = StatsUtils.plot_polynomial_fit(
                    X=X, y=y, model_results=model_results, degree=selected_degree,
                    x_label=self.independent_var_name, y_label=self.dependent_var_name,
                    with_confidence_intervals=True
                )

            if buf is not None:
                st.image(buf, caption=f"{'Hermite' if self.polynomial_type == 'hermite' else 'Regular'} Polynomial Fit (Degree {selected_degree})")
            else:
                st.warning("Could not generate plot. Try a different polynomial degree or type.")
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")

        # Display equation and coefficients
        self._display_equation(model_results[selected_degree])

    def _display_equation(self, model_info: Dict) -> None:
        """
        Display the model equation and coefficients.

        Args:
            model_info: Information about the selected model
        """
        st.markdown("<h3 style='color: #2c3e50;'>Model Equation</h3>", unsafe_allow_html=True)

        try:
            # Get coefficients and format equation
            coef = model_info['coefficients']
            intercept = model_info['intercept'] if 'intercept' in model_info else 0

            # Store for later use
            self.deterministic_coefs = (intercept, coef)

            # Format equation using HTML styling
            if self.polynomial_type == "hermite":
                equation = self._format_hermite_equation(coef, model_info)
            else:
                equation = self._format_regular_equation(intercept, coef)

            # Display the formatted equation in a styled container
            st.markdown(
                f"<div style='background-color: #f0f8ff; border-left: 4px solid #2c3e50; padding: 10px; border-radius: 5px; font-size: 1.1em;'>"
                f"{equation}"
                f"</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating equation: {str(e)}")

    def _format_hermite_equation(self, coef: np.ndarray, model_info: Dict) -> str:
        """
        Format a Hermite polynomial equation for display.

        Args:
            coef: Coefficient array
            model_info: Model information dictionary

        Returns:
            Formatted equation string with HTML
        """
        # Filter out very small coefficients for cleaner display
        significant_threshold = 1e-4

        equation = f"g({self.independent_var_name}) = {coef[0]:.4f}H₀(x)"
        for i in range(1, len(coef)):
            # Skip very small coefficients for cleaner display
            if abs(coef[i]) < significant_threshold:
                continue

            if coef[i] >= 0:
                equation += f" + {coef[i]:.4f}H₍{i}₎(x)"
            else:
                equation += f" - {abs(coef[i]):.4f}H₍{i}₎(x)"

        # Add standardization info
        X_mean = model_info['X_mean']
        X_std = model_info['X_std']
        equation += f"<br>where x = ({self.independent_var_name} - {X_mean:.4f}) / {X_std:.4f}"

        # Add explanation of Hermite polynomials for reference
        equation += f"<br><small>H₀(x) = 1, H₁(x) = 2x, H₂(x) = 4x² - 2, H₃(x) = 8x³ - 12x, ...</small>"
        return equation

    def _format_regular_equation(self, intercept: float, coef: np.ndarray) -> str:
        """
        Format a regular polynomial equation for display.

        Args:
            intercept: Intercept value
            coef: Coefficient array

        Returns:
            Formatted equation string with HTML
        """
        equation = f"g({self.independent_var_name}) = {intercept:.4f}"
        for i, c in enumerate(coef[1:]):  # Skip the intercept term
            term = f"{abs(c):.4f} {self.independent_var_name.replace(' ', '_')}<sup>{i+1}</sup>"
            if c >= 0:
                equation += f" + {term}"
            else:
                equation += f" - {term}"

        return equation

    def _calculate_and_display_residuals(self, X: np.ndarray, y: np.ndarray, model_results: Dict) -> None:
        """
        Calculate and display residuals analysis.

        Args:
            X: Independent variable data
            y: Dependent variable data
            model_results: Dictionary with model results
        """
        selected_degree = self.polynomial_degree
        selected_model = model_results[selected_degree]

        try:
            if self.polynomial_type == "hermite":
                y_pred, residuals = self._calculate_hermite_residuals(X, y, selected_model)
            else:
                y_pred, residuals = self._calculate_regular_residuals(X, y, selected_model)

            # Store residuals for later use
            self.residuals = residuals

            # Add residual diagnostics
            self._display_residual_diagnostics(residuals, y_pred, X.flatten())

        except Exception as e:
            st.error(f"Error calculating residuals: {str(e)}")
            self.residuals = np.zeros_like(y)  # Default to zeros on error

    def _calculate_hermite_residuals(self, X: np.ndarray, y: np.ndarray, model_info: Dict) -> Tuple[np.ndarray, np.ndarray]:

        """This method computes predictions and residuals for a Hermite polynomial model.
        It standardizes the input data, applies the Hermite polynomial using the stored
        coefficients, and returns both the predictions and the difference between actual
        and predicted values.

        The method includes robust error handling and fallback calculations in case
        the vectorized approach fails. It also includes warnings for potential numerical
        instability cases.

            X (np.ndarray): Independent variable data (input features)
            y (np.ndarray): Dependent variable data (target values)
            model_info (Dict): Dictionary containing model parameters including:
                - 'X_mean'      : Mean of training X values
                - 'X_std'       : Standard deviation of training X values
                - 'coefficients': Hermite polynomial coefficients

            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - y_pred   : Predicted values from the Hermite polynomial model
                - residuals: Difference between actual and predicted values (y - y_pred)

        Raises:
            Exception: Caught internally with fallback to zeros arrays

        """
        try:
            # For Hermite polynomials - using numpy's hermval for better stability
            X_mean = model_info['X_mean']
            X_std  = model_info['X_std']

            # Check for division by zero
            if np.isclose(X_std, 0):
                st.warning("Standard deviation of X is near zero. Using a small value to avoid division by zero.")
                X_std = 1e-8

            X_standardized = (X - X_mean) / X_std
            coef           = model_info['coefficients']

            # More stable approach using numpy's hermval directly with all coefficients at once
            # This is more efficient and numerically stable than calculating each term separately
            try:
                # First try the direct vectorized approach
                y_pred = hermval(X_standardized.flatten(), coef)

            except Exception:
                # Fall back to term-by-term calculation if the vectorized approach fails
                y_pred = np.zeros(X_standardized.shape[0])

                # Check for numerical stability
                if np.any(np.abs(X_standardized) > 10):
                    st.warning("Some standardized values are very large. This may cause numerical instability in Hermite polynomials.")

                for i in range(self.polynomial_degree + 1):
                    coef_array = np.zeros(i + 1)
                    coef_array[i] = 1.0
                    y_pred += coef[i] * hermval(X_standardized.flatten(), coef_array)

            return y_pred, y - y_pred

        except Exception as e:
            st.error(f"Error calculating Hermite residuals: {str(e)}")
            return np.zeros_like(y), np.zeros_like(y)

    def _calculate_regular_residuals(self, X: np.ndarray, y: np.ndarray, model_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate residuals for regular polynomial models.

        Args:
            X: Independent variable data
            y: Dependent variable data
            model_info: Model information dictionary

        Returns:
            Tuple of (predictions, residuals)
        """
        # For regular polynomials
        X_poly = model_info['poly'].transform(X)
        y_pred = model_info['model'].predict(X_poly)

        return y_pred, y - y_pred

    def _display_residual_diagnostics(self, residuals: np.ndarray, predicted: np.ndarray, independent_var: np.ndarray) -> None:
        """
        Display residual diagnostics.

        Args:
            residuals: Residual values
            predicted: Predicted values
            independent_var: Independent variable values
        """
        with st.expander("Residual Diagnostics"):
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) <= 5000 else (None, None)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean of Residuals", f"{mean_residual:.4f}")
                st.metric("Std. Dev. of Residuals", f"{std_residual:.4f}")

            with col2:
                if shapiro_p is not None:
                    st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4f}")
                    if shapiro_p < 0.05:
                        st.warning("Residuals may not be normally distributed (p < 0.05)")
                    else:
                        st.success("Residuals appear normally distributed (p >= 0.05)")
                else:
                    st.info("Shapiro-Wilk test not calculated (dataset too large)")

            # Add residual plots
            try:
                buf = StatsUtils.plot_residual_diagnostics(
                                            residuals       = residuals,
                                            predicted       = predicted,
                                            independent_var = independent_var )
                if buf is not None:
                    st.image(buf, caption="Residual Diagnostic Plots")
            except Exception as plot_err:
                st.warning(f"Could not generate residual plots: {str(plot_err)}")

    def _get_coefficient_dataframe(self, model_info: Dict) -> pd.DataFrame:
        """
        Create a dataframe of model coefficients.

        Args:
            model_info: Model information dictionary

        Returns:
            DataFrame with coefficient information or empty DataFrame on error
        """
        try:
            coef = model_info['coefficients']
            intercept = model_info['intercept'] if 'intercept' in model_info else 0

            if self.polynomial_type == "hermite":
                return pd.DataFrame({
                    'Term': [f"H_{i}(x)" for i in range(len(coef))],
                    'Coefficient': coef
                })
            else:
                return pd.DataFrame({
                    'Term': [f"{self.independent_var_name}^{i}" if i > 0 else "Intercept"
                            for i in range(len(coef))],
                    'Coefficient': [intercept] + list(coef[1:])
                })
        except Exception:
            return pd.DataFrame()

    def _add_download_options(self) -> None:
        """Add download buttons for metrics and coefficients."""
        # Get model info
        if not self.deterministic_model:
            return

        # Create metrics dataframe
        metrics_df = self._create_metrics_dataframe({i: self.deterministic_model for i in range(1, 6)})

        # Create coefficient dataframe
        coef_df = self._get_coefficient_dataframe(self.deterministic_model)

        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            # Download model metrics
            csv = metrics_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="polynomial_model_metrics.csv">Download Model Metrics (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)

        with col2:
            # Download coefficients
            if not coef_df.empty:
                csv = coef_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="polynomial_coefficients.csv">Download Coefficients (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
