import base64
from typing import TYPE_CHECKING, Dict, Any
import warnings

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermval
from scipy import stats

from wound_analysis.dashboard_components.stochastic_modeling_components.stats_utils import StatsUtils

class CreateDeterministicComponent:


    def __init__(self, df: pd.DataFrame, parent: 'StochasticModelingTab'):
        self.df                   = df
        self.CN                   = parent.CN

        # User defined variables
        self.independent_var      = parent.independent_var
        self.dependent_var        = parent.dependent_var
        self.independent_var_name = parent.independent_var_name
        self.dependent_var_name   = parent.dependent_var_name

        # Instance variables
        self.deterministic_model  = None
        self.residuals            = None
        self.polynomial_degree    = None
        self.deterministic_coefs  = None
        self.polynomial_type      = None  # Variable to track polynomial type

    def render(self) -> Dict[str, Any]:
        """
        Create and display the deterministic component analysis.

        Returns:
            Dict containing the deterministic model, residuals, polynomial degree,
            coefficients, and polynomial type.
        """
        st.subheader("Deterministic Component Analysis")

        with st.container():
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

            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)

            st.write(f"Analyzing the relationship between {self.dependent_var_name} and {self.independent_var_name}")

            # Add an explanation about polynomial types
            with st.expander("About Polynomial Types", expanded=False):
                st.markdown("""
                ### Regular vs. Hermite Polynomials

                **Regular Polynomials**:
                - Standard polynomials in the form: $g(X) = β₀ + β₁X + β₂X² + ... + βₙXⁿ$
                - Good for general trend modeling
                - Simple to interpret

                **Hermite Polynomials**:
                - Orthogonal polynomials defined by: $H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}(e^{-x^2})$
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
                st.markdown("""
                ### Deterministic Component

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
                st.markdown("""
                ### Deterministic Component with Hermite Polynomials

                The deterministic component is modeled using Hermite polynomials:

                $g(X) = c₀H₀(x) + c₁H₁(x) + c₂H₂(x) + ... + cₙHₙ(x)$

                where:
                - $g(X)$ is the predicted value of the dependent variable
                - $x$ is the standardized independent variable: $x = (X - μ_X) / σ_X$
                - $H_n(x)$ is the nth Hermite polynomial
                - $c₀, c₁, ..., cₙ$ are the coefficients
                - $n$ is the degree of the polynomial

                Hermite polynomials are particularly useful for modeling stochastic processes with Gaussian characteristics.
                """)

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
                return {
                    'deterministic_model': None,
                    'residuals'          : None,
                    'polynomial_degree'  : None,
                    'deterministic_coefs': None,
                    'polynomial_type'    : 'regular'
                }

            # Check for extreme values that might cause numerical issues
            if np.abs(X).max() > 1e5 or np.abs(y).max() > 1e5:
                st.warning("""
                ⚠️ Warning: Your data contains extreme values which may cause numerical instability.
                Consider preprocessing your data (e.g., scaling or log-transformation) before analysis.
                """)

            # Convert to appropriate shape for sklearn
            X = X.reshape(-1, 1)

            # Fit polynomial models of different degrees
            try:
                with st.spinner("Fitting polynomial models..."):
                    max_degree = 5
                    if self.polynomial_type == "Regular Polynomial":
                        model_results = StatsUtils.fit_polynomial_models(X=X, y=y, max_degree=max_degree)
                    else:
                        model_results = StatsUtils.fit_hermite_polynomial_models(X=X, y=y, max_degree=max_degree)
            except Exception as e:
                st.error(f"Error fitting polynomial models: {str(e)}")
                st.info("Try using a different polynomial type or inspect your data for irregularities.")
                return {
                    'deterministic_model': None,
                    'residuals'          : None,
                    'polynomial_degree'  : None,
                    'deterministic_coefs': None,
                    'polynomial_type'    : 'regular' if self.polynomial_type == "Regular Polynomial" else 'hermite'
                }

            # Display polynomial degree selection
            col1, col2 = st.columns([1, 1])

            with col1:
                selected_degree = st.slider("Select Polynomial Degree",
                                           min_value=1, max_value=max_degree,
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

            # Display model metrics for different degrees
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

            with col2:
                st.markdown("### Model Selection Metrics")
                st.write("Lower AIC and BIC values indicate better models, balancing fit and complexity.")
                best_aic = metrics_df['AIC'].idxmin() + 1
                best_bic = metrics_df['BIC'].idxmin() + 1

                # Find optimal degree balancing fit and complexity
                overfitting_threshold = 0.1  # Threshold for considering a model as overfitting
                potential_degrees = metrics_df[metrics_df['Overfitting'] < overfitting_threshold]

                if not potential_degrees.empty:
                    optimal_degree = potential_degrees['AIC'].idxmin() + 1
                else:
                    optimal_degree = best_aic

                st.write(f"Best degree by AIC: {best_aic}")
                st.write(f"Best degree by BIC: {best_bic}")
                st.write(f"Recommended degree (balancing fit and overfitting): {optimal_degree}")

                # Add recommendation highlighting if different from selected
                if optimal_degree != selected_degree:
                    st.info(f"💡 Consider using degree {optimal_degree} to balance fit quality and model complexity.")

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

            # Highlight multiple metrics in the dataframe
            st.dataframe(metrics_df.style
                        .highlight_min(subset=['AIC', 'BIC'], color='lightgreen')
                        .highlight_max(subset=['R² (Test)'], color='lightblue')
                        .highlight_between(subset=['Overfitting'], left=0, right=0.1, color='lightyellow'))

            # Display the polynomial plot
            st.markdown("### Polynomial Fit Visualization")

            try:
                if self.polynomial_type == "hermite":
                    buf = StatsUtils.plot_hermite_polynomial_fit(
                        X=X, y=y, model_results=model_results, degree=selected_degree,
                        x_label=self.independent_var_name, y_label=self.dependent_var_name,
                        with_confidence_intervals=True  # New parameter for confidence intervals
                    )
                else:
                    buf = StatsUtils.plot_polynomial_fit(
                        X=X, y=y, model_results=model_results, degree=selected_degree,
                        x_label=self.independent_var_name, y_label=self.dependent_var_name,
                        with_confidence_intervals=True  # New parameter for confidence intervals
                    )

                if buf is not None:
                    st.image(buf, caption=f"{'Hermite' if self.polynomial_type == 'hermite' else 'Regular'} Polynomial Fit (Degree {selected_degree})")
                else:
                    st.warning("Could not generate plot. Try a different polynomial degree or type.")
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")

            # Display equation and coefficients in a nicely formatted HTML block
            st.markdown("<h3 style='color: #2c3e50;'>Model Equation</h3>", unsafe_allow_html=True)

            # Get coefficients and format equation
            selected_model = model_results[selected_degree]

            try:
                coef = selected_model['coefficients']
                intercept = selected_model['intercept'] if 'intercept' in selected_model else 0

                # Store for later use
                self.deterministic_coefs = (intercept, coef)

                # Format equation using HTML styling
                if self.polynomial_type == "hermite":
                    equation = f"g({self.independent_var_name}) = {coef[0]:.4f}H₀(x)"
                    for i in range(1, len(coef)):
                        if coef[i] >= 0:
                            equation += f" + {coef[i]:.4f}H₍{i}₎(x)"
                        else:
                            equation += f" - {abs(coef[i]):.4f}H₍{i}₎(x)"

                    # Add standardization info
                    equation += f"<br>where x = ({self.independent_var_name} - {selected_model['X_mean']:.4f}) / {selected_model['X_std']:.4f}"
                else:
                    equation = f"g({self.independent_var_name}) = {intercept:.4f}"
                    for i, c in enumerate(coef[1:]):  # Skip the intercept term
                        term = f"{abs(c):.4f} {self.independent_var_name.replace(' ', '_')}<sup>{i+1}</sup>"
                        if c >= 0:
                            equation += f" + {term}"
                        else:
                            equation += f" - {term}"

                # Display the formatted equation in a styled container
                st.markdown(
                    f"<div style='background-color: #f0f8ff; border-left: 4px solid #2c3e50; padding: 10px; border-radius: 5px; font-size: 1.1em;'>"
                    f"{equation}"
                    f"</div>", unsafe_allow_html=True)

                # Prepare coefficient table
                if self.polynomial_type == "hermite":
                    coef_df = pd.DataFrame({
                        'Term': [f"H_{i}(x)" for i in range(len(coef))],
                        'Coefficient': coef
                    })
                else:
                    coef_df = pd.DataFrame({
                        'Term': [f"{self.independent_var_name}^{i}" if i > 0 else "Intercept"
                                for i in range(len(coef))],
                        'Coefficient': [intercept] + list(coef[1:])
                    })
            except Exception as e:
                st.error(f"Error generating equation: {str(e)}")
                coef_df = pd.DataFrame()

            # Calculate residuals for random component analysis
            try:
                if self.polynomial_type == "hermite":
                    # For Hermite polynomials - using numpy's hermval for better stability
                    X_mean = selected_model['X_mean']
                    X_std = selected_model['X_std']
                    X_standardized = (X - X_mean) / X_std

                    # More stable approach using numpy's hermval
                    y_pred = np.zeros(X_standardized.shape[0])
                    for i in range(selected_degree + 1):
                        coef_array = np.zeros(i + 1)
                        coef_array[i] = 1.0
                        y_pred += coef[i] * hermval(X_standardized.flatten(), coef_array)
                else:
                    # For regular polynomials
                    X_poly = selected_model['poly'].transform(X)
                    y_pred = selected_model['model'].predict(X_poly)

                residuals = y - y_pred

                # Store residuals for later use
                self.residuals = residuals

                # Add residual diagnostics
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
                            residuals=residuals,
                            predicted=y_pred,
                            independent_var=X.flatten()
                        )
                        if buf is not None:
                            st.image(buf, caption="Residual Diagnostic Plots")
                    except Exception as plot_err:
                        st.warning(f"Could not generate residual plots: {str(plot_err)}")

            except Exception as e:
                st.error(f"Error calculating residuals: {str(e)}")
                self.residuals = np.zeros_like(y)  # Default to zeros on error

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

            st.markdown('</div>', unsafe_allow_html=True)

        return {
            'deterministic_model': self.deterministic_model,
            'residuals'          : self.residuals,
            'polynomial_degree'  : self.polynomial_degree,
            'deterministic_coefs': self.deterministic_coefs,
            'polynomial_type'    : self.polynomial_type
        }
