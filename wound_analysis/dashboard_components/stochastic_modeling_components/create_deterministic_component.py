import base64
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab

import numpy as np
import pandas as pd
import streamlit as st

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

    def render(self) -> Dict[str, Any]:
        """
        Create and display the deterministic component analysis.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame to analyze
        dependent_var : str
            Column name of dependent variable
        independent_var : str
            Column name of independent variable
        dependent_var_name : str
            Display name of dependent variable
        independent_var_name : str
            Display name of independent variable
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

            # Get data for variables
            X = self.df[self.independent_var].values
            y = self.df[self.dependent_var].values

            # Remove rows with NaN values
            mask = ~(np.isnan(X) | np.isnan(y))
            X = X[mask]
            y = y[mask]

            if len(X) < 5:
                st.error(f"Not enough valid data points to perform polynomial fitting (minimum 5 required, got {len(X)})")
                return

            # Convert to appropriate shape for sklearn
            X = X.reshape(-1, 1)

            # Fit polynomial models of different degrees
            with st.spinner("Fitting polynomial models..."):
                max_degree = 5
                model_results = StatsUtils.fit_polynomial_models(X=X, y=y, max_degree=max_degree)

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

            with col2:
                st.markdown("### Model Selection Metrics")
                st.write("Lower AIC and BIC values indicate better models, balancing fit and complexity.")
                best_aic = metrics_df['AIC'].idxmin() + 1
                best_bic = metrics_df['BIC'].idxmin() + 1
                st.write(f"Best degree by AIC: {best_aic}")
                st.write(f"Best degree by BIC: {best_bic}")

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
                """)


            st.dataframe(metrics_df.style.highlight_min(subset=['AIC', 'BIC'], color='lightgreen'))

            # Display the polynomial plot
            st.markdown("### Polynomial Fit Visualization")

            buf = StatsUtils.plot_polynomial_fit(X=X, y=y, model_results=model_results, degree=selected_degree,
                                           x_label=self.independent_var_name, y_label=self.dependent_var_name)

            if buf is not None:
                st.image(buf, caption=f"Polynomial Fit (Degree {selected_degree})")

            # Display equation and coefficients in a nicely formatted HTML block
            st.markdown("<h3 style='color: #2c3e50;'>Model Equation</h3>", unsafe_allow_html=True)

            # Get coefficients and format equation
            selected_model = model_results[selected_degree]
            coef           = selected_model['coefficients']
            intercept      = selected_model['intercept']

            # Store for later use
            self.deterministic_coefs = (intercept, coef)

            # Format equation using HTML styling (using <sup> for exponents)
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

            # Display coefficient table
            # st.markdown("### Coefficient Values")

            coef_df = pd.DataFrame({
                'Term': [f"{self.independent_var_name}^{i}" if i > 0 else "Intercept"
                        for i in range(len(coef))],
                'Coefficient': [intercept] + list(coef[1:])
            })

            # st.dataframe(coef_df)

            # Calculate residuals for random component analysis
            X_poly = selected_model['poly'].transform(X)
            y_pred = selected_model['model'].predict(X_poly)
            residuals = y - y_pred

            # Store residuals for later use
            self.residuals = residuals

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
                csv = coef_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="polynomial_coefficients.csv">Download Coefficients (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        return {'deterministic_model': self.deterministic_model,
                'residuals'          : self.residuals,
                'polynomial_degree'  : self.polynomial_degree,
                'deterministic_coefs': self.deterministic_coefs}
