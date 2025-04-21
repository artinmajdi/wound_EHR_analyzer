"""
Model visualization components for stochastic modeling.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from ...utils.stochastic_modeling import PolynomialModeler
from scipy import stats


class ModelViewer:
    """
    A class for creating interactive visualizations of deterministic
    and probabilistic models.
    """

    def __init__(self, key_prefix: str = ""):
        """
        Initialize the model viewer.

        Parameters:
        ----------
        key_prefix : str, optional
            Prefix for Streamlit widget keys to avoid conflicts
        """
        self.key_prefix = key_prefix
        self.modeler = PolynomialModeler()

    def display_model_analysis(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             x_name: str,
                             y_name: str,
                             max_degree: int = 5,
                             criterion: str = 'aic') -> Dict:
        """
        Display polynomial modeling analysis results.

        Parameters:
        ----------
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        x_name : str
            Name of independent variable
        y_name : str
            Name of dependent variable
        max_degree : int, optional
            Maximum polynomial degree to consider
        criterion : str, optional
            Model selection criterion ('aic' or 'bic')

        Returns:
        -------
        Dict
            Dictionary containing model results
        """
        st.subheader("Polynomial Model Analysis")

        # Fit models and get results
        results = self.modeler.fit_models(x, y, max_degree, criterion)
        best_degree = self.modeler.get_best_model(results, criterion)['degree']

        # Display model comparison
        self._display_model_comparison(results, criterion)

        # Display best model details
        best_model_results = self._display_best_model(results[best_degree], x, y, x_name, y_name)

        # Display residual analysis
        residual_results = self._display_residual_analysis(results[best_degree], x, y, x_name)

        return {
            'model_results': results[best_degree],
            'residual_results': residual_results
        }

    def _display_model_comparison(self, results: Dict, criterion: str):
        """
        Display comparison of models with different degrees.

        Parameters:
        ----------
        results : Dict
            Dictionary of modeling results for each degree
        criterion : str
            Model selection criterion used
        """
        st.subheader("Model Comparison")

        # Create comparison DataFrame
        comparison_data = []
        for degree, model_info in results.items():
            row = {
                'Degree': degree,
                'R²': model_info['r2'],
                'Adjusted R²': model_info['adj_r2'],
                'AIC': model_info['aic'],
                'BIC': model_info['bic'],
                'RMSE': model_info['rmse']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Highlight best model based on criterion
        criterion_col = criterion.upper()
        best_value = df[criterion_col].min()

        def highlight_best(x):
            return ['background-color: lightgreen' if x[criterion_col] == best_value else '' for _ in x]

        st.dataframe(df.style.apply(highlight_best, axis=1))

    def _display_best_model(self, model_results: Dict,
                          x: np.ndarray,
                          y: np.ndarray,
                          x_name: str,
                          y_name: str) -> Dict:
        """
        Display details of the best fitting model.

        Parameters:
        ----------
        model_results : Dict
            Results for the best model
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        x_name : str
            Name of independent variable
        y_name : str
            Name of dependent variable

        Returns:
        -------
        Dict
            Dictionary containing visualization results
        """
        st.subheader("Best Model Details")

        # Create scatter plot with fitted curve
        fig = go.Figure()

        # Add scatter plot of data
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data',
            marker=dict(
                size=8,
                opacity=0.6
            )
        ))

        # Generate points for smooth curve
        x_smooth = np.linspace(min(x), max(x), 200)
        y_smooth = self.modeler.predict(model_results['model'], x_smooth)

        # Add fitted curve
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name='Fitted Model',
            line=dict(
                color='red',
                width=2
            )
        ))

        # Add confidence bands if available
        if 'confidence_bands' in model_results:
            lower, upper = model_results['confidence_bands']
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=lower,
                mode='lines',
                name='95% Confidence',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=upper,
                mode='lines',
                name='95% Confidence',
                line=dict(dash='dash', color='gray'),
                fill='tonexty'
            ))

        fig.update_layout(
            title=f'Polynomial Model: {y_name} vs {x_name}',
            xaxis_title=x_name,
            yaxis_title=y_name,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display model equation and coefficients
        st.subheader("Model Equation")
        equation = self._format_model_equation(model_results['coefficients'], x_name)
        st.latex(equation)

        # Display coefficient table
        coef_data = []
        for i, (coef, std_err, p_val) in enumerate(zip(
            model_results['coefficients'],
            model_results['std_errors'],
            model_results['p_values']
        )):
            coef_data.append({
                'Term': f'{x_name}^{i}' if i > 0 else 'Intercept',
                'Coefficient': f'{coef:.4f}',
                'Std. Error': f'{std_err:.4f}',
                'p-value': f'{p_val:.4f}'
            })

        st.dataframe(pd.DataFrame(coef_data))

        return {'figure': fig, 'equation': equation}

    def _display_residual_analysis(self, model_results: Dict,
                                x: np.ndarray,
                                y: np.ndarray,
                                x_name: str) -> Dict:
        """
        Display residual analysis plots and statistics.

        Parameters:
        ----------
        model_results : Dict
            Results for the model
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        x_name : str
            Name of independent variable

        Returns:
        -------
        Dict
            Dictionary containing residual analysis results
        """
        st.subheader("Residual Analysis")

        # Calculate residuals
        y_pred = self.modeler.predict(model_results['model'], x)
        residuals = y - y_pred

        # Create subplots for residual analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals vs Fitted',
                'Normal Q-Q Plot',
                'Scale-Location Plot',
                'Residuals vs Order'
            )
        )

        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals'
            ),
            row=1, col=1
        )

        # 2. Normal Q-Q Plot
        residuals_sorted = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(residuals))
        )

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=residuals_sorted,
                mode='markers',
                name='Q-Q Plot'
            ),
            row=1, col=2
        )

        # 3. Scale-Location Plot
        standardized_residuals = residuals / np.std(residuals)
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=np.sqrt(np.abs(standardized_residuals)),
                mode='markers',
                name='Scale-Location'
            ),
            row=2, col=1
        )

        # 4. Residuals vs Order
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(residuals)),
                y=residuals,
                mode='markers',
                name='vs Order'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Residual Diagnostic Plots"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Residual statistics
        st.subheader("Residual Statistics")
        stats_data = {
            'Mean': np.mean(residuals),
            'Std Dev': np.std(residuals),
            'Skewness': stats.skew(residuals),
            'Kurtosis': stats.kurtosis(residuals)
        }

        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        stats_data.update({
            'Shapiro-Wilk p-value': shapiro_p
        })

        st.dataframe(pd.DataFrame([stats_data]))

        return {
            'residuals': residuals,
            'statistics': stats_data,
            'normality_test': {'statistic': shapiro_stat, 'p_value': shapiro_p}
        }

    def _format_model_equation(self, coefficients: np.ndarray, x_name: str) -> str:
        """
        Format the model equation in LaTeX.

        Parameters:
        ----------
        coefficients : np.ndarray
            Model coefficients
        x_name : str
            Name of independent variable

        Returns:
        -------
        str
            LaTeX formatted equation
        """
        terms = []
        for i, coef in enumerate(coefficients):
            if abs(coef) < 1e-10:  # Skip terms with very small coefficients
                continue

            if i == 0:
                terms.append(f"{coef:.4f}")
            elif i == 1:
                terms.append(f"{coef:+.4f}{x_name}")
            else:
                terms.append(f"{coef:+.4f}{x_name}^{i}")

        equation = " ".join(terms)
        return f"f({x_name}) = {equation}"

    def create_polynomial_fit_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model_results: Dict,
        degree: int,
        x_name: str,
        y_name: str,
        width: int = 800,
        height: int = 500
    ) -> go.Figure:
        """
        Create an interactive plot showing polynomial fit.

        Args:
            x: Independent variable values
            y: Dependent variable values
            model_results: Dictionary containing model results
            degree: Polynomial degree to plot
            x_name: Name of independent variable
            y_name: Name of dependent variable
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))

        # Sort x for smooth curve
        x_sorted = np.sort(x)
        x_sorted_2d = x_sorted.reshape(-1, 1)

        # Get model info
        model_info = model_results[degree]
        x_poly = model_info['poly'].transform(x_sorted_2d)
        y_poly_pred = model_info['model'].predict(x_poly)

        # Add fitted curve
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_poly_pred,
            mode='lines',
            name=f'Polynomial (degree {degree})',
            line=dict(color='red', width=2)
        ))

        # Add equation to the plot
        coef = model_info['coefficients']
        intercept = model_info['intercept']

        equation = f"y = {intercept:.3f}"
        for i, c in enumerate(coef[1:]):
            if c >= 0:
                equation += f" + {c:.3f}x^{i+1}"
            else:
                equation += f" - {abs(c):.3f}x^{i+1}"

        # Add R² values
        r2_train = model_info['r2_train']
        r2_test = model_info['r2_test']

        # Add annotation
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=f"Equation: {equation}<br>R² (train): {r2_train:.3f}<br>R² (test): {r2_test:.3f}",
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        # Update layout
        fig.update_layout(
            title=f'Polynomial Fit (Degree {degree})',
            xaxis_title=x_name,
            yaxis_title=y_name,
            width=width,
            height=height,
            showlegend=True
        )

        return fig

    def create_complete_model_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        residual_std: float,
        x_name: str,
        y_name: str,
        width: int = 800,
        height: int = 500
    ) -> go.Figure:
        """
        Create an interactive plot showing the complete probabilistic model.

        Args:
            x: Independent variable values
            y: Dependent variable values
            y_pred: Predicted values
            residual_std: Standard deviation of residuals
            x_name: Name of independent variable
            y_name: Name of dependent variable
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))

        # Add prediction line
        fig.add_trace(go.Scatter(
            x=x,
            y=y_pred,
            mode='lines',
            name='Prediction',
            line=dict(color='red', width=2)
        ))

        # Calculate prediction intervals
        z_95 = 1.96  # 95% confidence
        z_80 = 1.28  # 80% confidence
        z_50 = 0.67  # 50% confidence

        intervals = {
            '95%': (y_pred - z_95 * residual_std,
                   y_pred + z_95 * residual_std),
            '80%': (y_pred - z_80 * residual_std,
                   y_pred + z_80 * residual_std),
            '50%': (y_pred - z_50 * residual_std,
                   y_pred + z_50 * residual_std)
        }

        # Add prediction intervals
        colors = ['rgba(0,100,80,0.2)',
                 'rgba(0,100,80,0.4)',
                 'rgba(0,100,80,0.6)']

        for (interval_name, (lower, upper)), color in zip(
            intervals.items(), colors
        ):
            fig.add_trace(go.Scatter(
                x=x,
                y=upper,
                mode='lines',
                name=f'{interval_name} Interval',
                line=dict(width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=x,
                y=lower,
                mode='lines',
                name=f'{interval_name} Interval',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=color
            ))

        # Update layout
        fig.update_layout(
            title=f'Complete Probabilistic Model: {y_name} vs. {x_name}',
            xaxis_title=x_name,
            yaxis_title=y_name,
            width=width,
            height=height,
            showlegend=True
        )

        return fig

    def display_model_metrics(
        self,
        model_results: Dict,
        container: st.container
    ) -> None:
        """
        Display model metrics in a Streamlit container.

        Args:
            model_results: Dictionary containing model results
            container: Streamlit container for displaying results
        """
        with container:
            st.markdown("### Model Selection Metrics")

            # Create metrics dataframe
            metrics_data = []
            for degree, results in model_results.items():
                metrics_data.append({
                    'Degree': degree,
                    'R² (Train)': results['r2_train'],
                    'R² (Test)': results['r2_test'],
                    'MSE (Train)': results['mse_train'],
                    'MSE (Test)': results['mse_test'],
                    'AIC': results['aic'],
                    'BIC': results['bic']
                })

            metrics_df = pd.DataFrame(metrics_data)

            # Display metrics with highlighting
            st.dataframe(
                metrics_df.style.highlight_min(
                    subset=['AIC', 'BIC'],
                    color='lightgreen'
                )
            )

            # Find best models
            best_aic = metrics_df.loc[metrics_df['AIC'].idxmin(), 'Degree']
            best_bic = metrics_df.loc[metrics_df['BIC'].idxmin(), 'Degree']

            st.write(f"Best degree by AIC: {best_aic}")
            st.write(f"Best degree by BIC: {best_bic}")

    def display_equation(
        self,
        model_info: Dict,
        x_name: str,
        container: st.container
    ) -> None:
        """
        Display model equation in a Streamlit container.

        Args:
            model_info: Dictionary containing model information
            x_name: Name of independent variable
            container: Streamlit container for displaying results
        """
        with container:
            st.markdown("### Model Equation")

            # Get coefficients
            coef = model_info['coefficients']
            intercept = model_info['intercept']

            # Format equation
            equation = f"g({x_name}) = {intercept:.4f}"
            for i, c in enumerate(coef[1:]):
                if c >= 0:
                    equation += f" + {c:.4f}{x_name}^{i+1}"
                else:
                    equation += f" - {abs(c):.4f}{x_name}^{i+1}"

            st.markdown(f"**{equation}**")

            # Display coefficient table
            coef_df = pd.DataFrame({
                'Term': [f"{x_name}^{i}" if i > 0 else "Intercept"
                        for i in range(len(coef))],
                'Coefficient': [intercept] + list(coef[1:])
            })

            st.dataframe(coef_df)

    def display_prediction_controls(
        self,
        x_range: Tuple[float, float],
        x_name: str,
        container: st.container
    ) -> Tuple[float, float]:
        """
        Display prediction controls in a Streamlit container.

        Args:
            x_range: Tuple of (min, max) values for x
            x_name: Name of independent variable
            container: Streamlit container for displaying controls

        Returns:
            Tuple of (x_value, confidence_level)
        """
        with container:
            st.markdown("### Prediction Controls")

            # Input for prediction value
            x_value = st.number_input(
                f"Enter {x_name} value for prediction:",
                min_value=float(x_range[0]),
                max_value=float(x_range[1]),
                value=float(np.mean(x_range)),
                step=0.1
            )

            # Input for confidence level
            confidence_level = st.slider(
                "Confidence level for prediction interval:",
                min_value=0.5,
                max_value=0.99,
                value=0.95,
                step=0.01
            )

            return x_value, confidence_level
