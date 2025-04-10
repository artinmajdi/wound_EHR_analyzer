from datetime import datetime
from io import BytesIO
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import streamlit as st
import plotly.graph_objects as go

from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab
from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.stochastic_modeling.advanced_statistics import AdvancedStatistics

class CreateCompleteModel:


    def __init__(self, df: pd.DataFrame, parent: 'StochasticModelingTab'):
        self.parent               = parent
        self.df                   = df
        self.CN                   = parent.CN
        self.deterministic_model  = parent.deterministic_model
        self.residuals            = parent.residuals
        self.fitted_distribution  = parent.fitted_distribution
        self.patient_id           = parent.patient_id
        self.independent_var      = parent.independent_var
        self.dependent_var        = parent.dependent_var
        self.independent_var_name = parent.independent_var_name
        self.dependent_var_name   = parent.dependent_var_name
        self.advanced_statistics  = parent.advanced_statistics


    def render(self):
        """
        Create and display the complete probabilistic model.

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
        df = self.df

        st.subheader("Complete Probabilistic Model")

        if self.deterministic_model is None or self.residuals is None:
            st.error("Please complete the Deterministic and Random Component analyses first.")
            return

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

            st.markdown("""
            ### Complete Two-Component Model

            The complete probabilistic model combines the deterministic component g(X) with the random component η:

            Y = g(X) + η

            where:
            - Y is the dependent variable (random variable)
            - g(X) is the deterministic component (polynomial function)
            - η is the random component (with probability distribution)

            This model provides:
            1. Expected value predictions via g(X)
            2. Uncertainty quantification via the distribution of η
            3. Prediction intervals based on the combined model
            """)

            # Get data for visualization
            X = df[self.independent_var].values
            y = df[self.dependent_var].values

            # Remove rows with NaN values
            mask = ~(np.isnan(X) | np.isnan(y))
            X = X[mask]
            y = y[mask]

            # Convert to appropriate shape for sklearn
            X_2d = X.reshape(-1, 1)

            # Get model and parameters
            selected_model = self.deterministic_model
            poly = selected_model['poly']
            model = selected_model['model']

            # Generate predictions from deterministic component
            X_poly = poly.transform(X_2d)
            y_pred = model.predict(X_poly)

            # Get standard deviation of residuals for prediction intervals
            residual_std = np.std(self.residuals)

            # Create new X values for smooth curve
            X_range = np.linspace(min(X), max(X), 100)
            X_range_2d = X_range.reshape(-1, 1)
            X_range_poly = poly.transform(X_range_2d)
            y_range_pred = model.predict(X_range_poly)

            # Create prediction intervals (assuming normal distribution for the random component)
            # This can be adapted based on the best-fitting distribution from the random component analysis
            z_95 = 1.96  # 95% confidence interval
            lower_95 = y_range_pred - z_95 * residual_std
            upper_95 = y_range_pred + z_95 * residual_std

            z_80 = 1.28  # 80% confidence interval
            lower_80 = y_range_pred - z_80 * residual_std
            upper_80 = y_range_pred + z_80 * residual_std

            z_50 = 0.67  # 50% confidence interval
            lower_50 = y_range_pred - z_50 * residual_std
            upper_50 = y_range_pred + z_50 * residual_std

            # Create interactive plot with Plotly
            fig = go.Figure()

            # Add scatter plot of data points
            fig.add_trace(go.Scatter(
                x=X, y=y,
                mode='markers',
                name='Data Points',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))

            # Add deterministic component curve
            fig.add_trace(go.Scatter(
                x=X_range, y=y_range_pred,
                mode='lines',
                name='Deterministic Component',
                line=dict(color='red', width=3)
            ))

            # Add 95% prediction interval
            fig.add_trace(go.Scatter(
                x=X_range, y=upper_95,
                mode='lines',
                name='95% Prediction Interval',
                line=dict(color='rgba(0, 100, 80, 0.2)', width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=X_range, y=lower_95,
                mode='lines',
                name='95% Prediction Interval',
                line=dict(color='rgba(0, 100, 80, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.2)'
            ))

            # Add 80% prediction interval
            fig.add_trace(go.Scatter(
                x=X_range, y=upper_80,
                mode='lines',
                name='80% Prediction Interval',
                line=dict(color='rgba(0, 100, 80, 0)', width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=X_range, y=lower_80,
                mode='lines',
                name='80% Prediction Interval',
                line=dict(color='rgba(0, 100, 80, 0)', width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.4)'
            ))

            # Add 50% prediction interval
            fig.add_trace(go.Scatter(
                x=X_range, y=upper_50,
                mode='lines',
                name='50% Prediction Interval',
                line=dict(color='rgba(0, 100, 80, 0)', width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=X_range, y=lower_50,
                mode='lines',
                name='50% Prediction Interval',
                line=dict(color='rgba(0, 100, 80, 0)', width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.6)'
            ))

            # Update layout
            fig.update_layout(
                title=f'Probabilistic Model: {self.dependent_var_name} vs. {self.independent_var_name}',
                xaxis_title=self.independent_var_name,
                yaxis_title=self.dependent_var_name,
                hovermode='closest',
                legend=dict(x=0.02, y=0.98),
                width=800,
                height=500
            )

            # Add annotation for the model equation
            intercept = model.intercept_
            coef = model.coef_

            equation = f"E[{self.dependent_var_name}] = {intercept:.3f}"
            for i, c in enumerate(coef[1:]):  # Skip the first coefficient (it's just 1)
                if c >= 0:
                    equation += f" + {c:.3f}{self.independent_var_name}^{i+1}"
                else:
                    equation += f" - {abs(c):.3f}{self.independent_var_name}^{i+1}"

            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                text=equation,
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Add controls for visualization
            st.markdown("### Visualization Controls")

            col1, col2 = st.columns(2)

            with col1:
                show_deterministic = st.checkbox("Show Deterministic Component", value=True)
                show_data_points = st.checkbox("Show Data Points", value=True)

            with col2:
                show_95_pi = st.checkbox("Show 95% Prediction Interval", value=True)
                show_80_pi = st.checkbox("Show 80% Prediction Interval", value=True)
                show_50_pi = st.checkbox("Show 50% Prediction Interval", value=True)

            # Create dynamic plot based on controls
            fig_dynamic = go.Figure()

            # Add scatter plot of data points
            if show_data_points:
                fig_dynamic.add_trace(go.Scatter(
                    x=X, y=y,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))

            # Add deterministic component curve
            if show_deterministic:
                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=y_range_pred,
                    mode='lines',
                    name='Deterministic Component',
                    line=dict(color='red', width=3)
                ))

            # Add 95% prediction interval
            if show_95_pi:
                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=upper_95,
                    mode='lines',
                    name='95% Prediction Interval',
                    line=dict(color='rgba(0, 100, 80, 0.2)', width=0),
                    showlegend=False
                ))

                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=lower_95,
                    mode='lines',
                    name='95% Prediction Interval',
                    line=dict(color='rgba(0, 100, 80, 0.2)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 80, 0.2)'
                ))

            # Add 80% prediction interval
            if show_80_pi:
                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=upper_80,
                    mode='lines',
                    name='80% Prediction Interval',
                    line=dict(color='rgba(0, 100, 80, 0)', width=0),
                    showlegend=False
                ))

                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=lower_80,
                    mode='lines',
                    name='80% Prediction Interval',
                    line=dict(color='rgba(0, 100, 80, 0)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 80, 0.4)'
                ))

            # Add 50% prediction interval
            if show_50_pi:
                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=upper_50,
                    mode='lines',
                    name='50% Prediction Interval',
                    line=dict(color='rgba(0, 100, 80, 0)', width=0),
                    showlegend=False
                ))

                fig_dynamic.add_trace(go.Scatter(
                    x=X_range, y=lower_50,
                    mode='lines',
                    name='50% Prediction Interval',
                    line=dict(color='rgba(0, 100, 80, 0)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 80, 0.6)'
                ))

            # Update layout
            fig_dynamic.update_layout(
                title=f'Probabilistic Model: {self.dependent_var_name} vs. {self.independent_var_name}',
                xaxis_title=self.independent_var_name,
                yaxis_title=self.dependent_var_name,
                hovermode='closest',
                legend=dict(x=0.02, y=0.98),
                width=800,
                height=500
            )

            # Display the dynamic plot
            st.plotly_chart(fig_dynamic, use_container_width=True)

            # Comparison between deterministic and probabilistic predictions
            st.markdown("### Comparison: Deterministic vs. Probabilistic Prediction")

            st.markdown("""
            #### Deterministic Approach:
            - Provides a single point prediction
            - Cannot quantify uncertainty
            - May lead to overconfidence in predictions

            #### Probabilistic Approach:
            - Provides a distribution of possible outcomes
            - Explicitly quantifies uncertainty
            - Enables risk assessment and decision-making under uncertainty
            - Accounts for natural biological variability
            """)

            # Example of prediction at a specific point
            st.markdown("### Interactive Prediction")

            # Let user select a value of the independent variable
            x_pred = st.slider(f"Select {self.independent_var_name} value for prediction:",
                              float(min(X)), float(max(X)), float((min(X) + max(X)) / 2))

            # Calculate deterministic prediction
            x_pred_2d = np.array([[x_pred]])
            x_pred_poly = poly.transform(x_pred_2d)
            y_pred_mean = model.predict(x_pred_poly)[0]

            # Calculate prediction intervals
            y_pred_lower_95 = y_pred_mean - z_95 * residual_std
            y_pred_upper_95 = y_pred_mean + z_95 * residual_std

            # Display the predictions
            st.markdown(f"#### For {self.independent_var_name} = {x_pred:.2f}:")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Deterministic Prediction:**")
                st.markdown(f"{self.dependent_var_name} = {y_pred_mean:.2f}")

            with col2:
                st.markdown("**Probabilistic Prediction:**")
                st.markdown(f"{self.dependent_var_name} = {y_pred_mean:.2f} ± {z_95 * residual_std:.2f}")
                st.markdown(f"95% Prediction Interval: [{y_pred_lower_95:.2f}, {y_pred_upper_95:.2f}]")

            # Create visualization of the probability distribution at the selected point
            fig_dist = go.Figure()

            # Create x values for the normal distribution
            x_dist = np.linspace(y_pred_mean - 4 * residual_std, y_pred_mean + 4 * residual_std, 1000)

            # Calculate normal PDF
            y_dist = stats.norm.pdf(x_dist, y_pred_mean, residual_std)

            # Add PDF curve
            fig_dist.add_trace(go.Scatter(
                x=x_dist, y=y_dist,
                mode='lines',
                name='Probability Density',
                line=dict(color='blue', width=2)
            ))

            # Add vertical line for mean
            fig_dist.add_trace(go.Scatter(
                x=[y_pred_mean, y_pred_mean],
                y=[0, max(y_dist)],
                mode='lines',
                name='Expected Value',
                line=dict(color='red', width=2)
            ))

            # Add 95% CI
            fig_dist.add_trace(go.Scatter(
                x=[y_pred_lower_95, y_pred_lower_95],
                y=[0, stats.norm.pdf(y_pred_lower_95, y_pred_mean, residual_std)],
                mode='lines',
                name='95% CI Lower',
                line=dict(color='green', width=2, dash='dash')
            ))

            fig_dist.add_trace(go.Scatter(
                x=[y_pred_upper_95, y_pred_upper_95],
                y=[0, stats.norm.pdf(y_pred_upper_95, y_pred_mean, residual_std)],
                mode='lines',
                name='95% CI Upper',
                line=dict(color='green', width=2, dash='dash')
            ))

            # Fill the area between the CI
            x_fill = np.linspace(y_pred_lower_95, y_pred_upper_95, 100)
            y_fill = stats.norm.pdf(x_fill, y_pred_mean, residual_std)

            fig_dist.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='tozeroy',
                name='95% Probability',
                line=dict(color='rgba(0, 100, 80, 0.2)'),
                fillcolor='rgba(0, 100, 80, 0.2)'
            ))

            # Update layout
            fig_dist.update_layout(
                title=f'Probability Distribution of {self.dependent_var_name} at {self.independent_var_name}={x_pred:.2f}',
                xaxis_title=self.dependent_var_name,
                yaxis_title='Probability Density',
                hovermode='closest',
                legend=dict(x=0.02, y=0.98),
                width=800,
                height=400
            )

            # Display the probability distribution
            st.plotly_chart(fig_dist, use_container_width=True)

            # Add explanation
            st.markdown("""
            The plot above shows the probability distribution of the dependent variable at the selected value
            of the independent variable. The vertical red line represents the expected value (deterministic prediction),
            while the green dashed lines represent the 95% prediction interval. The shaded area between the green lines
            represents 95% of the probability mass, indicating the range of values that the dependent variable is likely to take.
            """)

            # Evaluation metrics
            st.markdown("### Model Evaluation Metrics")

            # Calculate metrics for deterministic model
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)

            # Create metrics dataframe
            metrics_df = pd.DataFrame({
                'Metric': ['R²', 'MSE', 'RMSE', 'Residual Std Dev'],
                'Value': [r2, mse, rmse, residual_std]
            })

            st.dataframe(metrics_df)

            # Add download button for the model
            st.markdown("### Export Model")

            # Create a dictionary with model information
            model_info = {
                'dependent_var': self.dependent_var_name,
                'independent_var': self.independent_var_name,
                'polynomial_degree': self.polynomial_degree,
                'intercept': float(model.intercept_),
                'coefficients': [float(c) for c in model.coef_],
                'residual_std': float(residual_std)
            }

            # Convert to JSON
            import json
            model_json = json.dumps(model_info, indent=4)

            # Create download button
            st.download_button(
                label="Download Model as JSON",
                data=model_json,
                file_name="probabilistic_model.json",
                mime="application/json"
            )

            st.markdown('</div>', unsafe_allow_html=True)

