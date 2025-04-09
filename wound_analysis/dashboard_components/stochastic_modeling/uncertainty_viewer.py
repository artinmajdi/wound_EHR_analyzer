"""
Uncertainty visualization components for stochastic modeling.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from ...utils.stochastic_modeling import UncertaintyQuantifier
from scipy import stats


class UncertaintyViewer:
    """
    Handles visualization of uncertainty quantification results.
    """

    def __init__(self, key_prefix: str = ""):
        """
        Initialize the uncertainty viewer.

        Parameters:
        ----------
        key_prefix : str, optional
            Prefix for Streamlit widget keys to avoid conflicts
        """
        self.key_prefix = key_prefix
        self.quantifier = UncertaintyQuantifier()

    def display_uncertainty_analysis(self,
                                  model_results: Dict,
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  x_name: str,
                                  y_name: str) -> Dict:
        """
        Display uncertainty analysis results.

        Parameters:
        ----------
        model_results : Dict
            Results from polynomial modeling
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
            Dictionary containing uncertainty analysis results
        """
        st.subheader("Uncertainty Analysis")

        # Create tabs for different uncertainty views
        tab1, tab2, tab3 = st.tabs([
            "Prediction Intervals",
            "Risk Assessment",
            "Sensitivity Analysis"
        ])

        with tab1:
            pred_results = self._display_prediction_intervals(
                model_results, x, y, x_name, y_name
            )

        with tab2:
            risk_results = self._display_risk_assessment(
                model_results, x, y, x_name, y_name
            )

        with tab3:
            sensitivity_results = self._display_sensitivity_analysis(
                model_results, x, y, x_name, y_name
            )

        return {
            'prediction_intervals': pred_results,
            'risk_assessment': risk_results,
            'sensitivity': sensitivity_results
        }

    def _display_prediction_intervals(self,
                                   model_results: Dict,
                                   x: np.ndarray,
                                   y: np.ndarray,
                                   x_name: str,
                                   y_name: str) -> Dict:
        """
        Display prediction interval analysis.

        Parameters:
        ----------
        model_results : Dict
            Model results
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
            Dictionary containing prediction interval results
        """
        st.subheader("Prediction Intervals")

        # User inputs for prediction
        col1, col2 = st.columns(2)
        with col1:
            x_pred = st.number_input(
                f"Enter {x_name} value for prediction",
                min_value=float(min(x)),
                max_value=float(max(x)),
                value=float(np.median(x)),
                key=f"{self.key_prefix}_x_pred"
            )

        with col2:
            conf_level = st.slider(
                "Confidence Level (%)",
                min_value=50,
                max_value=99,
                value=95,
                step=5,
                key=f"{self.key_prefix}_conf_level"
            )

        # Calculate prediction interval
        pred_results = self.quantifier.calculate_prediction_interval(
            model_results['model'],
            x_pred,
            confidence_level=conf_level/100
        )

        # Display results
        st.write("### Prediction Results")
        results_df = pd.DataFrame({
            'Metric': ['Point Prediction', 'Lower Bound', 'Upper Bound'],
            'Value': [
                f"{pred_results['prediction']:.2f}",
                f"{pred_results['lower_bound']:.2f}",
                f"{pred_results['upper_bound']:.2f}"
            ]
        })
        st.dataframe(results_df)

        # Create visualization
        fig = self._create_prediction_plot(
            model_results, x, y,
            x_pred, pred_results,
            x_name, y_name
        )
        st.plotly_chart(fig, use_container_width=True)

        return pred_results

    def _display_risk_assessment(self,
                              model_results: Dict,
                              x: np.ndarray,
                              y: np.ndarray,
                              x_name: str,
                              y_name: str) -> Dict:
        """
        Display risk assessment analysis.

        Parameters:
        ----------
        model_results : Dict
            Model results
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
            Dictionary containing risk assessment results
        """
        st.subheader("Risk Assessment")

        # User inputs for risk thresholds
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.number_input(
                f"Risk Threshold for {y_name}",
                value=float(np.median(y)),
                key=f"{self.key_prefix}_risk_threshold"
            )

        with col2:
            direction = st.selectbox(
                "Risk Direction",
                options=['above', 'below'],
                key=f"{self.key_prefix}_risk_direction"
            )

        # Calculate risk probabilities
        risk_results = self.quantifier.calculate_risk_probability(
            model_results['model'],
            x,
            threshold,
            direction=direction
        )

        # Display risk levels
        st.write("### Risk Levels")
        risk_df = pd.DataFrame({
            'Risk Level': ['Low', 'Moderate', 'High', 'Very High'],
            'Probability Range': [
                '0-25%',
                '25-50%',
                '50-75%',
                '75-100%'
            ],
            'Clinical Interpretation': [
                'Regular monitoring',
                'Increased monitoring',
                'Intervention recommended',
                'Immediate intervention required'
            ]
        })
        st.dataframe(risk_df)

        # Create risk visualization
        fig = self._create_risk_plot(
            risk_results, x, threshold,
            x_name, y_name, direction
        )
        st.plotly_chart(fig, use_container_width=True)

        return risk_results

    def _display_sensitivity_analysis(self,
                                   model_results: Dict,
                                   x: np.ndarray,
                                   y: np.ndarray,
                                   x_name: str,
                                   y_name: str) -> Dict:
        """
        Display sensitivity analysis results.

        Parameters:
        ----------
        model_results : Dict
            Model results
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
            Dictionary containing sensitivity analysis results
        """
        st.subheader("Sensitivity Analysis")

        # Calculate sensitivity metrics
        sensitivity_results = self.quantifier.perform_sensitivity_analysis(
            model_results['model'],
            x,
            y
        )

        # Display global sensitivity indices
        st.write("### Global Sensitivity Indices")
        sens_df = pd.DataFrame({
            'Metric': list(sensitivity_results['global_indices'].keys()),
            'Value': list(sensitivity_results['global_indices'].values())
        })
        st.dataframe(sens_df)

        # Create sensitivity plots
        fig = self._create_sensitivity_plots(
            sensitivity_results, x_name, y_name
        )
        st.plotly_chart(fig, use_container_width=True)

        return sensitivity_results

    def _create_prediction_plot(self,
                             model_results: Dict,
                             x: np.ndarray,
                             y: np.ndarray,
                             x_pred: float,
                             pred_results: Dict,
                             x_name: str,
                             y_name: str) -> go.Figure:
        """
        Create prediction interval visualization.
        """
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

        # Add prediction point and interval
        fig.add_trace(go.Scatter(
            x=[x_pred],
            y=[pred_results['prediction']],
            mode='markers',
            name='Prediction',
            marker=dict(
                size=12,
                symbol='star',
                color='red'
            )
        ))

        # Add error bars for prediction interval
        fig.add_trace(go.Scatter(
            x=[x_pred, x_pred],
            y=[pred_results['lower_bound'], pred_results['upper_bound']],
            mode='lines',
            name='Prediction Interval',
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        ))

        fig.update_layout(
            title='Prediction Interval Visualization',
            xaxis_title=x_name,
            yaxis_title=y_name,
            showlegend=True
        )

        return fig

    def _create_risk_plot(self,
                        risk_results: Dict,
                        x: np.ndarray,
                        threshold: float,
                        x_name: str,
                        y_name: str,
                        direction: str) -> go.Figure:
        """
        Create risk assessment visualization.
        """
        fig = go.Figure()

        # Add probability curve
        fig.add_trace(go.Scatter(
            x=x,
            y=risk_results['probabilities'],
            mode='lines',
            name='Risk Probability',
            line=dict(color='blue')
        ))

        # Add risk zones
        risk_levels = [0.25, 0.5, 0.75]
        colors = ['rgba(0,255,0,0.1)', 'rgba(255,255,0,0.1)',
                'rgba(255,165,0,0.1)', 'rgba(255,0,0,0.1)']

        for i in range(len(risk_levels) + 1):
            lower = risk_levels[i-1] if i > 0 else 0
            upper = risk_levels[i] if i < len(risk_levels) else 1

            fig.add_trace(go.Scatter(
                x=x,
                y=[upper] * len(x),
                fill='tonexty',
                fillcolor=colors[i],
                line=dict(width=0),
                showlegend=False
            ))

        fig.update_layout(
            title=f'Risk Assessment: Probability of {y_name} {direction} {threshold}',
            xaxis_title=x_name,
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )

        return fig

    def _create_sensitivity_plots(self,
                               sensitivity_results: Dict,
                               x_name: str,
                               y_name: str) -> go.Figure:
        """
        Create sensitivity analysis visualizations.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'First-Order Effects',
                'Total Effects',
                'Parameter Interactions',
                'Convergence Analysis'
            )
        )

        # First-order effects
        fig.add_trace(
            go.Bar(
                x=[x_name],
                y=[sensitivity_results['first_order']],
                name='First-Order'
            ),
            row=1, col=1
        )

        # Total effects
        fig.add_trace(
            go.Bar(
                x=[x_name],
                y=[sensitivity_results['total_effect']],
                name='Total Effect'
            ),
            row=1, col=2
        )

        # Parameter interactions
        if 'interactions' in sensitivity_results:
            fig.add_trace(
                go.Heatmap(
                    z=sensitivity_results['interactions'],
                    x=[x_name],
                    y=[x_name],
                    colorscale='Viridis'
                ),
                row=2, col=1
            )

        # Convergence analysis
        if 'convergence' in sensitivity_results:
            fig.add_trace(
                go.Scatter(
                    y=sensitivity_results['convergence'],
                    mode='lines',
                    name='Convergence'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Sensitivity Analysis Results"
        )

        return fig

    def create_prediction_interval_plot(
        self,
        x_value: float,
        y_pred: float,
        residual_std: float,
        confidence_level: float,
        y_name: str,
        width: int = 800,
        height: int = 400
    ) -> go.Figure:
        """
        Create an interactive plot showing prediction interval.

        Args:
            x_value: Value of independent variable
            y_pred: Predicted value
            residual_std: Standard deviation of residuals
            confidence_level: Confidence level (0-1)
            y_name: Name of dependent variable
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Calculate interval bounds
        lower = y_pred - z_score * residual_std
        upper = y_pred + z_score * residual_std

        # Create range for distribution plot
        y_range = np.linspace(
            y_pred - 4 * residual_std,
            y_pred + 4 * residual_std,
            1000
        )

        # Calculate normal distribution PDF
        pdf = stats.norm.pdf(y_range, y_pred, residual_std)

        # Create figure
        fig = go.Figure()

        # Add distribution curve
        fig.add_trace(go.Scatter(
            x=y_range,
            y=pdf,
            mode='lines',
            name='Probability Distribution',
            line=dict(color='blue', width=2)
        ))

        # Add shaded interval region
        y_interval = y_range[
            (y_range >= lower) & (y_range <= upper)
        ]
        pdf_interval = stats.norm.pdf(
            y_interval, y_pred, residual_std
        )

        fig.add_trace(go.Scatter(
            x=y_interval,
            y=pdf_interval,
            fill='tozeroy',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(width=0),
            name=f'{confidence_level*100:.1f}% Prediction Interval'
        ))

        # Add vertical lines for prediction and bounds
        fig.add_vline(
            x=y_pred,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text='Prediction'
        )
        fig.add_vline(
            x=lower,
            line=dict(color='gray', width=1, dash='dot'),
            annotation_text='Lower Bound'
        )
        fig.add_vline(
            x=upper,
            line=dict(color='gray', width=1, dash='dot'),
            annotation_text='Upper Bound'
        )

        # Update layout
        fig.update_layout(
            title='Prediction Interval Distribution',
            xaxis_title=y_name,
            yaxis_title='Probability Density',
            width=width,
            height=height,
            showlegend=True
        )

        return fig

    def create_risk_assessment_plot(
        self,
        y_pred: float,
        residual_std: float,
        threshold: float,
        direction: str,
        y_name: str,
        width: int = 800,
        height: int = 400
    ) -> Tuple[go.Figure, float]:
        """
        Create an interactive plot showing risk assessment.

        Args:
            y_pred: Predicted value
            residual_std: Standard deviation of residuals
            threshold: Risk threshold value
            direction: Risk direction ('above' or 'below')
            y_name: Name of dependent variable
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Tuple of (Plotly figure, risk probability)
        """
        # Create range for distribution plot
        y_range = np.linspace(
            y_pred - 4 * residual_std,
            y_pred + 4 * residual_std,
            1000
        )

        # Calculate normal distribution PDF
        pdf = stats.norm.pdf(y_range, y_pred, residual_std)

        # Calculate risk probability
        if direction == 'above':
            risk_prob = 1 - stats.norm.cdf(
                threshold, y_pred, residual_std
            )
            risk_range = y_range[y_range >= threshold]
        else:  # below
            risk_prob = stats.norm.cdf(
                threshold, y_pred, residual_std
            )
            risk_range = y_range[y_range <= threshold]

        # Create figure
        fig = go.Figure()

        # Add distribution curve
        fig.add_trace(go.Scatter(
            x=y_range,
            y=pdf,
            mode='lines',
            name='Probability Distribution',
            line=dict(color='blue', width=2)
        ))

        # Add shaded risk region
        risk_pdf = stats.norm.pdf(risk_range, y_pred, residual_std)

        fig.add_trace(go.Scatter(
            x=risk_range,
            y=risk_pdf,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0),
            name='Risk Region'
        ))

        # Add vertical lines
        fig.add_vline(
            x=y_pred,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text='Prediction'
        )
        fig.add_vline(
            x=threshold,
            line=dict(color='orange', width=2),
            annotation_text='Risk Threshold'
        )

        # Update layout
        risk_direction = 'above' if direction == 'above' else 'below'
        fig.update_layout(
            title=f'Risk Assessment (Probability of {y_name} {risk_direction} threshold)',
            xaxis_title=y_name,
            yaxis_title='Probability Density',
            width=width,
            height=height,
            showlegend=True
        )

        return fig, risk_prob

    def display_prediction_results(
        self,
        x_value: float,
        y_pred: float,
        residual_std: float,
        confidence_level: float,
        x_name: str,
        y_name: str,
        container: st.container
    ) -> None:
        """
        Display prediction results in a Streamlit container.

        Args:
            x_value: Value of independent variable
            y_pred: Predicted value
            residual_std: Standard deviation of residuals
            confidence_level: Confidence level (0-1)
            x_name: Name of independent variable
            y_name: Name of dependent variable
            container: Streamlit container for displaying results
        """
        with container:
            st.markdown("### Prediction Results")

            # Calculate z-score and interval
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * residual_std
            lower = y_pred - margin
            upper = y_pred + margin

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Value",
                    f"{y_pred:.2f} {y_name}"
                )

            with col2:
                st.metric(
                    "Standard Error",
                    f"±{residual_std:.2f} {y_name}"
                )

            with col3:
                st.metric(
                    f"{confidence_level*100:.1f}% Interval",
                    f"({lower:.2f}, {upper:.2f}) {y_name}"
                )

            # Add interpretation
            st.markdown("""
            #### Interpretation
            - The predicted value represents the most likely outcome
            - The standard error indicates the typical deviation from predictions
            - The prediction interval shows the range where we expect the true
              value to fall with the specified confidence level
            """)

    def display_risk_assessment(
        self,
        risk_prob: float,
        threshold: float,
        direction: str,
        y_name: str,
        container: st.container
    ) -> None:
        """
        Display risk assessment results in a Streamlit container.

        Args:
            risk_prob: Probability of exceeding/falling below threshold
            threshold: Risk threshold value
            direction: Risk direction ('above' or 'below')
            y_name: Name of dependent variable
            container: Streamlit container for displaying results
        """
        with container:
            st.markdown("### Risk Assessment")

            # Calculate risk level
            risk_level = np.select([
                risk_prob < 0.25,
                risk_prob < 0.50,
                risk_prob < 0.75,
                True
            ], [
                "Low",
                "Moderate",
                "High",
                "Very High"
            ])

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Risk Probability",
                    f"{risk_prob*100:.1f}%"
                )

            with col2:
                st.metric(
                    "Risk Level",
                    risk_level,
                    delta=None,
                    delta_color="off"
                )

            # Add interpretation
            risk_direction = 'exceeding' if direction == 'above' else 'falling below'
            st.markdown(f"""
            #### Interpretation
            - There is a **{risk_prob*100:.1f}%** probability of {y_name}
              {risk_direction} {threshold}
            - This represents a **{risk_level.lower()} risk level**

            #### Clinical Recommendations
            """)

            if risk_level == "Low":
                st.success("""
                - Continue standard monitoring
                - No immediate intervention required
                - Schedule routine follow-up
                """)
            elif risk_level == "Moderate":
                st.warning("""
                - Increase monitoring frequency
                - Consider preventive measures
                - Review treatment plan
                """)
            elif risk_level == "High":
                st.error("""
                - Implement immediate preventive measures
                - Schedule urgent follow-up
                - Consider treatment modification
                """)
            else:  # Very High
                st.error("""
                - Immediate clinical attention required
                - Aggressive intervention recommended
                - Daily monitoring needed
                """)

    def display_export_options(
        self,
        model_info: Dict,
        container: st.container
    ) -> None:
        """
        Display model export options in a Streamlit container.

        Args:
            model_info: Dictionary containing model information
            container: Streamlit container for displaying options
        """
        with container:
            st.markdown("### Export Options")

            export_type = st.selectbox(
                "Select export format:",
                ["Python Function", "JSON", "Pickle"]
            )

            if export_type == "Python Function":
                st.code("""
                def predict_with_uncertainty(x, confidence_level=0.95):
                    # Transform input
                    x_poly = polynomial_features.transform([[x]])

                    # Get prediction
                    y_pred = model.predict(x_poly)[0]

                    # Calculate prediction interval
                    z_score = scipy.stats.norm.ppf((1 + confidence_level) / 2)
                    margin = z_score * residual_std

                    return {
                        'prediction': y_pred,
                        'lower_bound': y_pred - margin,
                        'upper_bound': y_pred + margin,
                        'standard_error': residual_std
                    }
                """, language="python")

            elif export_type == "JSON":
                st.code("""
                {
                    "model_type": "polynomial",
                    "degree": 3,
                    "coefficients": [...],
                    "intercept": 0.0,
                    "residual_std": 1.0,
                    "r2_score": 0.85,
                    "metadata": {
                        "date_created": "2024-03-21",
                        "data_range": {"min": 0, "max": 100},
                        "variables": {
                            "independent": "wound_size",
                            "dependent": "impedance"
                        }
                    }
                }
                """, language="json")

            else:  # Pickle
                st.code("""
                import pickle

                model_data = {
                    'model': model,
                    'polynomial_features': polynomial_features,
                    'residual_std': residual_std,
                    'metadata': {...}
                }

                with open('wound_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                """, language="python")

            st.download_button(
                "Download Model",
                data="Model data placeholder",
                file_name="wound_model.txt",
                mime="text/plain"
            )
