"""
Distribution visualization components for stochastic modeling.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from io import BytesIO
import base64
from ...utils.stochastic_modeling import DistributionAnalyzer


class DistributionViewer:
    """
    A class for creating interactive visualizations of probability distributions
    and statistical analysis results.
    """

    def __init__(self):
        """Initialize the DistributionViewer with default settings."""
        self.available_distributions = {
            'Normal': stats.norm,
            'Log-Normal': stats.lognorm,
            'Gamma': stats.gamma,
            'Weibull': stats.weibull_min,
            'Exponential': stats.expon
        }

    def create_distribution_plot(
        self,
        data: np.ndarray,
        fitted_results: Dict,
        var_name: str,
        width: int = 800,
        height: int = 500
    ) -> go.Figure:
        """
        Create an interactive plot showing data histogram and fitted distributions.

        Args:
            data: Array of data points
            fitted_results: Dictionary containing fitted distribution results
            var_name: Name of the variable being plotted
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add histogram of data
        fig.add_trace(go.Histogram(
            x=data,
            name='Data',
            nbinsx=30,
            histnorm='probability density',
            opacity=0.7
        ))

        # Sort distributions by AIC
        sorted_results = sorted(
            fitted_results.items(),
            key=lambda x: x[1]['aic']
        )

        # Create x values for distribution curves
        x = np.linspace(np.min(data), np.max(data), 1000)

        # Plot top 3 distributions
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        for i, (dist_name, result) in enumerate(sorted_results[:3]):
            if i >= len(colors):
                break

            distribution = result['distribution']
            params = result['params']

            try:
                if dist_name == 'Log-Normal':
                    pdf = distribution.pdf(x, result['params']['s'],
                                        result['params']['loc'],
                                        result['params']['scale'])
                else:
                    pdf = distribution.pdf(x, *params)

                fig.add_trace(go.Scatter(
                    x=x,
                    y=pdf,
                    name=f'{dist_name} (AIC: {result["aic"]:.2f})',
                    line=dict(color=colors[i], width=2)
                ))
            except Exception as e:
                st.warning(f"Could not plot {dist_name} distribution: {str(e)}")

        # Update layout
        fig.update_layout(
            title=f'Distribution Analysis for {var_name}',
            xaxis_title=var_name,
            yaxis_title='Density',
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    def create_qq_plot(
        self,
        data: np.ndarray,
        dist_name: str,
        params: tuple,
        width: int = 600,
        height: int = 400
    ) -> go.Figure:
        """
        Create a Q-Q plot for assessing distribution fit.

        Args:
            data: Array of data points
            dist_name: Name of the distribution
            params: Distribution parameters
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        # Calculate theoretical quantiles
        if dist_name in self.available_distributions:
            distribution = self.available_distributions[dist_name]
            theoretical_quantiles = distribution.ppf(
                np.linspace(0.01, 0.99, len(data)),
                *params
            )

            # Sort data for empirical quantiles
            empirical_quantiles = np.sort(data)

            # Create Q-Q plot
            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=empirical_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='blue', size=8)
            ))

            # Add reference line
            min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
            max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
            ref_line = np.linspace(min_val, max_val, 100)

            fig.add_trace(go.Scatter(
                x=ref_line,
                y=ref_line,
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ))

            # Update layout
            fig.update_layout(
                title=f'Q-Q Plot ({dist_name} Distribution)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                width=width,
                height=height,
                showlegend=True
            )

            return fig
        else:
            raise ValueError(f"Distribution {dist_name} not supported")

    def create_residual_plot(
        self,
        x: np.ndarray,
        residuals: np.ndarray,
        x_name: str,
        width: int = 800,
        height: int = 500
    ) -> go.Figure:
        """
        Create a residual plot with fitted variance function.

        Args:
            x: Independent variable values
            residuals: Model residuals
            x_name: Name of independent variable
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add scatter plot of residuals
        fig.add_trace(go.Scatter(
            x=x,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))

        # Add horizontal line at y=0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            name="Zero Line"
        )

        # Calculate moving average for trend visualization
        window = max(5, len(x) // 20)  # Window size: 5% of data points or at least 5
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        residuals_sorted = residuals[sorted_indices]

        # Calculate moving average
        moving_avg = np.convolve(
            residuals_sorted,
            np.ones(window)/window,
            mode='valid'
        )
        x_ma = x_sorted[window-1:]

        # Add moving average line
        fig.add_trace(go.Scatter(
            x=x_ma,
            y=moving_avg,
            mode='lines',
            name=f'Moving Average (window={window})',
            line=dict(color='green', width=2)
        ))

        # Update layout
        fig.update_layout(
            title=f'Residuals vs. {x_name}',
            xaxis_title=x_name,
            yaxis_title='Residuals',
            width=width,
            height=height,
            showlegend=True
        )

        return fig

    def create_prediction_interval_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        intervals: Dict[str, Tuple[np.ndarray, np.ndarray]],
        x_name: str,
        y_name: str,
        width: int = 800,
        height: int = 500
    ) -> go.Figure:
        """
        Create a plot showing prediction intervals.

        Args:
            x: Independent variable values
            y: Actual dependent variable values
            y_pred: Predicted values
            intervals: Dictionary of prediction intervals
            x_name: Name of independent variable
            y_name: Name of dependent variable
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add scatter plot of actual data
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

        # Add prediction intervals
        colors = ['rgba(0,100,80,0.2)', 'rgba(0,100,80,0.4)', 'rgba(0,100,80,0.6)']
        for (interval_name, (lower, upper)), color in zip(intervals.items(), colors):
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
            title=f'Prediction Intervals: {y_name} vs. {x_name}',
            xaxis_title=x_name,
            yaxis_title=y_name,
            width=width,
            height=height,
            showlegend=True
        )

        return fig

    def display_distribution_stats(
        self,
        fitted_results: Dict,
        container: st.container
    ) -> None:
        """
        Display distribution fitting statistics in a Streamlit container.

        Args:
            fitted_results: Dictionary containing fitted distribution results
            container: Streamlit container for displaying results
        """
        with container:
            st.markdown("### Distribution Fitting Statistics")

            # Create dataframe for statistics
            stats_data = []
            for dist_name, result in fitted_results.items():
                stats_data.append({
                    'Distribution': dist_name,
                    'AIC': result['aic'],
                    'KS Statistic': result['ks_stat'],
                    'p-value': result['p_value']
                })

            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.sort_values('AIC')

            # Display dataframe with highlighting
            st.dataframe(
                stats_df.style.highlight_min(
                    subset=['AIC'],
                    color='lightgreen'
                )
            )

            # Add download button
            csv = stats_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="distribution_statistics.csv">Download Statistics (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Display best fit parameters
            if len(stats_df) > 0:
                best_dist = stats_df.iloc[0]['Distribution']
                best_result = fitted_results[best_dist]

                st.markdown(f"### Best Fit: {best_dist} Distribution")

                # Display parameters based on distribution type
                if best_dist == "Normal":
                    st.write(f"μ (mean): {best_result['params'][0]:.4f}")
                    st.write(f"σ (std dev): {best_result['params'][1]:.4f}")
                elif best_dist == "Log-Normal":
                    st.write(f"σ (shape): {best_result['params']['s']:.4f}")
                    st.write(f"μ (scale): {best_result['params']['scale']:.4f}")
                else:
                    for i, param in enumerate(best_result['params']):
                        st.write(f"Parameter {i+1}: {param:.4f}")

    def display_basic_stats(
        self,
        data: np.ndarray,
        container: st.container
    ) -> None:
        """
        Display basic statistics in a Streamlit container.

        Args:
            data: Array of data points
            container: Streamlit container for displaying results
        """
        with container:
            st.markdown("### Basic Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean", f"{np.mean(data):.2f}")
            with col2:
                st.metric("Median", f"{np.median(data):.2f}")
            with col3:
                st.metric("Std Dev", f"{np.std(data):.2f}")
            with col4:
                st.metric("Sample Size", str(len(data)))

    def display_distribution_analysis(self, data: np.ndarray,
                                   variable_name: str,
                                   group_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Display distribution analysis results.

        Parameters:
        ----------
        data : np.ndarray
            Data to analyze
        variable_name : str
            Name of the variable being analyzed
        group_data : Dict[str, np.ndarray], optional
            Dictionary of data grouped by categories
        """
        st.subheader(f"Distribution Analysis: {variable_name}")

        # Fit distributions to overall data
        results = self.analyzer.fit_distributions(data)
        best_dist_name, best_dist_info = self.analyzer.get_best_distribution(results)

        if best_dist_name is None:
            st.error("Could not fit any distributions to the data.")
            return

        # Create distribution plot
        fig = self._create_distribution_plot(data, results, best_dist_name)
        st.plotly_chart(fig, use_container_width=True)

        # Display distribution parameters
        self._display_distribution_parameters(results, best_dist_name)

        # Group analysis if provided
        if group_data is not None:
            self._display_group_analysis(group_data, variable_name)

    def _create_distribution_plot(self, data: np.ndarray,
                                results: Dict,
                                best_dist_name: str) -> go.Figure:
        """
        Create a plot showing histogram and fitted distributions.

        Parameters:
        ----------
        data : np.ndarray
            Data to plot
        results : Dict
            Distribution fitting results
        best_dist_name : str
            Name of the best fitting distribution

        Returns:
        -------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=data,
            name='Data',
            histnorm='probability density',
            showlegend=True,
            opacity=0.7
        ))

        # Generate points for distribution curves
        x = np.linspace(min(data), max(data), 200)

        # Add fitted distributions
        for dist_name, dist_info in results.items():
            dist = dist_info['distribution']
            params = dist_info['params']

            if isinstance(params, dict):
                y = dist.pdf(x, **params)
            else:
                y = dist.pdf(x, *params)

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                name=f'{dist_name} fit',
                line=dict(
                    dash='dash' if dist_name != best_dist_name else 'solid',
                    width=2 if dist_name != best_dist_name else 3
                )
            ))

        fig.update_layout(
            title='Data Distribution with Fitted Curves',
            xaxis_title='Value',
            yaxis_title='Density',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    def _display_distribution_parameters(self, results: Dict, best_dist_name: str):
        """
        Display distribution parameters and goodness of fit statistics.

        Parameters:
        ----------
        results : Dict
            Distribution fitting results
        best_dist_name : str
            Name of the best fitting distribution
        """
        st.subheader("Distribution Parameters")

        # Create a DataFrame for the results
        data = []
        for dist_name, dist_info in results.items():
            row = {
                'Distribution': dist_name,
                'AIC': dist_info['aic'],
                'KS Statistic': dist_info['ks_stat'],
                'p-value': dist_info['p_value']
            }

            # Add parameters
            params = dist_info['params']
            if isinstance(params, dict):
                for param_name, param_value in params.items():
                    row[f'Parameter: {param_name}'] = param_value
            else:
                for i, param_value in enumerate(params):
                    row[f'Parameter {i+1}'] = param_value

            data.append(row)

        df = pd.DataFrame(data)

        # Highlight the best distribution
        def highlight_best(x):
            return ['background-color: lightgreen' if x.name == best_dist_name else '' for _ in x]

        st.dataframe(df.style.apply(highlight_best, axis=1))

    def _display_group_analysis(self, group_data: Dict[str, np.ndarray],
                              variable_name: str):
        """
        Display analysis of grouped data.

        Parameters:
        ----------
        group_data : Dict[str, np.ndarray]
            Dictionary of data grouped by categories
        variable_name : str
            Name of the variable being analyzed
        """
        st.subheader("Group Analysis")

        # Create box plot
        fig = go.Figure()
        for group_name, group_values in group_data.items():
            fig.add_trace(go.Box(
                y=group_values,
                name=group_name,
                boxpoints='outliers'
            ))

        fig.update_layout(
            title=f'{variable_name} Distribution by Group',
            yaxis_title=variable_name,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate and display group statistics
        stats_data = []
        for group_name, group_values in group_data.items():
            stats = self.analyzer.calculate_basic_stats(group_values)
            if stats:
                stats['Group'] = group_name
                stats_data.append(stats)

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.set_index('Group')
            st.dataframe(stats_df)

            # Perform and display statistical tests
            if len(group_data) >= 2:
                st.subheader("Statistical Tests")

                # ANOVA
                f_stat, p_value = stats.f_oneway(*group_data.values())
                st.write("One-way ANOVA:")
                st.write(f"F-statistic: {f_stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")

                # Bartlett's test
                stat, p_value = stats.bartlett(*group_data.values())
                st.write("\nBartlett's test for equal variances:")
                st.write(f"Statistic: {stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")
