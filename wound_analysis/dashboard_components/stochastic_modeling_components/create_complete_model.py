from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import eval_hermite
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import plotly.graph_objects as go
import json


class CreateCompleteModel:
    """
    Handles creation and visualization of complete probabilistic models.

    This class allows users to combine deterministic and random components
    to create a complete probabilistic model for analysis.
    """

    def __init__(self, df: pd.DataFrame, parent: 'StochasticModelingTab'):
        self.parent               = parent
        self.df                   = df
        self.CN                   = parent.CN

        # previously calculated variables
        self.residuals            = parent.residuals
        self.polynomial_degree    = parent.polynomial_degree
        self.deterministic_model  = parent.deterministic_model
        self.polynomial_type      = parent.polynomial_type

        # user defined variables
        self.independent_var      = parent.independent_var
        self.dependent_var        = parent.dependent_var
        self.independent_var_name = parent.independent_var_name
        self.dependent_var_name   = parent.dependent_var_name
        self.is_hermite           = False # Initialize


    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data by removing NaNs and reshaping."""
        X = self.df[self.independent_var].values
        y = self.df[self.dependent_var].values
        mask = ~(np.isnan(X) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        X_2d = X_clean.reshape(-1, 1)
        return X_clean, y_clean, X_2d

    def _predict(self, X_values_2d: np.ndarray) -> np.ndarray:
        """Generate predictions using the deterministic model."""
        selected_model = self.deterministic_model
        model = selected_model['model']

        if self.is_hermite:
            X_mean = selected_model['X_mean']
            X_std = selected_model['X_std']
            X_standardized = (X_values_2d - X_mean) / X_std
            X_transformed = np.zeros((X_standardized.shape[0], self.polynomial_degree + 1))
            for d in range(self.polynomial_degree + 1):
                X_transformed[:, d] = np.array([eval_hermite(d, x[0]) for x in X_standardized])
        else:
            poly = selected_model['poly']
            X_transformed = poly.transform(X_values_2d)

        return model.predict(X_transformed)

    def _calculate_prediction_intervals(self, y_pred: np.ndarray, residual_std: float) -> dict[str, np.ndarray]:
        """Calculate 50%, 80%, and 95% prediction intervals."""
        intervals = {}
        z_values = {'50': 0.67, '80': 1.28, '95': 1.96}
        for level, z in z_values.items():
            delta = z * residual_std
            intervals[f'lower_{level}'] = y_pred - delta
            intervals[f'upper_{level}'] = y_pred + delta
        return intervals

    def _get_model_equation_text(self) -> str:
        """Generate the model equation string for annotation."""
        selected_model = self.deterministic_model
        model = selected_model['model']
        if self.is_hermite:
            coefs = selected_model['coefficients']
            X_mean = selected_model['X_mean']
            X_std = selected_model['X_std']
            equation = f"E[{self.dependent_var_name}] = {coefs[0]:.3f}H₀(x)"
            for i in range(1, len(coefs)):
                sign = '+' if coefs[i] >= 0 else '-'
                equation += f" {sign} {abs(coefs[i]):.3f}H₍{i}₎(x)"
            # Use <br> for line break in plotly annotation HTML
            equation += f"<br>where x = ({self.independent_var_name} - {X_mean:.3f}) / {X_std:.3f}"
        else:
            intercept = model.intercept_
            coef = model.coef_
            equation = f"E[{self.dependent_var_name}] = {intercept:.3f}"
            # Use <sup> for superscript in plotly annotation HTML
            for i, c in enumerate(coef[1:]):
                sign = '+' if c >= 0 else '-'
                equation += f" {sign} {abs(c):.3f}{self.independent_var_name}<sup>{i+1}</sup>"
        return equation

    def _create_base_figure(self, title: str) -> go.Figure:
        """Create a base Plotly figure with common layout."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title=self.independent_var_name,
            yaxis_title=self.dependent_var_name,
            hovermode='closest',
            legend=dict(x=0.02, y=0.98),
            width=800,
            height=500
        )
        return fig

    def _add_scatter_trace(self, fig: go.Figure, x: np.ndarray, y: np.ndarray):
        """Add scatter plot trace for data points."""
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers', name='Data Points',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))

    def _add_deterministic_trace(self, fig: go.Figure, x_range: np.ndarray, y_range_pred: np.ndarray):
        """Add line trace for the deterministic component."""
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range_pred, mode='lines', name='Deterministic Component',
            line=dict(color='red', width=3)
        ))

    def _add_prediction_interval_traces(self, fig: go.Figure, x_range: np.ndarray, intervals: dict[str, np.ndarray], level: str, color_rgba: str):
        """Add traces for a specific prediction interval level."""
        upper_bound = intervals[f'upper_{level}']
        lower_bound = intervals[f'lower_{level}']
        # Ensure legend name is consistent and shows only once per interval level
        legend_name = f'{level}% Prediction Interval'
        fill_color = f'rgba({color_rgba}, {0.6 if level == "50" else (0.4 if level == "80" else 0.2)})'
        line_color = f'rgba({color_rgba}, 0)' # Make lines invisible for fill

        # Add upper bound trace (invisible line)
        fig.add_trace(go.Scatter(
            x=x_range, y=upper_bound, mode='lines', name=legend_name,
            line=dict(color=line_color, width=0), showlegend=False
        ))
        # Add lower bound trace (invisible line) with fill to upper bound
        fig.add_trace(go.Scatter(
            x=x_range, y=lower_bound, mode='lines', name=legend_name,
            line=dict(color=line_color, width=0), fill='tonexty', fillcolor=fill_color,
            showlegend=True # Show legend entry associated with the lower trace/fill area
        ))


    def _render_visualization_controls(self) -> dict[str, bool]:
        """Render checkboxes for plot controls."""
        st.markdown("### Visualization Controls")
        controls = {}
        col1, col2 = st.columns(2)
        with col1:
            controls['show_deterministic'] = st.checkbox("Show Deterministic Component", value=True, key="vis_det")
            controls['show_data_points'] = st.checkbox("Show Data Points", value=True, key="vis_data")
        with col2:
            controls['show_95_pi'] = st.checkbox("Show 95% Prediction Interval", value=True, key="vis_95")
            controls['show_80_pi'] = st.checkbox("Show 80% Prediction Interval", value=True, key="vis_80")
            controls['show_50_pi'] = st.checkbox("Show 50% Prediction Interval", value=True, key="vis_50")
        return controls

    def _plot_dynamic_model(self, X: np.ndarray, y: np.ndarray, X_range: np.ndarray, y_range_pred: np.ndarray, intervals: dict[str, np.ndarray], controls: dict[str, bool]):
        """Create and display the dynamic plot based on controls."""
        fig_dynamic = self._create_base_figure(f'Probabilistic Model: {self.dependent_var_name} vs. {self.independent_var_name}')

        if controls['show_data_points']:
            self._add_scatter_trace(fig_dynamic, X, y)
        if controls['show_deterministic']:
            self._add_deterministic_trace(fig_dynamic, X_range, y_range_pred)

        pi_color_rgba = '0, 100, 80' # Base color for prediction intervals
        if controls['show_95_pi']:
            self._add_prediction_interval_traces(fig_dynamic, X_range, intervals, '95', pi_color_rgba)
        if controls['show_80_pi']:
            self._add_prediction_interval_traces(fig_dynamic, X_range, intervals, '80', pi_color_rgba)
        if controls['show_50_pi']:
            self._add_prediction_interval_traces(fig_dynamic, X_range, intervals, '50', pi_color_rgba)

        # Add equation annotation to the dynamic plot
        equation = self._get_model_equation_text()
        fig_dynamic.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.02,
            text=equation, showarrow=False, font=dict(size=10), # Smaller font for annotation
            align='left', # Align text left
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1, borderpad=4
        )

        st.plotly_chart(fig_dynamic, use_container_width=True)


    def _render_interactive_prediction(self, X: np.ndarray, residual_std: float):
        """Render the interactive prediction section."""
        st.markdown("### Interactive Prediction")

        # Ensure X is not empty before finding min/max
        if len(X) == 0:
             st.warning("Cannot create prediction slider: No valid data points.")
             return

        min_x, max_x = float(min(X)), float(max(X))
        default_x = float((min_x + max_x) / 2)

        # Use a unique key for the slider
        x_pred_val = st.slider(f"Select {self.independent_var_name} value for prediction:",
                               min_x, max_x, default_x, key="interactive_slider")

        x_pred_2d = np.array([[x_pred_val]])
        y_pred_mean = self._predict(x_pred_2d)[0]

        intervals = self._calculate_prediction_intervals(np.array([y_pred_mean]), residual_std)
        y_pred_lower_95 = intervals['lower_95'][0]
        y_pred_upper_95 = intervals['upper_95'][0]
        z_95 = 1.96 # Hardcoded for display consistency

        st.markdown(f"#### For {self.independent_var_name} = {x_pred_val:.2f}:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Deterministic Prediction:**")
            st.metric(label=self.dependent_var_name, value=f"{y_pred_mean:.2f}")
        with col2:
            st.markdown("**Probabilistic Prediction (95% PI):**")
            st.metric(label=f"{self.dependent_var_name} Range", value=f"[{y_pred_lower_95:.2f}, {y_pred_upper_95:.2f}]", delta=f"± {z_95 * residual_std:.2f}", delta_color="off")


        self._plot_prediction_distribution(x_pred_val, y_pred_mean, residual_std, y_pred_lower_95, y_pred_upper_95)

    def _plot_prediction_distribution(self, x_pred_val: float, y_pred_mean: float, residual_std: float, lower_95: float, upper_95: float):
        """Plot the probability distribution at a specific point."""
        fig_dist_title = f'Predicted Distribution of {self.dependent_var_name} <br>at {self.independent_var_name}={x_pred_val:.2f}'
        fig_dist = self._create_base_figure(fig_dist_title)
        fig_dist.update_layout(xaxis_title=self.dependent_var_name, yaxis_title='Probability Density', height=400, showlegend=True)

        x_dist = np.linspace(y_pred_mean - 4 * residual_std, y_pred_mean + 4 * residual_std, 1000)
        y_dist = stats.norm.pdf(x_dist, y_pred_mean, residual_std)
        max_pdf = max(y_dist) if len(y_dist) > 0 else 0.1 # Avoid error if y_dist is empty

        # PDF curve
        fig_dist.add_trace(go.Scatter(x=x_dist, y=y_dist, mode='lines', name='Probability Density', line=dict(color='blue', width=2)))
        # Mean line
        fig_dist.add_trace(go.Scatter(x=[y_pred_mean, y_pred_mean], y=[0, max_pdf], mode='lines', name=f'Expected Value ({y_pred_mean:.2f})', line=dict(color='red', width=2)))
        # 95% CI lines
        ci_line_style = dict(color='green', width=2, dash='dash')
        fig_dist.add_trace(go.Scatter(x=[lower_95, lower_95], y=[0, stats.norm.pdf(lower_95, y_pred_mean, residual_std)], mode='lines', name=f'95% Lower ({lower_95:.2f})', line=ci_line_style))
        fig_dist.add_trace(go.Scatter(x=[upper_95, upper_95], y=[0, stats.norm.pdf(upper_95, y_pred_mean, residual_std)], mode='lines', name=f'95% Upper ({upper_95:.2f})', line=ci_line_style))
        # Fill 95% CI area
        x_fill = np.linspace(lower_95, upper_95, 100)
        y_fill = stats.norm.pdf(x_fill, y_pred_mean, residual_std)
        fig_dist.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', name='95% Probability Area', line=dict(color='rgba(0, 100, 80, 0)'), fillcolor='rgba(0, 100, 80, 0.2)'))

        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown(f"""
        The plot above shows the *predicted* probability distribution of **{self.dependent_var_name}** when **{self.independent_var_name}** is **{x_pred_val:.2f}**, assuming the random component follows a Normal distribution with standard deviation {residual_std:.3f}.
        The red line marks the expected value (mean prediction), and the green dashed lines show the 95% prediction interval.
        """) # Updated explanation


    def _render_model_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, residual_std: float):
        """Calculate and display model evaluation metrics."""
        st.markdown("### Model Evaluation Metrics")
        # Ensure y_true and y_pred have the same length and are not empty
        if len(y_true) == 0 or len(y_true) != len(y_pred):
            st.warning("Cannot calculate metrics: Invalid input data.")
            return

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        metrics_df = pd.DataFrame({
            'Metric': ['R²', 'MSE', 'RMSE', 'Residual Std Dev (η)'],
            'Value': [f"{r2:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{residual_std:.4f}"] # Format values
        })
        # Display dataframe without index
        st.dataframe(metrics_df.set_index('Metric'))


    def _render_export_model(self, residual_std: float):
        """Render the export model section with download button."""
        st.markdown("### Export Model")
        selected_model = self.deterministic_model
        model = selected_model['model']

        model_info = {
            'dependent_var': self.dependent_var_name,
            'independent_var': self.independent_var_name,
            'polynomial_degree': self.polynomial_degree,
            'polynomial_type': self.polynomial_type,
            'residual_std': float(residual_std)
        }

        if self.is_hermite:
            model_info.update({
                'coefficients': [float(c) for c in selected_model['coefficients']], # Use stored coefficients directly
                'X_mean': float(selected_model['X_mean']),
                'X_std': float(selected_model['X_std'])
            })
        else:
             # For regular polynomials, coefficients from LinearRegression include intercept if fit_intercept=True
             # PolynomialFeatures transformer adds a bias column (all ones) which corresponds to the intercept
             # Ensure we handle whether intercept was fitted or not
             coef_ = model.coef_
             intercept_ = model.intercept_ if hasattr(model, 'intercept_') else 0.0 # Handle cases where intercept is not fitted

             model_info.update({
                'intercept': float(intercept_),
                'coefficients': [float(c) for c in coef_[1:]] # Skip the first coef if it corresponds to the bias term added by PolynomialFeatures
                # Note: If PolynomialFeatures(include_bias=False) was used, coef_[0] would be the coef for x^1
                # Assuming include_bias=True (default) or intercept is handled by LinearRegression
            })


        model_json = json.dumps(model_info, indent=4)
        st.download_button(
            label="Download Model as JSON",
            data=model_json,
            file_name=f"probabilistic_model_{self.dependent_var_name}_vs_{self.independent_var_name}.json", # More specific filename
            mime="application/json",
            key="download_model_json" # Add key
        )


    def render(self):
        """Create and display the complete probabilistic model."""

        if self.deterministic_model is None or self.residuals is None:
            st.error("Please complete the Deterministic and Random Component analyses first.")
            return

        # Determine if using Hermite polynomials early on
        self.is_hermite = self.polynomial_type == 'hermite' or self.deterministic_model.get('is_hermite', False)

        with st.container():

            st.markdown("""
            ### Complete Probabilistic Two-Component Model
            The complete probabilistic model combines the deterministic component *g(X)* with the random component *η*:
            \[ Y = g(X) + \eta \]
            where:
            - *Y* is the dependent variable (random variable)
            - *g(X)* is the deterministic component (polynomial function representing the expected value)
            - *η* is the random component (capturing variability around the expected value)

            This model provides:
            1. **Expected value predictions** via *g(X)*.
            2. **Uncertainty quantification** via the distribution of *η*.
            3. **Prediction intervals** based on the combined model.
            """, unsafe_allow_html=True) # Allow HTML for italics/math

            # 1. Prepare Data
            X, y, X_2d = self._prepare_data()
            if len(X) == 0:
                st.warning("No valid data points found after removing NaNs. Cannot proceed.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            # 2. Calculate Predictions and Residual Standard Deviation
            y_pred_on_data = self._predict(X_2d) # Predictions on the original data points
            # Use the stored residuals' standard deviation
            residual_std = np.std(self.residuals)

            # 3. Prepare data for smooth curve and intervals
            X_range = np.linspace(min(X), max(X), 100)
            y_range_pred = self._predict(X_range.reshape(-1, 1)) # Predictions for the smooth curve
            intervals = self._calculate_prediction_intervals(y_range_pred, residual_std)

            # --- Visualization Section ---
            # No initial static plot - go straight to dynamic plot with controls

            # 4. Render Visualization Controls
            controls = self._render_visualization_controls()

            # 5. Plot Dynamic Model based on controls
            st.markdown("### Probabilistic Model Visualization")
            self._plot_dynamic_model(X, y, X_range, y_range_pred, intervals, controls)

            # 6. Comparison Text
            st.markdown("### Comparison: Deterministic vs. Probabilistic Prediction")
            st.markdown("""
            - **Deterministic Approach:** Provides a single point prediction (*g(X)*). It represents the *average* expected outcome but doesn't quantify the uncertainty or variability.
            - **Probabilistic Approach:** Provides a distribution of possible outcomes (*g(X) + η*). It explicitly quantifies uncertainty using the random component's distribution, enabling risk assessment and acknowledging natural variability.
            """)

            # 7. Interactive Prediction Section
            self._render_interactive_prediction(X, residual_std)

            # 8. Model Evaluation (using predictions on original data)
            self._render_model_evaluation(y, y_pred_on_data, residual_std)

            # 9. Export Model
            self._render_export_model(residual_std)

            st.markdown('</div>', unsafe_allow_html=True)

