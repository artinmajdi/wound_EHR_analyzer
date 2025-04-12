from datetime import datetime
import json
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab


class CreateUncertaintyQuantificationTools:

    def __init__(self, df: pd.DataFrame, parent: 'StochasticModelingTab'):
        self.parent               = parent
        self.df                   = df
        self.CN                   = parent.CN

        # previously calculated variables
        self.deterministic_model  = parent.deterministic_model
        self.residuals            = parent.residuals
        self.fitted_distribution  = parent.fitted_distribution

        # user defined variables
        self.patient_id           = parent.patient_id
        self.independent_var      = parent.independent_var
        self.dependent_var        = parent.dependent_var
        self.independent_var_name = parent.independent_var_name
        self.dependent_var_name   = parent.dependent_var_name


    def render(self):
        """
        Create and display uncertainty quantification tools for risk assessment and decision support.
        """
        df = self.df
        st.subheader("Uncertainty Quantification Tools")

        # Show info about uncertainty quantification
        st.info("""
        This section provides tools for quantifying uncertainty in wound healing predictions and
        generating risk assessments based on the probabilistic model. These tools are designed to
        support clinical decision-making under uncertainty.
        """)

        # Check if complete model is available
        if not hasattr(self, 'deterministic_model') or not hasattr(self, 'fitted_distribution'):
            st.warning("Please complete the model analysis first (run the Complete Model section).")
            return

        with st.container():
            st.markdown("""
            <style>
            .tools-section {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 15px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)

            # Create two-column section for the dashboard
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<div class="tools-section">', unsafe_allow_html=True)
                st.markdown("### Prediction Interval Calculator")

                # Input for prediction value
                prediction_x = st.number_input(
                    f"Enter {self.independent_var_name} value for prediction:",
                    min_value = float(df[self.independent_var].min()),
                    max_value = float(df[self.independent_var].max()),
                    value     = float(df[self.independent_var].median()),
                    step      = 0.1
                )

                # Input for confidence level
                confidence_level = st.slider(
                    "Confidence level for prediction interval:",
                    min_value = 0.5,
                    max_value = 0.99,
                    value     = 0.95,
                    step      = 0.01
                )

                # Calculate prediction
                if st.button("Calculate Prediction Interval"):
                    try:
                        # Get the best polynomial model
                        best_degree = self.deterministic_model['best_degree']
                        model = self.deterministic_model['models'][best_degree]['model']
                        poly = self.deterministic_model['poly']

                        # Transform input for prediction
                        X_pred = poly.transform([[prediction_x]])

                        # Get deterministic prediction
                        y_pred = model.predict(X_pred)[0]

                        # Get residual distribution parameters
                        if self.fitted_distribution is not None:
                            dist_name = self.fitted_distribution['best_distribution']
                            dist_params = self.fitted_distribution['params'][dist_name]

                            # Calculate prediction interval based on distribution
                            if dist_name == 'norm':
                                # For normal distribution
                                loc, scale = dist_params
                                lower_bound = y_pred + stats.norm.ppf((1-confidence_level)/2, loc=loc, scale=scale)
                                upper_bound = y_pred + stats.norm.ppf(1-(1-confidence_level)/2, loc=loc, scale=scale)
                            elif hasattr(stats, dist_name):
                                # For other distributions
                                dist = getattr(stats, dist_name)
                                lower_bound = y_pred + dist.ppf((1-confidence_level)/2, *dist_params)
                                upper_bound = y_pred + dist.ppf(1-(1-confidence_level)/2, *dist_params)
                            else:
                                st.error(f"Unsupported distribution: {dist_name}")
                                return

                            # Display results
                            st.markdown("#### Prediction Results")
                            st.markdown(f"**Input {self.independent_var_name}:** {prediction_x:.2f}")
                            st.markdown(f"**Predicted {self.dependent_var_name}:** {y_pred:.2f}")
                            st.markdown(f"**{confidence_level*100:.1f}% Prediction Interval:** [{lower_bound:.2f}, {upper_bound:.2f}]")

                            # Create visualization
                            fig = go.Figure()

                            # Add prediction point
                            fig.add_trace(go.Scatter(
                                x=[prediction_x],
                                y=[y_pred],
                                mode='markers',
                                marker=dict(size=10, color='blue'),
                                name='Predicted value'
                            ))

                            # Add prediction interval
                            fig.add_trace(go.Scatter(
                                x=[prediction_x, prediction_x],
                                y=[lower_bound, upper_bound],
                                mode='lines',
                                line=dict(width=2, color='red'),
                                name=f'{confidence_level*100:.1f}% Prediction Interval'
                            ))

                            # Add interval markers
                            fig.add_trace(go.Scatter(
                                x=[prediction_x, prediction_x],
                                y=[lower_bound, upper_bound],
                                mode='markers',
                                marker=dict(size=8, color='red'),
                                showlegend=False
                            ))

                            # Update layout
                            fig.update_layout(
                                title=f'Prediction with {confidence_level*100:.1f}% Confidence Interval',
                                xaxis_title=self.independent_var_name,
                                yaxis_title=self.dependent_var_name,
                                height=400,
                                width=400
                            )

                            st.plotly_chart(fig)

                            # Add interpretation
                            st.markdown("#### Interpretation")
                            st.markdown(f"""
                            - The predicted value of {self.dependent_var_name} is **{y_pred:.2f}**.
                            - With {confidence_level*100:.1f}% confidence, the actual value will be between **{lower_bound:.2f}** and **{upper_bound:.2f}**.
                            - The width of this interval reflects the uncertainty in the prediction.
                            """)
                    except Exception as e:
                        st.error(f"Error calculating prediction interval: {str(e)}")

                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="tools-section">', unsafe_allow_html=True)
                st.markdown("### Risk Assessment Tool")

                # Input for threshold
                threshold_type = st.selectbox(
                    "Select threshold type:",
                    ["Greater than", "Less than"],
                    index=0
                )

                threshold_value = st.number_input(
                    f"Threshold value for {self.dependent_var_name}:",
                    value=float(df[self.dependent_var].median()),
                    step=0.1
                )

                # Input for future X value
                future_x = st.number_input(
                    f"Enter future {self.independent_var_name} value:",
                    min_value=float(df[self.independent_var].min()),
                    max_value=float(df[self.independent_var].max() * 1.5),  # Allow some extrapolation
                    value=float(df[self.independent_var].median()),
                    step=0.1
                )

                # Calculate risk
                if st.button("Calculate Risk Probability"):
                    try:
                        # Get the best polynomial model
                        best_degree = self.deterministic_model['best_degree']
                        model = self.deterministic_model['models'][best_degree]['model']
                        poly = self.deterministic_model['poly']

                        # Transform input for prediction
                        X_pred = poly.transform([[future_x]])

                        # Get deterministic prediction
                        y_pred = model.predict(X_pred)[0]

                        # Get residual distribution parameters
                        if self.fitted_distribution is not None:
                            dist_name = self.fitted_distribution['best_distribution']
                            dist_params = self.fitted_distribution['params'][dist_name]

                            # Calculate probability based on distribution and threshold type
                            if hasattr(stats, dist_name):
                                dist = getattr(stats, dist_name)

                                # Calculate the standardized threshold
                                standardized_threshold = threshold_value - y_pred

                                # Calculate probability
                                if threshold_type == "Greater than":
                                    prob = 1 - dist.cdf(standardized_threshold, *dist_params)
                                else:  # "Less than"
                                    prob = dist.cdf(standardized_threshold, *dist_params)

                                # Display results
                                st.markdown("#### Risk Assessment Results")
                                st.markdown(f"**Input {self.independent_var_name}:** {future_x:.2f}")
                                st.markdown(f"**Predicted {self.dependent_var_name}:** {y_pred:.2f}")

                                # Format the probability as percentage
                                prob_percent = prob * 100

                                # Determine risk level based on probability
                                if prob_percent < 25:
                                    risk_level = "Low"
                                    color = "green"
                                elif prob_percent < 50:
                                    risk_level = "Moderate"
                                    color = "orange"
                                elif prob_percent < 75:
                                    risk_level = "High"
                                    color = "red"
                                else:
                                    risk_level = "Very High"
                                    color = "darkred"

                                # Display probability and risk level
                                st.markdown(f"""
                                **Probability of {self.dependent_var_name} being {threshold_type.lower()} {threshold_value}:**
                                """)

                                # Create a progress bar for visualization
                                st.progress(float(prob))

                                # Display the percentage and risk level
                                st.markdown(f"""
                                <div style='text-align: center; margin-top: -10px;'>
                                    <span style='font-size: 24px; color: {color};'>{prob_percent:.1f}%</span>
                                    <br>
                                    <span style='font-size: 18px; color: {color};'>({risk_level} Risk)</span>
                                </div>
                                """, unsafe_allow_html=True)

                                # Create visualization of probability
                                fig = go.Figure()

                                # Create x values for distribution curve
                                x_range = np.linspace(
                                    y_pred - 4 * (dist_params[-1] if len(dist_params) > 1 else 1),
                                    y_pred + 4 * (dist_params[-1] if len(dist_params) > 1 else 1),
                                    1000
                                )

                                # Calculate PDF values
                                if dist_name == 'norm':
                                    pdf_values = stats.norm.pdf(x_range - y_pred, *dist_params)
                                else:
                                    pdf_values = dist.pdf(x_range - y_pred, *dist_params)

                                # Add the PDF curve
                                fig.add_trace(go.Scatter(
                                    x=x_range,
                                    y=pdf_values,
                                    mode='lines',
                                    name='Probability Distribution',
                                    line=dict(color='blue', width=2)
                                ))

                                # Add vertical line for predicted value
                                fig.add_vline(
                                    x=y_pred,
                                    line_width=2,
                                    line_dash="dash",
                                    line_color="green",
                                    annotation_text="Predicted Value",
                                    annotation_position="top"
                                )

                                # Add vertical line for threshold
                                fig.add_vline(
                                    x=threshold_value,
                                    line_width=2,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text="Threshold",
                                    annotation_position="top"
                                )

                                # Shade the area representing the probability
                                x_shade = np.linspace(
                                    threshold_value if threshold_type == "Greater than" else x_range[0],
                                    x_range[-1] if threshold_type == "Greater than" else threshold_value,
                                    100
                                )

                                if dist_name == 'norm':
                                    y_shade = stats.norm.pdf(x_shade - y_pred, *dist_params)
                                else:
                                    y_shade = dist.pdf(x_shade - y_pred, *dist_params)

                                fig.add_trace(go.Scatter(
                                    x=x_shade,
                                    y=y_shade,
                                    fill='tozeroy',
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    line=dict(color='rgba(255, 0, 0, 0.2)'),
                                    name=f'Probability Area ({prob_percent:.1f}%)'
                                ))

                                # Update layout
                                fig.update_layout(
                                    title=f'Probability Distribution for {self.dependent_var_name} at {self.independent_var_name}={future_x}',
                                    xaxis_title=self.dependent_var_name,
                                    yaxis_title='Probability Density',
                                    height=400,
                                    width=400
                                )

                                st.plotly_chart(fig)

                                # Add interpretation
                                st.markdown("#### Interpretation")
                                interpretation_text = f"""
                                - At {self.independent_var_name} = {future_x}, the predicted {self.dependent_var_name} is **{y_pred:.2f}**.
                                - There is a **{prob_percent:.1f}%** probability that {self.dependent_var_name} will be {threshold_type.lower()} {threshold_value}.
                                - This represents a **{risk_level} risk** level.
                                """

                                # Add clinical recommendations based on risk level
                                recommendations = {
                                    "Low": "Regular monitoring should be sufficient.",
                                    "Moderate": "Consider increasing monitoring frequency and implementing preventive measures.",
                                    "High": "Implement intensive monitoring and interventions to mitigate risk.",
                                    "Very High": "Immediate attention and intervention recommended."
                                }

                                interpretation_text += f"""
                                - **Recommendation:** {recommendations[risk_level]}
                                """

                                st.markdown(interpretation_text)
                            else:
                                st.error(f"Unsupported distribution: {dist_name}")
                    except Exception as e:
                        st.error(f"Error calculating risk probability: {str(e)}")

                st.markdown('</div>', unsafe_allow_html=True)

        # Add section for exporting prediction model
        st.markdown('<div class="tools-section">', unsafe_allow_html=True)
        st.markdown("### Export Prediction Model")

        export_format = st.selectbox(
            "Select export format:",
            ["Python Function", "JSON", "Pickle"],
            index=0
        )

        if st.button("Generate Exportable Model"):
            if export_format == "Python Function":
                # Generate Python function
                best_degree = self.deterministic_model['best_degree']
                coeffs = self.deterministic_model['models'][best_degree]['model'].coef_
                intercept = self.deterministic_model['models'][best_degree]['model'].intercept_

                dist_name = self.fitted_distribution['best_distribution']
                dist_params = self.fitted_distribution['params'][dist_name]

                python_code = f"""
                                import numpy as np
                                from scipy import stats

                                def predict_wound_healing(x, confidence_level=0.95):
                                    \"\"\"
                                    Predict wound healing with confidence intervals.

                                    Parameters:
                                    ----------
                                    x : float
                                        The {self.independent_var_name} value to predict for
                                    confidence_level : float, optional
                                        Confidence level for prediction interval, default 0.95

                                    Returns:
                                    -------
                                    dict
                                        Dictionary containing prediction, confidence interval, and metadata
                                    \"\"\"
                                    # Polynomial model (degree {best_degree})
                                    coefficients = {list(coeffs)}
                                    intercept = {intercept}

                                    # Make polynomial features
                                    poly_features = [1]  # Intercept
                                    for i in range(1, {best_degree + 1}):
                                        poly_features.append(x ** i)

                                    # Make prediction
                                    y_pred = np.dot(coefficients, poly_features[1:]) + intercept

                                    # Distribution parameters for residuals
                                    dist_name = "{dist_name}"
                                    dist_params = {list(dist_params)}

                                    # Calculate prediction interval
                                    if dist_name == "norm":
                                        loc, scale = dist_params
                                        lower_bound = y_pred + stats.norm.ppf((1-confidence_level)/2, loc=loc, scale=scale)
                                        upper_bound = y_pred + stats.norm.ppf(1-(1-confidence_level)/2, loc=loc, scale=scale)
                                    else:
                                        # For other distributions, would need additional implementation
                                        lower_bound = None
                                        upper_bound = None

                                    # Return results
                                    return {{
                                        "prediction": y_pred,
                                        "lower_bound": lower_bound,
                                        "upper_bound": upper_bound,
                                        "confidence_level": confidence_level,
                                        "x_value": x,
                                        "model_info": {{
                                            "dependent_var": "{self.dependent_var_name}",
                                            "independent_var": "{self.independent_var_name}",
                                            "polynomial_degree": {best_degree},
                                            "residual_distribution": dist_name
                                        }}
                                    }}
                                """

                st.code(python_code, language="python")

                # Create a download button for the Python function
                st.download_button(
                    label="Download Python Function",
                    data=python_code,
                    file_name="wound_healing_prediction.py",
                    mime="text/plain"
                )

            elif export_format == "JSON":
                # Export model parameters as JSON
                best_degree = self.deterministic_model['best_degree']
                coeffs = self.deterministic_model['models'][best_degree]['model'].coef_.tolist()
                intercept = float(self.deterministic_model['models'][best_degree]['model'].intercept_)

                dist_name = self.fitted_distribution['best_distribution']
                dist_params = [float(p) for p in self.fitted_distribution['params'][dist_name]]

                json_data = {
                    "model_type": "polynomial",
                    "degree": best_degree,
                    "coefficients": coeffs,
                    "intercept": intercept,
                    "residual_distribution": {
                        "name": dist_name,
                        "parameters": dist_params
                    },
                    "variables": {
                        "dependent": self.dependent_var_name,
                        "independent": self.independent_var_name
                    },
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "description": f"Wound healing prediction model for {self.dependent_var_name} based on {self.independent_var_name}"
                    }
                }

                json_str = json.dumps(json_data, indent=2)
                st.code(json_str, language="json")

                # Create a download button for the JSON
                st.download_button(
                    label="Download JSON Model",
                    data=json_str,
                    file_name="wound_healing_model.json",
                    mime="application/json"
                )

            elif export_format == "Pickle":
                # Export model as pickle
                best_degree = self.deterministic_model['best_degree']
                model = self.deterministic_model['models'][best_degree]['model']

                export_data = {
                    "polynomial_model": {
                        "degree": best_degree,
                        "model": model
                    },
                    "residual_distribution": self.fitted_distribution,
                    "variables": {
                        "dependent": self.dependent_var_name,
                        "independent": self.independent_var_name
                    }
                }

                # Create a pickle file
                pickle_data = pickle.dumps(export_data)

                # Create a download button for the pickle
                st.download_button(
                    label="Download Pickle Model",
                    data=pickle_data,
                    file_name="wound_healing_model.pkl",
                    mime="application/octet-stream"
                )

        st.markdown("</div>", unsafe_allow_html=True)
