import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import json
from datetime import datetime
import streamlit as st

from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.dashboard_components.visualizer import Visualizer
from wound_analysis.utils.stochastic_modeling.distribution_analyzer import DistributionAnalyzer


# TODO: refactor this into smaller components (1st step: make the functions staticmethods. 2nd step: refactor into smaller components)
# TODO: Check if I have already done the hermit polynomial modeling.
# TODO: Ask the AI to review this and tell me in text all the things that is happening here (with mathematical equations).
# TODO: Add the advanced statistical tests to the output.

class StochasticModelingTab:
    """
    Renders the Stochastic Modeling tab in the Streamlit application.

    This tab provides probabilistic modeling capabilities for diabetic wound healing,
    treating measurements as random variables rather than fixed values.

    Parameters:
    ----------
    selected_patient : str
        The currently selected patient from the sidebar dropdown.
    wound_data_processor : WoundDataProcessor
        The data processor instance containing the filtered DataFrame and processing methods.
    """

    def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
        """
        Initialize the StochasticModelingTab with the given parameters.

        Args:
            selected_patient (str): The currently selected patient from the sidebar dropdown.
            wound_data_processor (WoundDataProcessor): The data processor instance containing the filtered DataFrame and processing methods.
        """
        # Store input parameters
        self.wound_data_processor = wound_data_processor
        self.patient_id           = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
        self.df                   = wound_data_processor.df
        self.CN                   = DColumns(df=self.df)

        # Initialize parameter lists for selections
        self._initialize_parameter_lists()

        # Available probability distributions for fitting
        self.available_distributions = {
            'Normal'     : stats.norm,
            'Log-Normal' : stats.lognorm,
            'Gamma'      : stats.gamma,
            'Weibull'    : stats.weibull_min,
            'Exponential': stats.expon
        }

        # Initialize session state for model parameters
        if 'deterministic_model' not in st.session_state:
            st.session_state.deterministic_model = None
        if 'polynomial_degree' not in st.session_state:
            st.session_state.polynomial_degree = None
        if 'deterministic_coefs' not in st.session_state:
            st.session_state.deterministic_coefs = None
        if 'residuals' not in st.session_state:
            st.session_state.residuals = None
        if 'fitted_distribution' not in st.session_state:
            st.session_state.fitted_distribution = None

        # Models for storing analysis results
        self.deterministic_model = st.session_state.deterministic_model
        self.polynomial_degree   = st.session_state.polynomial_degree
        self.deterministic_coefs = st.session_state.deterministic_coefs
        self.residuals           = st.session_state.residuals
        self.fitted_distribution = st.session_state.fitted_distribution

    def _initialize_parameter_lists(self):
        """Initialize parameter lists for dropdown selections based on the probabilistic modeling framework.

        Following the two-component model approach:
        - Dependent variables (Y): Physiological measurements (impedance, temperature, oxygenation)
        - Independent variables (X): Wound characteristics and time-related variables
        - Additional parameters: Control variables (patient demographics, medical history, etc.)
        """
        # Define available dependent variables (Y) - Physiological measurements
        # These are the outcomes we're modeling probabilistically
        self.dependent_variables = {
            # Impedance measurements at different frequencies
            'High Freq Impedance Absolute'   : self.CN.HIGHEST_FREQ_ABSOLUTE,
            'High Freq Impedance Real'       : self.CN.HIGHEST_FREQ_REAL,
            'High Freq Impedance Imaginary'  : self.CN.HIGHEST_FREQ_IMAGINARY,
            'Center Freq Impedance Absolute' : self.CN.CENTER_FREQ_ABSOLUTE,
            'Center Freq Impedance Real'     : self.CN.CENTER_FREQ_REAL,
            'Center Freq Impedance Imaginary': self.CN.CENTER_FREQ_IMAGINARY,
            'Low Freq Impedance Absolute'    : self.CN.LOWEST_FREQ_ABSOLUTE,
            'Low Freq Impedance Real'        : self.CN.LOWEST_FREQ_REAL,
            'Low Freq Impedance Imaginary'   : self.CN.LOWEST_FREQ_IMAGINARY,

            # Temperature measurements
            'Temperature Center'  : self.CN.CENTER_TEMP,
            'Temperature Edge'    : self.CN.EDGE_TEMP,
            'Temperature Peri'    : self.CN.PERI_TEMP,
            'Temperature Gradient': self.CN.TOTAL_GRADIENT,
            'Center-Edge Temp Gradient': self.CN.CENTER_EDGE_GRADIENT,
            'Edge-Peri Temp Gradient': self.CN.EDGE_PERI_GRADIENT,

            # Oxygenation measurements
            'Oxygenation': self.CN.OXYGENATION,
            'Hemoglobin': self.CN.HEMOGLOBIN,
            'Oxyhemoglobin': self.CN.OXYHEMOGLOBIN,
            'Deoxyhemoglobin': self.CN.DEOXYHEMOGLOBIN,
        }

        # Define available independent variables (X) - Wound characteristics and time variables
        # These are the primary predictors in our model
        self.independent_variables = {
            # Wound characteristics
            'Wound Area'  : self.CN.WOUND_AREA,
            'Wound Length': self.CN.LENGTH,
            'Wound Width' : self.CN.WIDTH,
            'Wound Depth' : self.CN.DEPTH,

            # Time-related variables
            'Days Since First Visit': self.CN.DAYS_SINCE_FIRST_VISIT,
            'Visit Number'          : self.CN.VISIT_NUMBER,

            # Healing metrics
            'Healing Rate': self.CN.HEALING_RATE,
        }

        # Define additional parameters (control variables)
        # These help isolate the effects of primary variables of interest
        self.additional_parameters = {
            # Patient-level controls
            'Age'         : self.CN.AGE,
            'Sex'         : self.CN.SEX,
            'BMI'         : self.CN.BMI,
            'BMI Category': self.CN.BMI_CATEGORY,
            'Race'        : self.CN.RACE,
            'Ethnicity'   : self.CN.ETHNICITY,

            # Medical history
            'Diabetes Status': self.CN.DIABETES,
            'A1C Value'      : self.CN.A1C,

            # Lifestyle factors
            'Smoking Status': self.CN.SMOKING_STATUS,
            'Alcohol Status': self.CN.ALCOHOL_STATUS,

            # Wound-specific factors
            'Wound Location'  : self.CN.WOUND_LOCATION,
            'Wound Type'      : self.CN.WOUND_TYPE,
            'Infection Status': self.CN.INFECTION,

            # Treatment-related
            'Current Wound Care': self.CN.CURRENT_WOUND_CARE,
        }

        # You might also want to add distribution types for the probabilistic modeling
        self.distribution_types = [
            'Normal',
            'Log-normal',
            'Gamma',
            'Weibull'
        ]

        # Polynomial degrees for the deterministic component
        self.polynomial_degrees = list(range(1, 6))  # Linear to 5th degree

    def _parameter_selection_ui(self) -> Tuple[str, str, List[str], Dict, bool]:
        """
        Create a visually appealing parameter selection UI.

        Returns:
        -------
        Tuple[str, str, List[str], Dict, bool]:
            - Selected dependent variable column name
            - Selected independent variable column name
            - List of selected additional parameter column names
            - Dictionary of filter parameters
            - Boolean indicating if the analysis should be run
        """
        st.subheader("📊 Parameter Selection")

        # Create a styled container
        with st.container():

            # Primary variable selection in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                dependent_var_name = st.selectbox(
                    "📈 Dependent Variable (Y)",
                    options=list(self.dependent_variables.keys()),
                    index=0,
                    help="The outcome variable to be modeled probabilistically"
                )
                dependent_var = self.dependent_variables[dependent_var_name]

            with col2:
                independent_var_name = st.selectbox(
                    "📉 Independent Variable (X)",
                    options=list(self.independent_variables.keys()),
                    index=0,
                    help="The primary predictor variable"
                )
                independent_var = self.independent_variables[independent_var_name]

            # Additional parameters section
            with col3:
                additional_params_names = st.multiselect(
                    "➕ Additional Parameters",
                    options=list(self.additional_parameters.keys()),
                    default=[],
                    help="These parameters will be used for conditional analysis and stratification"
                )
                additional_params = [self.additional_parameters[param] for param in additional_params_names]

            with col4:
                # Filter options section
                filters = {}

                # Patient filter
                if self.patient_id == "All Patients":

                    if st.checkbox("👤 Filter by Patient ID", value=False):
                        patient_ids = sorted(self.df[self.CN.RECORD_ID].unique())

                        selected_patients = st.multiselect(
                            "👤 Filter by Patient ID",
                            options=[f"Patient {id:d}" for id in patient_ids],
                            default=[],
                            help="Select specific patients to include in the analysis"
                        )
                        if selected_patients:
                            filters['patients'] = [int(p.split()[1]) for p in selected_patients]

                # Date range filter
                if st.checkbox("📅 Filter by Date Range", value=False):
                    col1a, col2a = st.columns(2)
                    with col1a:
                        min_date = self.df[self.CN.VISIT_DATE].min()
                        start_date = pd.to_datetime(st.date_input(
                            "Start Date",
                            value=min_date,
                            min_value=min_date,
                            max_value=self.df[self.CN.VISIT_DATE].max()
                        ))
                    with col2a:
                        max_date = self.df[self.CN.VISIT_DATE].max()
                        end_date = pd.to_datetime(st.date_input(
                            "End Date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date
                        ))
                    filters['date_range'] = (start_date, end_date)

            # Run analysis button
            run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        return dependent_var, independent_var, additional_params, filters, run_analysis

    def _filter_data(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Filter the dataframe based on user-selected filters.

        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to filter
        filters : Dict
            Dictionary of filter parameters

        Returns:
        -------
        pd.DataFrame
            Filtered dataframe
        """
        filtered_df = df.copy()

        # Filter by patients
        if 'patients' in filters and filters['patients']:
            filtered_df = filtered_df[filtered_df[self.CN.RECORD_ID].isin(filters['patients'])]

        # Filter by date range
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df[self.CN.VISIT_DATE]) >= start_date) &
                (pd.to_datetime(filtered_df[self.CN.VISIT_DATE]) <= end_date)
            ]

        return filtered_df


    def _fit_distributions(self, data: np.ndarray) -> Dict:
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

        for dist_name, distribution in self.available_distributions.items():
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
                    'aic': self._calculate_aic(distribution, params, data)
                }
            except Exception as e:
                st.warning(f"Could not fit {dist_name} distribution: {str(e)}")

        return results


    def _calculate_aic(self, distribution, params, data):
        """Calculate Akaike Information Criterion for the distribution fit."""
        k = len(params)
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        return 2 * k - 2 * log_likelihood

    def _plot_distribution_fit(self, data, results, var_name):
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

    def _create_distribution_analysis(self, df, dependent_var, dependent_var_name):
        """
        Create and display distribution analysis.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame to analyze
        dependent_var : str
            Column name of dependent variable
        dependent_var_name : str
            Display name of dependent variable
        """
        st.subheader("Distribution Analysis")

        # Helper function to build vertical tabs HTML for distribution parameters
        def build_vertical_tabs_html(dist_results, available_dists, best_dist):
            tabs_html = """
            <script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
            <style>
              .vertical-tabs {
                  display: flex;
                  border: 1px solid #ccc;
                  border-radius: 5px;
                  overflow: hidden;
                  font-family: Arial, sans-serif;
              }
              .tab-buttons {
                  display: flex;
                  flex-direction: column;
                  width: 200px;
                  background-color: #f9f9f9;
                  border-right: 1px solid #ccc;
              }
              .tab-buttons button {
                  background-color: inherit;
                  border: none;
                  outline: none;
                  padding: 12px 16px;
                  text-align: left;
                  cursor: pointer;
                  transition: background-color 0.3s;
                  font-weight: 600;
                  font-size: 14px;
              }
              .tab-buttons button:hover {
                  background-color: #ddd;
              }
              .tab-buttons button.active {
                  background-color: #ccc;
              }
              .tab-content-container {
                  flex-grow: 1;
                  padding: 20px;
              }
              .tab-content {
                  display: none;
              }
              .tab-content.active {
                  display: block;
              }
              .tab-content h3 {
                  margin-top: 0;
                  color: #333;
              }
              .tab-content p {
                  font-size: 16px;
                  margin: 10px 0;
              }
              .tab-content ul {
                  list-style: none;
                  padding: 0;
              }
              .tab-content ul li {
                  margin-bottom: 8px;
                  font-size: 14px;
                  line-height: 1.4;
              }
              .tab-content ul li strong {
                  color: #555;
              }
            </style>
            <div class="vertical-tabs">
              <div class="tab-buttons">
            """
            for idx, dist in enumerate(available_dists):
                active_class = "active" if idx == 0 else ""
                button_label = f"{dist} (Best Fit)" if dist == best_dist else dist
                tabs_html += f'<button class="tablinks {active_class}" onclick="openTab(event, \'{dist}\')">{button_label}</button>'
            tabs_html += """
              </div>
              <div class="tab-content-container">
            """
            for idx, dist in enumerate(available_dists):
                selected = dist_results[dist]
                active_class = "active" if idx == 0 else ""
                header_label = f"{dist} Distribution Parameters" + (" (Best Fit)" if dist == best_dist else "")
                content_html = f"<h3>{header_label}</h3>"
                if dist == "Normal":
                    content_html += """<p>$$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2} $$</p>"""
                    content_html += f"<ul><li><strong>μ (mean):</strong> {selected['params'][0]:.4f}</li>"
                    content_html += f"<li><strong>σ (std dev):</strong> {selected['params'][1]:.4f}</li></ul>"
                elif dist == "Log-Normal":
                    content_html += """<p>$$ f(x) = \\frac{1}{x\\,\\sigma\\sqrt{2\\pi}} e^{-\\frac{(\\ln x-\\mu)^2}{2\\sigma^2}} $$</p>"""
                    content_html += f"<ul><li><strong>σ (shape):</strong> {selected['params']['s']:.4f}</li>"
                    content_html += f"<li><strong>μ (scale):</strong> {selected['params']['scale']:.4f}</li></ul>"
                elif dist == "Weibull":
                    content_html += """<p>$$ f(x) = \\frac{k}{\\lambda} \\left(\\frac{x}{\\lambda}\\right)^{k-1} e^{-\\left(\\frac{x}{\\lambda}\\right)^k} $$</p>"""
                    content_html += f"<ul><li><strong>k (shape):</strong> {selected['params'][0]:.4f}</li>"
                    content_html += f"<li><strong>λ (scale):</strong> {selected['params'][1]:.4f}</li>"
                    if len(selected['params']) > 2:
                        content_html += f"<li><strong>Location (shift):</strong> {selected['params'][2]:.4f}</li>"
                    content_html += "</ul>"
                elif dist == "Exponential":
                    content_html += """<p>$$ f(x) = \\lambda e^{-\\lambda x} $$</p>"""
                    content_html += f"<ul><li><strong>λ (rate):</strong> {selected['params'][0]:.4f}</li>"
                    content_html += "<li><strong>Note:</strong> λ = 1/mean (average time between events)</li>"
                    if len(selected['params']) > 1:
                        content_html += f"<li><strong>Location (shift):</strong> {selected['params'][1]:.4f}</li>"
                    content_html += "</ul>"
                elif dist == "Gamma":
                    content_html += """<p>$$ f(x) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} x^{\\alpha-1} e^{-\\beta x} $$</p>"""
                    content_html += f"<ul><li><strong>α (shape):</strong> {selected['params'][0]:.4f}</li>"
                    content_html += f"<li><strong>β (rate):</strong> {selected['params'][1]:.4f}</li>"
                    content_html += "<li><strong>Note:</strong> β = 1/scale (average time between events)</li>"
                    if len(selected['params']) > 2:
                        content_html += f"<li><strong>Location (shift):</strong> {selected['params'][2]:.4f}</li>"
                    content_html += "</ul>"
                else:
                    content_html += "<p><strong>Distribution Parameters:</strong></p><ul>"
                    param_names = []
                    if hasattr(selected['distribution'], 'shapes') and selected['distribution'].shapes is not None:
                        param_names = selected['distribution'].shapes.split(",")
                    for i, param in enumerate(selected['params']):
                        param_name = param_names[i] if i < len(param_names) else f"Parameter {i+1}"
                        content_html += f"<li><strong>{param_name}:</strong> {param:.4f}</li>"
                    content_html += "</ul>"
                tabs_html += f'<div id="{dist}" class="tab-content {active_class}">{content_html}</div>'
            tabs_html += """
              </div>
            </div>
            <script>
              function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                  tabcontent[i].classList.remove("active");
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                  tablinks[i].classList.remove("active");
                }
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.classList.add("active");
              }
            </script>
            """
            return tabs_html

        with st.container():
            # Inject custom CSS for the analysis section
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

            st.write(f"Analyzing the distribution of {dependent_var_name}")

            # Extract and clean the data for the dependent variable
            data = df[dependent_var].values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                st.error(f"No valid data available for {dependent_var_name}")
                return

            # Display basic statistics
            st.markdown("### Basic Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{np.mean(valid_data):.2f}")
            col2.metric("Median", f"{np.median(valid_data):.2f}")
            col3.metric("Std Dev", f"{np.std(valid_data):.2f}")
            col4.metric("Sample Size", f"{len(valid_data)}")

            # Fit distributions to the data
            with st.spinner("Fitting probability distributions..."):
                dist_results = self._fit_distributions(data=valid_data)
                # dist_results = DistributionAnalyzer().fit_distributions(data=valid_data)

            if not dist_results:
                st.error("Failed to fit distributions to the data.")
                return

            # Display distribution fitting plot
            st.markdown("### Distribution Fitting")
            with st.expander("Understanding Probability Distributions"):
                st.markdown("""
                ### Common Probability Distributions and Their Interpretation

                **Weibull Distribution:**
                - **Description:** Models time-to-failure data and is widely used in reliability analysis
                - **Key Parameters:** Shape (k) and Scale (λ)
                - **Interpretation:**
                    - Shape < 1: Failure rate decreases over time (early failures)
                    - Shape = 1: Exponential distribution (constant failure rate)
                    - Shape > 1: Failure rate increases over time (aging/wear-out)
                - **Applications:** Survival analysis, reliability engineering, wind speed modeling

                **Normal (Gaussian) Distribution:**
                - **Description:** Symmetric bell-shaped distribution for continuous variables
                - **Key Parameters:** Mean (μ) and Standard Deviation (σ)
                - **Interpretation:**
                    - μ determines the center of the distribution
                    - σ controls the spread (larger σ = more variability)
                    - 68% of data within μ ± σ, 95% within μ ± 2σ
                - **Applications:** Natural phenomena, measurement errors, statistical inference

                **Gamma Distribution:**
                - **Description:** Models positive-valued, right-skewed data
                - **Key Parameters:** Shape (α) and Rate (β)
                - **Interpretation:**
                    - Shape controls the skewness (α < 1 = highly skewed, α > 1 = more symmetric)
                    - Rate controls the spread (higher β = more concentrated)
                - **Applications:** Waiting times, insurance claims, rainfall modeling
                """)
            buf = self._plot_distribution_fit(valid_data, dist_results, dependent_var_name)
            if buf is not None:
                st.image(buf, caption=f"Distribution Analysis for {dependent_var_name}")

            # Display goodness of fit statistics
            st.markdown("### Goodness of Fit Statistics")
            with st.expander("Understanding Goodness of Fit Metrics"):
                st.markdown("""
                ### Interpreting Goodness of Fit Statistics

                **Akaike Information Criterion (AIC):**
                - Measures the relative quality of statistical models for a given dataset
                - Lower AIC values indicate better model fit
                - Accounts for both model complexity and goodness of fit
                - Rule of thumb:
                    - ΔAIC < 2: Models are essentially equivalent
                    - 4 < ΔAIC < 7: Considerably less support for higher AIC model
                    - ΔAIC > 10: Strong evidence against higher AIC model
                - When comparing models, choose the one with the lowest AIC

                **Kolmogorov-Smirnov (KS) Statistic:**
                - Measures the maximum distance between empirical and theoretical distributions
                - Ranges from 0 to 1, where lower values indicate better fit
                - KS p-value interpretation:
                    - p < 0.05: Significant difference between distributions (reject null hypothesis)
                    - p ≥ 0.05: No significant difference (fail to reject null hypothesis)
                - Note: KS test is sensitive to sample size - large samples may show significant differences even for good fits

                **Using Both Metrics Together:**
                - First look at AIC to identify the best overall model
                - Then examine KS statistic and p-value to assess goodness of fit
                - A good model will have:
                    - Low AIC compared to other models
                    - Low KS statistic
                    - High KS p-value (≥ 0.05)
                """)
            # Build dataframe with fit statistics and display it
            fit_stats = []
            for dist_name, result in dist_results.items():
                fit_stats.append({
                    "Distribution": dist_name,
                    "AIC": result["aic"],
                    "KS Statistic": result["ks_stat"],
                    "p-value": result["p_value"]
                })
            fit_stats_df = pd.DataFrame(fit_stats).sort_values("AIC")
            st.dataframe(fit_stats_df.style.format({'p-value': lambda x: "{:.3f}".format(x) if x >= 0.001 else "{:.2e}".format(x)}))

            # Provide a CSV download link for the statistics
            csv = fit_stats_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="distribution_statistics.csv">Download Statistics (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Show distribution parameters using vertical tabs if statistics are available
            if not fit_stats_df.empty:
                available_dists = fit_stats_df['Distribution'].tolist()
                best_dist = fit_stats_df.iloc[0]["Distribution"]
                import streamlit.components.v1 as components
                tabs_html = build_vertical_tabs_html(dist_results, available_dists, best_dist)
                components.html(tabs_html, height=600)

            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def _fit_polynomial_models(X: np.ndarray, y: np.ndarray, max_degree: int = 5) -> Dict[int, Dict]:
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
    def _plot_polynomial_fit(X: np.ndarray, y: np.ndarray, model_results: Dict, degree: int,
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


    def _create_deterministic_component(self, df: pd.DataFrame, dependent_var: str, independent_var: str,
                                       dependent_var_name: str, independent_var_name: str):
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

            st.write(f"Analyzing the relationship between {dependent_var_name} and {independent_var_name}")
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
            X = df[independent_var].values
            y = df[dependent_var].values

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
                model_results = self._fit_polynomial_models(X=X, y=y, max_degree=max_degree)

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

            buf = self._plot_polynomial_fit(X, y, model_results, selected_degree,
                                           independent_var_name, dependent_var_name)

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
            equation = f"g({independent_var_name}) = {intercept:.4f}"
            for i, c in enumerate(coef[1:]):  # Skip the intercept term
                term = f"{abs(c):.4f} {independent_var_name.replace(' ', '_')}<sup>{i+1}</sup>"
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
                'Term': [f"{independent_var_name}^{i}" if i > 0 else "Intercept"
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


    def _create_random_component(self, independent_var_name: str):
        """
        Create and display the random component analysis.

        Parameters:
        ----------
        independent_var_name : str
            Display name of independent variable
        """
        st.subheader("Random Component Analysis")

        if self.residuals is None or self.deterministic_model is None:
            st.error("Please complete the Deterministic Component analysis first.")
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
            ### Random Component (η)

            After fitting the deterministic component g(X), the residuals represent the random component η:

            Y = g(X) + η

            where:
            - Y is the observed value of the dependent variable
            - g(X) is the deterministic component (polynomial function)
            - η is the random component representing natural variability

            This random component is modeled as a random variable with a probability distribution.
            """)

            # Residual analysis
            st.markdown("### Residual Analysis")

            # Create plot of residuals
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot histogram of residuals
            hist, bin_edges, _ = ax.hist(self.residuals, bins=20, density=True, alpha=0.6, color='gray', label='Residuals')

            # Fit normal distribution to residuals
            mu, std = stats.norm.fit(self.residuals)
            x = np.linspace(min(self.residuals), max(self.residuals), 1000)
            pdf = stats.norm.pdf(x, mu, std)

            # Plot the fitted normal distribution
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal: μ={mu:.3f}, σ={std:.3f}')

            # Add vertical line at x=0
            ax.axvline(x=0, color='green', linestyle='--', linewidth=1, label='Zero')

            ax.set_title('Distribution of Residuals (Random Component)')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Density')
            ax.legend()

            # Save plot to BytesIO
            buf_hist = BytesIO()
            fig.tight_layout()
            plt.savefig(buf_hist, format='png')
            buf_hist.seek(0)
            plt.close(fig)

            # Display the plot
            st.image(buf_hist, caption='Distribution of Residuals (Random Component)')

            # Basic statistics of residuals
            st.markdown("### Basic Statistics of Residuals")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{np.mean(self.residuals):.4f}")
            col2.metric("Median", f"{np.median(self.residuals):.4f}")
            col3.metric("Std Dev", f"{np.std(self.residuals):.4f}")
            col4.metric("Shapiro-Wilk p-value", f"{stats.shapiro(self.residuals)[1]:.4f}")

            # Interpretation of normality
            sw_pvalue = stats.shapiro(self.residuals)[1]
            if sw_pvalue < 0.05:
                st.warning("The residuals do not follow a normal distribution (Shapiro-Wilk p < 0.05).")
            else:
                st.success("The residuals appear to follow a normal distribution (Shapiro-Wilk p ≥ 0.05).")

            # Distribution fitting for residuals
            st.markdown("### Distribution Fitting for Residuals")

            # Fit multiple distributions to residuals
            with st.spinner("Fitting distributions to residuals..."):
                dist_results = self._fit_distributions(self.residuals)

            if not dist_results:
                st.error("Failed to fit distributions to the residuals.")
                return

            # Display goodness of fit statistics
            fit_stats = []
            for dist_name, result in dist_results.items():
                fit_stats.append({
                    "Distribution": dist_name,
                    "AIC": result["aic"],
                    "KS Statistic": result["ks_stat"],
                    "p-value": result["p_value"]
                })

            fit_stats_df = pd.DataFrame(fit_stats)
            fit_stats_df = fit_stats_df.sort_values("AIC")

            st.dataframe(fit_stats_df.style.highlight_min(subset=['AIC'], color='lightgreen'))

            # Best distribution for residuals
            best_dist = fit_stats_df.iloc[0]["Distribution"]
            st.markdown(f"### Best Fit: {best_dist} Distribution")

            # Store the best fit distribution
            best_dist_result = dist_results[best_dist]
            st.session_state.fitted_distribution = {
                'best_distribution': best_dist,
                'params'           : best_dist_result['params'],
                'aic'              : best_dist_result['aic'],
                'ks_stat'          : best_dist_result['ks_stat'],
                'p_value'          : best_dist_result['p_value']
            }
            self.fitted_distribution = st.session_state.fitted_distribution

            # Display plot of residuals with best fit
            buf_fit = self._plot_distribution_fit(self.residuals, dist_results, "Residuals")

            if buf_fit is not None:
                st.image(buf_fit, caption=f"Distribution Analysis for Residuals (Best Fit: {best_dist})")

            # Variance function modeling
            st.markdown("### Variance Function Modeling")

            st.markdown("""
            The variance of the random component may depend on the value of the independent variable.
            This relationship can be modeled as a variance function:

            Var(η) = σ² × (X)ᵏ

            where:
            - Var(η) is the variance of the random component
            - σ² is the base variance
            - X is the independent variable
            - k is the power parameter
            """)

            # Create plot to visualize residuals versus independent variable
            fig, ax = plt.subplots(figsize=(10, 6))

            try:
                # Try to get X values from the polynomial features
                feature_names = self.deterministic_model['poly'].get_feature_names_out()
                if len(feature_names) > 1:
                    X = feature_names[1]  # Get the name of the original feature
                    try:
                        # Safely extract numeric values from feature name
                        X_values = np.array([float(x.split('^')[0].split('_')[-1]) for x in X.split()])
                    except (IndexError, ValueError):
                        # If feature name parsing fails, use the original X values from the model fit
                        X_values = self.deterministic_model['model']._fit_X[:, 0]
                else:
                    # If only one feature exists, use the original X values from the model fit
                    X_values = self.deterministic_model['model']._fit_X[:, 0]
            except Exception as e:
                # Fallback to using the residuals' index if all else fails
                st.warning(f"Could not retrieve X values: {str(e)}. Using index as X values.")
                X_values = np.arange(len(self.residuals))

            # # Get X values from the deterministic model
            # X = self.deterministic_model['poly'].get_feature_names_out()[1]  # Get the name of the original feature
            # X_values = np.array([float(x.split('^')[0].split('_')[1]) for x in X.split()])

            # Plot residuals versus independent variable
            ax.scatter(X_values, self.residuals, color='blue', alpha=0.6)

            # Add horizontal line at y=0
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

            # Add loess smoothing or moving average to visualize trend
            try:
                # Sort X values and residuals together
                sorted_indices = np.argsort(X_values)
                sorted_X = X_values[sorted_indices]
                sorted_residuals = self.residuals[sorted_indices]

                # Compute moving average for visualization
                window_size = max(5, len(X_values) // 10)  # 10% of data points or at least 5
                moving_avg = np.convolve(sorted_residuals, np.ones(window_size)/window_size, mode='valid')
                moving_avg_X = sorted_X[window_size-1:]

                # Plot moving average
                ax.plot(moving_avg_X, moving_avg, color='green', linewidth=2, label=f'Moving Avg (window={window_size})')
            except Exception as e:
                st.warning(f"Could not compute moving average: {str(e)}")

            ax.set_title(f'Residuals vs. {independent_var_name}')
            ax.set_xlabel(independent_var_name)
            ax.set_ylabel('Residuals')
            ax.legend()

            # Save plot to BytesIO
            buf_resid = BytesIO()
            fig.tight_layout()
            plt.savefig(buf_resid, format='png')
            buf_resid.seek(0)
            plt.close(fig)

            # Display the plot
            st.image(buf_resid, caption=f'Residuals vs. {independent_var_name}')

            # Variance function estimation
            st.markdown("### Variance Function Estimation")

            # Create bins of X values and compute variance of residuals in each bin
            try:
                # Create bins of X values
                n_bins = min(10, len(X_values) // 5)  # At least 5 points per bin
                if n_bins < 3:
                    st.warning("Not enough data points to estimate variance function.")
                else:
                    bins = np.linspace(min(X_values), max(X_values), n_bins + 1)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    bin_variances = []

                    for i in range(n_bins):
                        # Get residuals in this bin
                        mask = (X_values >= bins[i]) & (X_values < bins[i+1])
                        bin_residuals = self.residuals[mask]

                        # Compute variance if there are enough points
                        if len(bin_residuals) >= 3:
                            bin_variances.append(np.var(bin_residuals))
                        else:
                            bin_variances.append(np.nan)

                    # Remove NaN values
                    valid_indices = ~np.isnan(bin_variances)
                    valid_centers = bin_centers[valid_indices]
                    valid_variances = np.array(bin_variances)[valid_indices]

                    if len(valid_centers) >= 3:
                        # Fit power function: variance = a * x^k
                        # Take logarithm: log(variance) = log(a) + k * log(x)
                        log_x = np.log(valid_centers)
                        log_var = np.log(valid_variances)

                        # Linear regression on log-transformed data
                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_var)

                        # Convert back to original scale
                        a = np.exp(intercept)
                        k = slope

                        # Create plot of variance function
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Plot bin variances
                        ax.scatter(valid_centers, valid_variances, color='blue', alpha=0.6, label='Bin Variances')

                        # Plot fitted variance function
                        x_range = np.linspace(min(valid_centers), max(valid_centers), 100)
                        variance_func = a * x_range**k
                        ax.plot(x_range, variance_func, color='red', linewidth=2,
                                label=f'Var(η) = {a:.3f} × ({independent_var_name})^{k:.3f}')

                        ax.set_title(f'Variance Function: Var(η) vs. {independent_var_name}')
                        ax.set_xlabel(independent_var_name)
                        ax.set_ylabel('Variance of Residuals')
                        ax.legend()

                        # Save plot to BytesIO
                        buf_var = BytesIO()
                        fig.tight_layout()
                        plt.savefig(buf_var, format='png')
                        buf_var.seek(0)
                        plt.close(fig)

                        # Display the plot
                        st.image(buf_var, caption=f'Variance Function: Var(η) vs. {independent_var_name}')

                        # Display the variance function equation
                        st.markdown(f"**Variance Function Equation:** Var(η) = {a:.4f} × ({independent_var_name})^{k:.4f}")

                        # Interpretation
                        if abs(k) < 0.1:
                            st.info("The variance appears to be approximately constant (homoscedasticity).")
                        elif k > 0:
                            st.info(f"The variance increases with {independent_var_name} (heteroscedasticity).")
                        else:
                            st.info(f"The variance decreases with {independent_var_name} (heteroscedasticity).")

                        # R-squared of the fit
                        st.metric("R² of Variance Function Fit", f"{r_value**2:.4f}")
                    else:
                        st.warning("Not enough valid bins to estimate variance function.")
            except Exception as e:
                st.error(f"Error estimating variance function: {str(e)}")

            # Add download button for residuals
            residuals_df = pd.DataFrame({
                "Residual": self.residuals
            })

            csv = residuals_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="residuals.csv">Download Residuals (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


    def _create_complete_model(self, df: pd.DataFrame, dependent_var: str, independent_var: str,
                               dependent_var_name: str, independent_var_name: str):
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
            X = df[independent_var].values
            y = df[dependent_var].values

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
                title=f'Probabilistic Model: {dependent_var_name} vs. {independent_var_name}',
                xaxis_title=independent_var_name,
                yaxis_title=dependent_var_name,
                hovermode='closest',
                legend=dict(x=0.02, y=0.98),
                width=800,
                height=500
            )

            # Add annotation for the model equation
            intercept = model.intercept_
            coef = model.coef_

            equation = f"E[{dependent_var_name}] = {intercept:.3f}"
            for i, c in enumerate(coef[1:]):  # Skip the first coefficient (it's just 1)
                if c >= 0:
                    equation += f" + {c:.3f}{independent_var_name}^{i+1}"
                else:
                    equation += f" - {abs(c):.3f}{independent_var_name}^{i+1}"

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
                title=f'Probabilistic Model: {dependent_var_name} vs. {independent_var_name}',
                xaxis_title=independent_var_name,
                yaxis_title=dependent_var_name,
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
            x_pred = st.slider(f"Select {independent_var_name} value for prediction:",
                              float(min(X)), float(max(X)), float((min(X) + max(X)) / 2))

            # Calculate deterministic prediction
            x_pred_2d = np.array([[x_pred]])
            x_pred_poly = poly.transform(x_pred_2d)
            y_pred_mean = model.predict(x_pred_poly)[0]

            # Calculate prediction intervals
            y_pred_lower_95 = y_pred_mean - z_95 * residual_std
            y_pred_upper_95 = y_pred_mean + z_95 * residual_std

            # Display the predictions
            st.markdown(f"#### For {independent_var_name} = {x_pred:.2f}:")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Deterministic Prediction:**")
                st.markdown(f"{dependent_var_name} = {y_pred_mean:.2f}")

            with col2:
                st.markdown("**Probabilistic Prediction:**")
                st.markdown(f"{dependent_var_name} = {y_pred_mean:.2f} ± {z_95 * residual_std:.2f}")
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
                title=f'Probability Distribution of {dependent_var_name} at {independent_var_name}={x_pred:.2f}',
                xaxis_title=dependent_var_name,
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
                'dependent_var': dependent_var_name,
                'independent_var': independent_var_name,
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

    def _create_advanced_statistics(self, df: pd.DataFrame, dependent_var: str, independent_var: str,
                                  dependent_var_name: str, independent_var_name: str):
        """
        Create and display advanced statistical methods for sophisticated analysis.

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
        st.subheader("Advanced Statistical Methods")

        # Show info about requirements
        st.info("""
        This section provides access to advanced statistical methods for more sophisticated analysis.
        Note that some of these methods require additional Python packages that may need to be installed.
        """)

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

            # Create expandable sections for each advanced method
            with st.expander("Polynomial Chaos Expansion (PCE)", expanded=False):
                st.markdown("""
                ### Polynomial Chaos Expansion

                Polynomial Chaos Expansion (PCE) is a method for representing random variables using orthogonal polynomials.
                It allows for efficient propagation of uncertainty through complex models. PCE represents the output
                random variable as a series expansion of orthogonal polynomials of the input random variables.

                PCE implementation requires the following packages:
                - `chaospy` for polynomial chaos expansions
                - `scikit-learn` for regression and cross-validation

                This method is particularly useful for:
                - Efficiently computing statistical moments of model outputs
                - Sensitivity analysis to identify important parameters
                - Creating surrogate models for computationally expensive simulations
                """)

                # Add placeholder for PCE implementation
                st.warning("PCE implementation will be added in a future update.")

                # Add reference
                st.markdown("""
                **References:**
                1. Xiu, D., & Karniadakis, G. E. (2002). The Wiener-Askey polynomial chaos for stochastic differential equations. SIAM Journal on Scientific Computing, 24(2), 619-644.
                2. Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. Reliability Engineering & System Safety, 93(7), 964-979.
                """)

            with st.expander("Bayesian Hierarchical Modeling", expanded=False):
                st.markdown("""
                ### Bayesian Hierarchical Modeling

                Bayesian hierarchical modeling is a statistical framework that allows for modeling of complex,
                multi-level data structures. It is particularly useful for medical data where measurements
                are nested within patients, and patients are nested within groups.

                Bayesian implementation requires the following packages:
                - `pymc3` or `pymc` for Bayesian modeling
                - `arviz` for visualization and diagnostics

                Benefits of Bayesian hierarchical models:
                - Accounts for patient-level variations and group-level effects
                - Provides full posterior distributions for all parameters
                - Allows for incorporation of prior knowledge
                - Handles missing data and unbalanced designs naturally
                """)

                # Add placeholder for Bayesian implementation
                st.warning("Bayesian hierarchical modeling implementation will be added in a future update.")

                # Add reference
                st.markdown("""
                **References:**
                1. Gelman, A., & Hill, J. (2006). Data analysis using regression and multilevel/hierarchical models. Cambridge University Press.
                2. McElreath, R. (2020). Statistical rethinking: A Bayesian course with examples in R and Stan. CRC press.
                """)

            with st.expander("Mixed-Effects Modeling", expanded=False):
                st.markdown("""
                ### Mixed-Effects Modeling

                Mixed-effects models (also called multilevel models) account for both fixed effects
                (population-level) and random effects (individual-level variations). These models
                are particularly useful for longitudinal data where measurements are taken from the
                same patients over time.

                Mixed-effects implementation requires the following packages:
                - `statsmodels` for basic mixed-effects models
                - `lme4` or `PyHLM` for more complex hierarchical linear models

                Key features:
                - Separates within-subject and between-subject variability
                - Accounts for correlation between repeated measurements
                - Handles unbalanced designs and missing data
                - Provides both population-level and individual-level estimates
                """)

                # Add placeholder for mixed-effects implementation
                st.warning("Mixed-effects modeling implementation will be added in a future update.")

                # Add reference
                st.markdown("""
                **References:**
                1. Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. Journal of Statistical Software, 67(1), 1-48.
                2. Pinheiro, J. C., & Bates, D. M. (2000). Mixed-effects models in S and S-PLUS. Springer.
                """)

            with st.expander("Variance Function Modeling", expanded=False):
                st.markdown("""
                ### Variance Function Modeling

                Variance function modeling extends standard regression by explicitly modeling how the
                variance of the response variable changes with predictor variables or fitted values.
                This approach is particularly useful for heteroscedastic data, where the variance is not constant.

                Our implementation models the variance as a power function of the independent variable:

                $Var(Y|X) = \sigma^2 \cdot |X|^p$

                where:
                - $Var(Y|X)$ is the conditional variance of Y given X
                - $\sigma^2$ is a base variance parameter
                - $X$ is the independent variable or predicted mean
                - $p$ is the power parameter determining how variance changes with X
                """)

                # Check if we have already computed variance function in random component
                if hasattr(self, 'residuals') and self.deterministic_model is not None:
                    st.info("Variance function modeling has already been performed in the Random Component section.")
                    # Add a button to navigate to that section
                    if st.button("Go to Variance Function in Random Component"):
                        st.session_state.active_stochastic_tab = "Random Component"
                else:
                    st.warning("Please complete the Deterministic and Random Component analyses first to access variance function modeling.")

                # Add reference
                st.markdown("""
                **References:**
                1. Carroll, R. J., & Ruppert, D. (1988). Transformation and weighting in regression. Chapman and Hall.
                2. Davidian, M., & Carroll, R. J. (1987). Variance function estimation. Journal of the American Statistical Association, 82(400), 1079-1091.
                """)

            with st.expander("Conditional Distribution Analysis", expanded=False):
                st.markdown("""
                ### Conditional Distribution Analysis

                Conditional distribution analysis examines how the distribution of the dependent variable
                changes across different values or ranges of the independent variable(s). This approach
                provides a more complete picture of the relationship between variables beyond summary statistics.

                This method includes:
                - Calculating conditional distributions for different ranges of X
                - Visualizing how distribution parameters change with X
                - Testing for differences in distributions across groups
                - Modeling the changes in distribution shape, not just location
                """)

                # Add implementation for conditional distribution analysis
                if self.deterministic_model is not None and self.residuals is not None and len(self.residuals) > 0:
                    try:
                        # Get data from the model
                        X = self.deterministic_model['poly'].get_feature_names_out()[1]  # Get the name of the original feature
                        X_values = np.array([float(x.split('^')[0].split('_')[1]) for x in X.split()])

                        # Create conditional distributions
                        st.markdown("#### Analyze Distribution Across Ranges")

                        # Let user define number of bins
                        n_bins = st.slider("Number of ranges to analyze", min_value=2, max_value=5, value=3)

                        # Create bins
                        bin_edges = np.linspace(min(X_values), max(X_values), n_bins + 1)

                        # Create figure for visualization
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Color palette
                        colors = ['blue', 'green', 'red', 'purple', 'orange']

                        # Process each bin
                        for i in range(n_bins):
                            # Get data in this bin
                            mask = (X_values >= bin_edges[i]) & (X_values < bin_edges[i+1])
                            bin_residuals = self.residuals[mask]

                            if len(bin_residuals) >= 5:  # Ensure enough points for distribution fitting
                                # Fit normal distribution
                                mu, std = stats.norm.fit(bin_residuals)

                                # Create x values for the distribution
                                x = np.linspace(min(self.residuals), max(self.residuals), 1000)

                                # Calculate PDF
                                pdf = stats.norm.pdf(x, mu, std)

                                # Plot the PDF
                                ax.plot(x, pdf, color=colors[i % len(colors)], linewidth=2,
                                        label=f'{independent_var_name} ∈ [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]: μ={mu:.3f}, σ={std:.3f}')

                                # Add histogram for this bin (scaled to match PDFs)
                                hist, bin_edges_hist = np.histogram(bin_residuals, bins=10, density=True)
                                bin_centers = (bin_edges_hist[:-1] + bin_edges_hist[1:]) / 2
                                ax.scatter(bin_centers, hist, color=colors[i % len(colors)], alpha=0.5, s=20)
                            else:
                                st.warning(f"Not enough data points in range [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}] for distribution fitting")

                        ax.set_title('Conditional Distributions of Residuals')
                        ax.set_xlabel('Residuals')
                        ax.set_ylabel('Density')
                        ax.legend()

                        # Save plot to BytesIO
                        buf = BytesIO()
                        fig.tight_layout()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        plt.close(fig)

                        # Display the plot
                        st.image(buf, caption='Conditional Distributions Across Ranges')

                        # Create a table of conditional statistics
                        st.markdown("#### Conditional Statistics")

                        # Create dataframe for statistics
                        stats_data = []

                        for i in range(n_bins):
                            # Get data in this bin
                            mask = (X_values >= bin_edges[i]) & (X_values < bin_edges[i+1])
                            bin_residuals = self.residuals[mask]
                            bin_x = X_values[mask]

                            if len(bin_residuals) >= 5:
                                # Calculate statistics
                                stats_data.append({
                                    'Range': f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]',
                                    'Count': len(bin_residuals),
                                    'Mean': np.mean(bin_residuals),
                                    'Std Dev': np.std(bin_residuals),
                                    'Median': np.median(bin_residuals),
                                    'Skewness': stats.skew(bin_residuals),
                                    'Kurtosis': stats.kurtosis(bin_residuals),
                                    'Mean X': np.mean(bin_x) if len(bin_x) > 0 else np.nan
                                })

                        # Create dataframe and display
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df)

                        # Test for equality of means across bins
                        if len(stats_data) >= 2:
                            st.markdown("#### Statistical Tests")

                            # Prepare data for ANOVA
                            bin_groups = []
                            for i in range(n_bins):
                                mask = (X_values >= bin_edges[i]) & (X_values < bin_edges[i+1])
                                bin_residuals = self.residuals[mask]
                                if len(bin_residuals) >= 5:
                                    bin_groups.append(bin_residuals)

                            if len(bin_groups) >= 2:
                                # Run ANOVA
                                f_stat, p_value = stats.f_oneway(*bin_groups)

                                st.markdown(f"**ANOVA test for equality of means across ranges:**")
                                st.markdown(f"F-statistic: {f_stat:.4f}")
                                st.markdown(f"p-value: {p_value:.4f}")

                                if p_value < 0.05:
                                    st.warning("The means of the residuals differ significantly across the ranges (p < 0.05). This suggests that the deterministic component may not fully capture the relationship between the variables.")
                                else:
                                    st.success("The means of the residuals do not differ significantly across the ranges (p ≥ 0.05). This suggests that the deterministic component adequately captures the relationship between the variables.")

                                # Run Bartlett's test for equal variances
                                stat, p_value = stats.bartlett(*bin_groups)

                                st.markdown(f"**Bartlett's test for equality of variances across ranges:**")
                                st.markdown(f"Test statistic: {stat:.4f}")
                                st.markdown(f"p-value: {p_value:.4f}")

                                if p_value < 0.05:
                                    st.warning("The variances of the residuals differ significantly across the ranges (p < 0.05). This suggests heteroscedasticity, which should be accounted for in the model.")
                                else:
                                    st.success("The variances of the residuals do not differ significantly across the ranges (p ≥ 0.05). This suggests homoscedasticity.")
                    except Exception as e:
                        st.error(f"Error in conditional distribution analysis: {str(e)}")
                else:
                    st.warning("Please complete the Deterministic and Random Component analyses first to access conditional distribution analysis.")

                # Add reference
                st.markdown("""
                **References:**
                1. Gelman, A., & Hill, J. (2006). Data analysis using regression and multilevel/hierarchical models. Cambridge University Press.
                2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
                """)

            st.markdown('</div>', unsafe_allow_html=True)

    def _create_uncertainty_quantification_tools(self, df: pd.DataFrame, dependent_var: str, independent_var: str,
                                                dependent_var_name: str, independent_var_name: str):
        """
        Create and display uncertainty quantification tools for risk assessment and decision support.

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
                    f"Enter {independent_var_name} value for prediction:",
                    min_value=float(df[independent_var].min()),
                    max_value=float(df[independent_var].max()),
                    value=float(df[independent_var].median()),
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
                            st.markdown(f"**Input {independent_var_name}:** {prediction_x:.2f}")
                            st.markdown(f"**Predicted {dependent_var_name}:** {y_pred:.2f}")
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
                                xaxis_title=independent_var_name,
                                yaxis_title=dependent_var_name,
                                height=400,
                                width=400
                            )

                            st.plotly_chart(fig)

                            # Add interpretation
                            st.markdown("#### Interpretation")
                            st.markdown(f"""
                            - The predicted value of {dependent_var_name} is **{y_pred:.2f}**.
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
                    f"Threshold value for {dependent_var_name}:",
                    value=float(df[dependent_var].median()),
                    step=0.1
                )

                # Input for future X value
                future_x = st.number_input(
                    f"Enter future {independent_var_name} value:",
                    min_value=float(df[independent_var].min()),
                    max_value=float(df[independent_var].max() * 1.5),  # Allow some extrapolation
                    value=float(df[independent_var].median()),
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
                                st.markdown(f"**Input {independent_var_name}:** {future_x:.2f}")
                                st.markdown(f"**Predicted {dependent_var_name}:** {y_pred:.2f}")

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
                                **Probability of {dependent_var_name} being {threshold_type.lower()} {threshold_value}:**
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
                                    title=f'Probability Distribution for {dependent_var_name} at {independent_var_name}={future_x}',
                                    xaxis_title=dependent_var_name,
                                    yaxis_title='Probability Density',
                                    height=400,
                                    width=400
                                )

                                st.plotly_chart(fig)

                                # Add interpretation
                                st.markdown("#### Interpretation")
                                interpretation_text = f"""
                                - At {independent_var_name} = {future_x}, the predicted {dependent_var_name} is **{y_pred:.2f}**.
                                - There is a **{prob_percent:.1f}%** probability that {dependent_var_name} will be {threshold_type.lower()} {threshold_value}.
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
                                        The {independent_var_name} value to predict for
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
                                            "dependent_var": "{dependent_var_name}",
                                            "independent_var": "{independent_var_name}",
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
                        "dependent": dependent_var_name,
                        "independent": independent_var_name
                    },
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "description": f"Wound healing prediction model for {dependent_var_name} based on {independent_var_name}"
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
                        "dependent": dependent_var_name,
                        "independent": independent_var_name
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

    def render(self) -> None:
        """
        Render the Stochastic Modeling tab content.
        """
        st.header("Probabilistic Modeling of Wound Healing")

        # Introduction section explaining the probabilistic modeling approach
        with st.expander("Introduction to Probabilistic Modeling", expanded=False):
            st.markdown("""
            ## Introduction to Probabilistic Modeling

            Traditional deterministic models of wound healing assume that measurements represent fixed,
            exact values. However, biological systems have inherent variability and uncertainty that
            deterministic models cannot fully capture.

            ### Deterministic vs. Probabilistic Approaches

            **Deterministic Approach:**
            - Treats measurements as fixed values
            - Produces single-valued predictions
            - Cannot account for natural biological variability
            - Limited ability to quantify uncertainty

            **Probabilistic Approach:**
            - Treats measurements as random variables with probability distributions
            - Produces probability distributions rather than point estimates
            - Accounts for natural biological variability
            - Provides robust uncertainty quantification
            - Enables risk assessment and decision-making under uncertainty

            This analysis implements a two-component model approach:
            1. A deterministic component representing the expected trend
            2. A random component representing the variability around that trend
            """)

        # Parameter selection UI
        dependent_var, independent_var, additional_params, filters, run_analysis = self._parameter_selection_ui()

        # Get display names for variables
        dependent_var_name = next((name for name, col in self.dependent_variables.items() if col == dependent_var), dependent_var)
        independent_var_name = next((name for name, col in self.independent_variables.items() if col == independent_var), independent_var)

        # Analysis results when the Run Analysis button is clicked
        if run_analysis or any([self.deterministic_model, self.residuals, self.fitted_distribution]):
            st.subheader("Analysis Results")

            # Filter the data
            with st.spinner("Filtering data..."):
                filtered_df = self._filter_data(self.df, filters)

            if filtered_df.empty:
                st.error("No data available after applying filters. Please adjust your filter criteria.")
                return

            # Display info about selected data
            st.write(f"Analysis based on {filtered_df.shape[0]} records")

            # Create tabs for different analysis components
            tabs = st.tabs([
                "Distribution Analysis",
                "Deterministic Component",
                "Random Component",
                "Complete Model",
                "Advanced Statistics",
                "Uncertainty Tools"
            ])

            # Distribution Analysis Tab
            with tabs[0]:
                self._create_distribution_analysis(filtered_df, dependent_var, dependent_var_name)

            # Deterministic Component Tab
            with tabs[1]:
                self._create_deterministic_component(filtered_df, dependent_var, independent_var, dependent_var_name, independent_var_name)

            # Random Component Tab
            with tabs[2]:
                self._create_random_component(independent_var_name)

            # Complete Model Tab
            with tabs[3]:
                self._create_complete_model(filtered_df, dependent_var, independent_var, dependent_var_name, independent_var_name)

            # Advanced Statistics Tab
            with tabs[4]:
                self._create_advanced_statistics(filtered_df, dependent_var, independent_var, dependent_var_name, independent_var_name)

            # Uncertainty Quantification Tools Tab
            with tabs[5]:
                self._create_uncertainty_quantification_tools(filtered_df, dependent_var, independent_var, dependent_var_name, independent_var_name)
