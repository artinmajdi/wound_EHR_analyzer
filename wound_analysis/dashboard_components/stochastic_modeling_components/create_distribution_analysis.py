import base64
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
from wound_analysis.dashboard_components.stochastic_modeling_components.stats_utils import StatsUtils


class CreateDistributionAnalysis:

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

            st.write(f"Analyzing the distribution of {self.dependent_var_name}")

            # Extract and clean the data for the dependent variable
            data = self.df[self.dependent_var].values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                st.error(f"No valid data available for {self.dependent_var_name}")
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
                dist_results = StatsUtils.fit_distributions(data=valid_data)
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
            buf = StatsUtils.plot_distribution_fit(data=valid_data, results=dist_results, var_name=self.dependent_var_name)

            if buf is not None:
                st.image(buf, caption=f"Distribution Analysis for {self.dependent_var_name}")

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

