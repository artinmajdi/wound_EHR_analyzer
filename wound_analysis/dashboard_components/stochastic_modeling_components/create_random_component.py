from io import BytesIO
import base64
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import streamlit as st

from wound_analysis.dashboard_components.stochastic_modeling_components.stats_utils import StatsUtils

class CreateRandomComponent:


    def __init__(self, parent: 'StochasticModelingTab'):
        self.parent               = parent
        self.CN                   = parent.CN

        # previously calculated variables
        self.deterministic_model     = parent.deterministic_model
        self.residuals               = parent.residuals
        self.independent_var_name    = parent.independent_var_name
        self.available_distributions = parent.available_distributions
        self.polynomial_type         = parent.polynomial_type
        self.independent_var         = parent.independent_var

        # Instance variables
        self.fitted_distribution  = None


    def render(self) -> Dict[str, Any]:
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
                dist_results = StatsUtils.fit_distributions(data=self.residuals, available_distributions=self.available_distributions)

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
            self.fitted_distribution = {
                'best_distribution': best_dist,
                'params'           : best_dist_result['params'],
                'aic'              : best_dist_result['aic'],
                'ks_stat'          : best_dist_result['ks_stat'],
                'p_value'          : best_dist_result['p_value']
            }

            # Display plot of residuals with best fit
            buf_fit = StatsUtils.plot_distribution_fit(data=self.residuals, results=dist_results, var_name="Residuals")

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

            # Get X values based on polynomial type
            is_hermite = self.polynomial_type == 'hermite' or self.deterministic_model.get('is_hermite', False)

            try:
                if is_hermite:
                    # For Hermite polynomials, we need to get the original X values
                    # We can use the fit data from the model
                    # First, try to get the original X from the parent's dataframe
                    from_df = self.parent.df
                    valid_mask = ~np.isnan(from_df[self.independent_var])
                    X_values = from_df[self.independent_var].values[valid_mask]

                    # Ensure we have the right number of X values
                    if len(X_values) != len(self.residuals):
                        # Fallback to just using indices
                        X_values = np.arange(len(self.residuals))
                else:
                    # For regular polynomials
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
            except Exception as e:
                # Fallback to using indices if everything else fails
                st.warning(f"Error retrieving X values: {str(e)}. Using index as X values.")
                X_values = np.arange(len(self.residuals))

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

            ax.set_title(f'Residuals vs. {self.independent_var_name}')
            ax.set_xlabel(self.independent_var_name)
            ax.set_ylabel('Residuals')
            ax.legend()

            # Save plot to BytesIO
            buf_resid = BytesIO()
            fig.tight_layout()
            plt.savefig(buf_resid, format='png')
            buf_resid.seek(0)
            plt.close(fig)

            # Display the plot
            st.image(buf_resid, caption=f'Residuals vs. {self.independent_var_name}')

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
                                label=f'Var(η) = {a:.3f} × ({self.independent_var_name})^{k:.3f}')

                        ax.set_title(f'Variance Function: Var(η) vs. {self.independent_var_name}')
                        ax.set_xlabel(self.independent_var_name)
                        ax.set_ylabel('Variance of Residuals')
                        ax.legend()

                        # Save plot to BytesIO
                        buf_var = BytesIO()
                        fig.tight_layout()
                        plt.savefig(buf_var, format='png')
                        buf_var.seek(0)
                        plt.close(fig)

                        # Display the plot
                        st.image(buf_var, caption=f'Variance Function: Var(η) vs. {self.independent_var_name}')

                        # Display the variance function equation
                        st.markdown(f"**Variance Function Equation:** Var(η) = {a:.4f} × ({self.independent_var_name})^{k:.4f}")

                        # Interpretation
                        if abs(k) < 0.1:
                            st.info("The variance appears to be approximately constant (homoscedasticity).")
                        elif k > 0:
                            st.info(f"The variance increases with {self.independent_var_name} (heteroscedasticity).")
                        else:
                            st.info(f"The variance decreases with {self.independent_var_name} (heteroscedasticity).")

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

        return self.fitted_distribution
