from datetime import datetime
from io import BytesIO
import json
import pickle
from typing import TYPE_CHECKING

from wound_analysis.utils.stochastic_modeling.advanced_statistics import AdvancedStatistics

if TYPE_CHECKING:
    from wound_analysis.dashboard_components.stochastic_modeling_tab import StochasticModelingTab

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import streamlit as st

class CreateAdvancedStatistics:

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

        self.advanced_statistics  = AdvancedStatistics()


    def render(self) -> None:
        """
        Create and display advanced statistical methods for sophisticated analysis.

        Parameters:
        ----------
        self.df : pd.DataFrame
            DataFrame to analyze
        self.dependent_var : str
            Column name of dependent variable
        self.independent_var : str
            Column name of independent variable
        self.dependent_var_name : str
            Display name of dependent variable
        self.independent_var_name : str
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

                # Add implementation
                try:
                    # Get data for PCE analysis
                    X = self.df[self.independent_var].values
                    y = self.df[self.dependent_var].values

                    # Create controls for PCE parameters
                    st.markdown("#### PCE Configuration")
                    degree = st.slider("Polynomial degree", min_value=1, max_value=5, value=3)
                    distribution = st.selectbox("Distribution type", ["normal", "uniform"], index=0)

                    # Run PCE analysis
                    pce_results = self.advanced_statistics.polynomial_chaos_expansion(
                        x=X,
                        y=y,
                        degree=degree,
                        distribution=distribution
                    )

                    # Display results
                    st.markdown("#### PCE Results")

                    # Show model performance metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{pce_results['r2']:.4f}")
                    with col2:
                        st.metric("Mean Squared Error", f"{pce_results['mse']:.4f}")

                    # Show sensitivity analysis
                    st.markdown("#### Sensitivity Analysis")
                    sensitivity = pce_results['sensitivity']

                    # Create bar plot for first-order indices
                    fig, ax = plt.subplots(figsize=(8, 4))
                    indices = list(sensitivity['first_order'].keys())
                    values = list(sensitivity['first_order'].values())

                    ax.bar(indices, values)
                    ax.set_title('First-order Sensitivity Indices')
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Sensitivity')

                    # Save plot to BytesIO
                    buf = BytesIO()
                    fig.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close(fig)

                    # Display the plot
                    st.image(buf, caption='PCE Sensitivity Analysis')

                    # Cross-validation analysis
                    st.markdown("#### Cross-validation Analysis")
                    n_folds = st.slider("Number of CV folds", min_value=2, max_value=10, value=5)

                    cv_results = self.advanced_statistics.cross_validate(
                        x=X,
                        y=y,
                        n_folds=n_folds,
                        degree=degree
                    )

                    # Display CV results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean CV Score", f"{cv_results['mean_score']:.4f}")
                    with col2:
                        st.metric("CV Score Std", f"{cv_results['std_score']:.4f}")

                    # Plot CV predictions
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Sort all points by x for plotting
                    sort_idx = np.argsort(X)
                    X_sorted = X[sort_idx]
                    y_sorted = y[sort_idx]

                    # Plot original data
                    ax.scatter(X_sorted, y_sorted, alpha=0.5, label='Original Data')

                    # Plot CV predictions
                    all_preds = []
                    for test_idx, y_pred in cv_results['predictions']:
                        all_preds.extend(list(zip(X[test_idx], y_pred)))

                    all_preds = np.array(all_preds)
                    sort_idx = np.argsort(all_preds[:, 0])
                    all_preds = all_preds[sort_idx]

                    ax.plot(all_preds[:, 0], all_preds[:, 1], 'r-', label='CV Predictions')

                    ax.set_title('Cross-validation Predictions')
                    ax.set_xlabel(self.independent_var_name)
                    ax.set_ylabel(self.dependent_var_name)
                    ax.legend()

                    # Save plot to BytesIO
                    buf = BytesIO()
                    fig.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close(fig)

                    # Display the plot
                    st.image(buf, caption='PCE Cross-validation Results')

                except Exception as e:
                    st.error(f"Error in PCE analysis: {str(e)}")
                    st.warning("Please ensure your data is suitable for PCE analysis (numeric, non-null values).")


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

                # Add reference
                st.markdown("""
                **References:**
                1. Gelman, A., & Hill, J. (2006). Data analysis using regression and multilevel/hierarchical models. Cambridge University Press.
                2. McElreath, R. (2020). Statistical rethinking: A Bayesian course with examples in R and Stan. CRC press.
                """)

                try:
                    import pymc3 as pm
                    import arviz as az

                    # Get data for Bayesian analysis
                    X      = self.df[self.independent_var].values
                    y      = self.df[self.dependent_var].values
                    groups = self.df[self.CN.RECORD_ID].values if self.patient_id == "All Patients" else None

                    # Create controls for Bayesian model
                    st.markdown("#### Bayesian Model Configuration")
                    n_samples   = st.slider("Number of MCMC samples", min_value=500, max_value=5000, value=2000)
                    n_tune      = st.slider("Number of tuning steps", min_value=100, max_value=2000, value=1000)
                    random_seed = st.number_input("Random seed", min_value=1, value=42)

                    # Run Bayesian analysis
                    results = self.advanced_statistics.bayesian_hierarchical_modeling(
                        x=X,
                        y=y,
                        groups=groups,
                        n_samples=n_samples,
                        n_tune=n_tune,
                        random_seed=random_seed
                    )

                    if results is not None:
                        # Display results
                        st.markdown("#### Bayesian Analysis Results")

                        # Summary statistics
                        st.dataframe(results['summary'])

                        # Plot posterior distributions
                        st.markdown("#### Posterior Distributions")
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                        # Plot for alpha
                        az.plot_posterior(results['trace'], var_names=['alpha'], ax=axes[0])
                        axes[0].set_title('Intercept (α)')

                        # Plot for beta
                        az.plot_posterior(results['trace'], var_names=['beta'], ax=axes[1])
                        axes[1].set_title('Slope (β)')

                        # Plot for sigma
                        az.plot_posterior(results['trace'], var_names=['sigma'], ax=axes[2])
                        axes[2].set_title('Standard Deviation (σ)')

                        # Save plot to BytesIO
                        buf = BytesIO()
                        fig.tight_layout()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        plt.close(fig)

                        # Display the plot
                        st.image(buf, caption='Posterior Distributions')

                        # Plot trace plots
                        st.markdown("#### Trace Plots")
                        fig = plt.figure(figsize=(15, 10))
                        az.plot_trace(results['trace'], var_names=['alpha', 'beta', 'sigma'])

                        # Save plot to BytesIO
                        buf = BytesIO()
                        fig.tight_layout()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        plt.close(fig)

                        # Display the plot
                        st.image(buf, caption='MCMC Trace Plots')

                        # Display model diagnostics
                        st.markdown("#### Model Diagnostics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{results['diagnostics']['r2_score']:.4f}")
                        with col2:
                            st.metric("WAIC", f"{results['diagnostics']['waic'].waic:.4f}")
                        with col3:
                            st.metric("LOO", f"{results['diagnostics']['loo'].loo:.4f}")

                except Exception as e:
                    st.error(f"Error in Bayesian analysis: {str(e)}")
                    st.warning("Please ensure your data is suitable for Bayesian analysis (numeric, non-null values).")

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

                # Add reference
                st.markdown("""
                **References:**
                1. Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. Journal of Statistical Software, 67(1), 1-48.
                2. Pinheiro, J. C., & Bates, D. M. (2000). Mixed-effects models in S and S-PLUS. Springer.
                """)

                try:
                    # Get data for mixed-effects analysis
                    X = self.df[self.independent_var].values
                    y = self.df[self.dependent_var].values
                    groups = self.df[self.CN.RECORD_ID].values

                    # Create controls for mixed-effects model
                    st.markdown("#### Mixed-Effects Model Configuration")

                    # Select random effects
                    available_effects = [self.independent_var] + (
                        [param for param in self.df.columns if param in self.additional_parameters.values()]
                    )
                    random_effects = st.multiselect(
                        "Select random effects",
                        options=available_effects,
                        default=[self.independent_var],
                        help="Variables that will have random effects across groups"
                    )

                    fixed_effects = st.multiselect(
                        "Select fixed effects",
                        options=available_effects,
                        default=[self.independent_var],
                        help="Variables that will have fixed effects"
                    )

                    # Run mixed-effects analysis
                    results = self.advanced_statistics.mixed_effects_modeling(
                        x=X,
                        y=y,
                        groups=groups,
                        random_effects=random_effects,
                        fixed_effects=fixed_effects
                    )

                    if results is not None:
                        # Display results
                        st.markdown("#### Mixed-Effects Model Results")
                        st.text(results['model'].summary().as_text())

                        # Display random effects
                        st.markdown("#### Random Effects")
                        st.dataframe(results['random_effects'])

                        # Display variance components
                        st.markdown("#### Variance Components")
                        variance_df = pd.DataFrame({
                            'Component': ['Random Effects', 'Residual'],
                            'Variance': [
                                results['variance_components']['random_effects'][0],
                                results['variance_components']['residual']
                            ]
                        })
                        st.dataframe(variance_df)

                        # Display model diagnostics
                        st.markdown("#### Model Diagnostics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{results['diagnostics']['r2_score']:.4f}")
                        with col2:
                            st.metric("AIC", f"{results['diagnostics']['aic']:.4f}")
                        with col3:
                            st.metric("BIC", f"{results['diagnostics']['bic']:.4f}")

                        # Plot predictions vs observations
                        st.markdown("#### Model Predictions")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y, results['predictions'], alpha=0.5)
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                        ax.set_xlabel("Observed Values")
                        ax.set_ylabel("Predicted Values")
                        ax.set_title("Observed vs. Predicted Values")

                        # Save plot to BytesIO
                        buf = BytesIO()
                        fig.tight_layout()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        plt.close(fig)

                        # Display the plot
                        st.image(buf, caption='Model Predictions')

                        # Display likelihood ratio test results
                        st.markdown("#### Likelihood Ratio Test")
                        st.write(f"Test statistic: {results['diagnostics']['lr_test']['statistic']:.4f}")
                        st.write(f"p-value: {results['diagnostics']['lr_test']['p_value']:.4f}")

                        if results['diagnostics']['lr_test']['p_value'] < 0.05:
                            st.success("The random effects are significant (p < 0.05). This suggests that the mixed-effects model is more appropriate than a simple fixed-effects model.")
                        else:
                            st.warning("The random effects are not significant (p ≥ 0.05). A simpler fixed-effects model might be sufficient.")

                except Exception as e:
                    st.error(f"Error in mixed-effects analysis: {str(e)}")
                    st.warning("Please ensure your data is suitable for mixed-effects analysis (numeric, non-null values, and valid grouping variable).")

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

                # Add implementation
                try:
                    # Get data for variance function modeling
                    X = self.df[self.independent_var].values
                    y = self.df[self.dependent_var].values

                    # Create controls for variance function modeling
                    st.markdown("#### Variance Function Configuration")
                    model_type = st.selectbox(
                        "Variance Function Type",
                        ["power", "exponential"],
                        help="Power: σ²(x) = a * x^b, Exponential: σ²(x) = a * exp(b*x)"
                    )

                    # First, fit a simple polynomial model to get residuals
                    X_poly = sm.add_constant(X)
                    model = sm.OLS(y, X_poly)
                    results = model.fit()
                    residuals = results.resid

                    # Fit variance function
                    variance_results = self.advanced_statistics.variance_function_modeling(
                        x=X,
                        residuals=residuals,
                        model_type=model_type
                    )

                    # Display results
                    st.markdown("#### Variance Function Results")

                    # Show parameters
                    st.markdown("**Model Parameters:**")
                    st.write(f"a = {variance_results['parameters']['a']:.4f}")
                    st.write(f"b = {variance_results['parameters']['b']:.4f}")
                    st.write(f"R² = {variance_results['r2']:.4f}")

                    # Plot variance function
                    st.markdown("#### Variance Function Plot")

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # Plot 1: Squared residuals vs. predictor
                    ax1.scatter(X, residuals**2, alpha=0.5, label='Squared Residuals')

                    # Sort X for smooth line plot
                    X_sorted = np.sort(X)
                    fitted_variance = variance_results['fitted_values'][np.argsort(X)]

                    ax1.plot(X_sorted, fitted_variance, 'r-', label='Fitted Variance Function')
                    ax1.set_title('Variance Function Fit')
                    ax1.set_xlabel(self.independent_var_name)
                    ax1.set_ylabel('Squared Residuals')
                    ax1.legend()

                    # Plot 2: Standardized residuals
                    standardized_residuals = residuals / np.sqrt(fitted_variance)
                    ax2.scatter(X, standardized_residuals, alpha=0.5)
                    ax2.axhline(y=0, color='r', linestyle='--')
                    ax2.set_title('Standardized Residuals')
                    ax2.set_xlabel(self.independent_var_name)
                    ax2.set_ylabel('Standardized Residuals')

                    # Save plot to BytesIO
                    buf = BytesIO()
                    fig.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close(fig)

                    # Display the plot
                    st.image(buf, caption='Variance Function Analysis')

                    # Additional diagnostics
                    st.markdown("#### Diagnostic Plots")

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # QQ plot of standardized residuals
                    stats.probplot(standardized_residuals, dist="norm", plot=ax1)
                    ax1.set_title('Q-Q Plot of Standardized Residuals')

                    # Histogram of standardized residuals
                    ax2.hist(standardized_residuals, bins=30, density=True, alpha=0.7)

                    # Add normal curve
                    xmin, xmax = ax2.get_xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, np.mean(standardized_residuals), np.std(standardized_residuals))
                    ax2.plot(x, p, 'k', linewidth=2)

                    ax2.set_title('Distribution of Standardized Residuals')
                    ax2.set_xlabel('Standardized Residuals')
                    ax2.set_ylabel('Density')

                    # Save plot to BytesIO
                    buf = BytesIO()
                    fig.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close(fig)

                    # Display the plot
                    st.image(buf, caption='Diagnostic Plots')

                    # Statistical tests
                    st.markdown("#### Statistical Tests")

                    # Test for normality of standardized residuals
                    _, p_value = stats.normaltest(standardized_residuals)
                    st.write(f"**Test for Normality of Standardized Residuals**")
                    st.write(f"p-value: {p_value:.4f}")

                    if p_value < 0.05:
                        st.warning("The standardized residuals appear to be non-normal (p < 0.05). "
                                 "Consider using a different variance function or transformation.")
                    else:
                        st.success("The standardized residuals appear to be normally distributed (p ≥ 0.05).")

                    # Test for remaining heteroscedasticity
                    _, p_value = stats.levene(standardized_residuals[X < np.median(X)],
                                            standardized_residuals[X >= np.median(X)])
                    st.write(f"**Test for Remaining Heteroscedasticity**")
                    st.write(f"p-value: {p_value:.4f}")

                    if p_value < 0.05:
                        st.warning("There may still be some heteroscedasticity remaining (p < 0.05). "
                                 "Consider using a different variance function.")
                    else:
                        st.success("The variance function appears to have successfully accounted for heteroscedasticity (p ≥ 0.05).")

                except Exception as e:
                    st.error(f"Error in variance function modeling: {str(e)}")
                    st.warning("Please ensure your data is suitable for variance function modeling (numeric, non-null values).")

            with st.expander("Conditional Distribution Analysis", expanded=False):
                self._conditional_distribution_analysis()




            st.markdown('</div>', unsafe_allow_html=True)

        # Add section for exporting prediction model
        st.markdown('<div class="tools-section">', unsafe_allow_html=True)
        st.markdown("### Export Prediction Model")

        export_format = st.selectbox(
            "Select export format for prediction model :",
            ["Python Function", "JSON", "Pickle"],
            index=0
        )

        if st.button("Generate Exportable Model for Prediction"):
            self._export_model(export_format)

        st.markdown("</div>", unsafe_allow_html=True)


    def _conditional_distribution_analysis(self):
        """Performs conditional distribution analysis on model residuals."""
        st.markdown("""
        ### Conditional Distribution Analysis

        Conditional distribution analysis examines how the dependent variable's distribution
        changes across different ranges of the independent variable(s). This provides deeper
        insights beyond summary statistics.

        Key analyses:
        - Conditional distributions for X ranges
        - Distribution parameter evolution
        - Distribution equality tests
        - Shape changes modeling
        """)

        if not (self.deterministic_model and self.residuals is not None and len(self.residuals) > 0):
            st.warning("Please complete the Deterministic and Random Component analyses first.")
            return

        try:
            # Get model data
            X = self.df[self.independent_var].values
            y = self.df[self.dependent_var].values

            # Analysis configuration
            n_bins = st.slider("Number of ranges", min_value=3, max_value=10, value=5,
                             help="Ranges to divide independent variable into")

            # Run analysis
            results = self.advanced_statistics.conditional_distribution_analysis(x=X, y=y, n_bins=n_bins)

            # Display statistics table
            st.markdown("#### Conditional Statistics")
            st.dataframe(pd.DataFrame(results['bin_statistics']))

            # Create distribution evolution plot
            self._plot_distribution_evolution(X, y, results)

            # Statistical tests
            self._display_statistical_tests(results)

            # Plot normality tests
            self._plot_normality_tests(results)

            # Plot parameter evolution
            self._plot_parameter_evolution(results)

        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.warning("Please check data is suitable (numeric, non-null values).")


    def _plot_distribution_evolution(self, X, y, results):
        """Creates and displays violin plot showing distribution evolution."""
        fig, ax = plt.subplots(figsize=(12, 6))

        bin_data = []
        bin_labels = []
        for i, stats in enumerate(results['bin_statistics']):
            bin_mask = (X >= results['bin_edges'][i]) & (X < results['bin_edges'][i+1])
            if np.sum(bin_mask) > 1:
                bin_data.append(y[bin_mask])
                bin_labels.append(f"{results['bin_edges'][i]:.2f}-{results['bin_edges'][i+1]:.2f}")

        parts = ax.violinplot(bin_data, points=100, showmeans=True)

        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('red')

        ax.set_title(f'Distribution of {self.dependent_var_name} Across {self.independent_var_name} Ranges')
        ax.set_ylabel(self.dependent_var_name)
        ax.set_xlabel(f'Ranges of {self.independent_var_name}')
        ax.set_xticks(range(1, len(bin_labels) + 1))
        ax.set_xticklabels(bin_labels, rotation=45)

        self._display_plot(fig, 'Distribution Evolution')


    def _display_statistical_tests(self, results):
        """Displays results of statistical tests."""
        st.markdown("#### Statistical Tests")
        st.write("**Test for Homogeneity of Variances (Levene's Test)**")
        st.write(f"p-value: {results['levene_p_value']:.4f}")

        if results['levene_p_value'] < 0.05:
            st.warning("Variances differ significantly across ranges (p < 0.05). Suggests heteroscedasticity.")
        else:
            st.success("Variances consistent across ranges (p ≥ 0.05). Suggests homoscedasticity.")


    def _plot_normality_tests(self, results):
        """Creates and displays normality test results plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        p_values = [s['normality_p_value'] for s in results['bin_statistics'] if 'normality_p_value' in s]
        ranges   = [f"{s['bin_center']:.2f}" for s in results['bin_statistics'] if 'normality_p_value' in s]

        bars = ax.bar(ranges, p_values, alpha=0.7)
        ax.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')

        for i, p_value in enumerate(p_values):
            bars[i].set_color('green' if p_value >= 0.05 else 'red')

        ax.set_title('Normality Test p-values Across Ranges')
        ax.set_xlabel(f'Center of {self.independent_var_name} Range')
        ax.set_ylabel('p-value')
        ax.set_xticklabels(ranges, rotation=45)
        ax.legend()

        self._display_plot(fig, 'Normality Tests Across Ranges')


    def _plot_parameter_evolution(self, results):
        """Creates and displays distribution parameter evolution plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        centers = [s['bin_center'] for s in results['bin_statistics']]
        means = [s['mean'] for s in results['bin_statistics']]
        stds = [s['std'] for s in results['bin_statistics']]

        ax1.plot(centers, means, 'bo-')
        ax1.set_title(f'Evolution of Mean {self.dependent_var_name}')
        ax1.set_xlabel(self.independent_var_name)
        ax1.set_ylabel(f'Mean {self.dependent_var_name}')

        ax2.plot(centers, stds, 'ro-')
        ax2.set_title(f'Evolution of {self.dependent_var_name} Standard Deviation')
        ax2.set_xlabel(self.independent_var_name)
        ax2.set_ylabel('Standard Deviation')

        self._display_plot(fig, 'Evolution of Distribution Parameters')


    def _display_plot(self, fig, caption):
        """Helper to display a matplotlib figure in Streamlit."""
        buf = BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        st.image(buf, caption=caption)


    def _export_model(self, export_format: str):
        """Export model in specified format."""
        best_degree = self.deterministic_model['best_degree']
        model       = self.deterministic_model['models'][best_degree]['model']
        coeffs      = model.coef_.tolist()
        intercept   = float(model.intercept_)
        dist_name   = self.fitted_distribution['best_distribution']
        dist_params = [float(p) for p in self.fitted_distribution['params'][dist_name]]

        if export_format == "Python Function":
            python_code = self._generate_python_function(best_degree, coeffs, intercept, dist_name, dist_params)
            st.code(python_code, language="python")
            st.download_button(
                label     = "Download Python Function",
                data      = python_code,
                file_name = "wound_healing_prediction.py",
                mime      = "text/plain"
            )

        elif export_format == "JSON":
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
            st.download_button(
                label     = "Download JSON Model",
                data      = json_str,
                file_name = "wound_healing_model.json",
                mime      = "application/json"
            )

        elif export_format == "Pickle":
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
            pickle_data = pickle.dumps(export_data)
            st.download_button(
                label     = "Download Pickle Model",
                data      = pickle_data,
                file_name = "wound_healing_model.pkl",
                mime      = "application/octet-stream"
            )


    def _generate_python_function(self, best_degree, coeffs, intercept, dist_name, dist_params):
        """Generate Python function code for model."""
        return f"""
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
                # Polynomial features
                poly_features = [1] + [x ** i for i in range(1, {best_degree + 1})]

                # Make prediction
                y_pred = np.dot({coeffs}, poly_features[1:]) + {intercept}

                # Calculate prediction interval
                if "{dist_name}" == "norm":
                    loc, scale = {dist_params}
                    lower_bound = y_pred + stats.norm.ppf((1-confidence_level)/2, loc=loc, scale=scale)
                    upper_bound = y_pred + stats.norm.ppf(1-(1-confidence_level)/2, loc=loc, scale=scale)
                else:
                    lower_bound = upper_bound = None

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
                        "residual_distribution": "{dist_name}"
                    }}
                }}
                """


    def _create_advanced_statistics_original(self):
        """ Create and display advanced statistical methods for sophisticated analysis. """

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
                                        label=f'{self.independent_var_name} ∈ [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]: μ={mu:.3f}, σ={std:.3f}')

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
