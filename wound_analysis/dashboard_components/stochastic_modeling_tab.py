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

from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.dashboard_components.visualizer import Visualizer


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
        self.wound_data_processor = wound_data_processor
        self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
        self.df = wound_data_processor.df
        self.CN = DColumns(df=self.df)

        # Initialize parameter lists for selections
        self._initialize_parameter_lists()

        # Available probability distributions for fitting
        self.available_distributions = {
            'Normal': stats.norm,
            'Log-Normal': stats.lognorm,
            'Gamma': stats.gamma,
            'Weibull': stats.weibull_min,
            'Exponential': stats.expon
        }

        # Models for storing analysis results
        self.deterministic_model = None
        self.polynomial_degree = None
        self.deterministic_coefs = None
        self.residuals = None

    def _initialize_parameter_lists(self):
        """Initialize parameter lists for dropdown selections."""
        # Define available dependent variables (outcomes)
        self.dependent_variables = {
            'Impedance Absolute' : self.CN.HIGHEST_FREQ_ABSOLUTE,
            'Impedance Real'     : self.CN.HIGHEST_FREQ_REAL,
            'Impedance Imaginary': self.CN.HIGHEST_FREQ_IMAGINARY,
            'Wound Area'         : self.CN.WOUND_AREA,
            'Temperature Center' : self.CN.CENTER_TEMP,
            'Temperature Edge'   : self.CN.EDGE_TEMP,
            'Temperature Peri'   : self.CN.PERI_TEMP,
        }

        # Define available independent variables (predictors)
        self.independent_variables = {
            'Wound Size': self.CN.WOUND_AREA,
            'Days Since First Visit': self.CN.DAYS_SINCE_FIRST_VISIT,
            'Time (days)': self.CN.DAYS_SINCE_FIRST_VISIT,
            'Age': self.CN.AGE,
            'BMI': self.CN.BMI
        }

        # Define additional parameters for multi-select
        self.additional_parameters = {
            'Age': self.CN.AGE,
            'BMI': self.CN.BMI,
            'Sex': self.CN.SEX,
            'Diabetes Status': self.CN.DIABETES,
            'A1C Value': self.CN.A1C,
            'Smoking Status': self.CN.SMOKING_STATUS,
            'Alcohol Status': self.CN.ALCOHOL_STATUS,
        }

    def _parameter_selection_ui(self) -> Tuple[str, str, List[str], Dict, bool]:
        """
        Create the parameter selection UI.

        Returns:
        -------
        Tuple[str, str, List[str], Dict, bool]:
            - Selected dependent variable column name
            - Selected independent variable column name
            - List of selected additional parameter column names
            - Dictionary of filter parameters
            - Boolean indicating if the analysis should be run
        """
        st.subheader("Parameter Selection")

        # Create a container with a border
        with st.container():
            st.markdown("""
            <style>
            .parameter-section {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 15px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="parameter-section">', unsafe_allow_html=True)

            # Create two columns for primary variable selection
            col1, col2 = st.columns(2)

            with col1:
                # Dependent variable selection
                dependent_var_name = st.selectbox(
                    "Dependent Variable (Y)",
                    options=list(self.dependent_variables.keys()),
                    index=0,
                    help="The outcome variable to be modeled probabilistically"
                )
                dependent_var = self.dependent_variables[dependent_var_name]

            with col2:
                # Independent variable selection
                independent_var_name = st.selectbox(
                    "Independent Variable (X)",
                    options=list(self.independent_variables.keys()),
                    index=0,
                    help="The primary predictor variable"
                )
                independent_var = self.independent_variables[independent_var_name]

            # Additional parameters multi-select
            st.markdown("### Additional Parameters")
            st.write("Select additional parameters to include in the analysis:")

            additional_params_names = st.multiselect(
                "Additional Parameters",
                options=list(self.additional_parameters.keys()),
                default=[],
                help="These parameters will be used for conditional analysis and stratification"
            )
            additional_params = [self.additional_parameters[param] for param in additional_params_names]

            # Filter options
            st.markdown("### Data Filtering")

            filters = {}

            # Patient filter (if All Patients is selected)
            if self.patient_id == "All Patients":
                patient_ids = sorted(self.df[self.CN.RECORD_ID].unique())
                selected_patients = st.multiselect(
                    "Filter by Patient ID",
                    options=[f"Patient {id:d}" for id in patient_ids],
                    default=[],
                    help="Select specific patients to include in the analysis"
                )
                if selected_patients:
                    filters['patients'] = [int(p.split()[1]) for p in selected_patients]

            # Date range filter
            date_filter = st.checkbox("Filter by Date Range", value=False)
            if date_filter:
                col1, col2 = st.columns(2)
                with col1:
                    min_date = self.df[self.CN.VISIT_DATE].min()
                    start_date = pd.to_datetime(st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=self.df[self.CN.VISIT_DATE].max()
                    ))
                with col2:
                    max_date = self.df[self.CN.VISIT_DATE].max()
                    end_date = pd.to_datetime(st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    ))
                filters['date_range'] = (start_date, end_date)

            # Run analysis button
            run_analysis = st.button("Run Analysis", type="primary")

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

            st.write(f"Analyzing the distribution of {dependent_var_name}")

            # Get data for dependent variable
            data = df[dependent_var].values

            # Remove NaN values
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                st.error(f"No valid data available for {dependent_var_name}")
                return

            # Show basic statistics
            st.markdown("### Basic Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{np.mean(valid_data):.2f}")
            col2.metric("Median", f"{np.median(valid_data):.2f}")
            col3.metric("Std Dev", f"{np.std(valid_data):.2f}")
            col4.metric("Sample Size", f"{len(valid_data)}")

            # Fit distributions
            with st.spinner("Fitting probability distributions..."):
                dist_results = self._fit_distributions(valid_data)

            if not dist_results:
                st.error("Failed to fit distributions to the data.")
                return

            # Display distribution plots
            st.markdown("### Distribution Fitting")

            buf = self._plot_distribution_fit(valid_data, dist_results, dependent_var_name)

            if buf is not None:
                st.image(buf, caption=f"Distribution Analysis for {dependent_var_name}")

            # Display goodness of fit statistics
            st.markdown("### Goodness of Fit Statistics")

            # Create dataframe for display
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

            st.dataframe(fit_stats_df)

            # Add download button for the statistics
            csv = fit_stats_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="distribution_statistics.csv">Download Statistics (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Show best fit distribution parameters
            if fit_stats_df.shape[0] > 0:
                best_dist = fit_stats_df.iloc[0]["Distribution"]
                st.markdown(f"### Best Fit: {best_dist} Distribution")

                best_result = dist_results[best_dist]
                param_names = best_result['distribution'].shapes.split(",") if hasattr(best_result['distribution'], 'shapes') else []

                if best_dist == "Normal":
                    st.write(f"μ (mean): {best_result['params'][0]:.4f}")
                    st.write(f"σ (std dev): {best_result['params'][1]:.4f}")
                elif best_dist == "Log-Normal":
                    st.write(f"σ (shape): {best_result['params']['s']:.4f}")
                    st.write(f"μ (scale): {best_result['params']['scale']:.4f}")
                else:
                    # Generic parameter display
                    for i, param in enumerate(best_result['params']):
                        param_name = param_names[i] if i < len(param_names) else f"Parameter {i+1}"
                        st.write(f"{param_name}: {param:.4f}")

            st.markdown('</div>', unsafe_allow_html=True)

    def _fit_polynomial_models(self, X: np.ndarray, y: np.ndarray, max_degree: int = 5) -> Dict[int, Dict]:
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

        for degree in range(1, max_degree + 1):
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)

            # Fit the model
            model = LinearRegression()
            model.fit(X_poly_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_poly_train)
            y_pred_test = model.predict(X_poly_test)

            # Calculate metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)

            # Calculate AIC and BIC
            n_train = len(y_train)
            k = degree + 1  # number of parameters
            aic = n_train * np.log(mse_train) + 2 * k
            bic = n_train * np.log(mse_train) + k * np.log(n_train)

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

    def _plot_polynomial_fit(self, X: np.ndarray, y: np.ndarray, model_results: Dict, degree: int,
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
                model_results = self._fit_polynomial_models(X, y, max_degree)

            # Display polynomial degree selection
            col1, col2 = st.columns([1, 1])

            with col1:
                selected_degree = st.slider("Select Polynomial Degree",
                                           min_value=1, max_value=max_degree,
                                           value=2, step=1,
                                           help="Higher degrees can fit the data better but may overfit")

            # Store the selected model for later use
            self.deterministic_model = model_results[selected_degree]
            self.polynomial_degree = selected_degree

            # Display model metrics for different degrees
            metrics_df = pd.DataFrame({
                'Degree': list(range(1, max_degree + 1)),
                'R² (Train)': [model_results[d]['r2_train'] for d in range(1, max_degree + 1)],
                'R² (Test)': [model_results[d]['r2_test'] for d in range(1, max_degree + 1)],
                'MSE (Train)': [model_results[d]['mse_train'] for d in range(1, max_degree + 1)],
                'MSE (Test)': [model_results[d]['mse_test'] for d in range(1, max_degree + 1)],
                'AIC': [model_results[d]['aic'] for d in range(1, max_degree + 1)],
                'BIC': [model_results[d]['bic'] for d in range(1, max_degree + 1)]
            })

            with col2:
                st.markdown("### Model Selection Metrics")
                st.write("Lower AIC and BIC values indicate better models, balancing fit and complexity.")
                best_aic = metrics_df['AIC'].idxmin() + 1
                best_bic = metrics_df['BIC'].idxmin() + 1
                st.write(f"Best degree by AIC: {best_aic}")
                st.write(f"Best degree by BIC: {best_bic}")

            st.dataframe(metrics_df.style.highlight_min(subset=['AIC', 'BIC'], color='lightgreen'))

            # Display the polynomial plot
            st.markdown("### Polynomial Fit Visualization")

            buf = self._plot_polynomial_fit(X, y, model_results, selected_degree,
                                           independent_var_name, dependent_var_name)

            if buf is not None:
                st.image(buf, caption=f"Polynomial Fit (Degree {selected_degree})")

            # Display equation and coefficients
            st.markdown("### Model Equation")

            # Get coefficients and format equation
            selected_model = model_results[selected_degree]
            coef = selected_model['coefficients']
            intercept = selected_model['intercept']

            # Store for later use
            self.deterministic_coefs = (intercept, coef)

            # Format equation
            equation = f"g({independent_var_name}) = {intercept:.4f}"
            for i, c in enumerate(coef[1:]):  # Skip the first coefficient (it's just 1)
                if c >= 0:
                    equation += f" + {c:.4f}{independent_var_name}^{i+1}"
                else:
                    equation += f" - {abs(c):.4f}{independent_var_name}^{i+1}"

            st.markdown(f"**{equation}**")

            # Display coefficient table
            st.markdown("### Coefficient Values")

            coef_df = pd.DataFrame({
                'Term': [f"{independent_var_name}^{i}" if i > 0 else "Intercept"
                        for i in range(len(coef))],
                'Coefficient': [intercept] + list(coef[1:])
            })

            st.dataframe(coef_df)

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

            # Get X values from the deterministic model
            X = self.deterministic_model['poly'].get_feature_names_out()[1]  # Get the name of the original feature
            X_values = np.array([float(x.split('^')[0].split('_')[1]) for x in X.split()])

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
                        st.session_state.active_tab = "Random Component"
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

            This tab implements a two-component model approach: a deterministic component representing
            the expected trend, and a random component representing the variability around that trend.
            """)

            # Display a conceptual diagram (placeholder)
            st.image("https://via.placeholder.com/800x300?text=Deterministic+vs+Probabilistic+Modeling",
                    caption="Conceptual comparison of deterministic vs. probabilistic modeling approaches")

        # Parameter selection UI
        dependent_var, independent_var, additional_params, filters, run_analysis = self._parameter_selection_ui()

        # Get display names for variables
        dependent_var_name = next((name for name, col in self.dependent_variables.items() if col == dependent_var), dependent_var)
        independent_var_name = next((name for name, col in self.independent_variables.items() if col == independent_var), independent_var)

        # Analysis results when the Run Analysis button is clicked
        if run_analysis:
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
            analysis_tabs = st.tabs([
                "Distribution Analysis",
                "Deterministic Component",
                "Random Component",
                "Complete Model",
                "Advanced Statistics",
                "Uncertainty Tools"
            ])

            # Distribution Analysis Tab
            with analysis_tabs[0]:
                self._create_distribution_analysis(filtered_df, dependent_var, dependent_var_name)

            # Deterministic Component Tab
            with analysis_tabs[1]:
                self._create_deterministic_component(filtered_df, dependent_var, independent_var,
                                                    dependent_var_name, independent_var_name)

            # Random Component Tab
            with analysis_tabs[2]:
                self._create_random_component(independent_var_name)

            # Complete Model Tab
            with analysis_tabs[3]:
                self._create_complete_model(filtered_df, dependent_var, independent_var,
                                          dependent_var_name, independent_var_name)

            # Advanced Statistics Tab
            with analysis_tabs[4]:
                self._create_advanced_statistics(filtered_df, dependent_var, independent_var,
                                               dependent_var_name, independent_var_name)

            # Uncertainty Quantification Tools Tab
            with analysis_tabs[5]:
                self._create_uncertainty_quantification_tools(filtered_df, dependent_var, independent_var,
                                                           dependent_var_name, independent_var_name)
