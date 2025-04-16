import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Union
from datetime import datetime
from scipy import stats

from wound_analysis.dashboard_components.stochastic_modeling_components import (
	CreateAdvancedStatistics,
	CreateCompleteModel,
	CreateDeterministicComponent,
	CreateDistributionAnalysis,
	CreateRandomComponent,
	CreateUncertaintyQuantificationTools
)
from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor


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
		if 'polynomial_type' not in st.session_state:
			st.session_state.polynomial_type = 'regular'

		# Models for storing analysis results
		self.deterministic_model = st.session_state.deterministic_model
		self.polynomial_degree   = st.session_state.polynomial_degree
		self.deterministic_coefs = st.session_state.deterministic_coefs
		self.residuals           = st.session_state.residuals
		self.fitted_distribution = st.session_state.fitted_distribution
		self.polynomial_type     = st.session_state.polynomial_type

		self.independent_var      = None
		self.dependent_var        = None
		self.independent_var_name = None
		self.dependent_var_name   = None
		self.additional_params    = None


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


	def _parameter_selection_ui(self) -> Tuple[Dict[str, Union[Tuple[datetime, datetime], List[int]]], str]:
		"""
		Create a visually appealing parameter selection UI.

		Returns:
		-------
		Tuple[Dict[str, Union[str, Tuple[datetime, datetime], List[int]]], str]:
			- Dictionary of filter parameters
			- Boolean indicating if the analysis should be run
		"""
		st.subheader("📊 Parameter Selection")

		# Create a styled container
		with st.container():

			# Primary variable selection in columns
			col1, col2, col3, col4 = st.columns(4)

			with col1:
				self.dependent_var_name = st.selectbox(
					"📈 Dependent Variable (Y)",
					options=list(self.dependent_variables.keys()),
					index=0,
					help="The outcome variable to be modeled probabilistically"
				)
				self.dependent_var = self.dependent_variables[self.dependent_var_name]

			with col2:
				self.independent_var_name = st.selectbox(
					"📉 Independent Variable (X)",
					options=list(self.independent_variables.keys()),
					index=0,
					help="The primary predictor variable"
				)
				self.independent_var = self.independent_variables[self.independent_var_name]

			# Additional parameters section
			with col3:
				additional_params_names = st.multiselect(
					"➕ Additional Parameters",
					options=list(self.additional_parameters.keys()),
					default=[],
					help="These parameters will be used for conditional analysis and stratification"
				)
				self.additional_params = [self.additional_parameters[param] for param in additional_params_names]

			with col4:
				# Filter options section
				filters: Dict[str, Union[Tuple[datetime, datetime], List[int]]] = {}

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
			run_analysis: str = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

			st.markdown('</div>', unsafe_allow_html=True)

		return filters, run_analysis


	def _filter_data(self, df: pd.DataFrame, filters: Dict[str, Union[str, Tuple[datetime, datetime], List[int]]]) -> pd.DataFrame:
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


	def apply_outlier_threshold(self, df: pd.DataFrame, features_to_analyze: List[str]) -> pd.DataFrame:

		cols = st.columns([2, 3])

		with cols[0]:
			outlier_threshold = st.number_input(
				"Stochastic Outlier Threshold (Quantile)",
				min_value = 0.0,
				max_value = 0.9,
				value     = 0.0,
				step      = 0.05,
				help      = "Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		# Create a copy of the dataframe with only the features we want to analyze
		analysis_df = df[features_to_analyze].copy().dropna()

		# Remove outliers if threshold is set
		if outlier_threshold > 0:
			for col in analysis_df.columns:
				q_low       = analysis_df[col].quantile(outlier_threshold)
				q_high      = analysis_df[col].quantile(1 - outlier_threshold)
				analysis_df = analysis_df[ (analysis_df[col] >= q_low) & (analysis_df[col] <= q_high) ]

		if analysis_df.empty or len(analysis_df) < 2:
			st.warning("Not enough data after outlier removal for correlation analysis.")
			return pd.DataFrame()

		# Return the original dataframe but keep only the rows that survived the outlier filtering
		return df.loc[analysis_df.index]

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
		filters, run_analysis = self._parameter_selection_ui()


		# Get display names for variables
		self.dependent_var_name   = next((name for name, col in self.dependent_variables.items()   if col == self.dependent_var),   self.dependent_var)
		self.independent_var_name = next((name for name, col in self.independent_variables.items() if col == self.independent_var), self.independent_var)

		# Apply outlier threshold
		# TODO: need to apply outlier threshold to actual variables: self.dependent_var, self.independent_var, *self.additional_params
		self.df = self.apply_outlier_threshold(df=self.df, features_to_analyze=[self.CN.WOUND_AREA, self.CN.HIGHEST_FREQ_ABSOLUTE])

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
				CreateDistributionAnalysis(df=filtered_df, parent=self).render()

			# Deterministic Component Tab
			with tabs[1]:
				cdc = CreateDeterministicComponent(df=filtered_df, parent=self).render()
				self.residuals           = cdc['residuals']
				self.polynomial_degree   = cdc['polynomial_degree']
				self.deterministic_coefs = cdc['deterministic_coefs']
				self.deterministic_model = cdc['deterministic_model']
				self.polynomial_type     = cdc['polynomial_type']

				st.session_state.residuals           = self.residuals
				st.session_state.polynomial_degree   = self.polynomial_degree
				st.session_state.deterministic_coefs = self.deterministic_coefs
				st.session_state.deterministic_model = self.deterministic_model
				st.session_state.polynomial_type     = self.polynomial_type

			# Random Component Tab
			with tabs[2]:
				self.fitted_distribution = CreateRandomComponent(parent=self).render()

				st.session_state.fitted_distribution = self.fitted_distribution

			# Complete Model Tab
			with tabs[3]:
				CreateCompleteModel(df=filtered_df, parent=self).render()

			# Advanced Statistics Tab
			with tabs[4]:
				CreateAdvancedStatistics(df=filtered_df, parent=self).render()
				# CreateAdvancedStatistics(df=filtered_df, parent=self)._create_advanced_statistics_original()

			# Uncertainty Quantification Tools Tab
			with tabs[5]:
				CreateUncertaintyQuantificationTools(df=filtered_df, parent=self).render()


