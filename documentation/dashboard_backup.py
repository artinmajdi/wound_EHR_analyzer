# Standard library imports
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / '.env')

# Third-party imports
import numpy as np
import pandas as pd
import sklearn
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# from plotly.subplots import make_subplots

# Local application imports
from wound_analysis.utils.data_processor import DataManager, ImpedanceAnalyzer, WoundDataProcessor
from wound_analysis.utils.llm_interface import WoundAnalysisLLM
from wound_analysis.utils.statistical_analysis import CorrelationAnalysis
from wound_analysis.dashboard_components.settings import DashboardSettings
from wound_analysis.dashboard_components.visualizer import Visualizer


# Debug mode disabled
st.set_option('client.showErrorDetails', True)

def debug_log(message):
	"""Write debug messages to a file and display in Streamlit"""
	with open('/app/debug.log', 'a') as f:
		f.write(f"{message}\n")
	st.sidebar.text(message)


class Dashboard:
	"""Main dashboard class for the Wound Analysis application.

	This class serves as the core controller for the wound analysis dashboard, integrating
	data processing, visualization, and analysis components. It handles the initialization,
	setup, and rendering of the Streamlit application interface.

	The dashboard provides comprehensive wound analysis features including:
	- Overview of patient data and population statistics
	- Impedance analysis with clinical interpretation
	- Temperature gradient analysis
	- Tissue oxygenation assessment
	- Exudate characterization
	- Risk factor evaluation
	- LLM-powered wound analysis

	Attributes:
		DashboardSettings (DashboardSettings): Configuration settings for the application
		data_manager (DataManager): Handles data loading and processing operations
		visualizer (Visualizer): Creates data visualizations for the dashboard
		impedance_analyzer (ImpedanceAnalyzer): Processes and interprets impedance measurements
		llm_platform (str): Selected platform for LLM analysis (e.g., "ai-verde")
		llm_model (str): Selected LLM model for analysis
		csv_dataset_path (str): Path to the uploaded CSV dataset
		data_processor (WoundDataProcessor): Processes wound data for analysis
		impedance_freq_sweep_path (pathlib.Path): Path to impedance frequency sweep data files

	Methods:
		setup(): Initializes the Streamlit page configuration and sidebar
		load_data(uploaded_file): Loads data from the uploaded CSV file
		run(): Main execution method that runs the dashboard application
		_create_dashboard_tabs(): Creates and manages the main dashboard tabs
		_overview_tab(): Renders the overview tab for patient data
		_impedance_tab(): Renders impedance analysis visualization and interpretation
		_temperature_tab(): Renders temperature analysis visualization
		_oxygenation_tab(): Renders oxygenation analysis visualization
		_exudate_tab(): Renders exudate analysis visualization
		_risk_factors_tab(): Renders risk factors analysis visualization
		_llm_analysis_tab(): Renders the LLM-powered analysis interface
		_create_left_sidebar(): Creates the sidebar with configuration options
		"""
	def __init__(self):
		"""Initialize the dashboard."""
		self.DashboardSettings             = DashboardSettings()
		self.data_manager       = DataManager()
		self.visualizer         = Visualizer()
		self.impedance_analyzer = ImpedanceAnalyzer()

		# LLM configuration placeholders
		self.llm_platform   = None
		self.llm_model      = None
		self.csv_dataset_path  = None
		self.data_processor = None
		self.impedance_freq_sweep_path: pathlib.Path = None

	def setup(self) -> None:
		"""Set up the dashboard configuration."""
		st.set_page_config(
			page_title = self.DashboardSettings.PAGE_TITLE,
			page_icon  = self.DashboardSettings.PAGE_ICON,
			layout     = self.DashboardSettings.LAYOUT
		)
		DashboardSettings.initialize()
		self._create_left_sidebar()

	@staticmethod
	# @st.cache_data
	def load_data(uploaded_file) -> Optional[pd.DataFrame]:
		"""
		Loads data from an uploaded file into a pandas DataFrame.

		This function serves as a wrapper around the DataManager's load_data method,
		providing consistent data loading functionality for the dashboard.

		Args:
			uploaded_file: The file uploaded by the user through the application interface (typically a csv, excel, or other supported format)

		Returns:
			Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data, or None if the file couldn't be loaded

		Note:
			The actual loading logic is handled by the DataManager class.
		"""
		df = DataManager.load_data(uploaded_file)
		return df

	def run(self) -> None:
		"""
		Run the main dashboard application.

		This method initializes the dashboard, loads the dataset, processes wound data,
		sets up the page layout including title and patient selection dropdown,
		and creates the dashboard tabs.

		If no CSV file is uploaded, displays an information message.
		If data loading fails, displays an error message.

		Returns:
			None
		"""

		self.setup()
		if not self.csv_dataset_path:
			st.info("Please upload a CSV file to proceed.")
			return

		df = self.load_data(self.csv_dataset_path)

		if df is None:
			st.error("Failed to load data. Please check the CSV file.")
			return

		self.data_processor = WoundDataProcessor(df=df, impedance_freq_sweep_path=self.impedance_freq_sweep_path)

		# Header
		st.title(self.DashboardSettings.PAGE_TITLE)

		# Patient selection
		patient_ids = sorted(df['Record ID'].unique())
		patient_options = ["All Patients"] + [f"Patient {id:d}" for id in patient_ids]
		selected_patient = st.selectbox("Select Patient", patient_options)

		# Create tabs
		self._create_dashboard_tabs(df, selected_patient)

	def _create_dashboard_tabs(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
		Create and manage dashboard tabs for displaying patient wound data.

		This method sets up the main dashboard interface with multiple tabs for different
		wound analysis categories. Each tab is populated with specific visualizations and
		data analyses related to the selected patient.

		Parameters:
		-----------
		df : pd.DataFrame
			The dataframe containing wound data for analysis
		selected_patient : str
			The identifier of the currently selected patient

		Returns:
		--------
		None
			This method updates the Streamlit UI directly without returning values

		Notes:
		------
		The following tabs are created:
		- Overview          : General patient information and wound summary
		- Impedance Analysis: Electrical measurements of wound tissue
		- Temperature       : Thermal measurements and analysis
		- Oxygenation       : Oxygen saturation and related metrics
		- Exudate           : Analysis of wound drainage
		- Risk Factors      : Patient-specific risk factors for wound healing
		- LLM Analysis      : Natural language processing analysis of wound data
		"""

		tabs = st.tabs([
			"Overview",
			"Impedance Analysis",
			"Temperature",
			"Oxygenation",
			"Exudate",
			"Risk Factors",
			"LLM Analysis"
		])

		with tabs[0]:
			self._overview_tab(df, selected_patient)
		with tabs[1]:
			self._impedance_tab(df, selected_patient)
		with tabs[2]:
			self._temperature_tab(df, selected_patient)
		with tabs[3]:
			self._oxygenation_tab(df, selected_patient)
		with tabs[4]:
			self._exudate_tab(df, selected_patient)
		with tabs[5]:
			self._risk_factors_tab(df, selected_patient)
		with tabs[6]:
			self._llm_analysis_tab(selected_patient)

	def _overview_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
		Renders the Overview tab in the Streamlit application.

		This method displays different content based on whether all patients or a specific patient is selected.
		For all patients, it renders a summary overview of all patients' data.
		For a specific patient, it renders that patient's individual overview and a wound area over time plot.

		Parameters:
		----------
		df : pd.DataFrame
			The dataframe containing all wound data
		selected_patient : str
			The currently selected patient from the sidebar dropdown. Could be "All Patients"
			or a specific patient name in the format "Patient X" where X is the patient ID.

		Returns:
		-------
		None
			This method directly renders content to the Streamlit UI and doesn't return any value.
		"""

		st.header("Overview")

		if selected_patient == "All Patients":
			self._render_all_patients_overview(df)
		else:
			patient_id = int(selected_patient.split(" ")[1])
			self._render_patient_overview(df, patient_id)
			st.subheader("Wound Area Over Time")
			fig = self.visualizer.create_wound_area_plot(DataManager.get_patient_data(df, patient_id), patient_id)
			st.plotly_chart(fig, use_container_width=True)

	def _impedance_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
			Render the impedance analysis tab in the Streamlit application.

			This method creates a comprehensive data analysis and visualization tab focused on bioelectrical
			impedance measurement data. It provides different views depending on whether the analysis is for
			all patients combined or a specific patient.

			For all patients:
			- Creates a scatter plot correlating impedance to healing rate with outlier control
			- Displays statistical correlation coefficients
			- Shows impedance components over time
			- Shows average impedance by wound type

			For individual patients:
			- Provides three detailed sub-tabs:
				1. Overview: Basic impedance measurements over time with mode selection
				2. Clinical Analysis: Detailed per-visit assessment including tissue health index, infection risk assessment, tissue composition analysis, and changes since previous visit
				3. Advanced Interpretation: Sophisticated analysis including healing trajectory, wound healing stage classification, tissue electrical properties, and clinical insights

			Parameters:
			----------
			df : pd.DataFrame
					The dataframe containing all patient data with impedance measurements
			selected_patient : str
					Either "All Patients" or a specific patient identifier (e.g., "Patient 43")

			Returns:
			-------
			None
					This method renders UI elements directly to the Streamlit app
		"""

		st.header("Impedance Analysis")

		if selected_patient == "All Patients":
			self._render_population_impedance_analysis(df)
		else:
			patient_id = int(selected_patient.split(" ")[1])
			self._render_patient_impedance_analysis(df, patient_id)

	def _render_population_impedance_analysis(self, df: pd.DataFrame) -> None:
		"""
		Renders the population-level impedance analysis section of the dashboard.

		This method creates visualizations and controls for analyzing impedance data across the entire patient population. It includes
		correlation analysis with filtering controls, a scatter plot of relationships between variables, and additional charts that provide
		population-level insights about impedance measurements.

		Parameters:
		----------
		df : pd.DataFrame
			The input dataframe containing patient impedance data to be analyzed.
			Expected to contain columns related to impedance measurements and patient information.

		Returns:
		-------
		None
			This method directly renders components to the Streamlit dashboard and doesn't return values.
		"""
		# Use the generalized feature analysis function with impedance as the primary feature
		primary_feature = "Skin Impedance (kOhms) - Z"
		default_features = [
			primary_feature,
			"Calculated Wound Area",
			"Healing Rate (%)"
		]

		filtered_df = self._render_feature_analysis(
			df=df,
			primary_feature=primary_feature,
			default_correlated_features=default_features,
			feature_name_for_ui="Impedance"
		)

		# Render any additional impedance-specific charts here
		if not filtered_df.empty:
			self._render_population_impedance_charts(filtered_df)

	def _render_population_impedance_charts(self, df: pd.DataFrame) -> None:
		"""
		Renders additional population-level impedance charts.

		Parameters:
		----------
		df : pd.DataFrame
			The filtered dataframe containing impedance data
		"""
		if "Skin Impedance (kOhms) - Z" not in df.columns:
			st.warning("Impedance data not available for additional charts.")
			return

		# Drop rows with missing impedance values for these charts
		chart_df = df.dropna(subset=["Skin Impedance (kOhms) - Z"])

		# Check if we have enough data to create meaningful visualizations
		if len(chart_df) <= 1:
			st.warning("Insufficient data for additional impedance analysis.")
			return

		st.subheader("Additional Impedance Insights")
		col1, col2 = st.columns(2)

		with col1:
			# Create boxplot of impedance by wound type
			if "Wound Type" in chart_df.columns:
				wound_box_df = chart_df.dropna(subset=["Wound Type"])
				if not wound_box_df.empty:
					fig = px.box(
						wound_box_df,
						x="Wound Type",
						y="Skin Impedance (kOhms) - Z",
						title="Impedance by Wound Type",
						points="all"
					)
					fig.update_layout(
						xaxis_title="Wound Type",
						yaxis_title="Impedance (kOhms)"
					)
					st.plotly_chart(fig, use_container_width=True, key="impedance_wound_type_boxplot")
				else:
					st.info("No wound type data available for boxplot.")
			else:
				st.info("Wound type information not available.")

		with col2:
			# Create histogram of impedance distribution
			fig = px.histogram(
				chart_df,
				x="Skin Impedance (kOhms) - Z",
				nbins=20,
				title="Distribution of Impedance Measurements"
			)
			fig.update_layout(
				xaxis_title="Impedance (kOhms)",
				yaxis_title="Count"
			)
			st.plotly_chart(fig, use_container_width=True, key="impedance_distribution_histogram")

	def _render_patient_impedance_analysis(self, df: pd.DataFrame, patient_id: int) -> None:
		"""
		Renders the impedance analysis section for a specific patient in the dashboard.

		This method creates a tabbed interface to display different perspectives on a patient's
		impedance data, organized into Overview, Clinical Analysis, and Advanced Interpretation tabs.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing all patient data (may be filtered)
		patient_id : int
			The unique identifier for the patient to analyze

		Returns:
		-------
		None
			This method renders UI elements directly to the Streamlit dashboard
		"""

		# Get patient visits
		patient_data = self.data_processor.get_patient_visits(record_id=patient_id)
		visits = patient_data['visits']

		# Create tabs for different analysis views
		tab1, tab2, tab3 = st.tabs([
			"Overview",
			"Clinical Analysis",
			"Advanced Interpretation"
		])

		with tab1:
			self._render_patient_impedance_overview(visits)

		with tab2:
			self._render_patient_clinical_analysis(visits)

		with tab3:
			self._render_patient_advanced_analysis(visits)

	def _render_patient_impedance_overview(self, visits) -> None:
		"""
			Renders an overview section for patient impedance measurements.

			This method creates a section in the Streamlit application showing impedance measurements
			over time, allowing users to view different types of impedance data (absolute impedance,
			resistance, or capacitance).

			Parameters:
				visits (list): A list of patient visit data containing impedance measurements

			Returns:
			-------
			None
					This method renders UI elements directly to the Streamlit app

			Note:
				The visualization includes a selector for different measurement modes and
				displays an explanatory note about the measurement types and frequency effects.
		"""

		st.subheader("Impedance Measurements Over Time")

		# Add measurement mode selector
		measurement_mode = st.selectbox(
			"Select Measurement Mode:",
			["Absolute Impedance (|Z|)", "Resistance", "Capacitance"],
			key="impedance_mode_selector"
		)

		# Create impedance chart with selected mode
		fig = self.visualizer.create_impedance_chart(visits, measurement_mode=measurement_mode)
		st.plotly_chart(fig, use_container_width=True)

		# Logic behind analysis
		st.markdown("""
		<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
		<p style="margin-top:0; color:#666; font-weight:bold;">ABOUT THIS ANALYSIS:</p>
		<strong>Measurement Types:</strong><br>
		• <strong>|Z|</strong>: Total opposition to current flow<br>
		• <strong>Resistance</strong>: Opposition from ionic content<br>
		• <strong>Capacitance</strong>: Opposition from cell membranes<br><br>
		<strong>Frequency Effects:</strong><br>
		• <strong>Low (100Hz)</strong>: Measures extracellular fluid<br>
		• <strong>High (80000Hz)</strong>: Measures both intra/extracellular properties
		</div>
		""", unsafe_allow_html=True)


	def _render_patient_clinical_analysis(self, visits) -> None:
		"""
			Renders the bioimpedance clinical analysis section for a patient's wound data.

			This method creates a tabbed interface showing clinical analysis for each visit.
			For each visit (except the first one), it performs a comparative analysis with
			the previous visit to track changes in wound healing metrics.

			Args:
				visits (list): A list of dictionaries containing visit data. Each dictionary
						should have at least a 'visit_date' key and other wound measurement data.

			Returns:
			-------
			None
				The method updates the Streamlit UI directly

			Notes:
				- At least two visits are required for comprehensive analysis
				- Creates a tab for each visit date
				- Analysis is performed using the impedance_analyzer component
		"""

		st.subheader("Bioimpedance Clinical Analysis")

		# Only analyze if we have at least two visits
		if len(visits) < 2:
			st.warning("At least two visits are required for comprehensive clinical analysis")
			return

		# Create tabs for each visit
		visit_tabs = st.tabs([f"{visit.get('visit_date', 'N/A')}" for visit in visits])

		for visit_idx, visit_tab in enumerate(visit_tabs):
			with visit_tab:
				# Get current and previous visit data
				current_visit = visits[visit_idx]
				previous_visit = visits[visit_idx-1] if visit_idx > 0 else None

				# Generate comprehensive clinical analysis
				analysis = self.impedance_analyzer.generate_clinical_analysis(
					current_visit, previous_visit
				)

				# Display results in a structured layout
				self._display_clinical_analysis_results(analysis, previous_visit is not None)

	def _display_clinical_analysis_results(self, analysis, has_previous_visit):
		"""
			Display the clinical analysis results in the Streamlit UI using a structured layout.

			This method organizes the display of analysis results into two sections, each with two columns:
			1. Top section: Tissue health and infection risk assessments
			2. Bottom section: Tissue composition analysis and comparison with previous visits (if available)

			The method also adds an explanatory note about the color coding and significance markers used in the display.

			Parameters
			----------
			analysis : dict
				A dictionary containing the analysis results with the following keys:
				- 'tissue_health': Data for tissue health assessment
				- 'infection_risk': Data for infection risk assessment
				- 'frequency_response': Data for tissue composition analysis
				- 'changes': Observed changes since previous visit (if available)
				- 'significant_changes': Boolean flags indicating clinically significant changes

			has_previous_visit : bool
				Flag indicating whether there is data from a previous visit available for comparison

			Returns:
			-------
			None
		"""

		# Display Tissue Health and Infection Risk in a two-column layout
		col1, col2 = st.columns(2)

		with col1:
			self._display_tissue_health_assessment(analysis['tissue_health'])

		with col2:
			self._display_infection_risk_assessment(analysis['infection_risk'])

		st.markdown('---')

		# Display Tissue Composition and Changes in a two-column layout
		col1, col2 = st.columns(2)

		with col2:
			self._display_tissue_composition_analysis(analysis['frequency_response'])

		with col1:
			if has_previous_visit and 'changes' in analysis:
				self._display_visit_changes(
					analysis['changes'],
					analysis['significant_changes']
				)
			else:
				st.info("This is the first visit. No previous data available for comparison.")

		st.markdown("""
			<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
			<p style="margin-top:0; color:#666; font-weight:bold;">ABOUT THIS ANALYSIS:</p>
			<p>Color indicates direction of change: <span style="color:#FF4B4B">red = increase</span>, <span style="color:#00CC96">green = decrease</span>.<br>
			Asterisk (*) marks changes exceeding clinical thresholds: Resistance >15%, Capacitance >20%, Z >15%.</p>
			</div>
			""", unsafe_allow_html=True)

	def _display_tissue_health_assessment(self, tissue_health):
		"""
			Displays the tissue health assessment in the Streamlit UI.

			This method renders the tissue health assessment section including:
			- A header with tooltip explanation of how the tissue health index is calculated
			- The numerical health score with color-coded display (red: <40, orange: 40-60, green: >60)
			- A textual interpretation of the health score
			- A warning message if health score data is insufficient

			Parameters
			----------
			tissue_health : tuple
				A tuple containing (health_score, health_interp) where:
				- health_score (float or None): A numerical score from 0-100 representing tissue health
				- health_interp (str): A textual interpretation of the health score

			Returns:
			-------
			None
				This method updates the Streamlit UI but does not return a value
		"""

		st.markdown("### Tissue Health Assessment", help="The tissue health index is calculated using multi-frequency impedance ratios. The process involves: "
										"1) Extracting impedance data from sensor readings. "
										"2) Calculating the ratio of low to high frequency impedance (LF/HF ratio). "
										"3) Calculating phase angle if resistance and capacitance data are available. "
										"4) Normalizing the LF/HF ratio and phase angle to a 0-100 scale. "
										"5) Combining these scores with weightings to produce a final health score. "
										"6) Providing an interpretation based on the health score. "
										"The final score ranges from 0-100, with higher scores indicating better tissue health.")

		health_score, health_interp = tissue_health

		if health_score is not None:
			# Create a color scale for the health score
			color = "red" if health_score < 40 else "orange" if health_score < 60 else "green"
			st.markdown(f"**Tissue Health Index:** <span style='color:{color};font-weight:bold'>{health_score:.1f}/100</span>", unsafe_allow_html=True)
			st.markdown(f"**Interpretation:** {health_interp}")
		else:
			st.warning("Insufficient data for tissue health calculation")

	def _display_infection_risk_assessment(self, infection_risk):
		"""
			Displays the infection risk assessment information in the Streamlit app.

			This method presents the infection risk score, risk level, and contributing factors
			in a formatted way with color coding based on the risk severity.

			Parameters
			----------
			infection_risk : dict
				Dictionary containing infection risk assessment results with the following keys:
				- risk_score (float): A numerical score from 0-100 indicating infection risk
				- risk_level (str): A categorical assessment of risk (e.g., "Low", "Moderate", "High")
				- contributing_factors (list): List of factors that contribute to the infection risk

			Notes
			-----
			Risk score is color-coded:
			- Green: scores < 30 (low risk)
			- Orange: scores between 30 and 60 (moderate risk)
			- Red: scores ≥ 60 (high risk)

			The method includes a help tooltip that explains the factors used in risk assessment:
			1. Low/high frequency impedance ratio
			2. Sudden increase in low-frequency resistance
			3. Phase angle measurements
		"""

		st.markdown("### Infection Risk Assessment", help="The infection risk assessment is based on three key factors: "
			"1. Low/high frequency impedance ratio: A ratio > 15 is associated with increased infection risk."
			"2. Sudden increase in low-frequency resistance: A >30% increase may indicate an inflammatory response, "
			"which could be a sign of infection. This is because inflammation causes changes in tissue"
			"electrical properties, particularly at different frequencies."
			"3. Phase angle: Lower phase angles (<3°) indicate less healthy or more damaged tissue,"
			"which may be more susceptible to infection."
			"The risk score is a weighted sum of these factors, providing a quantitative measure of infection risk."
			"The final score ranges from 0-100, with higher scores indicating higher infection risk.")

		risk_score = infection_risk["risk_score"]
		risk_level = infection_risk["risk_level"]

		# Create a color scale for the risk score
		risk_color = "green" if risk_score < 30 else "orange" if risk_score < 60 else "red"
		st.markdown(f"**Infection Risk Score:** <span style='color:{risk_color};font-weight:bold'>{risk_score:.1f}/100</span>", unsafe_allow_html=True)
		st.markdown(f"**Risk Level:** {risk_level}")

		# Display contributing factors if any
		factors = infection_risk["contributing_factors"]
		if factors:
			st.markdown(f"**Contributing Factors:** {', '.join(factors)}")

	def _display_tissue_composition_analysis(self, freq_response):
		"""
			Displays the tissue composition analysis results in the Streamlit app based on frequency response data.

			This method presents the bioelectrical impedance analysis (BIA) results including alpha and beta
			dispersion values with their interpretation. It creates a section with explanatory headers and
			displays the calculated tissue composition metrics.

			Parameters
			-----------
			freq_response : dict
				Dictionary containing frequency response analysis results with the following keys:
				- 'alpha_dispersion': float, measurement of low to center frequency response
				- 'beta_dispersion': float, measurement of center to high frequency response
				- 'interpretation': str, clinical interpretation of the frequency response data

			Returns:
			--------
			None
				The method updates the Streamlit UI directly

			Note:
				- The method includes a help tooltip explaining the principles behind BIA and the significance
				of alpha and beta dispersion values.
		"""

		st.markdown("### Tissue Composition Analysis", help="""This analysis utilizes bioelectrical impedance analysis (BIA) principles to evaluatetissue characteristics based on the frequency-dependent response to electrical current.
		It focuses on two key dispersion regions:
		1. Alpha Dispersion (low to center frequency): Occurs in the kHz range, reflects extracellular fluid content and cell membrane permeability.
		Large alpha dispersion may indicate edema or inflammation.
		2. Beta Dispersion (center to high frequency): Beta dispersion is a critical phenomenon in bioimpedance analysis, occurring in the MHz frequency range (0.1–100 MHz) and providing insights into cellular structures. It reflects cell membrane properties (such as membrane capacitance and polarization, which govern how high-frequency currents traverse cells) and intracellular fluid content (including ionic conductivity and cytoplasmic resistance146. For example, changes in intracellular resistance (Ri) or membrane integrity directly alter the beta dispersion profile).
		Changes in beta dispersion can indicate alterations in cell structure or function.""")

		# Display tissue composition analysis from frequency response
		st.markdown("#### Analysis Results:")
		if 'alpha_dispersion' in freq_response and 'beta_dispersion' in freq_response:
			st.markdown(f"**Alpha Dispersion:** {freq_response['alpha_dispersion']:.3f}")
			st.markdown(f"**Beta Dispersion:** {freq_response['beta_dispersion']:.3f}")

		# Display interpretation with more emphasis
		st.markdown(f"**Tissue Composition Interpretation:** {freq_response['interpretation']}")

	def _display_visit_changes(self, changes, significant_changes):
		"""
			Display analysis of changes between visits.

			Args:
				changes: Dictionary mapping parameter names to percentage changes
				significant_changes: Dictionary mapping parameter names to boolean values
					indicating clinically significant
		"""
		st.markdown("#### Changes Since Previous Visit", help="""The changes since previous visit are based on bioelectrical impedance analysis (BIA) principles to evaluate the composition of biological tissues based on the frequency-dependent response to electrical current.""")

		if not changes:
			st.info("No comparable data from previous visit")
			return

		# Create a structured data dictionary to organize by frequency and parameter
		data_by_freq = {
			"Low Freq" : {"Z": None, "Resistance": None, "Capacitance": None},
			"Mid Freq" : {"Z": None, "Resistance": None, "Capacitance": None},
			"High Freq": {"Z": None, "Resistance": None, "Capacitance": None},
		}

		# Fill in the data from changes
		for key, change in changes.items():
			param_parts = key.split('_')
			param_name = param_parts[0].capitalize()
			freq_type = ' '.join(param_parts[1:]).replace('_', ' ')

			# Map to our standardized names
			if 'low frequency' in freq_type:
				freq_name = "Low Freq"
			elif 'center frequency' in freq_type:
				freq_name = "Mid Freq"
			elif 'high frequency' in freq_type:
				freq_name = "High Freq"
			else:
				continue

			# Check if this change is significant
			is_significant = significant_changes.get(key, False)

			# Format as percentage with appropriate sign and add asterisk if significant
			if change is not None:
				formatted_change = f"{change*100:+.1f}%"
				if is_significant:
					formatted_change = f"{formatted_change}*"
			else:
				formatted_change = "N/A"

			# Store in our data structure
			if param_name in ["Z", "Resistance", "Capacitance"]:
				data_by_freq[freq_name][param_name] = formatted_change

		# Convert to DataFrame for display
		change_df = pd.DataFrame(data_by_freq).T  # Transpose to get frequencies as rows

		# Reorder columns if needed
		if all(col in change_df.columns for col in ["Z", "Resistance", "Capacitance"]):
			change_df = change_df[["Z", "Resistance", "Capacitance"]]

		# Add styling
		def color_cells(val):
			try:
				if val is None or val == "N/A":
					return ''

				# Check if there's an asterisk and remove it for color calculation
				num_str = val.replace('*', '').replace('%', '')

				# Get numeric value by stripping % and sign
				num_val = float(num_str)

				# Determine colors based on value
				if num_val > 0:
					return 'color: #FF4B4B'  # Red for increases
				else:
					return 'color: #00CC96'  # Green for decreases
			except:
				return ''

		# Apply styling
		styled_df = change_df.style.map(color_cells).set_properties(**{
			'text-align': 'center',
			'font-size': '14px',
			'border': '1px solid #EEEEEE'
		})

		# Display as a styled table with a caption
		st.write("**Percentage Change by Parameter and Frequency:**")
		st.dataframe(styled_df)
		st.write("   (*) indicates clinically significant change")



	def _render_patient_advanced_analysis(self, visits) -> None:
		"""
			Renders the advanced bioelectrical analysis section for a patient's wound data.

			This method displays comprehensive bioelectrical analysis results including healing
			trajectory, wound healing stage classification, tissue electrical properties, and
			clinical insights derived from the impedance data. The analysis requires at least
			three visits to generate meaningful patterns and trends.

			Parameters:
				visits (list): A list of visit data objects containing impedance measurements and
							other wound assessment information.

			Returns:
			-------
			None
				The method updates the Streamlit UI directly

			Notes:
				- Displays a warning if fewer than three visits are available
				- Shows healing trajectory analysis with progression indicators
				- Presents wound healing stage classification based on impedance patterns
				- Displays Cole-Cole parameters representing tissue electrical properties if available
				- Provides clinical insights to guide treatment decisions
				- Includes reference information about bioimpedance interpretation
		"""

		st.subheader("Advanced Bioelectrical Interpretation")

		if len(visits) < 3:
			st.warning("At least three visits are required for advanced analysis")
			return

		# Generate advanced analysis
		analysis = self.impedance_analyzer.generate_advanced_analysis(visits)

		# Display healing trajectory analysis if available
		if 'healing_trajectory' in analysis and analysis['healing_trajectory']['status'] == 'analyzed':
			self._display_high_freq_impedance_healing_trajectory_analysis(analysis['healing_trajectory'])

		# Display wound healing stage classification
		self._display_wound_healing_stage(analysis['healing_stage'])

		# Display Cole-Cole parameters if available
		if 'cole_parameters' in analysis and analysis['cole_parameters']:
			self._display_tissue_electrical_properties(analysis['cole_parameters'])

		# Display clinical insights
		self._display_clinical_insights(analysis['insights'])

		# Reference information
		st.markdown("""
		<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
		<p style="margin-top:0; color:#666; font-weight:bold;">ABOUT THIS ANALYSIS:</p>

		<p style="font-weight:bold; margin-bottom:5px;">Frequency Significance:</p>
		<ul style="margin-top:0; padding-left:20px;">
		<li><strong>Low Frequency (100Hz):</strong> Primarily reflects extracellular fluid and tissue properties</li>
		<li><strong>Center Frequency:</strong> Reflects the maximum reactance point, varies based on tissue composition</li>
		<li><strong>High Frequency (80000Hz):</strong> Penetrates cell membranes, reflects total tissue properties</li>
		</ul>

		<p style="font-weight:bold; margin-bottom:5px;">Clinical Correlations:</p>
		<ul style="margin-top:0; padding-left:20px;">
		<li><strong>Decreasing High-Frequency Impedance:</strong> Often associated with improved healing</li>
		<li><strong>Increasing Low-to-High Frequency Ratio:</strong> May indicate inflammation or infection</li>
		<li><strong>Decreasing Phase Angle:</strong> May indicate deterioration in cellular health</li>
		<li><strong>Increasing Alpha Parameter:</strong> Often indicates increasing tissue heterogeneity</li>
		</ul>

		<p style="font-weight:bold; margin-bottom:5px;">Reference Ranges:</p>
		<ul style="margin-top:0; padding-left:20px;">
		<li><strong>Healthy Tissue Low/High Ratio:</strong> 5-12</li>
		<li><strong>Optimal Phase Angle:</strong> 5-7 degrees</li>
		<li><strong>Typical Alpha Range:</strong> 0.6-0.8</li>
		</ul>
		</div>
		""", unsafe_allow_html=True)

	def _display_high_freq_impedance_healing_trajectory_analysis(self, trajectory):
		"""
			Display the healing trajectory analysis with charts and statistics.

			This method visualizes the wound healing trajectory over time, including:
			- A line chart showing impedance values across visits
			- A trend line indicating the overall direction of change
			- Statistical analysis results (slope, p-value, R² value)
			- Interpretation of the healing trajectory

			Parameters
			----------
			trajectory : dict
				Dictionary containing healing trajectory data with the following keys:
				- dates : list of str
					Dates of the measurements
				- values : list of float
					Impedance values corresponding to each date
				- slope : float
					Slope of the trend line
				- p_value : float
					Statistical significance of the trend
				- r_squared : float
					R-squared value indicating goodness of fit
				- interpretation : str
					Text interpretation of the healing trajectory

			Returns:
			-------
			None
				This method displays its output directly in the Streamlit UI
		"""

		st.markdown("### Healing Trajectory Analysis")

		# Get trajectory data
		dates = trajectory['dates']
		values = trajectory['values']

		# Create trajectory chart
		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=list(range(len(dates))),
			y=values,
			mode='lines+markers',
			name='Impedance',
			line=dict(color='blue'),
			hovertemplate='%{y:.1f} kOhms'
		))

		# Add trend line
		x = np.array(range(len(values)))
		y = trajectory['slope'] * x + np.mean(values)
		fig.add_trace(go.Scatter(
			x=x,
			y=y,
			mode='lines',
			name='Trend',
			line=dict(color='red', dash='dash')
		))

		fig.update_layout(
			title="Impedance Trend Over Time",
			xaxis_title="Visit Number",
			yaxis_title="High-Frequency Impedance (kOhms)",
			hovermode="x unified"
		)

		st.plotly_chart(fig, use_container_width=True)

		# Display statistical results
		col1, col2 = st.columns(2)
		with col1:
			slope_color = "green" if trajectory['slope'] < 0 else "red"
			st.markdown(f"**Trend Slope:** <span style='color:{slope_color}'>{trajectory['slope']:.4f}</span>", unsafe_allow_html=True)
			st.markdown(f"**Statistical Significance:** p = {trajectory['p_value']:.4f}")

		with col2:
			st.markdown(f"**R² Value:** {trajectory['r_squared']:.4f}")
			st.info(trajectory['interpretation'])

		st.markdown("----")

	def _display_wound_healing_stage(self, healing_stage):
		"""
			Display wound healing stage classification in the Streamlit interface.

			This method renders the wound healing stage information with color-coded stages
			(red for Inflammatory, orange for Proliferative, green for Remodeling) and
			displays the confidence level and characteristics associated with the stage.

				healing_stage (dict): Dictionary containing wound healing stage analysis with keys:
					- 'stage': String indicating the wound healing stage (Inflammatory, Proliferative, or Remodeling)
					- 'confidence': Numeric or string value indicating confidence in the classification
					- 'characteristics': List of strings describing characteristics of the current healing stage

			Returns:
			-------
			None
				This method renders content to the Streamlit UI but does not return any value.
		"""
		st.markdown("### Wound Healing Stage Classification")

		stage_colors = {
			"Inflammatory": "red",
			"Proliferative": "orange",
			"Remodeling": "green"
		}

		stage_color = stage_colors.get(healing_stage['stage'], "blue")
		st.markdown(f"**Current Stage:** <span style='color:{stage_color};font-weight:bold'>{healing_stage['stage']}</span> (Confidence: {healing_stage['confidence']})", unsafe_allow_html=True)

		if healing_stage['characteristics']:
			st.markdown("**Characteristics:**")
			for char in healing_stage['characteristics']:
				st.markdown(f"- {char}")

	def _display_tissue_electrical_properties(self, cole_params):
		"""
		Displays the tissue electrical properties derived from Cole parameters in the Streamlit interface.

		This method creates a section in the UI that shows the key electrical properties of tissue,
		including extracellular resistance (R₀), total resistance (R∞), membrane capacitance (Cm),
		and tissue heterogeneity (α). Values are formatted with appropriate precision and units.

		Parameters
		----------
		cole_params : dict
			Dictionary containing the Cole-Cole parameters and related tissue properties.
			Expected keys include 'R0', 'Rinf', 'Cm', 'Alpha', and optionally 'tissue_homogeneity'.

		Returns:
		-------
		None
			The function directly updates the Streamlit UI and does not return a value.
		"""

		st.markdown("### Tissue Electrical Properties")

		col1, col2 = st.columns(2)

		with col1:
			if 'R0' in cole_params:
				st.markdown(f"**Extracellular Resistance (R₀):** {cole_params['R0']:.2f} Ω")
			if 'Rinf' in cole_params:
				st.markdown(f"**Total Resistance (R∞):** {cole_params['Rinf']:.2f} Ω")

		with col2:
			if 'Cm' in cole_params:
				st.markdown(f"**Membrane Capacitance:** {cole_params['Cm']:.2e} F")
			if 'Alpha' in cole_params:
				st.markdown(f"**Tissue Heterogeneity (α):** {cole_params['Alpha']:.2f}")
				st.info(cole_params.get('tissue_homogeneity', ''))

	def _display_clinical_insights(self, insights):
		"""
		Displays clinical insights in an organized expandable format using Streamlit components.

		This method renders clinical insights with their associated confidence levels, recommendations,
		supporting factors, and clinical interpretations. If no insights are available, it displays
		an informational message.

		Parameters
		----------
		insights : list
			A list of dictionaries, where each dictionary contains insight information with keys:
			- 'insight': str, the main insight text
			- 'confidence': str, confidence level of the insight
			- 'recommendation': str, optional, suggested actions based on the insight
			- 'supporting_factors': list, optional, factors supporting the insight
			- 'clinical_meaning': str, optional, clinical interpretation of the insight

		Returns:
		-------
		None
			This method renders UI components directly to the Streamlit UI but does not return any value.
		"""

		st.markdown("### Clinical Insights")

		if not insights:
			st.info("No significant clinical insights generated from current data.")
			return

		for i, insight in enumerate(insights):
			with st.expander(f"Clinical Insight {i+1}: {insight['insight'][:50]}...", expanded=i==0):
				st.markdown(f"**Insight:** {insight['insight']}")
				st.markdown(f"**Confidence:** {insight['confidence']}")

				if 'recommendation' in insight:
					st.markdown(f"**Recommendation:** {insight['recommendation']}")

				if 'supporting_factors' in insight and insight['supporting_factors']:
					st.markdown("**Supporting Factors:**")
					for factor in insight['supporting_factors']:
						st.markdown(f"- {factor}")

				if 'clinical_meaning' in insight:
					st.markdown(f"**Clinical Interpretation:** {insight['clinical_meaning']}")

	def _temperature_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
		Renders the temperature analysis tab of the dashboard.

		Parameters:
		----------
		df : pd.DataFrame
			The dataframe containing temperature data
		selected_patient : str
			Either "All Patients" or a specific patient ID
		"""
		st.header("Temperature Analysis")

		if selected_patient == "All Patients":
			# Use the generalized feature analysis function with temperature as the primary feature
			primary_feature = "Center of Wound Temperature (Fahrenheit)"
			default_features = [
				primary_feature,
				"Skin Impedance (kOhms) - Z",
				"Healing Rate (%)"
			]

			filtered_df = self._render_feature_analysis(
				df=df,
				primary_feature=primary_feature,
				default_correlated_features=default_features,
				feature_name_for_ui="Temperature"
			)

			# Render any additional temperature-specific visualizations here

		else:
			# Individual patient temperature analysis code (unchanged)
			df_temp = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
			df_temp['Visit date'] = pd.to_datetime(df_temp['Visit date']).dt.strftime('%m-%d-%Y')
			st.header(f"Temperature Analysis for Patient {selected_patient.split(' ')[1]}")

			# Get patient visits
			visits = self.data_processor.get_patient_visits(int(selected_patient.split(" ")[1]))

			# Create tabs
			trends_tab, visit_analysis_tab, overview_tab = st.tabs([
				"Temperature Trends",
				"Visit-by-Visit Analysis",
				"Overview & Clinical Guidelines"
			])

			with trends_tab:
				st.markdown("### Temperature Trends Over Time")
				fig = Visualizer.create_temperature_chart(df=df_temp)
				st.plotly_chart(fig, use_container_width=True)

				# Add statistical analysis
				temp_data = pd.DataFrame([
					{
						'date': visit['visit_date'],
						'center': visit['sensor_data']['temperature']['center'],
						'edge': visit['sensor_data']['temperature']['edge'],
						'peri': visit['sensor_data']['temperature']['peri']
					}
					for visit in visits['visits']
				])

				if not temp_data.empty:
					st.markdown("### Statistical Summary")
					col1, col2, col3 = st.columns(3)
					with col1:
						avg_center = temp_data['center'].mean()
						st.metric("Average Center Temp", f"{avg_center:.1f}°F")
					with col2:
						avg_edge = temp_data['edge'].mean()
						st.metric("Average Edge Temp", f"{avg_edge:.1f}°F")
					with col3:
						avg_peri = temp_data['peri'].mean()
						st.metric("Average Peri Temp", f"{avg_peri:.1f}°F")

			with visit_analysis_tab:
				st.markdown("### Visit-by-Visit Temperature Analysis")

				# Create tabs for each visit
				visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits['visits']])

				for tab, visit in zip(visit_tabs, visits['visits']):
					with tab:
						temp_data = visit['sensor_data']['temperature']

						# Display temperature readings
						st.markdown("#### Temperature Readings")
						col1, col2, col3 = st.columns(3)
						with col1:
							st.metric("center", f"{temp_data['center']}°F")
						with col2:
							st.metric("edge", f"{temp_data['edge']}°F")
						with col3:
							st.metric("peri", f"{temp_data['peri']}°F")

						# Calculate and display gradients
						if all(v is not None for v in temp_data.values()):

							st.markdown("#### Temperature Gradients")

							gradients = {
								'center-edge': temp_data['center'] - temp_data['edge'],
								'edge-peri': temp_data['edge'] - temp_data['peri'],
								'Total': temp_data['center'] - temp_data['peri']
							}

							col1, col2, col3 = st.columns(3)
							with col1:
								st.metric("center-edge", f"{gradients['center-edge']:.1f}°F")
							with col2:
								st.metric("edge-peri", f"{gradients['edge-peri']:.1f}°F")
							with col3:
								st.metric("Total Gradient", f"{gradients['Total']:.1f}°F")

						# Clinical interpretation
						st.markdown("#### Clinical Assessment")
						if temp_data['center'] is not None:
							center_temp = float(temp_data['center'])
							if center_temp < 93:
								st.warning("⚠️ Center temperature is below 93°F. This can significantly slow healing due to reduced blood flow and cellular activity.")
							elif 93 <= center_temp < 98:
								st.info("ℹ️ Center temperature is below optimal range. Mild warming might be beneficial.")
							elif 98 <= center_temp <= 102:
								st.success("✅ Center temperature is in the optimal range for wound healing.")
							else:
								st.error("❗ Center temperature is above 102°F. This may cause tissue damage and impair healing.")

						# Temperature gradient interpretation
						if all(v is not None for v in temp_data.values()):
							st.markdown("#### Gradient Analysis")
							if abs(gradients['Total']) > 4:
								st.warning(f"⚠️ Large temperature gradient ({gradients['Total']:.1f}°F) between center and periwound area may indicate inflammation or poor circulation.")
							else:
								st.success("✅ Temperature gradients are within normal range.")

			with overview_tab:
				st.markdown("### Clinical Guidelines for Temperature Assessment")
				st.markdown("""
					Temperature plays a crucial role in wound healing. Here's what the measurements indicate:
					- Optimal healing occurs at normal body temperature (98.6°F)
					- Temperatures below 93°F significantly slow healing
					- Temperatures between 98.6-102°F can promote healing
					- Temperatures above 102°F may damage tissues
				""")

				st.markdown("### Key Temperature Zones")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.error("❄️ Below 93°F")
					st.markdown("- Severely impaired healing\n- Reduced blood flow\n- Low cellular activity")
				with col2:
					st.info("🌡️ 93-98°F")
					st.markdown("- Suboptimal healing\n- May need warming\n- Monitor closely")
				with col3:
					st.success("✅ 98-102°F")
					st.markdown("- Optimal healing range\n- Good blood flow\n- Active metabolism")
				with col4:
					st.error("🔥 Above 102°F")
					st.markdown("- Tissue damage risk\n- Possible infection\n- Requires attention")

	def _oxygenation_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
		Renders the oxygenation analysis tab of the dashboard.

		Parameters:
		----------
		df : pd.DataFrame
			The dataframe containing oxygenation data
		selected_patient : str
			Either "All Patients" or a specific patient ID
		"""
		st.header("Oxygenation Analysis")

		if selected_patient == "All Patients":
			# Use the generalized feature analysis function with oxygenation as the primary feature
			primary_feature = "Oxygenation (%)"
			default_features = [
				primary_feature,
				"Skin Impedance (kOhms) - Z",
				"Healing Rate (%)"
			]

			filtered_df = self._render_feature_analysis(
				df=df,
				primary_feature=primary_feature,
				default_correlated_features=default_features,
				feature_name_for_ui="Oxygenation"
			)

			# Render any additional oxygenation-specific visualizations here

		else:
			# Individual patient oxygenation analysis code (unchanged)
			df_oxy = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
			df_oxy['Visit date'] = pd.to_datetime(df_oxy['Visit date']).dt.strftime('%m-%d-%Y')
			st.header(f"Oxygenation Analysis for Patient {selected_patient.split(' ')[1]}")

			# Get patient visits
			visits = self.data_processor.get_patient_visits(int(selected_patient.split(" ")[1]))

			# Create tabs
			trends_tab, visit_analysis_tab, overview_tab = st.tabs([
				"Oxygenation Trends",
				"Visit-by-Visit Analysis",
				"Overview & Clinical Guidelines"
			])

			with trends_tab:
				st.markdown("### Oxygenation Trends Over Time")
				fig = Visualizer.create_oxygenation_chart(df=df_oxy)
				st.plotly_chart(fig, use_container_width=True)

				# Add statistical analysis
				oxy_data = pd.DataFrame([
					{
						'date': visit['visit_date'],
						'center': visit['sensor_data']['oxygenation']['center'],
						'edge': visit['sensor_data']['oxygenation']['edge'],
						'peri': visit['sensor_data']['oxygenation']['peri']
					}
					for visit in visits['visits']
				])

				if not oxy_data.empty:
					st.markdown("### Statistical Summary")
					col1, col2, col3 = st.columns(3)
					with col1:
						avg_center = oxy_data['center'].mean()
						st.metric("Average Center Oxygenation", f"{avg_center:.1f}%")
					with col2:
						avg_edge = oxy_data['edge'].mean()
						st.metric("Average Edge Oxygenation", f"{avg_edge:.1f}%")
					with col3:
						avg_peri = oxy_data['peri'].mean()
						st.metric("Average Peri Oxygenation", f"{avg_peri:.1f}%")

			with visit_analysis_tab:
				st.markdown("### Visit-by-Visit Oxygenation Analysis")

				# Create tabs for each visit
				visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits['visits']])

				for tab, visit in zip(visit_tabs, visits['visits']):
					with tab:
						oxy_data = visit['sensor_data']['oxygenation']

						# Display oxygenation readings
						st.markdown("#### Oxygenation Readings")
						col1, col2, col3 = st.columns(3)
						with col1:
							st.metric("center", f"{oxy_data['center']}%")
						with col2:
							st.metric("edge", f"{oxy_data['edge']}%")
						with col3:
							st.metric("peri", f"{oxy_data['peri']}%")

						# Calculate and display gradients
						if all(v is not None for v in oxy_data.values()):

							st.markdown("#### Oxygenation Gradients")

							gradients = {
								'center-edge': oxy_data['center'] - oxy_data['edge'],
								'edge-peri': oxy_data['edge'] - oxy_data['peri'],
								'Total': oxy_data['center'] - oxy_data['peri']
							}

							col1, col2, col3 = st.columns(3)
							with col1:
								st.metric("center-edge", f"{gradients['center-edge']:.1f}%")
							with col2:
								st.metric("edge-peri", f"{gradients['edge-peri']:.1f}%")
							with col3:
								st.metric("Total Gradient", f"{gradients['Total']:.1f}%")

						# Clinical interpretation
						st.markdown("#### Clinical Assessment")
						if oxy_data['center'] is not None:
							center_oxy = float(oxy_data['center'])
							if center_oxy < 93:
								st.warning("⚠️ Center oxygenation is below 93%. This can significantly slow healing due to reduced blood flow and cellular activity.")
							elif 93 <= center_oxy < 98:
								st.info("ℹ️ Center oxygenation is below optimal range. Mild warming might be beneficial.")
							elif 98 <= center_oxy <= 102:
								st.success("✅ Center oxygenation is in the optimal range for wound healing.")
							else:
								st.error("❗ Center oxygenation is above 102%. This may cause tissue damage and impair healing.")

						# Oxygenation gradient interpretation
						if all(v is not None for v in oxy_data.values()):
							st.markdown("#### Gradient Analysis")
							if abs(gradients['Total']) > 4:
								st.warning(f"⚠️ Large oxygenation gradient ({gradients['Total']:.1f}%) between center and periwound area may indicate inflammation or poor circulation.")
							else:
								st.success("✅ Oxygenation gradients are within normal range.")

			with overview_tab:
				st.markdown("### Clinical Guidelines for Oxygenation Assessment")
				st.markdown("""
					Oxygenation plays a crucial role in wound healing. Here's what the measurements indicate:
					- Optimal healing occurs at normal body oxygenation (98.6%)
					- Oxygen levels below 93% significantly slow healing
					- Oxygen levels between 98.6-102% can promote healing
					- Oxygen levels above 102% may damage tissues
				""")

				st.markdown("### Key Oxygenation Zones")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.error("❄️ Below 93%")
					st.markdown("- Severely impaired healing\n- Reduced blood flow\n- Low cellular activity")
				with col2:
					st.info("🌡️ 93-98%")
					st.markdown("- Suboptimal healing\n- May need warming\n- Monitor closely")
				with col3:
					st.success("✅ 98-102%")
					st.markdown("- Optimal healing range\n- Good blood flow\n- Active metabolism")
				with col4:
					st.error("🔥 Above 102%")
					st.markdown("- Tissue damage risk\n- Possible infection\n- Requires attention")

	def _exudate_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
		Renders the exudate analysis tab of the dashboard.

		Parameters:
		----------
		df : pd.DataFrame
			The dataframe containing exudate data
		selected_patient : str
			Either "All Patients" or a specific patient ID
		"""
		st.header("Exudate Analysis")

		if selected_patient == "All Patients":
			# Use the generalized feature analysis function with exudate as the primary feature
			# For exudate, Hemoglobin Level can be a primary metric
			primary_feature = "Hemoglobin Level"
			default_features = [
				primary_feature,
				"Skin Impedance (kOhms) - Z",
				"Healing Rate (%)"
			]

			filtered_df = self._render_feature_analysis(
				df=df,
				primary_feature=primary_feature,
				default_correlated_features=default_features,
				feature_name_for_ui="Exudate"
			)

			# Render any additional exudate-specific visualizations here

		else:
			# Individual patient exudate analysis code (unchanged)
			df_exudate = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
			df_exudate['Visit date'] = pd.to_datetime(df_exudate['Visit date']).dt.strftime('%m-%d-%Y')
			st.header(f"Exudate Analysis for Patient {selected_patient.split(' ')[1]}")

			# Get patient visits
			visits = self.data_processor.get_patient_visits(int(selected_patient.split(" ")[1]))

			# Create tabs
			trends_tab, visit_analysis_tab, overview_tab = st.tabs([
				"Exudate Trends",
				"Visit-by-Visit Analysis",
				"Overview & Clinical Guidelines"
			])

			with trends_tab:
				st.markdown("### Exudate Trends Over Time")
				fig = Visualizer.create_exudate_chart(df_exudate)
				st.plotly_chart(fig, use_container_width=True)

				# Add statistical analysis
				exudate_data = pd.DataFrame([
					{
						'date': visit['visit_date'],
						'volume': visit['wound_info']['exudate'].get('volume', 'N/A'),
						'viscosity': visit['wound_info']['exudate'].get('viscosity', 'N/A'),
						'type': visit['wound_info']['exudate'].get('type', 'N/A')
					}
					for visit in visits['visits']
				])

				if not exudate_data.empty:
					st.markdown("### Statistical Summary")
					col1, col2, col3 = st.columns(3)
					with col1:
						avg_volume = exudate_data['volume'].mean()
						st.metric("Average Exudate Volume", f"{avg_volume:.1f} ml")
					with col2:
						avg_viscosity = exudate_data['viscosity'].mean()
						st.metric("Average Exudate Viscosity", f"{avg_viscosity:.1f} cP")
					with col3:
						avg_type = exudate_data['type'].mode()[0]
						st.metric("Most Common Exudate Type", f"{avg_type}")

			with visit_analysis_tab:
				st.markdown("### Visit-by-Visit Exudate Analysis")

				# Create tabs for each visit
				visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits['visits']])

				for tab, visit in zip(visit_tabs, visits['visits']):
					with tab:
						exudate_data = visit['wound_info']['exudate']

						# Display exudate data
						st.markdown("#### Exudate Readings")
						col1, col2, col3 = st.columns(3)
						with col1:
							st.metric("Volume", f"{exudate_data['volume']} ml")
						with col2:
							st.metric("Viscosity", f"{exudate_data['viscosity']} cP")
						with col3:
							st.metric("Type", f"{exudate_data['type']}")

						# Clinical interpretation
						st.subheader("Clinical Interpretation of Exudate Characteristics")

						# Create tabs for each exudate type
						exudate_type_tabs = st.tabs([
							"Low", "Medium", "High"
						])

						for exudate_type_tab, exudate_type in zip(exudate_type_tabs, ["Low", "Medium", "High"]):
							with exudate_type_tab:
								st.markdown(f"**Exudate Type: {exudate_type}**")
								st.markdown(f"**Hemoglobin Level:** {exudate_data['hemoglobin_level']:.2f} g/dL")
								st.markdown(f"**Exudate Volume:** {exudate_data['volume']} ml")
								st.markdown(f"**Exudate Viscosity:** {exudate_data['viscosity']} cP")
								st.markdown(f"**Exudate Type:** {exudate_data['type']}")

						# Exudate Type Analysis
						st.markdown('----')
						col1, col2 = st.columns(2)

						with col1:
							st.markdown("### Exudate Type Distribution")
							exudate_type_counts = exudate_data['type'].value_counts()
							fig_pie = px.pie(
								values=exudate_type_counts.values,
								names=exudate_type_counts.index,
								title="Exudate Type Distribution"
							)
							st.plotly_chart(fig_pie, use_container_width=True)

						with col2:
							st.markdown("### Exudate Volume and Viscosity Trends")
							fig_volume = px.line(
								exudate_data,
								x='Visit date',
								y='volume',
								title="Exudate Volume Over Time",
								labels={'volume': 'Volume (ml)'}
							)
							fig_volume.update_layout(yaxis_range=[0, exudate_data['volume'].max() + 1])
							st.plotly_chart(fig_volume, use_container_width=True)

							fig_viscosity = px.line(
								exudate_data,
								x='Visit date',
								y='viscosity',
								title="Exudate Viscosity Over Time",
								labels={'viscosity': 'Viscosity (cP)'}
							)
							fig_viscosity.update_layout(yaxis_range=[0, exudate_data['viscosity'].max() + 1])
							st.plotly_chart(fig_viscosity, use_container_width=True)

			with overview_tab:
				st.markdown("### Clinical Guidelines for Exudate Management")
				st.markdown("""
					Exudate management is crucial for wound healing. Here's what the measurements indicate:
					- Low exudate volume and viscosity is generally favorable for healing.
					- High exudate volume or viscosity may indicate increased inflammation or infection.
					- Changes in exudate type can suggest different underlying conditions.
				""")

				st.markdown("### Key Exudate Characteristics")
				col1, col2, col3 = st.columns(3)
				with col1:
					st.error("❄️ Low Exudate Volume")
					st.markdown("- Favorable for healing\n- Reduced risk of infection")
				with col2:
					st.info("🌡️ Normal Exudate Viscosity")
					st.markdown("- Ideal for wound healing\n- Reduced risk of infection")
				with col3:
					st.success("✅ High Exudate Volume")
					st.markdown("- Increased risk of infection\n- May require special care")

	def _risk_factors_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
		Renders the Risk Factors Analysis tab in the Streamlit application.
		This tab provides analysis of how different risk factors (diabetes, smoking, BMI)
		affect wound healing rates. For the aggregate view ('All Patients'), it shows
		statistical distributions and comparisons across different patient groups.
		For individual patients, it displays a personalized risk profile and assessment.
		Features:
		- For all patients: Interactive tabs showing wound healing statistics across diabetes status, smoking status, and BMI categories, with visualizations and statistical summaries
		- For individual patients: Risk profile with factors like diabetes, smoking status and BMI, plus a computed risk score and estimated healing time
		Args:
			df (pd.DataFrame): The dataframe containing all patient wound data
			selected_patient (str): The currently selected patient ID as a string, or "All Patients" for aggregate view
		Returns:
			None: This method renders UI components directly in Streamlit
		"""

		st.header("Risk Factors Analysis")

		if selected_patient == "All Patients":
			risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

			valid_df = df.dropna(subset=['Healing Rate (%)', 'Visit Number']).copy()

			# Add detailed warning for outliers with statistics
			# outliers = valid_df[abs(valid_df['Healing Rate (%)']) > 100]
			# if not outliers.empty:
			# 	st.warning(
			# 		f"⚠️ Data Quality Alert:\n\n"
			# 		f"Found {len(outliers)} measurements ({(len(outliers)/len(valid_df)*100):.1f}% of data) "
			# 		f"with healing rates outside the expected range (-100% to 100%).\n\n"
			# 		f"Statistics:\n"
			# 		f"- Minimum value: {outliers['Healing Rate (%)'].min():.1f}%\n"
			# 		f"- Maximum value: {outliers['Healing Rate (%)'].max():.1f}%\n"
			# 		f"- Mean value: {outliers['Healing Rate (%)'].mean():.1f}%\n"
			# 		f"- Number of unique patients affected: {len(outliers['Record ID'].unique())}\n\n"
			# 		"These values will be clipped to [-100%, 100%] range for visualization purposes."
			# 	)

			# Clip healing rates to reasonable range
			# valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

			for col in ['Diabetes?', 'Smoking status', 'BMI']:
				# Add consistent diabetes status for each patient
				first_diabetes_status = valid_df.groupby('Record ID')[col].first()
				valid_df[col] = valid_df['Record ID'].map(first_diabetes_status)

			valid_df['Healing_Color'] = valid_df['Healing Rate (%)'].apply(
				lambda x: 'green' if x < 0 else 'red'
			)

			with risk_tab1:

				st.subheader("Impact of Diabetes on Wound Healing")

				# Ensure diabetes status is consistent for each patient
				valid_df['Diabetes?'] = valid_df['Diabetes?'].fillna('No')

				# Compare average healing rates by diabetes status
				diab_stats = valid_df.groupby('Diabetes?').agg({ 'Healing Rate (%)': ['mean', 'count', 'std'] }).round(2)

				# Create a box plot for healing rates with color coding
				fig1 = px.box(
					valid_df,
					x='Diabetes?',
					y='Healing Rate (%)',
					title="Healing Rate Distribution by Diabetes Status",
					color='Healing_Color',
					color_discrete_map={'green': 'green', 'red': 'red'},
					points='all'
				)
				fig1.update_layout(
					xaxis_title="Diabetes Status",
					yaxis_title="Healing Rate (%)",
					showlegend=True,
					legend_title="Wound Status",
					legend={'traceorder': 'reversed'},
					yaxis=dict(
						range=[-100, 100],
						tickmode='linear',
						tick0=-100,
						dtick=25
					)
				)

				# Update legend labels
				fig1.for_each_trace(lambda t: t.update(name='Improving' if t.name == 'green' else 'Worsening'))
				st.plotly_chart(fig1, use_container_width=True)

				# Display statistics
				st.write("**Statistical Summary:**")
				for status in diab_stats.index:
					stats_data = diab_stats.loc[status]
					improvement_rate = (valid_df[valid_df['Diabetes?'] == status]['Healing Rate (%)'] < 0).mean() * 100
					st.write(f"- {status}: Average Healing Rate = {stats_data[('Healing Rate (%)', 'mean')]}% "
							"(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
							"SD={stats_data[('Healing Rate (%)', 'std')]}, "
							"Improvement Rate={improvement_rate:.1f}%)")

				# Compare wound types distribution
				wound_diab = pd.crosstab(valid_df['Diabetes?'], valid_df['Wound Type'], normalize='index') * 100
				fig2 = px.bar(
					wound_diab.reset_index().melt(id_vars='Diabetes?', var_name='Wound Type', value_name='Percentage'),
					x='Diabetes?',
					y='Percentage',
					color='Wound Type',
					title="Wound Type Distribution by Diabetes Status",
					labels={'Percentage': 'Percentage of Wounds (%)'}
				)
				st.plotly_chart(fig2, use_container_width=True)

			with risk_tab2:
				st.subheader("Impact of Smoking on Wound Healing")

				# Clean smoking status
				valid_df['Smoking status'] = valid_df['Smoking status'].fillna('Never')

				# Create healing rate distribution by smoking status with color coding
				fig1 = px.box(
					valid_df,
					x='Smoking status',
					y='Healing Rate (%)',
					title="Healing Rate Distribution by Smoking Status",
					color='Healing_Color',
					color_discrete_map={'green': 'green', 'red': 'red'},
					points='all'
				)
				fig1.update_layout(
					showlegend=True,
					legend_title="Wound Status",
					legend={'traceorder': 'reversed'},
					yaxis=dict(
						range=[-100, 100],
						tickmode='linear',
						tick0=-100,
						dtick=25
					)
				)
				# Update legend labels
				fig1.for_each_trace(lambda t: t.update(name='Improving' if t.name == 'green' else 'Worsening'))
				st.plotly_chart(fig1, use_container_width=True)

				# Calculate and display statistics
				smoke_stats = valid_df.groupby('Smoking status').agg({
					'Healing Rate (%)': ['mean', 'count', 'std']
				}).round(2)

				st.write("**Statistical Summary:**")
				for status in smoke_stats.index:
					stats_data = smoke_stats.loc[status]
					improvement_rate = (valid_df[valid_df['Smoking status'] == status]['Healing Rate (%)'] < 0).mean() * 100
					st.write(f"- {status}: Average Healing Rate = {stats_data[('Healing Rate (%)', 'mean')]}% "
							"(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
							"SD={stats_data[('Healing Rate (%)', 'std')]}, "
							"Improvement Rate={improvement_rate:.1f}%)")

				# Wound type distribution by smoking status
				wound_smoke = pd.crosstab(valid_df['Smoking status'], valid_df['Wound Type'], normalize='index') * 100
				fig2 = px.bar(
					wound_smoke.reset_index().melt(id_vars='Smoking status', var_name='Wound Type', value_name='Percentage'),
					x='Smoking status',
					y='Percentage',
					color='Wound Type',
					title="Wound Type Distribution by Smoking Status",
					labels={'Percentage': 'Percentage of Wounds (%)'}
				)
				st.plotly_chart(fig2, use_container_width=True)

			with risk_tab3:
				st.subheader("Impact of BMI on Wound Healing")

				# Create BMI categories
				bins = [0, 18.5, 24.9, 29.9, float('inf')]
				labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
				valid_df['BMI Category'] = pd.cut(valid_df['BMI'], bins=bins, labels=labels)

				# Create healing rate distribution by BMI category with color coding
				fig1 = px.box(
					valid_df,
					x='BMI Category',
					y='Healing Rate (%)',
					title="Healing Rate Distribution by BMI Category",
					color='Healing_Color',
					color_discrete_map={'green': 'green', 'red': 'red'},
					points='all'
				)
				fig1.update_layout(
					showlegend=True,
					legend_title="Wound Status",
					legend={'traceorder': 'reversed'},
					yaxis=dict(
						range=[-100, 100],
						tickmode='linear',
						tick0=-100,
						dtick=25
					)
				)
				# Update legend labels
				fig1.for_each_trace(lambda t: t.update(name='Improving' if t.name == 'green' else 'Worsening'))
				st.plotly_chart(fig1, use_container_width=True)

				# Calculate and display statistics
				bmi_stats = valid_df.groupby('BMI Category', observed=False).agg({
					'Healing Rate (%)': ['mean', 'count', 'std']
				}).round(2)

				st.write("**Statistical Summary:**")
				for category in bmi_stats.index:
					stats_data = bmi_stats.loc[category]
					improvement_rate = (valid_df[valid_df['BMI Category'] == category]['Healing Rate (%)'] < 0).mean() * 100
					st.write(f"- {category}: Average Healing Rate = {stats_data[('Healing Rate (%)', 'mean')]}% "
							"(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
							"SD={stats_data[('Healing Rate (%)', 'std')]}, "
							"Improvement Rate={improvement_rate:.1f}%)")

				# Wound type distribution by BMI category
				wound_bmi = pd.crosstab(valid_df['BMI Category'], valid_df['Wound Type'], normalize='index') * 100
				fig2 = px.bar(
					wound_bmi.reset_index().melt(id_vars='BMI Category', var_name='Wound Type', value_name='Percentage'),
					x='BMI Category',
					y='Percentage',
					color='Wound Type',
					title="Wound Type Distribution by BMI Category",
					labels={'Percentage': 'Percentage of Wounds (%)'}
				)
				st.plotly_chart(fig2, use_container_width=True)

		else:
			# For individual patient
			df_temp = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
			patient_data = df_temp.iloc[0]

			# Create columns for the metrics
			col1, col2 = st.columns(2)

			with col1:
				st.subheader("Patient Risk Profile")

				# Display key risk factors
				st.info(f"**Diabetes Status:** {patient_data['Diabetes?']}")
				st.info(f"**Smoking Status:** {patient_data['Smoking status']}")
				st.info(f"**BMI:** {patient_data['BMI']:.1f}")

				# BMI category
				bmi = patient_data['BMI']
				if bmi < 18.5:
					bmi_category = "Underweight"
				elif bmi < 25:
					bmi_category = "Normal"
				elif bmi < 30:
					bmi_category = "Overweight"
				elif bmi < 35:
					bmi_category = "Obese Class I"
				else:
					bmi_category = "Obese Class II-III"

				st.info(f"**BMI Category:** {bmi_category}")

			with col2:
				st.subheader("Risk Assessment")

				# Create a risk score based on known factors
				# This is a simplified example - in reality would be based on clinical evidence
				risk_factors = []
				risk_score = 0

				if patient_data['Diabetes?'] == 'Yes':
					risk_factors.append("Diabetes")
					risk_score += 3

				if patient_data['Smoking status'] == 'Current':
					risk_factors.append("Current smoker")
					risk_score += 2
				elif patient_data['Smoking status'] == 'Former':
					risk_factors.append("Former smoker")
					risk_score += 1

				if patient_data['BMI'] >= 30:
					risk_factors.append("Obesity")
					risk_score += 2
				elif patient_data['BMI'] >= 25:
					risk_factors.append("Overweight")
					risk_score += 1

				# Temperature gradient risk
				temp_gradient = patient_data['Center of Wound Temperature (Fahrenheit)'] - patient_data['Peri-wound Temperature (Fahrenheit)']
				if temp_gradient > 3:
					risk_factors.append("High temperature gradient")
					risk_score += 2

				# Impedance risk
				if patient_data['Skin Impedance (kOhms) - Z'] > 140:
					risk_factors.append("High impedance")
					risk_score += 2

				# Calculate risk category
				if risk_score >= 6:
					risk_category = "High"
					risk_color = "red"
				elif risk_score >= 3:
					risk_category = "Moderate"
					risk_color = "orange"
				else:
					risk_category = "Low"
					risk_color = "green"

				# Display risk category
				st.markdown(f"**Risk Category:** <span style='color:{risk_color};font-weight:bold'>{risk_category}</span> ({risk_score} points)", unsafe_allow_html=True)

				# Display risk factors
				if risk_factors:
					st.markdown("**Risk Factors:**")
					for factor in risk_factors:
						st.markdown(f"- {factor}")
				else:
					st.markdown("**Risk Factors:** None identified")

				# Estimated healing time based on risk score and wound size
				wound_area = patient_data['Calculated Wound Area']
				base_healing_weeks = 2 + wound_area/2  # Simple formula: 2 weeks + 0.5 weeks per cm²
				risk_multiplier = 1 + (risk_score * 0.1)  # Each risk point adds 10% to healing time
				est_healing_weeks = base_healing_weeks * risk_multiplier

				st.markdown(f"**Estimated Healing Time:** {est_healing_weeks:.1f} weeks")

	def _llm_analysis_tab(self, selected_patient: str) -> None:
		"""
		Creates and manages the LLM analysis tab in the Streamlit application.

		This method sets up the interface for running LLM-powered wound analysis
		on either all patients collectively or on a single selected patient.
		It handles the initialization of the LLM, retrieval of patient data,
		generation of analysis, and presentation of results.

		Parameters
		----------
		selected_patient : str
			The currently selected patient identifier (e.g., "Patient 1") or "All Patients"

		Returns:
		-------
		None
			The method updates the Streamlit UI directly

		Notes:
		-----
		- Analysis results are stored in session state to persist between reruns
		- The method supports two analysis modes:
			1. Population analysis (when "All Patients" is selected)
			2. Individual patient analysis (when a specific patient is selected)
		- Analysis results and prompts are displayed in separate tabs
		- A download button is provided for exporting reports as Word documents
		"""

		def _display_llm_analysis(llm_reports: dict):
			if 'thinking_process' in llm_reports and llm_reports['thinking_process'] is not None:
				tab1, tab2, tab3 = st.tabs(["Analysis", "Prompt", "Thinking Process"])
			else:
				tab1, tab2 = st.tabs(["Analysis", "Prompt"])
				tab3 = None

			with tab1:
				st.markdown(llm_reports['analysis_results'])
			with tab2:
				st.markdown(llm_reports['prompt'])

			if tab3 is not None:
				st.markdown("### Model's Thinking Process")
				st.markdown(llm_reports['thinking_process'])

			# Add download button for the report
			DataManager.download_word_report(st=st, report_path=llm_reports['report_path'])


		def _run_llm_analysis(prompt: str, patient_metadata: dict=None, analysis_results: dict=None) -> dict:

			if self.llm_model == "deepseek-r1":

				# Create a container for the analysis
				analysis_container = st.empty()
				analysis_container.markdown("### Analysis\n\n*Generating analysis...*")

				# Create a container for the thinking process

				thinking_container.markdown("### Thinking Process\n\n*Thinking...*")

				# Update the analysis container with final results
				analysis_container.markdown(f"### Analysis\n\n{analysis}")

				thinking_process = llm.get_thinking_process()

			else:
				thinking_process = None


			# Store analysis in session state for this patient
			report_path = DataManager.create_and_save_report(patient_metadata=patient_metadata, analysis_results=analysis_results, report_path=None)

			# Return data dictionary
			return dict(
				analysis_results = analysis,
				patient_metadata = patient_metadata,
				prompt           = prompt,
				report_path      = report_path,
				thinking_process = thinking_process
			)

		def stream_callback(data):
			if data["type"] == "thinking":
				thinking_container.markdown(f"### Thinking Process\n\n{data['content']}")

		st.header("LLM-Powered Wound Analysis")

		thinking_container = st.empty()

		# Initialize reports dictionary in session state if it doesn't exist
		if 'llm_reports' not in st.session_state:
			st.session_state.llm_reports = {}

		if self.csv_dataset_path is not None:

			llm = WoundAnalysisLLM(platform=self.llm_platform, model_name=self.llm_model)

			callback = stream_callback if self.llm_model == "deepseek-r1" else None

			if selected_patient == "All Patients":
				if st.button("Run Analysis", key="run_analysis"):
					population_data = self.data_processor.get_population_statistics()
					prompt = llm._format_population_prompt(population_data=population_data)
					analysis = llm.analyze_population_data(population_data=population_data, callback=callback)
					st.session_state.llm_reports['all_patients'] = _run_llm_analysis(prompt=prompt, analysis_results=analysis, patient_metadata=population_data)

				# Display analysis if it exists for this patient
				if 'all_patients' in st.session_state.llm_reports:
					_display_llm_analysis(st.session_state.llm_reports['all_patients'])

			else:
				patient_id = selected_patient.split(' ')[1]
				st.subheader(f"Patient {patient_id}")

				if st.button("Run Analysis", key="run_analysis"):
					patient_data = self.data_processor.get_patient_visits(int(patient_id))
					prompt = llm._format_per_patient_prompt(patient_data=patient_data)
					analysis = llm.analyze_patient_data(patient_data=patient_data, callback=callback)
					st.session_state.llm_reports[patient_id] = _run_llm_analysis(prompt=prompt, analysis_results=analysis, patient_metadata=patient_data['patient_metadata'])

				# Display analysis if it exists for this patient
				if patient_id in st.session_state.llm_reports:
					_display_llm_analysis(st.session_state.llm_reports[patient_id])
		else:
			st.warning("Please upload a patient data file from the sidebar to enable LLM analysis.")

	def _get_input_user_data(self) -> None:
		"""
		Get user inputs from Streamlit interface for data paths and validate them.

		This method provides UI components for users to:
		1. Upload a CSV file containing patient data
		2. Specify a path to the folder containing impedance frequency sweep XLSX files
		3. Validate the path and check for the existence of XLSX files

		The method populates:
		- self.csv_dataset_path: The uploaded CSV file
		- self.impedance_freq_sweep_path: Path to the folder containing impedance XLSX files

		Returns:
			None
		"""

		self.csv_dataset_path = st.file_uploader("Upload Patient Data (CSV)", type=['csv'])

		default_path = str(pathlib.Path(__file__).parent.parent / "dataset/impedance_frequency_sweep")

		if self.csv_dataset_path is not None:
			# Text input for dataset path
			dataset_path_input = st.text_input(
				"Path to impedance_frequency_sweep folder",
				value=default_path,
				help="Enter the absolute path to the folder containing impedance frequency sweep XLSX files",
				key="dataset_path_input_1"
			)

			# Convert to Path object
			self.impedance_freq_sweep_path = pathlib.Path(dataset_path_input)

			# Button to check if files exist
			# if st.button("Check XLSX Files"):
			try:
				# Check if path exists
				if not self.impedance_freq_sweep_path.exists():
					st.error(f"Path does not exist: {self.impedance_freq_sweep_path}")
				else:
					# Count xlsx files
					xlsx_files = list(self.impedance_freq_sweep_path.glob("**/*.xlsx"))

					if xlsx_files:
						st.success(f"Found {len(xlsx_files)} XLSX files in the directory")
						# Show files in an expander
						with st.expander("View Found Files"):
							for file in xlsx_files:
								st.text(f"- {file.name}")
					else:
						st.warning(f"No XLSX files found in {self.dataset_path}")
			except Exception as e:
				st.error(f"Error checking path: {e}")

	def _create_left_sidebar(self) -> None:
		"""
		Creates the left sidebar of the Streamlit application.

		This method sets up the sidebar with model configuration options, file upload functionality,
		and informational sections about the dashboard. The sidebar includes:

		1. Model Configuration section:
			- File uploader for patient data (CSV files)
			- Platform selector (defaulting to ai-verde)
			- Model selector with appropriate defaults based on the platform
			- Advanced settings expandable section for API keys and base URLs

		2. Information sections:
			- About This Dashboard: Describes the purpose and data visualization focus
			- Statistical Methods: Outlines analytical approaches used in the dashboard

		Returns:
			None
		"""

		with st.sidebar:
			st.markdown("### Dataset Configuration")
			self._get_input_user_data()

			st.markdown("---")
			st.subheader("Model Configuration")

			platform_options = WoundAnalysisLLM.get_available_platforms()

			self.llm_platform = st.selectbox( "Select Platform", platform_options,
				index=platform_options.index("ai-verde") if "ai-verde" in platform_options else 0,
				help="Hugging Face models are currently disabled. Please use AI Verde models."
			)

			if self.llm_platform == "huggingface":
				st.warning("Hugging Face models are currently disabled. Please use AI Verde models.")
				self.llm_platform = "ai-verde"

			available_models = WoundAnalysisLLM.get_available_models(self.llm_platform)
			default_model = "llama-3.3-70b-fp8" if self.llm_platform == "ai-verde" else "medalpaca-7b"
			self.llm_model = st.selectbox( "Select Model", available_models,
				index=available_models.index(default_model) if default_model in available_models else 0
			)

			# Add warning for deepseek-r1 model
			if self.llm_model == "deepseek-r1":
				st.warning("**Warning:** The DeepSeek R1 model is currently experiencing connection issues. Please select a different model for reliable results.", icon="⚠️")
				self.llm_model = "llama-3.3-70b-fp8"

			with st.expander("Advanced Model Settings"):

				api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
				if api_key:
					os.environ["OPENAI_API_KEY"] = api_key

				if self.llm_platform == "ai-verde":
					base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", ""))
					if base_url:
						os.environ["OPENAI_BASE_URL"] = base_url



	def _render_all_patients_overview(self, df: pd.DataFrame) -> None:
		"""
		Renders the overview dashboard for all patients in the dataset.

		This method creates a population statistics section with a wound area progression
		plot and key metrics including average days in study, estimated treatment duration,
		average healing rate, and overall improvement rate.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing wound data for all patients with columns:
			- 'Record ID': Patient identifier
			- 'Days_Since_First_Visit': Number of days elapsed since first visit
			- 'Estimated_Days_To_Heal': Predicted days until wound heals (if available)
			- 'Healing Rate (%)': Rate of healing in cm²/day
			- 'Overall_Improvement': 'Yes' or 'No' indicating if wound is improving

		Returns:
		-------
		None
			This method renders content to the Streamlit app interface
		"""

		st.subheader("Population Statistics")

		# Display wound area progression for all patients
		st.plotly_chart(
			self.visualizer.create_wound_area_plot(df),
			use_container_width=True
		)

		col1, col2, col3, col4 = st.columns(4)

		with col1:
			# Calculate average days in study (actual duration so far)
			avg_days_in_study = df.groupby('Record ID')['Days_Since_First_Visit'].max().mean()
			st.metric("Average Days in Study", f"{avg_days_in_study:.1f} days")

		with col2:
			try:
				# Calculate average estimated treatment duration for improving wounds
				estimated_days = df.groupby('Record ID')['Estimated_Days_To_Heal'].mean()
				valid_estimates = estimated_days[estimated_days.notna()]
				if len(valid_estimates) > 0:
					avg_estimated_duration = valid_estimates.mean()
					st.metric("Est. Treatment Duration", f"{avg_estimated_duration:.1f} days")
				else:
					st.metric("Est. Treatment Duration", "N/A")
			except (KeyError, AttributeError):
				st.metric("Est. Treatment Duration", "N/A")

		with col3:
			# Calculate average healing rate excluding zeros and infinite values
			healing_rates = df['Healing Rate (%)']
			valid_rates = healing_rates[(healing_rates != 0) & (np.isfinite(healing_rates))]
			avg_healing_rate = np.mean(valid_rates) if len(valid_rates) > 0 else 0
			st.metric("Average Healing Rate", f"{abs(avg_healing_rate):.2f} cm²/day")

		with col4:
			try:
				# Calculate improvement rate using only the last visit for each patient
				if 'Overall_Improvement' not in df.columns:
					df['Overall_Improvement'] = np.nan

				# Get the last visit for each patient and calculate improvement rate
				last_visits = df.groupby('Record ID').agg({
					'Overall_Improvement': 'last',
					'Healing Rate (%)': 'last'
				})

				# Calculate improvement rate from patients with valid improvement status
				valid_improvements = last_visits['Overall_Improvement'].dropna()
				if len(valid_improvements) > 0:
					improvement_rate = (valid_improvements == 'Yes').mean() * 100
					st.metric("Improvement Rate", f"{improvement_rate:.1f}%")
				else:
					st.metric("Improvement Rate", "N/A")

			except Exception as e:
				st.metric("Improvement Rate", "N/A")
				print(f"Error calculating improvement rate: {e}")

	def _render_patient_overview(self, df: pd.DataFrame, patient_id: int) -> None:
		"""
		Render the patient overview section in the Streamlit application.

		This method displays a comprehensive overview of a patient's profile, including demographics,
		medical history, diabetes status, and detailed wound information from their first visit.
		The information is organized into multiple columns and sections for better readability.

		Parameters
		----------
		df : pd.DataFrame
			The DataFrame containing all patient data.
		patient_id : int
			The unique identifier for the patient whose data should be displayed.

		Returns:
		-------
		None
			This method renders content to the Streamlit UI and doesn't return any value.

		Notes
		-----
		The method handles cases where data might be missing by displaying placeholder values.
		It organizes information into three main sections:
		1. Patient demographics and medical history
		2. Wound details including location, type, and care
		3. Specific wound characteristics like infection, granulation, and undermining

		The method will display an error message if no data is available for the specified patient.
		"""


		def get_metric_value(metric_name: str) -> str:
			# Check for None, nan, empty string, whitespace, and string 'nan'
			if metric_name is None or str(metric_name).lower().strip() in ['', 'nan', 'none']:
				return '---'
			return str(metric_name)


		patient_df = DataManager.get_patient_data(df, patient_id)
		if patient_df.empty:
			st.error("No data available for this patient.")
			return

		patient_data = patient_df.iloc[0]

		metadata = DataManager._extract_patient_metadata(patient_data)

		col1, col2, col3 = st.columns(3)

		with col1:
			st.subheader("Patient Demographics")
			st.write(f"Age: {metadata.get('age')} years")
			st.write(f"Sex: {metadata.get('sex')}")
			st.write(f"BMI: {metadata.get('bmi')}")
			st.write(f"Race: {metadata.get('race')}")
			st.write(f"Ethnicity: {metadata.get('ethnicity')}")

		with col2:
			st.subheader("Medical History")

			# Display active medical conditions
			if medical_history := metadata.get('medical_history'):
				active_conditions = {
					condition: status
					for condition, status in medical_history.items()
					if status and status != 'None'
				}
				for condition, status in active_conditions.items():
					st.write(f"{condition}: {status}")

			smoking     = get_metric_value(patient_data.get("Smoking status"))
			med_history = get_metric_value(patient_data.get("Medical History (select all that apply)"))

			st.write("Smoking Status:", "Yes" if smoking == "Current" else "No")
			st.write("Hypertension:", "Yes" if "Cardiovascular" in str(med_history) else "No")
			st.write("Peripheral Vascular Disease:", "Yes" if "PVD" in str(med_history) else "No")

		with col3:
			st.subheader("Diabetes Status")
			st.write(f"Status: {get_metric_value(patient_data.get('Diabetes?'))}")
			st.write(f"HbA1c: {get_metric_value(patient_data.get('Hemoglobin A1c (%)'))}%")
			st.write(f"A1c available: {get_metric_value(patient_data.get('A1c  available within the last 3 months?'))}")


		st.title("Wound Details (present at 1st visit)")
		patient_data = self.data_processor.get_patient_visits(record_id=patient_id)
		wound_info = patient_data['visits'][0]['wound_info']

		# Create columns for wound details
		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Basic Information")
			st.markdown(f"**Location:** {get_metric_value(wound_info.get('location'))}")
			st.markdown(f"**Type:** {get_metric_value(wound_info.get('type'))}")
			st.markdown(f"**Current Care:** {get_metric_value(wound_info.get('current_care'))}")
			st.markdown(f"**Clinical Events:** {get_metric_value(wound_info.get('clinical_events'))}")

		with col2:
			st.subheader("Wound Characteristics")

			# Infection information in an info box
			with st.container():
				infection = wound_info.get('infection')
				if 'Status' in infection and not (infection['Status'] is None or str(infection['Status']).lower().strip() in ['', 'nan', 'none']):
					st.markdown("**Infection**")
					st.info(
						f"Status: {get_metric_value(infection.get('status'))}\n\n"
						"WiFi Classification: {get_metric_value(infection.get('wifi_classification'))}"
					)

			# Granulation information in a success box
			with st.container():
				granulation = wound_info.get('granulation')
				if not (granulation is None or str(granulation).lower().strip() in ['', 'nan', 'none']):
					st.markdown("**Granulation**")
					st.success(
						f"Coverage: {get_metric_value(granulation.get('coverage'))}\n\n"
						f"Quality: {get_metric_value(granulation.get('quality'))}"
					)

			# necrosis information in a warning box
			with st.container():
				necrosis = wound_info.get('necrosis')
				if not (necrosis is None or str(necrosis).lower().strip() in ['', 'nan', 'none']):
					st.markdown("**Necrosis**")
					st.warning(f"**Necrosis Present:** {necrosis}")

		# Create a third section for undermining details
		st.subheader("Undermining Details")
		undermining = wound_info.get('undermining', {})
		cols = st.columns(3)

		with cols[0]:
			st.metric("Present", get_metric_value(undermining.get('present')))
		with cols[1]:
			st.metric("Location", get_metric_value(undermining.get('location')))
		with cols[2]:
			st.metric("Tunneling", get_metric_value(undermining.get('tunneling')))

	def _render_feature_analysis(self, df: pd.DataFrame, primary_feature: str,
                               default_correlated_features: list = None,
                               feature_name_for_ui: str = None) -> None:
		"""
		Renders comprehensive feature analysis including clustering and correlation analysis.

		This method can be reused across different tabs (impedance, temperature, etc.)
		for analyzing any feature and its correlations with other features.

		Parameters:
		----------
		df : pd.DataFrame
			The input dataframe containing data to be analyzed.
		primary_feature : str
			The name of the primary feature being analyzed (e.g., "Skin Impedance (kOhms) - Z")
		default_correlated_features : list, optional
			Default selection of features to correlate with the primary feature
		feature_name_for_ui : str, optional
			Display name for the feature in UI elements (defaults to primary_feature if None)

		Returns:
		-------
		None
			This method directly renders components to the Streamlit dashboard.
		"""
		# Use the provided name or default to the column name
		feature_ui_name = feature_name_for_ui or primary_feature

		# Create a key prefix based on feature name to ensure unique widget IDs
		key_prefix = feature_ui_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')

		# Create a copy of the dataframe for analysis
		analysis_df = df.copy()

		# Initialize working_df to analysis_df by default (will be used if no clustering is done)
		working_df = analysis_df

		# Get standard feature list for analysis
		standard_features = [
			"Skin Impedance (kOhms) - Z",
			"Calculated Wound Area",
			"Center of Wound Temperature (Fahrenheit)",
			"Oxygenation (%)",
			"Hemoglobin Level",
			"Calculated Age at Enrollment",
			"BMI",
			"Days_Since_First_Visit",
			"Healing Rate (%)"
		]

		# Filter to only include features that exist in the dataframe
		available_features = [f for f in standard_features if f in analysis_df.columns]

		# Set default correlated features if none provided
		if default_correlated_features is None:
			if primary_feature in available_features:
				available_features.remove(primary_feature)
			default_correlated_features = available_features[:2]  # Take first two by default
			if primary_feature not in default_correlated_features:
				default_correlated_features = [primary_feature] + default_correlated_features

		# Create an expander for clustering options
		with st.expander(f"{feature_ui_name} Data Clustering", expanded=True):
			st.markdown("### Cluster Analysis Settings")

			# Create columns for clustering controls
			col1, col2, col3 = st.columns([1, 1, 1])

			with col1:
				# Number of clusters selection with unique key
				n_clusters = st.slider(
					"Number of Clusters",
					min_value=2,
					max_value=10,
					value=3,
					key=f"{key_prefix}_n_clusters",
					help="Select the number of clusters to divide patient data into"
				)

			with col2:
				# Features for clustering selection with unique key
				cluster_features = st.multiselect(
					"Features for Clustering",
					options=available_features,
					default=default_correlated_features,
					key=f"{key_prefix}_cluster_features",
					help=f"Select features to be used for clustering patients"
				)

			with col3:
				# Method selection with unique key
				clustering_method = st.selectbox(
					"Clustering Method",
					options=["K-Means", "Hierarchical", "DBSCAN"],
					index=0,
					key=f"{key_prefix}_clustering_method",
					help="Select the clustering algorithm to use"
				)

				# Add button to run clustering with unique key
				run_clustering = st.button("Run Clustering", key=f"{key_prefix}_run_clustering")

		# Session state for clusters - use feature-specific keys
		cluster_key = f"{key_prefix}_clusters"
		cluster_df_key = f"{key_prefix}_cluster_df"
		selected_cluster_key = f"{key_prefix}_selected_cluster"
		feature_importance_key = f"{key_prefix}_feature_importance"

		if cluster_key not in st.session_state:
			st.session_state[cluster_key] = None
			st.session_state[cluster_df_key] = None
			st.session_state[selected_cluster_key] = None
			st.session_state[feature_importance_key] = None

		# Run clustering if requested
		if run_clustering and len(cluster_features) > 0:
			try:
				# Create a feature dataframe for clustering
				clustering_df = analysis_df[cluster_features].copy()

				# Drop rows with NaN values
				clustering_df = clustering_df.dropna()

				if clustering_df.empty:
					st.error("No valid data for clustering after removing missing values. Please select different features.")
					return

				# Standardize the data
				from sklearn.preprocessing import StandardScaler
				scaler = StandardScaler()
				scaled_data = scaler.fit_transform(clustering_df)

				# Perform clustering based on selected method
				clusters = None
				if clustering_method == "K-Means":
					from sklearn.cluster import KMeans
					model = KMeans(n_clusters=n_clusters, random_state=42)
					clusters = model.fit_predict(scaled_data)
				elif clustering_method == "Hierarchical":
					from sklearn.cluster import AgglomerativeClustering
					model = AgglomerativeClustering(n_clusters=n_clusters)
					clusters = model.fit_predict(scaled_data)
				elif clustering_method == "DBSCAN":
					from sklearn.cluster import DBSCAN
					model = DBSCAN(eps=0.5, min_samples=5)
					clusters = model.fit_predict(scaled_data)

				# Add cluster assignments to the dataframe
				clustering_df["Cluster"] = clusters

				# Store in session state
				st.session_state[cluster_key] = clusters
				st.session_state[cluster_df_key] = clustering_df

				# Feature importance (for K-Means only)
				if clustering_method == "K-Means":
					# Calculate distances to cluster centers
					centers = model.cluster_centers_
					feature_importance = {}

					for i, feature in enumerate(cluster_features):
						# Calculate variance of the feature across cluster centers
						importance = np.var([center[i] for center in centers])
						feature_importance[feature] = importance

					# Normalize importance scores
					total = sum(feature_importance.values())
					if total > 0:  # Avoid division by zero
						for feature in feature_importance:
							feature_importance[feature] /= total

					st.session_state[feature_importance_key] = feature_importance

				# Display success message
				st.success(f"Clustering complete with {len(set(clusters))} clusters.")

			except Exception as e:
				st.error(f"Clustering failed: {str(e)}")
				return

		# Display cluster analysis if available
		if st.session_state[cluster_key] is not None and st.session_state[cluster_df_key] is not None:
			st.subheader("Cluster Analysis Results")

			# Create columns for selection and visualization
			col1, col2 = st.columns([1, 3])

			with col1:
				# Dropdown to select a specific cluster for analysis with unique key
				unique_clusters = sorted(set(st.session_state[cluster_df_key]["Cluster"]))
				selected_cluster = st.selectbox(
					"Select Cluster to Analyze",
					options=["All Clusters"] + [f"Cluster {i}" for i in unique_clusters],
					key=f"{key_prefix}_select_cluster"
				)

				# Store the selected cluster
				st.session_state[selected_cluster_key] = selected_cluster

				# Display feature importance if available
				if st.session_state[feature_importance_key] is not None:
					st.subheader("Feature Importance")

					# Convert to DataFrame for display
					importance_df = pd.DataFrame({
						"Feature": list(st.session_state[feature_importance_key].keys()),
						"Importance": list(st.session_state[feature_importance_key].values())
					})

					# Sort by importance
					importance_df = importance_df.sort_values("Importance", ascending=False)

					# Display as a bar chart
					fig = px.bar(
						importance_df,
						x="Importance",
						y="Feature",
						orientation="h",
						title="Feature Importance in Clustering"
					)

					st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_importance_chart")

			with col2:
				# Visualization of clusters
				if len(cluster_features) >= 2:
					# We can create a 2D plot with unique keys for the selectboxes
					feature1 = st.selectbox(
						"X-axis feature",
						options=cluster_features,
						index=0,
						key=f"{key_prefix}_feature1"
					)
					feature2 = st.selectbox(
						"Y-axis feature",
						options=cluster_features,
						index=min(1, len(cluster_features)-1),
						key=f"{key_prefix}_feature2"
					)

					# Create scatter plot
					fig = px.scatter(
						st.session_state[cluster_df_key],
						x=feature1,
						y=feature2,
						color="Cluster",
						title=f"Cluster Visualization: {feature1} vs {feature2}"
					)

					st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_cluster_scatter")

			# Display cluster statistics
			st.subheader("Cluster Statistics")

			if selected_cluster != "All Clusters":
				# Extract the cluster number
				cluster_num = int(selected_cluster.split(" ")[1])

				# Filter data for the selected cluster
				cluster_data = st.session_state[cluster_df_key][st.session_state[cluster_df_key]["Cluster"] == cluster_num]

				# Compare cluster means to overall population
				summary_stats = []

				for feature in cluster_features:
					try:
						cluster_mean = cluster_data[feature].mean()
						overall_mean = st.session_state[cluster_df_key][feature].mean()

						# Calculate percent difference
						if overall_mean != 0:
							diff_pct = ((cluster_mean - overall_mean) / overall_mean) * 100
						else:
							diff_pct = 0

						summary_stats.append({
							"Feature": feature,
							"Cluster Mean": f"{cluster_mean:.2f}",
							"Population Mean": f"{overall_mean:.2f}",
							"Difference": f"{diff_pct:+.1f}%",
							"Significant": abs(diff_pct) > 15
						})
					except:
						pass

				if summary_stats:
					summary_df = pd.DataFrame(summary_stats)

					# Create a copy of the styling DataFrame to avoid the KeyError
					styled_df = summary_df.copy()

					# Define the highlight function that uses a custom attribute instead of accessing the DataFrame
					def highlight_significant(row):
						is_significant = row['Significant'] if 'Significant' in row else False
						# Return styling for all columns except 'Significant'
						return ['background-color: yellow' if is_significant else '' for _ in range(len(row))]

					# Apply styling to all columns, then drop the 'Significant' column for display
					styled_df = styled_df.style.apply(highlight_significant, axis=1)
					styled_df.hide(axis="columns", names=["Significant"])

					# Display the styled DataFrame
					st.table(styled_df)
					st.info("Highlighted rows indicate features where this cluster differs from the overall population by >15%")

				# If a specific cluster is selected, use that cluster's data for further analysis
				working_df = st.session_state[cluster_df_key][st.session_state[cluster_df_key]["Cluster"] == cluster_num]
			else:
				# If "All Clusters" is selected, use all data with cluster assignments
				if st.session_state[cluster_df_key] is not None:
					working_df = st.session_state[cluster_df_key]
				else:
					working_df = analysis_df
		# else is removed as working_df is already initialized at the top

		# Add outlier threshold control and calculate correlation
		filtered_df = self._display_feature_correlation_controls(working_df, primary_feature, key_prefix)

		# Create scatter plot if we have valid data
		if not filtered_df.empty:
			self._render_feature_scatter_plot(filtered_df, primary_feature, key_prefix)
		else:
			st.warning("No valid data available for the scatter plot.")

		return filtered_df

	def _display_feature_correlation_controls(self, df: pd.DataFrame, primary_feature: str, key_prefix: str) -> pd.DataFrame:
		"""
		Displays comprehensive statistical analysis between selected features.

		This method creates UI controls for configuring outlier thresholds and displays:
		1. Correlation matrix between all selected features
		2. Statistical significance (p-values)
		3. Effect sizes and confidence intervals
		4. Basic descriptive statistics for each feature

		Parameters
		----------
		df : pd.DataFrame
			The input dataframe containing data with all selected features
		primary_feature : str
			The name of the primary feature being analyzed (e.g., "Skin Impedance (kOhms) - Z")
		key_prefix : str
			Prefix for unique Streamlit widget keys

		Returns
		-------
		pd.DataFrame
			Processed dataframe with outliers removed
		"""
		col1, _, col3 = st.columns([2, 3, 3])

		with col1:
			outlier_threshold = st.number_input(
				"Outlier Threshold",
				min_value=0.0,
				max_value=0.9,
				value=0.0,
				step=0.05,
				key=f"{key_prefix}_outlier_threshold",
				help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		# Get the standard features for analysis
		features_to_analyze = [
			"Skin Impedance (kOhms) - Z",
			"Calculated Wound Area",
			"Center of Wound Temperature (Fahrenheit)",
			"Oxygenation (%)",
			"Hemoglobin Level",
			"Healing Rate (%)"
		]

		# Filter features that exist in the dataframe
		features_to_analyze = [f for f in features_to_analyze if f in df.columns]

		# Ensure primary feature is included if it exists in the dataframe
		if primary_feature in df.columns and primary_feature not in features_to_analyze:
			features_to_analyze.insert(0, primary_feature)

		# Create a copy of the dataframe with only the features we want to analyze
		analysis_df = df[features_to_analyze].copy()

		# Remove outliers if threshold is set
		if outlier_threshold > 0:
			for col in analysis_df.columns:
				q_low = analysis_df[col].quantile(outlier_threshold)
				q_high = analysis_df[col].quantile(1 - outlier_threshold)
				analysis_df = analysis_df[
					(analysis_df[col] >= q_low) &
					(analysis_df[col] <= q_high)
				]

		# Calculate correlation matrix
		corr_matrix = analysis_df.corr()

		# Calculate p-values for correlations
		def calculate_pvalue(x, y):
			from scipy import stats
			mask = ~(np.isnan(x) | np.isnan(y))
			if np.sum(mask) < 2:
				return np.nan
			return stats.pearsonr(x[mask], y[mask])[1]

		p_values = pd.DataFrame(
			[[calculate_pvalue(analysis_df[col1], analysis_df[col2])
			  for col2 in analysis_df.columns]
			 for col1 in analysis_df.columns],
			columns=analysis_df.columns,
			index=analysis_df.columns
		)

		# Display correlation heatmap
		st.subheader("Feature Correlation Analysis")

		# Create correlation heatmap
		fig = px.imshow(
			corr_matrix,
			labels=dict(color="Correlation"),
			x=corr_matrix.columns,
			y=corr_matrix.columns,
			color_continuous_scale="RdBu",
			aspect="auto"
		)
		fig.update_layout(
			title="Correlation Matrix Heatmap",
			width=800,
			height=600
		)
		st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_corr_heatmap")

		# Display detailed statistics
		st.subheader("Statistical Summary")

		# Create tabs for different statistical views with unique keys
		tab1, tab2, tab3 = st.tabs(["Correlation Details", "Descriptive Stats", "Effect Sizes"])

		with tab1:
			# Display significant correlations
			st.markdown("#### Significant Correlations (p < 0.05)")
			significant_corrs = []
			for i in range(len(features_to_analyze)):
				for j in range(i+1, len(features_to_analyze)):
					if p_values.iloc[i,j] < 0.05:
						significant_corrs.append({
							"Feature 1": features_to_analyze[i],
							"Feature 2": features_to_analyze[j],
							"Correlation": f"{corr_matrix.iloc[i,j]:.3f}",
							"p-value": f"{p_values.iloc[i,j]:.3e}"
						})

			if significant_corrs:
				st.table(pd.DataFrame(significant_corrs))
			else:
				st.info("No significant correlations found.")

		with tab2:
			# Display descriptive statistics
			st.markdown("#### Descriptive Statistics")
			desc_stats = analysis_df.describe()
			desc_stats.loc["skew"] = analysis_df.skew()
			desc_stats.loc["kurtosis"] = analysis_df.kurtosis()
			st.dataframe(desc_stats, key=f"{key_prefix}_desc_stats")

		with tab3:
			# Calculate and display effect sizes
			st.markdown(f"#### Effect Sizes (Cohen's d) relative to {primary_feature}")
			from scipy import stats

			effect_sizes = []
			if primary_feature in features_to_analyze:
				for col in features_to_analyze:
					if col != primary_feature:
						# Calculate Cohen's d
						d = (analysis_df[col].mean() - analysis_df[primary_feature].mean()) / \
							np.sqrt((analysis_df[col].var() + analysis_df[primary_feature].var()) / 2)

						# Calculate 95% confidence interval for Cohen's d
						n = len(analysis_df)
						se = np.sqrt((4/n) * (1 + d**2/8))
						ci_lower = d - 1.96 * se
						ci_upper = d + 1.96 * se

						# Interpret effect size
						if abs(d) < 0.2:
							interpretation = "Negligible"
						elif abs(d) < 0.5:
							interpretation = "Small"
						elif abs(d) < 0.8:
							interpretation = "Medium"
						else:
							interpretation = "Large"

						effect_sizes.append({
							"Feature": col,
							"Cohen's d": f"{d:.3f}",
							"95% CI": f"({ci_lower:.3f}, {ci_upper:.3f})",
							"Interpretation": interpretation
						})

			if effect_sizes:
				st.table(pd.DataFrame(effect_sizes))
			else:
				st.info("No effect sizes available.")

		return analysis_df

	def _render_feature_scatter_plot(self, df: pd.DataFrame, primary_feature: str, key_prefix: str = None) -> None:
		"""
		Render scatter plot showing relationship between primary feature and healing rate.

		Args:
			df: DataFrame containing feature data
			primary_feature: The primary feature being analyzed
			key_prefix: Prefix for unique Streamlit widget keys
		"""
		# Create a copy to avoid modifying the original dataframe
		plot_df = df.copy()

		# If no key_prefix provided, create one
		if key_prefix is None:
			key_prefix = primary_feature.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')

		# Handle missing values in Calculated Wound Area
		if 'Calculated Wound Area' in plot_df.columns:
			# Fill NaN with the mean, or 1 if all values are NaN
			mean_area = plot_df['Calculated Wound Area'].mean()
			plot_df['Calculated Wound Area'] = plot_df['Calculated Wound Area'].fillna(mean_area if pd.notnull(mean_area) else 1)

		# Define hover data columns we want to show if available
		hover_columns = ['Record ID', 'Event Name', 'Wound Type']
		available_hover = [col for col in hover_columns if col in plot_df.columns]

		# Only proceed if both primary feature and healing rate are available
		if primary_feature in plot_df.columns and 'Healing Rate (%)' in plot_df.columns:
			fig = px.scatter(
				plot_df,
				x=primary_feature,
				y='Healing Rate (%)',
				color='Diabetes?' if 'Diabetes?' in plot_df.columns else None,
				size='Calculated Wound Area' if 'Calculated Wound Area' in plot_df.columns else None,
				size_max=30,
				hover_data=available_hover,
				title=f"{primary_feature} vs Healing Rate Correlation"
			)

			fig.update_layout(
				xaxis_title=primary_feature,
				yaxis_title="Healing Rate (% reduction per visit)"
			)

			st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_scatter_plot")
		else:
			st.warning(f"Cannot create scatter plot: missing {primary_feature} or Healing Rate data.")

def main():
	"""Main application entry point."""
	dashboard = Dashboard()
	dashboard.run()

if __name__ == "__main__":
	main()
