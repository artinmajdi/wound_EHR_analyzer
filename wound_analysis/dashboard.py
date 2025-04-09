# Standard library imports
import os
import pathlib
from typing import Optional
import logging

# Third-party imports
import pandas as pd
import streamlit as st

from wound_analysis.dashboard_components import (
	ExudateTab,
	ImpedanceTab,
	LLMAnalysisTab,
	OverviewTab,
	OxygenationTab,
	RiskFactorsTab,
	TemperatureTab,
	DashboardSettings,
	Visualizer
)
from wound_analysis.utils import (
	CorrelationAnalysis,
	DataManager,
	ImpedanceAnalyzer,
	WoundAnalysisLLM,
	WoundDataProcessor,
	DColumns
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: add password for the deployed streamlit app

# Debug mode disabled
st.set_option('client.showErrorDetails', True)

def debug_log(message):
	"""Write debug messages to a file and display in Streamlit"""
	try:
		# Only attempt to write to file in environments where it might work
		if os.environ.get('WRITE_DEBUG_LOG', 'false').lower() == 'true':
			with open('/app/debug.log', 'a') as f:
				f.write(f"{message}\n")
	except Exception:
		pass
	# Always show in sidebar for debugging
	st.sidebar.text(message)


class Dashboard:
	"""
	Main dashboard class for the Wound Analysis application.

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
			test(): Placeholder for testing purposes
			setup(): Initializes the Streamlit page configuration and sidebar
			load_data(uploaded_file): Loads data from the uploaded CSV file
			run(): Main execution method that runs the dashboard application
			_create_dashboard_tabs(): Creates and manages the main dashboard tabs
			_create_left_sidebar(): Creates the sidebar with configuration options
	"""
	def __init__(self):
		"""Initialize the dashboard."""
		self.DashboardSettings  = DashboardSettings()
		self.data_manager       = DataManager()
		self.visualizer         = Visualizer()
		self.impedance_analyzer: Optional[ImpedanceAnalyzer] = None

		# LLM configuration placeholders
		self.llm_platform:              Optional[str] = None
		self.llm_model:                 Optional[str] = None
		self.csv_dataset_path:          Optional[pathlib.Path] = None
		self.wound_data_processor:      Optional[WoundDataProcessor] = None
		self.impedance_freq_sweep_path: Optional[pathlib.Path] = None
		self.CN: Optional[DColumns] = None

		self.filtered_df: Optional[pd.DataFrame] = None


	def setup(self) -> None:
		"""
		Set up the dashboard configuration.

		This method configures the Streamlit page settings and sidebar.
		"""
		st.set_page_config(
			page_title = self.DashboardSettings.PAGE_TITLE,
			page_icon  = self.DashboardSettings.PAGE_ICON,
			layout     = self.DashboardSettings.LAYOUT
		)
		DashboardSettings.initialize()
		self._create_left_sidebar()


	def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
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
		if df is not None:
			self.CN = DColumns(df=df)

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

		# Header
		st.title(self.DashboardSettings.PAGE_TITLE)

		# add two columns
		cols = st.columns((1,1,2))

		with cols[0]:
			# Patient selection
			st.markdown("""
			<style>
			.filter-container {
				border: 1px solid #e0e0e0;
				border-radius: 5px;
				padding: 10px;
				background-color: #f8f9fa;
				margin-bottom: 10px;
			}
			.filter-title {
				font-weight: bold;
				margin-bottom: 5px;
				color: #4b5563;
			}
			</style>
			""", unsafe_allow_html=True)
			# Create a container for the filter controls
			st.markdown('<div class="filter-container">', unsafe_allow_html=True)
			st.markdown('<div class="filter-title">Patient Filter</div>', unsafe_allow_html=True)

			patient_ids      = sorted(df[self.CN.RECORD_ID].unique())
			patient_options  = ["All Patients"] + [f"Patient {id:d}" for id in patient_ids]
			selected_patient = st.selectbox("Select Patients", patient_options, label_visibility="hidden")

		with cols[-1]:
			# Add CSS for the filter container
			st.markdown("""
			<style>
			.filter-container {
				border: 1px solid #e0e0e0;
				border-radius: 5px;
				padding: 10px;
				background-color: #f8f9fa;
				margin-bottom: 10px;
			}
			.filter-title {
				font-weight: bold;
				margin-bottom: 5px;
				color: #4b5563;
			}
			</style>
			""", unsafe_allow_html=True)

			# Create a container for the filter controls
			st.markdown('<div class="filter-container">', unsafe_allow_html=True)
			st.markdown('<div class="filter-title">Date Filter</div>', unsafe_allow_html=True)

			# Create two columns for the filter controls
			filter_cols = st.columns(2)

			with filter_cols[0]:
				filter_mode = st.radio("Mode", ["After", "Before"], key="filter_mode")

			with filter_cols[1]:
				min_date = df[self.CN.VISIT_DATE].min()
				max_date = df[self.CN.VISIT_DATE].max()
				filteration_date = pd.to_datetime(st.date_input("Date", value=min_date, min_value=min_date, max_value=max_date))

			# Close the container
			st.markdown('</div>', unsafe_allow_html=True)

		# Apply the date filter
		if filter_mode == "Before":
			self.filtered_df = df[pd.to_datetime(df[self.CN.VISIT_DATE]) <= filteration_date]
		else:
			self.filtered_df = df[pd.to_datetime(df[self.CN.VISIT_DATE]) >= filteration_date]

		# Initialize data processor with filtered data and reuse the impedance_analyzer
		if self.impedance_analyzer is not None:
			logger.debug("Initializing new wound data processor with impedance analyzer")
			self.wound_data_processor = WoundDataProcessor( df=self.filtered_df, impedance_freq_sweep_path=self.impedance_freq_sweep_path, impedance_analyzer=self.impedance_analyzer )
		else:
			logger.debug("Initializing new wound data processor without impedance analyzer")
			self.wound_data_processor = WoundDataProcessor( df=self.filtered_df, impedance_freq_sweep_path=self.impedance_freq_sweep_path )

		# Create tabs with filtered data
		self._create_dashboard_tabs(df=self.filtered_df, selected_patient=selected_patient)


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

		argsv = dict(selected_patient=selected_patient, wound_data_processor=self.wound_data_processor)
		with tabs[0]:
			OverviewTab(**argsv).render()
		with tabs[1]:
			ImpedanceTab(**argsv).render()
			# ImpedanceTabOriginal(**argsv).render()
		with tabs[2]:
			TemperatureTab(**argsv).render()
		with tabs[3]:
			OxygenationTab(**argsv).render()
		with tabs[4]:
			ExudateTab(**argsv).render()
		with tabs[5]:
			RiskFactorsTab(**argsv).render()
		with tabs[6]:
			LLMAnalysisTab(selected_patient=selected_patient, wound_data_processor=self.wound_data_processor, llm_platform=self.llm_platform, llm_model=self.llm_model).render()


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
		logger.debug("Getting user input for dataset path and impedance frequency sweep path")
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

						self.impedance_analyzer = ImpedanceAnalyzer(impedance_freq_sweep_path=self.impedance_freq_sweep_path)

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


def main():
	"""Main application entry point."""
	dashboard = Dashboard()
	dashboard.run()

if __name__ == "__main__":
	main()
