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

		Notes:
		-----
		The method performs the following operations:
		1. Creates a filtered dataset based on user-controlled outlier thresholds
		2. Allows clustering of data based on selected features
		3. Renders a scatter plot showing relationships between impedance variables
		4. Displays additional charts for population-level impedance analysis
		"""

		# Create a copy of the dataframe for analysis
		analysis_df = df.copy()

		# Create an expander for clustering options
		with st.expander("Patient Data Clustering", expanded=True):
			st.markdown("### Cluster Analysis Settings")

			# Create two columns for clustering controls
			col1, col2, col3, col4 = st.columns([1, 1, 1,2])

			with col1:
				# Number of clusters selection
				# n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3, help="Select the number of clusters to divide patient data into")
				n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3, help="Select the number of clusters to divide patient data into")

			with col2:
				# Features for clustering selection
				cluster_features = st.multiselect(
					"Features for Clustering",
					options=[
						"Skin Impedance (kOhms) - Z",
						"Calculated Wound Area",
						"Center of Wound Temperature (Fahrenheit)",
						"Oxygenation (%)",
						"Hemoglobin Level",
						"Calculated Age at Enrollment",
						"BMI",
						"Days_Since_First_Visit",
						"Healing Rate (%)"
					],
					default=["Skin Impedance (kOhms) - Z", "Calculated Wound Area", "Healing Rate (%)"],
					help="Select features to be used for clustering patients"
				)


			with col3:
				# Method selection
				clustering_method = st.selectbox(
					"Clustering Method",
					options=["K-Means", "Hierarchical", "DBSCAN"],
					index=0,
					help="Select the clustering algorithm to use"
				)

				# Add button to run clustering
				run_clustering = st.button("Run Clustering")

		# Session state for clusters
		if 'clusters' not in st.session_state:
			st.session_state.clusters = None
			st.session_state.cluster_df = None
			st.session_state.selected_cluster = None
			st.session_state.feature_importance = None

		# Run clustering if requested
		if run_clustering and len(cluster_features) > 0:
			try:
				# Create a feature dataframe for clustering
				clustering_df = analysis_df[cluster_features].copy()

				# Handle missing values
				clustering_df = clustering_df.fillna(clustering_df.mean())

				# Standardize the features
				from sklearn.preprocessing import StandardScaler
				from sklearn.cluster import KMeans, DBSCAN
				from sklearn.metrics import silhouette_score
				from scipy.cluster.hierarchy import linkage, fcluster
				import numpy as np

				# Drop rows with any remaining NaN values
				clustering_df = clustering_df.dropna()

				if len(clustering_df) > n_clusters:  # Ensure we have more data points than clusters
					# Get indices of valid rows to map back to original dataframe
					valid_indices = clustering_df.index

					# Standardize the data
					scaler = StandardScaler()
					scaled_features = scaler.fit_transform(clustering_df)

					# Perform clustering based on selected method
					if clustering_method == "K-Means":
						clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
						cluster_labels = clusterer.fit_predict(scaled_features)

						# Calculate feature importance for K-Means
						centers = clusterer.cluster_centers_
						feature_importance = {}
						for i, feature in enumerate(cluster_features):
							# Calculate the variance of this feature across cluster centers
							variance = np.var([center[i] for center in centers])
							feature_importance[feature] = variance

						# Normalize the feature importance
						max_importance = max(feature_importance.values())
						feature_importance = {k: v/max_importance for k, v in feature_importance.items()}

					elif clustering_method == "Hierarchical":
						# Perform hierarchical clustering
						Z = linkage(scaled_features, 'ward')
						cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # Adjust to 0-based

						# For hierarchical clustering, use silhouette coefficients for feature importance
						feature_importance = {}
						for i, feature in enumerate(cluster_features):
							# Create single-feature clustering and measure its quality
							single_feature = scaled_features[:, i:i+1]
							if len(np.unique(single_feature)) > 1:  # Only if feature has variation
								temp_clusters = fcluster(linkage(single_feature, 'ward'), n_clusters, criterion='maxclust')
								try:
									score = silhouette_score(single_feature, temp_clusters)
									feature_importance[feature] = max(0, score)  # Ensure non-negative
								except:
									feature_importance[feature] = 0.01  # Fallback value
							else:
								feature_importance[feature] = 0.01

						# Normalize the feature importance
						if max(feature_importance.values()) > 0:
							max_importance = max(feature_importance.values())
							feature_importance = {k: v/max_importance for k, v in feature_importance.items()}

					else:  # DBSCAN
						# Calculate epsilon based on data
						from sklearn.neighbors import NearestNeighbors
						neigh = NearestNeighbors(n_neighbors=3)
						neigh.fit(scaled_features)
						distances, _ = neigh.kneighbors(scaled_features)
						distances = np.sort(distances[:, 2], axis=0)  # Distance to 3rd nearest neighbor
						epsilon = np.percentile(distances, 90)  # Use 90th percentile as epsilon

						clusterer = DBSCAN(eps=epsilon, min_samples=max(3, len(scaled_features)//30))
						cluster_labels = clusterer.fit_predict(scaled_features)

						# For DBSCAN, count points in each cluster as a measure of feature importance
						from collections import Counter
						counts = Counter(cluster_labels)

						# Adjust n_clusters to actual number found by DBSCAN
						n_clusters = len([k for k in counts.keys() if k >= 0])  # Exclude noise points (label -1)

						# Calculate feature importance for DBSCAN using variance within clusters
						feature_importance = {}
						for i, feature in enumerate(cluster_features):
							variances = []
							for label in set(cluster_labels):
								if label >= 0:  # Exclude noise points
									cluster_data = scaled_features[cluster_labels == label, i]
									if len(cluster_data) > 1:
										variances.append(np.var(cluster_data))
							if variances:
								feature_importance[feature] = 1.0 - min(1.0, np.mean(variances)/np.var(scaled_features[:, i]))
							else:
								feature_importance[feature] = 0.01

						# Normalize feature importance
						if max(feature_importance.values()) > 0:
							max_importance = max(feature_importance.values())
							feature_importance = {k: v/max_importance for k, v in feature_importance.items()}

					# Create a new column in the original dataframe with cluster labels
					cluster_mapping = pd.Series(cluster_labels, index=valid_indices)
					analysis_df.loc[valid_indices, 'Cluster'] = cluster_mapping

					# Handle any NaN in cluster column (rows that were dropped during clustering)
					analysis_df['Cluster'] = analysis_df['Cluster'].fillna(-1).astype(int)

					# Store clustering results in session state
					st.session_state.clusters = sorted(analysis_df['Cluster'].unique())
					st.session_state.cluster_df = analysis_df
					st.session_state.feature_importance = feature_importance
					st.session_state.selected_cluster = None  # Reset selected cluster

					# Display success message
					st.success(f"Successfully clustered data into {n_clusters} clusters using {clustering_method}!")

					# Display cluster distribution
					cluster_counts = analysis_df['Cluster'].value_counts().sort_index()

					# Filter out noise points (label -1) for visualization
					if -1 in cluster_counts:
						noise_count = cluster_counts[-1]
						cluster_counts = cluster_counts[cluster_counts.index >= 0]
						st.info(f"Note: {noise_count} points were classified as noise (only applies to DBSCAN)")

					# Create a bar chart for cluster sizes
					fig = px.bar(
						x=cluster_counts.index,
						y=cluster_counts.values,
						labels={'x': 'Cluster', 'y': 'Number of Patients/Visits'},
						title=f"Cluster Distribution",
						color=cluster_counts.index,
						text=cluster_counts.values
					)

					fig.update_traces(textposition='outside')
					fig.update_layout(showlegend=False)
					st.plotly_chart(fig, use_container_width=True)

					# Create a spider/radar chart showing feature importance for clustering
					if feature_importance:
						# Create a radar chart for feature importance
						categories = list(feature_importance.keys())
						values = list(feature_importance.values())

						fig = go.Figure()
						fig.add_trace(go.Scatterpolar(
							r=values,
							theta=categories,
							fill='toself',
							name='Feature Importance'
						))

						fig.update_layout(
							title="Feature Importance in Clustering",
							polar=dict(
								radialaxis=dict(visible=True, range=[0, 1]),
							),
							showlegend=False
						)

						st.plotly_chart(fig, use_container_width=True)
				else:
					st.error("Not enough valid data points for clustering. Try selecting different features or reducing the number of clusters.")

			except Exception as e:
				st.error(f"Error during clustering: {str(e)}")
				import traceback
				st.error(traceback.format_exc())

		# Check if clustering has been performed
		if st.session_state.clusters is not None and st.session_state.cluster_df is not None:
			# Create selection for which cluster to analyze
			st.markdown("### Cluster Selection")

			cluster_options = [f"All Data"]
			for cluster_id in sorted([c for c in st.session_state.clusters if c >= 0]):
				cluster_count = len(st.session_state.cluster_df[st.session_state.cluster_df['Cluster'] == cluster_id])
				cluster_options.append(f"Cluster {cluster_id} (n={cluster_count})")

			selected_option = st.selectbox(
				"Select cluster to analyze:",
				options=cluster_options,
				index=0
			)

			# Update the selected cluster in session state
			if selected_option == "All Data":
				st.session_state.selected_cluster = None
				working_df = analysis_df
			else:
				cluster_id = int(selected_option.split(" ")[1].split("(")[0])
				st.session_state.selected_cluster = cluster_id
				working_df = st.session_state.cluster_df[st.session_state.cluster_df['Cluster'] == cluster_id].copy()

				# Display cluster characteristics
				st.markdown(f"### Characteristics of Cluster {cluster_id}")

				# Create summary statistics for this cluster vs. overall population
				summary_stats = []

				for feature in cluster_features:
					if feature in working_df.columns:
						try:
							cluster_mean = working_df[feature].mean()
							overall_mean = analysis_df[feature].mean()
							diff_pct = ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0

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
		else:
			working_df = analysis_df

		# Add outlier threshold control and calculate correlation
		filtered_df = self._display_impedance_correlation_controls(working_df)

		# Create scatter plot if we have valid data
		if not filtered_df.empty:
			self._render_impedance_scatter_plot(filtered_df)
		else:
			st.warning("No valid data available for the scatter plot.")

		# Create additional visualizations in a two-column layout
		self._render_population_impedance_charts(working_df)

	def _display_impedance_correlation_controls(self, df: pd.DataFrame) -> pd.DataFrame:
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
			The input dataframe containing wound data with all selected features

		Returns
		-------
		pd.DataFrame
			Processed dataframe with outliers removed
		"""
		col1, _, col3 = st.columns([2, 3, 3])

		with col1:
			outlier_threshold = st.number_input(
				"Impedance Outlier Threshold (Quantile)",
				min_value=0.0,
				max_value=0.9,
				value=0.0,
				step=0.05,
				help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		# Get the selected features for analysis
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
		st.plotly_chart(fig, use_container_width=True)

		# Display detailed statistics
		st.subheader("Statistical Summary")

		# Create tabs for different statistical views
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
			st.dataframe(desc_stats)

		with tab3:
			# Calculate and display effect sizes
			st.markdown("#### Effect Sizes (Cohen's d) relative to Impedance")
			from scipy import stats

			effect_sizes = []
			impedance_col = "Skin Impedance (kOhms) - Z"

			if impedance_col in features_to_analyze:
				for col in features_to_analyze:
					if col != impedance_col:
						# Calculate Cohen's d
						d = (analysis_df[col].mean() - analysis_df[impedance_col].mean()) / \
							np.sqrt((analysis_df[col].var() + analysis_df[impedance_col].var()) / 2)

						effect_sizes.append({
							"Feature": col,
							"Cohen's d": f"{d:.3f}",
							"Effect Size": "Large" if abs(d) > 0.8 else "Medium" if abs(d) > 0.5 else "Small",
							"95% CI": f"[{d-1.96*np.sqrt(4/len(analysis_df)):.3f}, {d+1.96*np.sqrt(4/len(analysis_df)):.3f}]"
						})

				if effect_sizes:
					st.table(pd.DataFrame(effect_sizes))
				else:
					st.info("No effect sizes could be calculated.")
			else:
				st.info("Impedance measurements not available for effect size calculation.")

		return analysis_df

	def _render_impedance_scatter_plot(self, df: pd.DataFrame) -> None:
		"""
		Render scatter plot showing relationship between impedance and healing rate.

		Args:
			df: DataFrame containing impedance and healing rate data
		"""
		# Create a copy to avoid modifying the original dataframe
		plot_df = df.copy()

		# Handle missing values in Calculated Wound Area
		if 'Calculated Wound Area' in plot_df.columns:
			# Fill NaN with the mean, or 1 if all values are NaN
			mean_area = plot_df['Calculated Wound Area'].mean()
			plot_df['Calculated Wound Area'] = plot_df['Calculated Wound Area'].fillna(mean_area if pd.notnull(mean_area) else 1)

		# Define hover data columns we want to show if available
		hover_columns = ['Record ID', 'Event Name', 'Wound Type']
		available_hover = [col for col in hover_columns if col in plot_df.columns]

		fig = px.scatter(
			plot_df,
			x='Skin Impedance (kOhms) - Z',
			y='Healing Rate (%)',
			color='Diabetes?' if 'Diabetes?' in plot_df.columns else None,
			size='Calculated Wound Area' if 'Calculated Wound Area' in plot_df.columns else None,
			size_max=30,
			hover_data=available_hover,
			title="Impedance vs Healing Rate Correlation"
		)

		fig.update_layout(
			xaxis_title="Impedance Z (kOhms)",
			yaxis_title="Healing Rate (% reduction per visit)"
		)

		st.plotly_chart(fig, use_container_width=True)

	def _render_population_impedance_charts(self, df: pd.DataFrame) -> None:
		"""
		Renders two charts showing population-level impedance statistics:
		1. A line chart showing average impedance components (Z, Z', Z'') over time by visit number
		2. A bar chart showing average impedance values by wound type

		This method uses the impedance_analyzer to calculate the relevant statistics
		from the provided dataframe and then creates visualizations using Plotly Express.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing impedance measurements and visit information
			for multiple patients

		Returns
		-------
		None
			The method renders charts directly to the Streamlit UI
		"""

		# Get prepared statistics
		avg_impedance, avg_by_type = self.impedance_analyzer.prepare_population_stats(df)

		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Impedance Components Over Time")
			fig1 = px.line(
				avg_impedance,
				x='Visit Number',
				y=["Skin Impedance (kOhms) - Z", "Skin Impedance (kOhms) - Z'", "Skin Impedance (kOhms) - Z''"],
				title="Average Impedance Components by Visit",
				markers=True
			)
			fig1.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)")
			st.plotly_chart(fig1, use_container_width=True)

		with col2:
			st.subheader("Impedance by Wound Type")
			fig2 = px.bar(
				avg_by_type,
				x='Wound Type',
				y="Skin Impedance (kOhms) - Z",
				title="Average Impedance by Wound Type",
				color='Wound Type'
			)
			fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Average Impedance Z (kOhms)")
			st.plotly_chart(fig2, use_container_width=True)

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
			Temperature tab display component for wound analysis application.

			This method creates and displays the temperature tab content in a Streamlit app, showing
			temperature gradient analysis and visualization based on user selection. It handles both
			aggregate data for all patients and detailed analysis for individual patients.

			Parameters
			----------
			df : pd.DataFrame
				The dataframe containing wound data for all patients.
			selected_patient : str
				The patient identifier to filter data. "All Patients" for aggregate view.

			Returns:
			-------
			None
				The method renders Streamlit UI components directly.

			Notes:
			-----
			For "All Patients" view, displays:
			- Temperature gradient analysis across wound types
			- Statistical correlation between temperature and healing rate
			- Scatter plot of temperature gradient vs healing rate

			For individual patient view, provides:
			- Temperature trends over time
			- Visit-by-visit detailed temperature analysis
			- Clinical guidelines for temperature assessment
			- Statistical summary with visual indicators
		"""

		if selected_patient == "All Patients":
			st.header("Temperature Gradient Analysis")

			# Create a temperature gradient dataframe
			temp_df = df.copy()

			temp_df['Visit date'] = pd.to_datetime(temp_df['Visit date']).dt.strftime('%m-%d-%Y')

			# Remove skipped visits
			# temp_df = temp_df[temp_df['Skipped Visit?'] != 'Yes']

			# Define temperature column names
			temp_cols = [	'Center of Wound Temperature (Fahrenheit)',
							'Edge of Wound Temperature (Fahrenheit)',
							'Peri-wound Temperature (Fahrenheit)']

			# Drop rows with missing temperature data
			temp_df = temp_df.dropna(subset=temp_cols)

			# Calculate temperature gradients if all required columns exist
			if all(col in temp_df.columns for col in temp_cols):
				temp_df['Center-Edge Gradient'] = temp_df[temp_cols[0]] - temp_df[temp_cols[1]]
				temp_df['Edge-Peri Gradient']   = temp_df[temp_cols[1]] - temp_df[temp_cols[2]]
				temp_df['Total Gradient']       = temp_df[temp_cols[0]] - temp_df[temp_cols[2]]


			# Add outlier threshold control
			col1, _, col3 = st.columns([2,3,3])

			with col1:
				outlier_threshold = st.number_input(
					"Temperature Outlier Threshold",
					min_value=0.0,
					max_value=0.9,
					value=0.2,
					step=0.05,
					help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
				)

			# Calculate correlation with outlier handling
			stats_analyzer = CorrelationAnalysis(data=temp_df, x_col='Center of Wound Temperature (Fahrenheit)', y_col='Healing Rate (%)', outlier_threshold=outlier_threshold)
			temp_df, r, p = stats_analyzer.calculate_correlation()

			# Display correlation statistics
			with col3:
				st.info(stats_analyzer.get_correlation_text())

			# Create boxplot of temperature gradients by wound type
			gradient_cols = ['Center-Edge Gradient', 'Edge-Peri Gradient', 'Total Gradient']
			fig = px.box(
				temp_df,
				x='Wound Type',
				y=gradient_cols,
				title="Temperature Gradients by Wound Type",
				points="all"
			)

			fig.update_layout(
				xaxis_title="Wound Type",
				yaxis_title="Temperature Gradient (°F)",
				boxmode='group'
			)
			st.plotly_chart(fig, use_container_width=True)

			# Scatter plot of total gradient vs healing rate
			# temp_df = temp_df.copy() # [df['Healing Rate (%)'] < 0].copy()
			temp_df['Healing Rate (%)'] = temp_df['Healing Rate (%)'].clip(-100, 100)
			temp_df['Calculated Wound Area'] = temp_df['Calculated Wound Area'].fillna(temp_df['Calculated Wound Area'].mean())

			fig = px.scatter(
				temp_df,#[temp_df['Healing Rate (%)'] > 0],  # Exclude first visits
				x='Total Gradient',
				y='Healing Rate (%)',
				color='Wound Type',
				size='Calculated Wound Area',#'Hemoglobin Level', #
				size_max=30,
				hover_data=['Record ID', 'Event Name'],
				title="Temperature Gradient vs. Healing Rate"
			)
			fig.update_layout(xaxis_title="Temperature Gradient (Center to Peri-wound, °F)", yaxis_title="Healing Rate (% reduction per visit)")
			st.plotly_chart(fig, use_container_width=True)

		else:
			# For individual patient
			df_temp = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
			df_temp['Visit date'] = pd.to_datetime(df_temp['Visit date']).dt.strftime('%m-%d-%Y')
			st.header(f"Temperature Gradient Analysis for Patient {selected_patient.split(' ')[1]}")

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
			Renders the oxygenation analysis tab in the dashboard.

			This tab displays visualizations related to wound oxygenation data, showing either
			aggregate statistics for all patients or detailed analysis for a single selected patient.

			For all patients:
			- Scatter plot showing relationship between oxygenation and healing rate
			- Box plot comparing oxygenation levels across different wound types
			- Statistical correlation between oxygenation and healing rate

			For individual patients:
			- Bar chart showing oxygenation levels across visits
			- Line chart tracking oxygenation over time

			Parameters
			-----------
			df : pd.DataFrame
				The dataframe containing all wound care data
			selected_patient : str
				The selected patient (either "All Patients" or a specific patient identifier)

			Returns:
			--------
			None
				This method renders Streamlit components directly to the app
		"""
		st.header("Oxygenation Analysis")

		if selected_patient == "All Patients":

			valid_df = df.copy()
			# valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

			valid_df['Hemoglobin Level'] = pd.to_numeric(valid_df['Hemoglobin Level'], errors='coerce')#.fillna(valid_df['Hemoglobin Level'].astype(float).mean())

			valid_df['Oxygenation (%)'] = pd.to_numeric(valid_df['Oxygenation (%)'], errors='coerce')
			valid_df['Healing Rate (%)'] = pd.to_numeric(valid_df['Healing Rate (%)'], errors='coerce')

			valid_df = valid_df.dropna(subset=['Oxygenation (%)', 'Healing Rate (%)', 'Hemoglobin Level'])

			# Add outlier threshold control
			col1, _, col3 = st.columns([2, 3, 3])

			with col1:
				outlier_threshold = st.number_input(
					"Oxygenation Outlier Threshold",
					min_value=0.0,
					max_value=0.9,
					value=0.2,
					step=0.05,
					help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
				)

			# Calculate correlation with outlier handling
			stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Oxygenation (%)', y_col='Healing Rate (%)', outlier_threshold=outlier_threshold)
			valid_df, r, p = stats_analyzer.calculate_correlation()

			# Display correlation statistics
			with col3:
				st.info(stats_analyzer.get_correlation_text())

			# Add consistent diabetes status for each patient
			# first_diabetes_status = valid_df.groupby('Record ID')['Diabetes?'].first()
			# valid_df['Diabetes?'] = valid_df['Record ID'].map(first_diabetes_status)
			valid_df['Healing Rate (%)']      = valid_df['Healing Rate (%)'].clip(-100, 100)
			valid_df['Calculated Wound Area'] = valid_df['Calculated Wound Area'].fillna(valid_df['Calculated Wound Area'].mean())

			if not valid_df.empty:
				fig1 = px.scatter(
					valid_df,
					x='Oxygenation (%)',
					y='Healing Rate (%)',
					# color='Diabetes?',
					size='Calculated Wound Area',#'Hemoglobin Level', #
					size_max=30,
					hover_data=['Record ID', 'Event Name', 'Wound Type'],
					title="Relationship Between Oxygenation and Healing Rate (size=Hemoglobin Level)"
				)
				fig1.update_layout(xaxis_title="Oxygenation (%)", yaxis_title="Healing Rate (% reduction per visit)")
				st.plotly_chart(fig1, use_container_width=True)
			else:
				st.warning("Insufficient data for oxygenation analysis.")

			# Create boxplot for oxygenation levels
			valid_box_df = df.dropna(subset=['Oxygenation (%)', 'Wound Type'])
			if not valid_box_df.empty:
				fig2 = px.box(
					valid_box_df,
					x='Wound Type',
					y='Oxygenation (%)',
					title="Oxygenation Levels by Wound Type",
					color='Wound Type',
					points="all"
				)
				fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Oxygenation (%)")
				st.plotly_chart(fig2, use_container_width=True)
			else:
				st.warning("Insufficient data for wound type comparison.")

		else:
			patient_data = DataManager.get_patient_data(df, int(selected_patient.split(" ")[1])).sort_values('Visit Number')

			# Convert Visit date to string for display
			patient_data['Visit date'] = pd.to_datetime(patient_data['Visit date']).dt.strftime('%m-%d-%Y')
			visits = self.data_processor.get_patient_visits(record_id=int(selected_patient.split(" ")[1]))['visits']

			fig_bar, fig_line = Visualizer.create_oxygenation_chart(patient_data, visits)

			st.plotly_chart(fig_bar, use_container_width=True)
			st.plotly_chart(fig_line, use_container_width=True)

	def _exudate_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""
			Create and display the exudate analysis tab in the wound management dashboard.

			This tab provides detailed analysis of wound exudate characteristics including volume,
			viscosity, and type. For aggregate patient data, it shows statistical correlations and
			visualizations comparing exudate properties across different wound types. For individual
			patients, it displays a timeline of exudate changes and provides clinical interpretations
			for each visit.

			Parameters
			----------
			df : pd.DataFrame
				The dataframe containing wound assessment data for all patients
			selected_patient : str
				The currently selected patient ID or "All Patients"

			Returns:
			-------
			None
				The method updates the Streamlit UI directly

			Notes:
			-----
			For aggregate analysis, this method:
			- Calculates correlations between exudate characteristics and healing rates
			- Creates boxplots comparing exudate properties across wound types
			- Generates scatter plots to visualize relationships between variables
			- Shows distributions of exudate types by wound category

			For individual patient analysis, this method:
			- Displays a timeline chart of exudate changes
			- Provides clinical interpretations for each visit
			- Offers treatment recommendations based on exudate characteristics
		"""

		st.header("Exudate Analysis")

		if selected_patient == "All Patients":
			# Create a copy of the dataframe for exudate analysis
			exudate_df = df.copy()

			# Define exudate columns
			exudate_cols = ['Exudate Volume', 'Exudate Viscosity', 'Exudate Type']

			# Drop rows with missing exudate data
			exudate_df = exudate_df.dropna(subset=exudate_cols)

			# Create numerical mappings for volume and viscosity
			level_mapping = {
				'Low': 1,
				'Medium': 2,
				'High': 3
			}

			# Convert volume and viscosity to numeric values
			exudate_df['Volume_Numeric']    = exudate_df['Exudate Volume'].map(level_mapping)
			exudate_df['Viscosity_Numeric'] = exudate_df['Exudate Viscosity'].map(level_mapping)

			if not exudate_df.empty:
				# Create two columns for volume and viscosity analysis
				col1, col2 = st.columns(2)

				with col1:
					st.subheader("Volume Analysis")
					# Calculate correlation between volume and healing rate
					valid_df = exudate_df.dropna(subset=['Volume_Numeric', 'Healing Rate (%)'])

					if len(valid_df) > 1:
						stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Volume_Numeric', y_col='Healing Rate (%)', REMOVE_OUTLIERS=False)
						_, _, _ = stats_analyzer.calculate_correlation()
						st.info(stats_analyzer.get_correlation_text(text="Volume correlation vs Healing Rate"))

					# Boxplot of exudate volume by wound type
					fig_vol = px.box(
						exudate_df,
						x='Wound Type',
						y='Exudate Volume',
						title="Exudate Volume by Wound Type",
						points="all"
					)

					fig_vol.update_layout(
						xaxis_title="Wound Type",
						yaxis_title="Exudate Volume",
						boxmode='group'
					)
					st.plotly_chart(fig_vol, use_container_width=True)

				with col2:
					st.subheader("Viscosity Analysis")
					# Calculate correlation between viscosity and healing rate
					valid_df = exudate_df.dropna(subset=['Viscosity_Numeric', 'Healing Rate (%)'])

					if len(valid_df) > 1:
						stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Viscosity_Numeric', y_col='Healing Rate (%)', REMOVE_OUTLIERS=False)
						_, _, _ = stats_analyzer.calculate_correlation()
						st.info(stats_analyzer.get_correlation_text(text="Viscosity correlation vs Healing Rate"))


					# Boxplot of exudate viscosity by wound type
					fig_visc = px.box(
						exudate_df,
						x='Wound Type',
						y='Exudate Viscosity',
						title="Exudate Viscosity by Wound Type",
						points="all"
					)
					fig_visc.update_layout(
						xaxis_title="Wound Type",
						yaxis_title="Exudate Viscosity",
						boxmode='group'
					)
					st.plotly_chart(fig_visc, use_container_width=True)

				# Create scatter plot matrix for volume, viscosity, and healing rate
				st.subheader("Relationship Analysis")

				exudate_df['Healing Rate (%)']      = exudate_df['Healing Rate (%)'].clip(-100, 100)
				exudate_df['Calculated Wound Area'] = exudate_df['Calculated Wound Area'].fillna(exudate_df['Calculated Wound Area'].mean())

				fig_scatter = px.scatter(
					exudate_df,
					x='Volume_Numeric',
					y='Healing Rate (%)',
					color='Wound Type',
					size='Calculated Wound Area',
					hover_data=['Record ID', 'Event Name', 'Exudate Volume', 'Exudate Viscosity', 'Exudate Type'],
					title="Exudate Characteristics vs. Healing Rate"
				)
				fig_scatter.update_layout(
					xaxis_title="Exudate Volume (1=Low, 2=Medium, 3=High)",
					yaxis_title="Healing Rate (% reduction per visit)"
				)
				st.plotly_chart(fig_scatter, use_container_width=True)

				# Display distribution of exudate types
				st.subheader("Exudate Type Distribution")
				col1, col2 = st.columns(2)

				with col1:
					# Distribution by wound type
					type_by_wound = pd.crosstab(exudate_df['Wound Type'], exudate_df['Exudate Type'])
					fig_type = px.bar(
						type_by_wound.reset_index().melt(id_vars='Wound Type', var_name='Exudate Type', value_name='Percentage'),
						x='Wound Type',
						y='Percentage',
						color='Exudate Type',
					)

					st.plotly_chart(fig_type, use_container_width=True)

				with col2:
					# Overall distribution
					type_counts = exudate_df['Exudate Type'].value_counts()
					fig_pie = px.pie(
						values=type_counts.values,
						names=type_counts.index,
						title="Overall Distribution of Exudate Types"
					)
					st.plotly_chart(fig_pie, use_container_width=True)

			else:
				st.warning("No valid exudate data available for analysis.")
		else:
			visits = self.data_processor.get_patient_visits(record_id=int(selected_patient.split(" ")[1]))['visits']

			# Display the exudate chart
			fig = Visualizer.create_exudate_chart(visits)
			st.plotly_chart(fig, use_container_width=True)

			# Clinical interpretation section
			st.subheader("Clinical Interpretation of Exudate Characteristics")

			# Create tabs for each visit
			visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits])

			# Process each visit in its corresponding tab
			for tab, visit in zip(visit_tabs, visits):
				with tab:
					col1, col2 = st.columns(2)
					volume           = visit['wound_info']['exudate'].get('volume', 'N/A')
					viscosity        = visit['wound_info']['exudate'].get('viscosity', 'N/A')
					exudate_type_str = str(visit['wound_info']['exudate'].get('type', 'N/A'))
					exudate_analysis = DashboardSettings.get_exudate_analysis(volume=volume, viscosity=viscosity, exudate_types=exudate_type_str)

					with col1:
						st.markdown("### Volume Analysis")
						st.write(f"**Current Level:** {volume}")
						st.info(exudate_analysis['volume_analysis'])

					with col2:
						st.markdown("### Viscosity Analysis")
						st.write(f"**Current Level:** {viscosity}")
						if viscosity == 'High':
							st.warning(exudate_analysis['viscosity_analysis'])
						elif viscosity == 'Low':
							st.info(exudate_analysis['viscosity_analysis'])

					# Exudate Type Analysis
					st.markdown('----')
					col1, col2 = st.columns(2)

					with col1:
						st.markdown("### Type Analysis")
						if exudate_type_str != 'N/A':
							exudate_types = [t.strip() for t in exudate_type_str.split(',')]
							st.write(f"**Current Types:** {exudate_types}")
						else:
							exudate_types = ['N/A']
							st.write("**Current Type:** N/A")

						# Process each exudate type
						highest_severity = 'info'  # Track highest severity for overall implications
						for exudate_type in exudate_types:
							if exudate_type in DashboardSettings.EXUDATE_TYPE_INFO:
								info = DashboardSettings.EXUDATE_TYPE_INFO[exudate_type]
								message = f"""
								**Description:** {info['description']} \n
								**Clinical Indication:** {info['indication']}
								"""
								if info['severity'] == 'error':
									st.error(message)
									highest_severity = 'error'
								elif info['severity'] == 'warning' and highest_severity != 'error':
									st.warning(message)
									highest_severity = 'warning'
								else:
									st.info(message)

						# Overall Clinical Assessment based on multiple types
						if len(exudate_types) > 1 and 'N/A' not in exudate_types:
							st.markdown("#### Overall Clinical Assessment")
							if highest_severity == 'error':
								st.error("⚠️ Multiple exudate types present with signs of infection. Immediate clinical attention recommended.")
							elif highest_severity == 'warning':
								st.warning("⚠️ Mixed exudate characteristics suggest active wound processes. Close monitoring required.")
							else:
								st.info("Multiple exudate types present. Continue regular monitoring of wound progression.")

					with col2:
						st.markdown("### Treatment Implications")
						if exudate_analysis['treatment_implications']:
							st.write("**Recommended Actions:**")
							st.success("\n".join(exudate_analysis['treatment_implications']))

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

def main():
	"""Main application entry point."""
	dashboard = Dashboard()
	dashboard.run()

if __name__ == "__main__":
	main()
