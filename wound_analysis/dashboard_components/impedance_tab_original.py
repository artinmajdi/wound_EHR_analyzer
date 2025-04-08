# Standard library imports
import traceback
from collections import Counter
from typing import List
from venv import logger

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Local application imports
from wound_analysis.utils.data_processor import ImpedanceAnalyzer, WoundDataProcessor, VisitsDataType, VisitsMetadataType
from wound_analysis.utils.column_schema import DColumns


class ImpedanceTab:
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

	def __init__(self, df: pd.DataFrame, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.wound_data_processor = wound_data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = df
		self.CN = DColumns(df=df)

	def render(self) -> None:

		st.header("Impedance Analysis")

		if self.patient_id == "All Patients":
			ImpedanceTab._render_population(df=self.df)
		else:
			visits_meta_data: VisitsMetadataType   = self.wound_data_processor.get_patient_visits(record_id=self.patient_id)
			visits          : List[VisitsDataType] = visits_meta_data['visits']

			ImpedanceTab._render_patient(visits=visits)

	@staticmethod
	def _render_population(df: pd.DataFrame) -> None:
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

		CN = DColumns(df=df)

		# Create a copy of the dataframe for analysis
		analysis_df = df.copy()

		# Create an expander for clustering options
		with st.expander("Patient Data Clustering", expanded=True):
			st.markdown("### Cluster Analysis Settings")

			# Create two columns for clustering controls
			col1, col2, col3 = st.columns([1, 2, 1])

			with col1:
				# Number of clusters selection
				# n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3, help="Select the number of clusters to divide patient data into")
				n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3, help="Select the number of clusters to divide patient data into")

			with col2:
				# Features for clustering selection
				cluster_features = st.multiselect(
					"Features for Clustering",
					options=[
						CN.HIGHEST_FREQ_ABSOLUTE,
						CN.WOUND_AREA,
						CN.CENTER_TEMP,
						CN.OXYGENATION,
						CN.HEMOGLOBIN,
						CN.AGE,
						CN.BMI,
						CN.DAYS_SINCE_FIRST_VISIT,
						CN.HEALING_RATE
					],
					default=[CN.HIGHEST_FREQ_ABSOLUTE, CN.WOUND_AREA, CN.HEALING_RATE],
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
								except Exception as e:
									st.error(f"Error calculating silhouette score: {str(e)}")
									feature_importance[feature] = 0.01  # Fallback value
							else:
								feature_importance[feature] = 0.01

						# Normalize the feature importance
						if max(feature_importance.values()) > 0:
							max_importance = max(feature_importance.values())
							feature_importance = {k: v/max_importance for k, v in feature_importance.items()}

					else:  # DBSCAN
						# Calculate epsilon based on data
						neigh = NearestNeighbors(n_neighbors=3)
						neigh.fit(scaled_features)
						distances, _ = neigh.kneighbors(scaled_features)
						distances = np.sort(distances[:, 2], axis=0)  # Distance to 3rd nearest neighbor
						epsilon = np.percentile(distances, 90)  # Use 90th percentile as epsilon

						clusterer = DBSCAN(eps=epsilon, min_samples=max(3, len(scaled_features)//30))
						cluster_labels = clusterer.fit_predict(scaled_features)

						# For DBSCAN, count points in each cluster as a measure of feature importance
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
						title="Cluster Distribution",
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
				st.error(traceback.format_exc())

		# Check if clustering has been performed
		if st.session_state.clusters is not None and st.session_state.cluster_df is not None:
			# Create selection for which cluster to analyze
			st.markdown("### Cluster Selection")

			cluster_options = ["All Data"]
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
								"Feature"        : feature,
								"Cluster Mean"   : f"{cluster_mean:.2f}",
								"Population Mean": f"{overall_mean:.2f}",
								"Difference"     : f"{diff_pct:+.1f}%",
								"Significant"    : abs(diff_pct) > 15
							})
						except Exception as e:
							st.error(f"Error calculating summary statistics: {str(e)}")
							st.error(traceback.format_exc())

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
		filtered_df = ImpedanceTab._display_correlation_controls(working_df)

		# Create scatter plot if we have valid data
		if not filtered_df.empty:
			ImpedanceTab._scatter_plot(df=filtered_df)
		else:
			st.warning("No valid data available for the scatter plot.")

		# Create additional visualizations in a two-column layout
		ImpedanceTab._population_charts(df=working_df)

	@staticmethod
	def _display_correlation_controls(df_for_cluster: pd.DataFrame) -> pd.DataFrame:
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

		CN = DColumns(df=df_for_cluster)

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
			CN.HIGHEST_FREQ_ABSOLUTE,
			CN.WOUND_AREA,
			CN.CENTER_TEMP,
			CN.OXYGENATION,
			CN.HEMOGLOBIN,
			CN.HEALING_RATE
		]

		# Filter features that exist in the dataframe
		features_to_analyze = [f for f in features_to_analyze if f in df_for_cluster.columns]

		# Create a copy of the dataframe with only the features we want to analyze
		analysis_df = df_for_cluster[features_to_analyze].copy()

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

	@staticmethod
	def _scatter_plot(df: pd.DataFrame) -> None:
		"""
		Render scatter plot showing relationship between impedance and healing rate.

		Args:
			df: DataFrame containing impedance and healing rate data
		"""

		CN = DColumns(df=df)

		# Create a copy to avoid modifying the original dataframe
		plot_df = df.copy()

		# Handle missing values in Calculated Wound Area
		if CN.WOUND_AREA in plot_df.columns:
			# Fill NaN with the mean, or 1 if all values are NaN
			mean_area = plot_df[CN.WOUND_AREA].mean()
			plot_df[CN.WOUND_AREA] = plot_df[CN.WOUND_AREA].fillna(mean_area if pd.notnull(mean_area) else 1)

		# Define hover data columns we want to show if available
		hover_columns = [CN.RECORD_ID, CN.EVENT_NAME, CN.WOUND_TYPE]
		available_hover = [col for col in hover_columns if col in plot_df.columns]

		fig = px.scatter(
			plot_df,
			x=CN.HIGHEST_FREQ_ABSOLUTE,
			y=CN.HEALING_RATE,
			color=CN.DIABETES if CN.DIABETES in plot_df.columns else None,
			size=CN.WOUND_AREA if CN.WOUND_AREA in plot_df.columns else None,
			size_max=30,
			hover_data=available_hover,
			title="Impedance vs Healing Rate Correlation"
		)

		fig.update_layout(
			xaxis_title="Impedance Z (kOhms)",
			yaxis_title="Healing Rate (% reduction per visit)"
		)

		st.plotly_chart(fig, use_container_width=True)

	@staticmethod
	def _population_charts(df: pd.DataFrame) -> None:
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

		CN = DColumns(df=df)

		# Get prepared statistics
		avg_impedance, avg_by_type = ImpedanceAnalyzer.prepare_population_stats(df=df)

		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Impedance Components Over Time")
			fig1 = px.line(
				avg_impedance,
				x=CN.VISIT_NUMBER,
				y=[CN.HIGHEST_FREQ_ABSOLUTE, CN.HIGHEST_FREQ_REAL, CN.HIGHEST_FREQ_IMAGINARY],
				title="Average Impedance Components by Visit",
				markers=True
			)
			fig1.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)")
			st.plotly_chart(fig1, use_container_width=True)

		with col2:
			st.subheader("Impedance by Wound Type")
			fig2 = px.bar(
				avg_by_type,
				x=CN.WOUND_TYPE,
				y=CN.HIGHEST_FREQ_ABSOLUTE,
				title="Average Impedance by Wound Type",
				color=CN.WOUND_TYPE
			)
			fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Average Impedance Z (kOhms)")
			st.plotly_chart(fig2, use_container_width=True)

	@staticmethod
	def _render_patient(visits: List[VisitsDataType]) -> None:
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
		# Create tabs for different analysis views
		tab1, tab2, tab3 = st.tabs([
			"Overview",
			"Clinical Analysis",
			"Advanced Interpretation"
		])

		with tab1:
			ImpedanceTab._patient_overview(visits=visits)

		with tab2:
			ImpedanceTab._render_patient_clinical_analysis(visits=visits)

		with tab3:
			ImpedanceTab._render_patient_advanced_analysis(visits=visits)

	@staticmethod
	def _patient_overview(visits: List[VisitsDataType]) -> None:
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
		fig = ImpedanceTab.create_impedance_chart(visits=visits, measurement_mode=measurement_mode)
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

	@staticmethod
	def _render_patient_clinical_analysis(visits: List[VisitsDataType]) -> None:
		"""
			Renders the bioimpedance clinical analysis section for a patient's wound data.

			This method creates a tabbed interface showing clinical analysis for each visit.
			For each visit (except the first one), it performs a comparative analysis with
			the previous visit to track changes in wound healing metrics.

			Args:
				visits (list): A list of dictionaries containing visit data. Each dictionary
						should have at least a CN.VISIT_DATE key and other wound measurement data.

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
		VISIT_DATE_TAG = WoundDataProcessor.get_visit_date_tag(visits)

		visit_tabs = st.tabs([f"{visit.get(VISIT_DATE_TAG, 'N/A')}" for visit in visits])

		for visit_idx, visit_tab in enumerate(visit_tabs):
			with visit_tab:
				# Get current and previous visit data
				current_visit = visits[visit_idx]
				previous_visit = visits[visit_idx-1] if visit_idx > 0 else None

				# Generate comprehensive clinical analysis
				analysis = ImpedanceAnalyzer.generate_clinical_analysis(current_visit=current_visit, previous_visit=previous_visit)

				# Display results in a structured layout
				ImpedanceTab._display_clinical_analysis_results(analysis=analysis, has_previous_visit=previous_visit is not None)

	@staticmethod
	def _render_patient_advanced_analysis(visits: List[VisitsDataType]) -> None:
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
		analysis = ImpedanceAnalyzer.generate_advanced_analysis(visits=visits)

		# Display healing trajectory analysis if available
		if 'healing_trajectory' in analysis and analysis['healing_trajectory']['status'] == 'analyzed':
			ImpedanceTab._display_high_freq_impedance_healing_trajectory_analysis(trajectory=analysis['healing_trajectory'])

		# Display wound healing stage classification
		ImpedanceTab._display_wound_healing_stage(healing_stage=analysis['healing_stage'])

		# Display Cole-Cole parameters if available
		if 'cole_parameters' in analysis and analysis['cole_parameters']:
			ImpedanceTab._display_tissue_electrical_properties(cole_params=analysis['cole_parameters'])

		# Display clinical insights
		ImpedanceTab._display_clinical_insights(insights=analysis['insights'])

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

	@staticmethod
	def _display_clinical_analysis_results(analysis: dict, has_previous_visit: bool) -> None:
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
			ImpedanceTab._display_tissue_health_assessment(analysis['tissue_health'])

		with col2:
			ImpedanceTab._display_infection_risk_assessment(analysis['infection_risk'])

		st.markdown('---')

		# Display Tissue Composition and Changes in a two-column layout
		col1, col2 = st.columns(2)

		with col2:
			ImpedanceTab._display_tissue_composition_analysis(analysis['frequency_response'])

		with col1:
			if has_previous_visit and 'changes' in analysis:
				ImpedanceTab._display_visit_changes(
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

	@staticmethod
	def _display_tissue_health_assessment(tissue_health):
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

	@staticmethod
	def _display_infection_risk_assessment(infection_risk):
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

	@staticmethod
	def _display_tissue_composition_analysis(freq_response):
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

	@staticmethod
	def _display_visit_changes(changes, significant_changes):
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
			except Exception as e:
				logger.error(f"Error in color_cells: {e}")
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

	@staticmethod
	def _display_high_freq_impedance_healing_trajectory_analysis(trajectory):
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

	@staticmethod
	def _display_wound_healing_stage(healing_stage: dict) -> None:
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
			"Inflammatory" : "red",
			"Proliferative": "orange",
			"Remodeling"   : "green"
		}

		stage_color = stage_colors.get(healing_stage['stage'], "blue")
		st.markdown(f"**Current Stage:** <span style='color:{stage_color};font-weight:bold'>{healing_stage['stage']}</span> (Confidence: {healing_stage['confidence']})", unsafe_allow_html=True)

		if healing_stage['characteristics']:
			st.markdown("**Characteristics:**")
			for char in healing_stage['characteristics']:
				st.markdown(f"- {char}")

	@staticmethod
	def _display_tissue_electrical_properties(cole_params: dict) -> None:
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

	@staticmethod
	def _display_clinical_insights(insights: list) -> None:
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

	@staticmethod
	def create_impedance_chart(visits:List[VisitsDataType], measurement_mode:str = "Absolute Impedance (|Z|)") -> go.Figure:
		"""Create an interactive chart displaying impedance measurements over time.

		Args:
			visits: List of visit dicts containing sensor impedance data
			measurement_mode: Type of measurement to display (|Z|, Resistance, Capacitance)

		Returns:
			Plotly Figure object showing impedance measurements over time
		"""

		# Define measurement types and their corresponding data fields
		MEASUREMENT_FIELDS = {
			"Absolute Impedance (|Z|)": "Z",
			"Resistance"              : "resistance",
			"Capacitance"             : "capacitance"
		}

		# Frequency labels
		FREQUENCY_LABELS = {
			"highest_freq"  : "Highest Freq",
			"center_freq": "Center Freq (Max Phase Angle)",
			"lowest_freq"   : "Lowest Freq"
		}

		# Line styles for different frequencies
		LINE_STYLES = {
			"highest_freq"  : None,
			"center_freq": dict(dash='dot'),
			"lowest_freq"   : dict(dash='dash')
		}

		# Initialize data containers
		dates = []
		measurements = { freq: {field: [] 	for field in MEASUREMENT_FIELDS.values()}
											for freq  in FREQUENCY_LABELS }

		VISIT_DATE_TAG = WoundDataProcessor.get_visit_date_tag(visits)

		# Process each visit
		for visit in visits:
			dates.append(visit[VISIT_DATE_TAG])

			sensor_data    = visit.get('sensor_data', {})
			impedance_data = sensor_data.get('impedance', {})

			# Process each frequency
			for freq in FREQUENCY_LABELS:
				freq_data = impedance_data.get(freq, {})
				for field in MEASUREMENT_FIELDS.values():
					try:
						val = float(freq_data.get(field)) if freq_data.get(field) not in (None, '') else None
					except (ValueError, TypeError):
						val = None
					measurements[freq][field].append(val)

		# Create figure
		fig = go.Figure()
		field = MEASUREMENT_FIELDS[measurement_mode]

		# Add traces for each frequency
		for freq in FREQUENCY_LABELS:
			values = measurements[freq][field]
			if any(v is not None for v in values):
				fig.add_trace(go.Scatter(
					x    = dates,
					y    = values,
					name = f'{measurement_mode.split()[0]} ({FREQUENCY_LABELS[freq]})',
					mode = 'lines+markers',
					line = LINE_STYLES[freq]
				))

		# Configure layout
		fig.update_layout(
			title       = 'Impedance Measurements Over Time',
			xaxis_title = VISIT_DATE_TAG,
			yaxis_title = f'{measurement_mode.split()[0]} Values',
			hovermode   = 'x unified',
			showlegend  = True,
			height      = 600,
			yaxis       = dict(type='log')
		)

		return fig
