# Standard library imports
import traceback
from typing import List, Dict, Any, Tuple, Optional
import logging

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
from wound_analysis.utils.data_processor import WoundDataProcessor, VisitsDataType, VisitsMetadataType
from wound_analysis.utils.column_schema import DColumns, ExcelSheetColumns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImpedanceTab:
	"""
	A class for managing and rendering the Impedance Analysis tab in the wound analysis dashboard.

	This class contains methods to display impedance analysis for both population-level data
	and individual patient data, including impedance measurements, frequency response analysis,
	and clinical interpretations.
	"""

	def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.wound_data_processor = wound_data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)

	def render(self) -> None:
		"""
		Render the impedance analysis tab based on patient selection.

		Delegates to either PopulationImpedanceRenderer or PatientImpedanceRenderer
		based on whether all patients or a specific patient is selected.
		"""
		st.header("Impedance Analysis")

		if self.patient_id == "All Patients":
			logger.debug("Rendering population-level impedance analysis")
			PopulationImpedanceRenderer(df=self.df, wound_data_processor=self.wound_data_processor).render()

		else:
			logger.debug(f"Rendering patient-level impedance analysis for patient {self.patient_id}")

			visits_meta_data: VisitsMetadataType   = self.wound_data_processor.get_patient_visits(record_id=self.patient_id)
			visits          : List[VisitsDataType] = visits_meta_data['visits']

			PatientImpedanceRenderer(visits=visits, wound_data_processor=self.wound_data_processor).render()


class PopulationImpedanceRenderer:
	"""
	Renderer for population-level impedance analysis.

	This class handles the rendering of population-level impedance analysis,
	including clustering, correlation analysis, and visualization of impedance data
	across the entire patient population.

	Attributes:
		df (pd.DataFrame): DataFrame containing all patient data
		CN (DColumns): Column name accessor for the DataFrame
	"""

	def __init__(self, df: pd.DataFrame, wound_data_processor: WoundDataProcessor):
		"""
		Initialize the PopulationImpedanceRenderer with patient data.

		Args:
			df: DataFrame containing all patient data
		"""
		self.df = df
		self.CN = DColumns(df=df)
		self.wound_data_processor = wound_data_processor

	def render(self) -> None:
		"""
		Render the population-level impedance analysis section.

		This method orchestrates the rendering of the population-level analysis,
		including clustering, correlation analysis, and visualization.
		"""
		# Create a copy of the dataframe for analysis
		analysis_df = self.df.copy()

		# Render clustering options and perform clustering if requested
		working_df = self._render_clustering_section(analysis_df)

		# Add outlier threshold control and calculate correlation
		filtered_df = self._display_correlation_controls(working_df)

		# Create scatter plot if we have valid data
		if not filtered_df.empty:
			self._render_scatter_plot(df=filtered_df)
		else:
			st.warning("No valid data available for the scatter plot.")

		# Create additional visualizations in a two-column layout
		self._render_population_charts(df=working_df)

	def _render_clustering_section(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
		"""
		Render the clustering section and perform clustering if requested.

		Args:
			analysis_df: DataFrame to perform clustering on

		Returns:
			DataFrame to use for further analysis (either clustered or original)
		"""
		with st.expander("Patient Data Clustering", expanded=True):
			st.markdown("### Cluster Analysis Settings")

			# Create columns for clustering controls
			col1, col2, col3 = st.columns([1, 2, 1])

			with col1:
				n_clusters = st.number_input(
					"Number of Clusters",
					min_value=2,
					max_value=10,
					value=3,
					help="Select the number of clusters to divide patient data into"
				)

			with col2:
				cluster_features = st.multiselect(
					"Features for Clustering",
					options=[
						self.CN.HIGHEST_FREQ_ABSOLUTE,
						self.CN.WOUND_AREA,
						self.CN.CENTER_TEMP,
						self.CN.OXYGENATION,
						self.CN.HEMOGLOBIN,
						self.CN.AGE,
						self.CN.BMI,
						self.CN.DAYS_SINCE_FIRST_VISIT,
						self.CN.HEALING_RATE
					],
					default=[self.CN.HIGHEST_FREQ_ABSOLUTE, self.CN.WOUND_AREA, self.CN.HEALING_RATE],
					help="Select features to be used for clustering patients"
				)

			with col3:
				clustering_method = st.selectbox(
					"Clustering Method",
					options=["K-Means", "Hierarchical", "DBSCAN"],
					index=0,
					help="Select the clustering algorithm to use"
				)

				run_clustering = st.button("Run Clustering")

			# Initialize session state for clusters if not already present
			if 'clusters' not in st.session_state:
				st.session_state.clusters = None
				st.session_state.cluster_df = None
				st.session_state.selected_cluster = None
				st.session_state.feature_importance = None

			# Run clustering if requested
			if run_clustering and cluster_features:
				try:
					self._perform_clustering(
						analysis_df=analysis_df,
						cluster_features=cluster_features,
						n_clusters=n_clusters,
						clustering_method=clustering_method
					)
				except Exception as e:
					st.error(f"Error during clustering: {str(e)}")
					st.error(traceback.format_exc())

			# Render cluster selection and characteristics if clustering has been performed
			working_df = self._render_cluster_selection(analysis_df, cluster_features)

		return working_df

	def _perform_clustering(self, analysis_df: pd.DataFrame, cluster_features: List[str],
							n_clusters: int, clustering_method: str) -> None:
		"""
		Perform clustering on the data using the specified method and features.

		Args:
			analysis_df: DataFrame to perform clustering on
			cluster_features: List of features to use for clustering
			n_clusters: Number of clusters to create
			clustering_method: Method to use for clustering (K-Means, Hierarchical, or DBSCAN)
		"""
		# Create a feature dataframe for clustering
		clustering_df = analysis_df[cluster_features].copy()

		# Handle missing values
		clustering_df = clustering_df.fillna(clustering_df.mean())

		# Drop rows with any remaining NaN values
		clustering_df = clustering_df.dropna()

		if len(clustering_df) <= n_clusters:
			st.error("Not enough valid data points for clustering. Try selecting different features or reducing the number of clusters.")
			return

		# Get indices of valid rows to map back to original dataframe
		valid_indices = clustering_df.index

		# Standardize the data
		scaler = StandardScaler()
		scaled_features = scaler.fit_transform(clustering_df)

		# Perform clustering based on selected method
		cluster_labels, feature_importance = self._apply_clustering_algorithm(
			scaled_features=scaled_features,
			cluster_features=cluster_features,
			n_clusters=n_clusters,
			clustering_method=clustering_method
		)

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
		self._display_cluster_distribution(analysis_df)

		# Display feature importance
		if feature_importance:
			self._display_feature_importance(feature_importance)

	def _apply_clustering_algorithm(self, scaled_features: np.ndarray, cluster_features: List[str],
									n_clusters: int, clustering_method: str) -> Tuple[np.ndarray, Dict[str, float]]:
		"""
		Apply the specified clustering algorithm to the data.

		Args:
			scaled_features: Standardized features to cluster
			cluster_features: Names of the features being clustered
			n_clusters: Number of clusters to create
			clustering_method: Method to use for clustering

		Returns:
			Tuple of (cluster_labels, feature_importance)
		"""
		feature_importance = {}

		if clustering_method == "K-Means":
			clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
			cluster_labels = clusterer.fit_predict(scaled_features)

			# Calculate feature importance for K-Means
			centers = clusterer.cluster_centers_
			for i, feature in enumerate(cluster_features):
				# Calculate the variance of this feature across cluster centers
				variance = np.var([center[i] for center in centers])
				feature_importance[feature] = variance

		elif clustering_method == "Hierarchical":
			# Perform hierarchical clustering
			Z = linkage(scaled_features, 'ward')
			cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # Adjust to 0-based

			# For hierarchical clustering, use silhouette coefficients for feature importance
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

		else:  # DBSCAN
			# Calculate epsilon based on data
			neigh = NearestNeighbors(n_neighbors=3)
			neigh.fit(scaled_features)
			distances, _ = neigh.kneighbors(scaled_features)
			distances = np.sort(distances[:, 2], axis=0)  # Distance to 3rd nearest neighbor
			epsilon = np.percentile(distances, 90)  # Use 90th percentile as epsilon

			clusterer = DBSCAN(eps=epsilon, min_samples=max(3, len(scaled_features)//30))
			cluster_labels = clusterer.fit_predict(scaled_features)

			# For DBSCAN, calculate feature importance using variance within clusters
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
		if feature_importance and max(feature_importance.values()) > 0:
			max_importance = max(feature_importance.values())
			feature_importance = {k: v/max_importance for k, v in feature_importance.items()}

		return cluster_labels, feature_importance

	def _display_cluster_distribution(self, analysis_df: pd.DataFrame) -> None:
		"""
		Display the distribution of data points across clusters.

		Args:
			analysis_df: DataFrame with cluster assignments
		"""
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

	def _display_feature_importance(self, feature_importance: Dict[str, float]) -> None:
		"""
		Display a radar chart showing feature importance in clustering.

		Args:
			feature_importance: Dictionary mapping feature names to importance values
		"""
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

	def _render_cluster_selection(self, analysis_df: pd.DataFrame, cluster_features: List[str]) -> pd.DataFrame:
		"""
		Render the cluster selection dropdown and display cluster characteristics.

		Args:
			analysis_df: Original DataFrame
			cluster_features: Features used for clustering

		Returns:
			DataFrame to use for further analysis (either filtered by cluster or original)
		"""
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
				self._display_cluster_characteristics(cluster_id, working_df, analysis_df, cluster_features)

			return working_df
		else:
			return analysis_df

	def _display_cluster_characteristics(self, cluster_id: int, cluster_df: pd.DataFrame,
										full_df: pd.DataFrame, features: List[str]) -> None:
		"""
		Display characteristics of the selected cluster compared to the overall population.

		Args:
			cluster_id: ID of the selected cluster
			cluster_df: DataFrame filtered to the selected cluster
			full_df: Full DataFrame with all data
			features: Features to compare
		"""
		st.markdown(f"### Characteristics of Cluster {cluster_id}")

		# Create summary statistics for this cluster vs. overall population
		summary_stats = []

		for feature in features:
			if feature in cluster_df.columns:
				try:
					cluster_mean = cluster_df[feature].mean()
					overall_mean = full_df[feature].mean()
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

	def _display_correlation_controls(self, df_for_cluster: pd.DataFrame) -> pd.DataFrame:
		"""
		Display controls for correlation analysis and perform the analysis.

		Args:
			df_for_cluster: DataFrame to analyze

		Returns:
			Filtered DataFrame with outliers removed
		"""
		cols = st.columns([2, 3])

		with cols[0]:
			outlier_threshold = st.number_input(
				"Impedance Outlier Threshold (Quantile)",
				min_value = 0.0,
				max_value = 0.9,
				value     = 0.0,
				step      = 0.05,
				help      = "Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		# Get the selected features for analysis
		features_to_analyze = [
			self.CN.HIGHEST_FREQ_ABSOLUTE,
			self.CN.WOUND_AREA,
			self.CN.CENTER_TEMP,
			self.CN.OXYGENATION,
			self.CN.HEMOGLOBIN,
			self.CN.HEALING_RATE
		]

		# Filter features that exist in the dataframe
		features_to_analyze = [f for f in features_to_analyze if f in df_for_cluster.columns]

		# Create a copy of the dataframe with only the features we want to analyze
		analysis_df = df_for_cluster[features_to_analyze].copy().dropna()

		# Remove outliers if threshold is set
		if outlier_threshold > 0:
			for col in analysis_df.columns:
				q_low       = analysis_df[col].quantile(outlier_threshold)
				q_high      = analysis_df[col].quantile(1 - outlier_threshold)
				analysis_df = analysis_df[ (analysis_df[col] >= q_low) & (analysis_df[col] <= q_high) ]

		if analysis_df.empty or len(analysis_df) < 2:
			st.warning("Not enough data after outlier removal for correlation analysis.")
			return pd.DataFrame()

		# Calculate correlation matrix
		corr_matrix = analysis_df.corr()

		# Calculate p-values for correlations
		p_values = self._calculate_correlation_pvalues(analysis_df)

		# Create correlation heatmap
		fig = px.imshow(
			abs(corr_matrix),
			labels=dict(color="Correlation"),
			color_continuous_scale="RdBu",
			aspect="auto",
			text_auto=".2f",
			title="Correlation Matrix Heatmap"
		)

		st.plotly_chart(fig, use_container_width=True)

		# Display detailed statistics
		st.subheader("Statistical Summary")

		# Create tabs for different statistical views
		tab1, tab2, tab3 = st.tabs(["Correlation Details", "Descriptive Stats", "Effect Sizes"])

		with tab1:
			self._display_significant_correlations(features_to_analyze, corr_matrix, p_values)

		with tab2:
			self._display_descriptive_statistics(analysis_df)

		with tab3:
			self._display_effect_sizes(analysis_df, features_to_analyze)

		return analysis_df

	def _calculate_correlation_pvalues(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Calculate p-values for correlations between all columns in the DataFrame.

		Args:
			df: DataFrame to analyze

		Returns:
			DataFrame of p-values with the same shape as the correlation matrix
		"""
		def calculate_pvalue(x, y):
			mask = ~(np.isnan(x) | np.isnan(y))
			if np.sum(mask) < 2:
				return np.nan
			return stats.pearsonr(x[mask], y[mask]).pvalue

		return pd.DataFrame(
			[[  stats.pearsonr(df[col1], df[col2]).pvalue # calculate_pvalue(df[col1], df[col2])
						for col2 in df.columns]
						for col1 in df.columns],
				columns = df.columns,
				index   = df.columns
		)

	def _display_significant_correlations(self, features: List[str], corr_matrix: pd.DataFrame, p_values: pd.DataFrame) -> None:
		"""
		Display significant correlations between features.

		Args:
			features: List of feature names
			corr_matrix: Correlation matrix
			p_values: Matrix of p-values
		"""
		st.markdown("#### Significant Correlations (p < 0.05)")
		significant_corrs = []
		for i in range(len(features)):
			for j in range(i+1, len(features)):
				if p_values.iloc[i,j] < 0.05:
					significant_corrs.append({
						"Feature 1": features[i],
						"Feature 2": features[j],
						"Correlation": f"{corr_matrix.iloc[i,j]:.3f}",
						"p-value": f"{p_values.iloc[i,j]:.3e}"
					})

		if significant_corrs:
			st.table(pd.DataFrame(significant_corrs))
		else:
			st.info("No significant correlations found.")

	def _display_descriptive_statistics(self, df: pd.DataFrame) -> None:
		"""
		Display descriptive statistics for the DataFrame.

		Args:
			df: DataFrame to analyze
		"""
		st.markdown("#### Descriptive Statistics")
		desc_stats = df.describe()
		desc_stats.loc["skew"] = df.skew()
		desc_stats.loc["kurtosis"] = df.kurtosis()
		st.dataframe(desc_stats)

	def _display_effect_sizes(self, df: pd.DataFrame, features: List[str]) -> None:
		"""
		Display effect sizes relative to impedance.

		Args:
			df: DataFrame to analyze
			features: List of feature names
		"""
		st.markdown("#### Effect Sizes (Cohen's d) relative to Impedance")

		effect_sizes = []
		impedance_col = "Skin Impedance (kOhms) - Z"

		if impedance_col in features:
			for col in features:
				if col != impedance_col:
					# Calculate Cohen's d
					d = (df[col].mean() - df[impedance_col].mean()) / \
						np.sqrt((df[col].var() + df[impedance_col].var()) / 2)

					effect_sizes.append({
						"Feature": col,
						"Cohen's d": f"{d:.3f}",
						"Effect Size": "Large" if abs(d) > 0.8 else "Medium" if abs(d) > 0.5 else "Small",
						"95% CI": f"[{d-1.96*np.sqrt(4/len(df)):.3f}, {d+1.96*np.sqrt(4/len(df)):.3f}]"
					})

			if effect_sizes:
				st.table(pd.DataFrame(effect_sizes))
			else:
				st.info("No effect sizes could be calculated.")
		else:
			st.info("Impedance measurements not available for effect size calculation.")

	def _render_scatter_plot(self, df: pd.DataFrame) -> None:
		"""
		Render a scatter plot showing the relationship between impedance and healing rate.

		Args:
			df: DataFrame to plot
		"""
		# Create a copy to avoid modifying the original dataframe
		plot_df = df.copy()

		# Handle missing values in Calculated Wound Area
		if self.CN.WOUND_AREA in plot_df.columns:
			# Fill NaN with the mean, or 1 if all values are NaN
			mean_area = plot_df[self.CN.WOUND_AREA].mean()
			plot_df[self.CN.WOUND_AREA] = plot_df[self.CN.WOUND_AREA].fillna(mean_area if pd.notnull(mean_area) else 1)

		# Define hover data columns we want to show if available
		hover_columns = [self.CN.RECORD_ID, self.CN.EVENT_NAME, self.CN.WOUND_TYPE]
		available_hover = [col for col in hover_columns if col in plot_df.columns]

		fig = px.scatter(
			plot_df,
			x=self.CN.HIGHEST_FREQ_ABSOLUTE,
			y=self.CN.HEALING_RATE,
			color=self.CN.DIABETES if self.CN.DIABETES in plot_df.columns else None,
			size=self.CN.WOUND_AREA if self.CN.WOUND_AREA in plot_df.columns else None,
			size_max=30,
			hover_data=available_hover,
			title="Impedance vs Healing Rate Correlation"
		)

		fig.update_layout(
			xaxis_title="Impedance Z (kOhms)",
			yaxis_title="Healing Rate (% reduction per visit)"
		)

		st.plotly_chart(fig, use_container_width=True)

	def _render_population_charts(self, df: pd.DataFrame) -> None:
		"""
		Render charts showing population-level impedance statistics.

		Args:
			df: DataFrame containing impedance measurements and visit information
		"""
		# Get prepared statistics

		avg_impedance, avg_by_type = self.wound_data_processor.impedance_analyzer.prepare_population_stats(df=df)

		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Impedance Components Over Time")
			fig1 = px.line(
				avg_impedance,
				x=self.CN.VISIT_NUMBER,
				y=[self.CN.HIGHEST_FREQ_ABSOLUTE, self.CN.HIGHEST_FREQ_REAL, self.CN.HIGHEST_FREQ_IMAGINARY],
				title="Average Impedance Components by Visit",
				markers=True
			)
			fig1.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)")
			st.plotly_chart(fig1, use_container_width=True)

		with col2:
			st.subheader("Impedance by Wound Type")
			fig2 = px.bar(
				avg_by_type,
				x=self.CN.WOUND_TYPE,
				y=self.CN.HIGHEST_FREQ_ABSOLUTE,
				title="Average Impedance by Wound Type",
				color=self.CN.WOUND_TYPE
			)
			fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Average Impedance Z (kOhms)")
			st.plotly_chart(fig2, use_container_width=True)


class PatientImpedanceRenderer:
	"""
	Renderer for patient-level impedance analysis.

	This class handles the rendering of patient-level impedance analysis,
	including overview, clinical analysis, and advanced interpretation tabs.

	Attributes:
		visits (List[VisitsDataType]): List of visit data for the patient
	"""

	def __init__(self, visits: List[VisitsDataType], wound_data_processor: WoundDataProcessor):
		"""
		Initialize the PatientImpedanceRenderer with visit data.

		Args:
			visits: List of visit data for the patient
		"""
		self.visits = visits
		self.wound_data_processor = wound_data_processor

		logger.debug("Fetching visit date tag")
		self.visit_date_tag = self.wound_data_processor.get_visit_date_tag(visits)

	def render(self) -> None:
		"""
		Render the patient-level impedance analysis section.

		This method creates a tabbed interface to display different perspectives
		on a patient's impedance data, organized into Overview, Clinical Analysis,
		and Advanced Interpretation tabs.
		"""
		# Create tabs for different analysis views
		tab1, tab2, tab3 = st.tabs([
			"Overview",
			"Clinical Analysis",
			"Advanced Interpretation"
		])

		with tab1:
			self._render_overview()

		with tab2:
			self._render_clinical_analysis()

		with tab3:
			self._render_advanced_analysis()

	def _render_overview(self) -> None:
		"""
		Render the overview section for patient impedance measurements.

		This method creates a section showing impedance measurements over time,
		allowing users to view different types of impedance data.
		"""
		st.subheader("Impedance Measurements Over Time")

		tabs = st.tabs(["High/Center/Low", "Entire Freq Sweep"])

		with tabs[0]:
			# Add measurement mode selector
			measurement_mode = st.selectbox(
				"Select Measurement Mode:",
				["Absolute Impedance (|Z|)", "Resistance", "Capacitance"],
				key="impedance_mode_selector"
			)

			# Create impedance chart with selected mode
			fig = self._create_impedance_chart(measurement_mode=measurement_mode)
			st.plotly_chart(fig, use_container_width=True)

		with tabs[1]:
			# Create impedance chart with entire frequency sweep
			self._create_impedance_chart_entire_freq_sweep()

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

	def _render_clinical_analysis(self) -> None:
		"""
		Render the bioimpedance clinical analysis section for a patient's wound data.

		This method creates a tabbed interface showing clinical analysis for each visit.
		For each visit (except the first one), it performs a comparative analysis with
		the previous visit to track changes in wound healing metrics.
		"""
		st.subheader("Bioimpedance Clinical Analysis")

		# Only analyze if we have at least two visits
		if len(self.visits) < 2:
			st.warning("At least two visits are required for comprehensive clinical analysis")
			return

		# Create tabs for each visit
		visit_tabs = st.tabs([f"{visit.get(self.visit_date_tag, 'N/A')}" for visit in self.visits])

		for visit_idx, visit_tab in enumerate(visit_tabs):
			with visit_tab:
				# Get current and previous visit data
				current_visit = self.visits[visit_idx]
				previous_visit = self.visits[visit_idx-1] if visit_idx > 0 else None

				try:
					# Generate comprehensive clinical analysis
					analysis = self.wound_data_processor.impedance_analyzer.generate_clinical_analysis(
						current_visit=current_visit,
						previous_visit=previous_visit
					)

					# Display results in a structured layout
					self._display_clinical_analysis_results(
						analysis=analysis,
						has_previous_visit=previous_visit is not None
					)

				except Exception as e:
					st.error(f"Error analyzing visit data: {str(e)}")
					st.error(traceback.format_exc())

	def _render_advanced_analysis(self) -> None:
		"""
		Render the advanced bioelectrical analysis section for a patient's wound data.

		This method displays comprehensive bioelectrical analysis results including healing
		trajectory, wound healing stage classification, tissue electrical properties, and
		clinical insights derived from the impedance data.
		"""
		st.subheader("Advanced Bioelectrical Interpretation")

		if len(self.visits) < 3:
			st.warning("At least three visits are required for advanced analysis")
			return

		try:
			# Generate advanced analysis
			logger.debug("Generating advanced analysis")
			analysis = self.wound_data_processor.impedance_analyzer.generate_advanced_analysis(visits=self.visits)
			logger.debug("Generating advanced analysis complete")

			# Display healing trajectory analysis if available
			if 'healing_trajectory' in analysis and analysis['healing_trajectory']['status'] == 'analyzed':
				self._display_healing_trajectory(trajectory=analysis['healing_trajectory'])

			# Display wound healing stage classification
			self._display_wound_healing_stage(healing_stage=analysis['healing_stage'])

			# Display Cole-Cole parameters if available
			if 'cole_parameters' in analysis and analysis['cole_parameters']:
				self._display_tissue_electrical_properties(cole_params=analysis['cole_parameters'])

			# Display clinical insights
			self._display_clinical_insights(insights=analysis['insights'])

			# Reference information
			self._display_advanced_analysis_info()
		except Exception as e:
			st.error(f"Error performing advanced analysis: {str(e)}")
			st.error(traceback.format_exc())

	def _display_advanced_analysis_info(self) -> None:
		"""
		Display the informational box for the advanced analysis tab.
		"""
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


	# --- Clinical Analysis Display Helpers ---

	def _display_clinical_analysis_results(self, analysis: Dict[str, Any], has_previous_visit: bool) -> None:
		"""
		Display the clinical analysis results in a structured layout.

		Args:
			analysis: Dictionary containing the analysis results
			has_previous_visit: Flag indicating whether there is data from a previous visit
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

		self._display_clinical_analysis_info()

	def _display_tissue_health_assessment(self, tissue_health: Tuple[Optional[float], str]) -> None:

		st.markdown("### Tissue Health Assessment", help="The tissue health index is calculated using multi-frequency impedance ratios.")

		health_score, health_interp = tissue_health

		if health_score is not None:
			# Create a color scale for the health score
			color = "red" if health_score < 40 else "orange" if health_score < 60 else "green"
			st.markdown(f"**Tissue Health Index:** <span style='color:{color};font-weight:bold'>{health_score:.1f}/100</span>", unsafe_allow_html=True)
			st.markdown(f"**Interpretation:** {health_interp}")
		else:
			st.warning("Insufficient data for tissue health calculation")

	def _display_infection_risk_assessment(self, infection_risk: Dict[str, Any]) -> None:

		st.markdown("### Infection Risk Assessment", help="The infection risk assessment is based on impedance measurements.")

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

	def _display_tissue_composition_analysis(self, freq_response: Dict[str, Any]) -> None:

		st.markdown("### Tissue Composition Analysis", help="This analysis utilizes bioelectrical impedance analysis principles.")

		# Display tissue composition analysis from frequency response
		st.markdown("#### Analysis Results:")
		if 'alpha_dispersion' in freq_response and 'beta_dispersion' in freq_response:
			st.markdown(f"**Alpha Dispersion:** {freq_response['alpha_dispersion']:.3f}")
			st.markdown(f"**Beta Dispersion:** {freq_response['beta_dispersion']:.3f}")

		# Display interpretation with more emphasis
		st.markdown(f"**Tissue Composition Interpretation:** {freq_response['interpretation']}")

	def _display_visit_changes(self, changes: Dict[str, float], significant_changes: Dict[str, bool]) -> None:
		"""
		Display analysis of changes between visits.

		Args:
			changes: Dictionary mapping parameter names to percentage changes
			significant_changes: Dictionary mapping parameter names to boolean values
				indicating clinically significant changes
		"""
		st.markdown("#### Changes Since Previous Visit", help="The changes since previous visit are based on bioelectrical impedance analysis principles.")

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
			except Exception:
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

	def _display_clinical_analysis_info(self) -> None:
		"""
		Display the informational box for the clinical analysis tab.
		"""
		st.markdown("""
			<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
			<p style="margin-top:0; color:#666; font-weight:bold;">ABOUT THIS ANALYSIS:</p>
			<p>Color indicates direction of change: <span style="color:#FF4B4B">red = increase</span>, <span style="color:#00CC96">green = decrease</span>.<br>
			Asterisk (*) marks changes exceeding clinical thresholds: Resistance >15%, Capacitance >20%, Z >15%.</p>
			</div>
			""", unsafe_allow_html=True)


	# --- Advanced Analysis Display Helpers ---

	def _display_healing_trajectory(self, trajectory: Dict[str, Any]) -> None:
		"""
		Display the healing trajectory analysis with charts and statistics.

		Args:
			trajectory: Dictionary containing healing trajectory data
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

	def _display_wound_healing_stage(self, healing_stage: Dict[str, Any]) -> None:
		"""
		Display wound healing stage classification.

		Args:
			healing_stage: Dictionary containing wound healing stage analysis
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


	def _display_tissue_electrical_properties(self, cole_params: Dict[str, Any]) -> None:
		"""
		Display the tissue electrical properties derived from Cole parameters.

		Args:
			cole_params: Dictionary containing the Cole-Cole parameters
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

	def _display_clinical_insights(self, insights: List[Dict[str, Any]]) -> None:
		"""
		Display clinical insights in an organized expandable format.

		Args:
			insights: List of dictionaries containing insight information
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


	# --- Chart Creation ---

	def _create_impedance_chart(self, measurement_mode: str = "Absolute Impedance (|Z|)") -> go.Figure:
		"""
		Create an interactive chart displaying impedance measurements over time.

		Args:
			measurement_mode: Type of measurement to display (|Z|, Resistance, Capacitance)

		Returns:
			Plotly Figure object showing impedance measurements over time
		"""
		# Define measurement types and their corresponding data fields
		measurement_fields = {
			"Absolute Impedance (|Z|)": "Z",
			"Resistance"              : "resistance",
			"Capacitance"             : "capacitance"
		}

		# Frequency labels
		frequency_labels = {
			"highest_freq": "Highest Freq",
			"center_freq" : "Center Freq (Max Phase Angle)",
			"lowest_freq" : "Lowest Freq"
		}

		# Line styles for different frequencies
		line_styles = {
			"highest_freq": None,
			"center_freq" : dict(dash='dot'),
			"lowest_freq" : dict(dash='dash')
		}

		# Initialize data containers
		dates = []
		measurements = {freq: {field: [] for field in measurement_fields.values()}
						for freq in frequency_labels}

		# Process each visit
		for visit in self.visits:
			visit_date = visit[self.visit_date_tag]

			dates.append(visit_date)

			sensor_data = visit.get('sensor_data', {})
			impedance_data = sensor_data.get('impedance', {})

			# try:
			# Process each frequency
			for freq in frequency_labels:
				freq_data = impedance_data.get(freq, {})
				for field in measurement_fields.values():
					try:
						val = float(freq_data.get(field)) if freq_data.get(field) not in (None, '') else None
					except (ValueError, TypeError) as e:
						raise ValueError(f"Invalid {field} in visit {visit_date}: {impedance_data} {str(e)}")

					measurements[freq][field].append(val)

			# except Exception as e:
			# 	raise ValueError(f"Invalid {impedance_data} {str(e)}")

		# Create figure
		fig = go.Figure()
		field = measurement_fields[measurement_mode]

		# Add traces for each frequency
		for freq in frequency_labels:
			values = measurements[freq][field]
			if any(v is not None for v in values):
				fig.add_trace(go.Scatter(
					x=dates,
					y=values,
					name=f'{measurement_mode.split()[0]} ({frequency_labels[freq]})',
					mode='lines+markers',
					line=line_styles[freq]
				))

		# Configure layout
		fig.update_layout(
			title='Impedance Measurements Over Time',
			xaxis_title=self.visit_date_tag,
			yaxis_title=f'{measurement_mode.split()[0]} Values',
			hovermode='x unified',
			showlegend=True,
			height=600,
			yaxis=dict(type='log' if measurement_mode == "Capacitance" else 'linear')
		)

		return fig

	def _create_impedance_chart_entire_freq_sweep(self) -> None:
		"""
		Create multiple tabs for different visit dates, each showing the entire frequency sweep plot.
		Each plot displays the real and imaginary impedance components against frequency.

		Features:
		- Interactive plots with hover information and zoom capabilities
		- Nyquist plot (Real vs. Imaginary) for impedance analysis
		- Frequency response analysis with key metrics
		- Data table with sortable columns
		- Error handling for missing or invalid data
		"""
		try:
			# Initialize data containers
			dates = []
			freq_sweep: Dict[str, pd.DataFrame] = {}
			valid_data_count = 0

			# Process each visit and collect data
			for visit in self.visits:
				try:
					visit_date = visit[self.visit_date_tag]
					dates.append(visit_date)

					sensor_data = visit.get('sensor_data', {}) or {}
					impedance_data = sensor_data.get('impedance', {}) or {}
					df = impedance_data.get('entire_freq_sweep', None)

					# Validate dataframe
					if isinstance(df, pd.DataFrame) and not df.empty:
						# Ensure required columns exist
						required_columns = [ExcelSheetColumns.FREQ.value,
										ExcelSheetColumns.REAL.value,
										ExcelSheetColumns.IMAGINARY.value]

						if all(col in df.columns for col in required_columns):
							# Sort by frequency for consistent plotting
							df = df.sort_values(by=ExcelSheetColumns.FREQ.value)
							freq_sweep[visit_date] = df
							valid_data_count += 1
						else:
							freq_sweep[visit_date] = None
							logger.warning(f"Missing required columns for visit {visit_date}. Required: {required_columns}")
					else:
						freq_sweep[visit_date] = None
				except Exception as e:
					logger.error(f"Error processing visit data: {str(e)}")
					continue

			# Check if we have any valid data
			if not dates:
				st.warning("No visit data available for frequency sweep analysis.")
				return

			if valid_data_count == 0:
				st.warning("No valid frequency sweep data found in any visits.")
				return

			# Create tabs for each visit date
			visit_tabs = st.tabs([f"{visit_date}" for visit_date in dates])

			# For each tab, create plots for that visit date
			for i, visit_date in enumerate(dates):
				with visit_tabs[i]:
					df = freq_sweep[visit_date]

					if df is None or df.empty:
						st.warning(f"No frequency sweep data available for visit date: {visit_date}")
						continue

					# Create frequency response plot
					st.subheader("Frequency Response")
					fig_freq = self._create_frequency_response_plot(df, visit_date)
					st.plotly_chart(fig_freq, use_container_width=True)

					# Create two columns for additional plots and metrics
					col1, col2 = st.columns(2)

					with col1:
						# Create Nyquist plot (Real vs Imaginary)
						st.subheader("Nyquist Plot")
						fig_nyquist = self._create_nyquist_plot(df, visit_date)
						st.plotly_chart(fig_nyquist, use_container_width=True)

					with col2:
						# Display key metrics
						st.subheader("Key Metrics")
						self._display_impedance_metrics(df, visit_date)

					# Display data table with frequency sweep values
					with st.expander("View Frequency Sweep Data Table"):
						# Format the dataframe for better display
						display_df = df.copy()

						# Round numeric columns to 2 decimal places for display
						for col in display_df.select_dtypes(include=['float64']).columns:
							display_df[col] = display_df[col].round(2)

						st.dataframe(
							display_df,
							use_container_width=True,
							hide_index=True
						)

						# Add download button for the data
						csv = df.to_csv(index=False)
						st.download_button(
							label="Download CSV",
							data=csv,
							file_name=f"impedance_sweep_{visit_date}.csv",
							mime="text/csv"
						)
		except Exception as e:
			st.error(f"Error generating frequency sweep plots: {str(e)}")
			logger.error(f"Error in _create_impedance_chart_entire_freq_sweep: {traceback.format_exc()}")

	def _create_frequency_response_plot(self, df: pd.DataFrame, visit_date: str) -> go.Figure:
		"""
		Create a frequency response plot showing real and imaginary components.

		Args:
			df: DataFrame containing frequency sweep data
			visit_date: Date of the visit

		Returns:
			Plotly Figure object with the frequency response plot
		"""
		# Create figure
		fig = go.Figure()

		# Add real component trace with improved styling
		fig.add_trace(go.Scatter(
			x=df[ExcelSheetColumns.FREQ.value],
			y=df[ExcelSheetColumns.REAL.value],
			name=f"Real Component (Z')",
			mode='lines+markers',
			line=dict(color='blue', width=2),
			marker=dict(size=6),
			hovertemplate=(
				"<b>Frequency:</b> %{x:.2f} Hz<br>"
				"<b>Real:</b> %{y:.2f} Ohm<br>"
			)
		))

		# Add imaginary component trace with improved styling
		fig.add_trace(go.Scatter(
			x=df[ExcelSheetColumns.FREQ.value],
			y=df[ExcelSheetColumns.IMAGINARY.value],
			name=f"Imaginary Component (-Z'')",
			mode='lines+markers',
			line=dict(color='red', width=2),
			marker=dict(size=6),
			hovertemplate=(
				"<b>Frequency:</b> %{x:.2f} Hz<br>"
				"<b>Imaginary:</b> %{y:.2f} Ohm<br>"
			)
		))

		# Add absolute impedance if available
		if ExcelSheetColumns.ABSOLUTE.value in df.columns:
			fig.add_trace(go.Scatter(
				x=df[ExcelSheetColumns.FREQ.value],
				y=df[ExcelSheetColumns.ABSOLUTE.value],
				name=f"Absolute Impedance (|Z|)",
				mode='lines+markers',
				line=dict(color='green', width=2),
				marker=dict(size=6),
				hovertemplate=(
					"<b>Frequency:</b> %{x:.2f} Hz<br>"
					"<b>|Z|:</b> %{y:.2f} Ohm<br>"
				)
			))

		# Configure layout with improved styling
		fig.update_layout(
			title={
				'text': f'Impedance Components vs. Frequency - {visit_date}',
				'font': dict(size=18)
			},
			xaxis_title='Frequency (Hz)',
			yaxis_title='Impedance (Ohm)',
			xaxis=dict(
				type='log',  # Log scale for frequency
				title_font=dict(size=14),
				gridcolor='rgba(211, 211, 211, 0.3)'
			),
			yaxis=dict(
				title_font=dict(size=14),
				gridcolor='rgba(211, 211, 211, 0.3)'
			),
			hovermode='closest',
			showlegend=True,
			legend=dict(
				yanchor="top",
				y=0.99,
				xanchor="right",
				x=0.99,
				bgcolor='rgba(255, 255, 255, 0.8)'
			),
			height=450,
			margin=dict(l=50, r=50, t=80, b=50),
			paper_bgcolor='white',
			plot_bgcolor='white',
			font=dict(family="Arial, sans-serif")
		)

		# Add range slider for better interactivity
		fig.update_layout(
			xaxis=dict(
				rangeslider=dict(visible=True),
				type='log'
			)
		)

		return fig

	def _create_nyquist_plot(self, df: pd.DataFrame, visit_date: str) -> go.Figure:
		"""
		Create a Nyquist plot (Real vs. Imaginary components).

		Args:
			df: DataFrame containing frequency sweep data
			visit_date: Date of the visit

		Returns:
			Plotly Figure object with the Nyquist plot
		"""
		# Create figure
		fig = go.Figure()

		# Add Nyquist plot (Real vs. Imaginary)
		fig.add_trace(go.Scatter(
			x=df[ExcelSheetColumns.REAL.value],
			y=df[ExcelSheetColumns.IMAGINARY.value],
			mode='lines+markers',
			name='Nyquist Plot',
			line=dict(color='purple', width=2),
			marker=dict(
				size=8,
				color=df[ExcelSheetColumns.FREQ.value],
				colorscale='Viridis',
				colorbar=dict(title='Frequency (Hz)'),
				showscale=True
			),
			hovertemplate=(
				"<b>Real:</b> %{x:.2f} Ohm<br>"
				"<b>Imaginary:</b> %{y:.2f} Ohm<br>"
				"<b>Frequency:</b> %{marker.color:.2f} Hz<br>"
			)
		))

		# Configure layout
		fig.update_layout(
			title={
				'text': f'Nyquist Plot - {visit_date}',
				'font': dict(size=16)
			},
			xaxis_title="Real Component (Z') [Ohm]",
			yaxis_title="Imaginary Component (-Z'') [Ohm]",
			xaxis=dict(
				title_font=dict(size=14),
				gridcolor='rgba(211, 211, 211, 0.3)'
			),
			yaxis=dict(
				title_font=dict(size=14),
				gridcolor='rgba(211, 211, 211, 0.3)'
			),
			hovermode='closest',
			height=400,
			margin=dict(l=50, r=50, t=80, b=50),
			paper_bgcolor='white',
			plot_bgcolor='white',
			font=dict(family="Arial, sans-serif")
		)

		return fig

	def _display_impedance_metrics(self, df: pd.DataFrame, visit_date: str) -> None:
		"""
		Display key metrics from the impedance data.

		Args:
			df: DataFrame containing frequency sweep data
			visit_date: Date of the visit
		"""
		try:
			# Inject custom CSS to adjust st.metric font sizes
			st.markdown(
				"""
				<style>
				[data-testid="stMetricValue"] {
					font-size: 14px !important;
				}
				[data-testid="stMetricLabel"] {
					font-size: 12px !important;
					font-weight: bold !important;
				}
				</style>
				""", unsafe_allow_html=True)

			# Calculate key metrics
			metrics = {}

			# Frequency range
			metrics['min_freq'] = df[ExcelSheetColumns.FREQ.value].min()
			metrics['max_freq'] = df[ExcelSheetColumns.FREQ.value].max()

			# Impedance ranges
			metrics['min_real'] = df[ExcelSheetColumns.REAL.value].min()
			metrics['max_real'] = df[ExcelSheetColumns.REAL.value].max()
			metrics['min_imag'] = df[ExcelSheetColumns.IMAGINARY.value].min()
			metrics['max_imag'] = df[ExcelSheetColumns.IMAGINARY.value].max()

			# Calculate phase angle if negative phase is available
			if ExcelSheetColumns.NEG_PHASE.value in df.columns:
				# Find frequency with maximum phase angle
				max_phase_idx = df[ExcelSheetColumns.NEG_PHASE.value].idxmax()
				metrics['max_phase'] = df.loc[max_phase_idx, ExcelSheetColumns.NEG_PHASE.value]
				metrics['max_phase_freq'] = df.loc[max_phase_idx, ExcelSheetColumns.FREQ.value]

			# Display metrics
			st.metric("Frequency Range", f"{metrics['min_freq']:.1f} - {metrics['max_freq']:.1f} Hz")
			st.metric("Real Impedance Range", f"{metrics['min_real']:.1f} - {metrics['max_real']:.1f} Ohm")
			st.metric("Imaginary Impedance Range", f"{metrics['min_imag']:.1f} - {metrics['max_imag']:.1f} Ohm")

			if 'max_phase' in metrics:
				st.metric("Max Phase Angle", f"{metrics['max_phase']:.2f}°")
				st.metric("Frequency at Max Phase", f"{metrics['max_phase_freq']:.1f} Hz")

			# Add data points count
			st.metric("Data Points", f"{len(df)}")

		except Exception as e:
			logger.error(f"Error calculating impedance metrics: {str(e)}")
			st.warning("Could not calculate impedance metrics due to an error.")
