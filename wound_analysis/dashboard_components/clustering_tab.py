# Standard library imports
import logging
import traceback
from typing import Dict, List, Literal, Union, Optional, Tuple, Any

# Third-party imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io

# Local application imports
from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.statistical_analysis import ClusteringAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringTab:
	"""
	A class for managing and rendering the Clustering tab in the wound analysis dashboard.

	This class contains methods to display clustering analysis for both population-level data
	and individual patient data, including clustering, correlation analysis, visualization,
	and clinical interpretations. It also includes SHAP (SHapley Additive exPlanations) analysis
	for model interpretability.

	Attributes:
		df (pd.DataFrame): The input DataFrame containing wound data
		CN (DColumns): Column name schema for the DataFrame
		selected_patient (str): Currently selected patient ID
		df_w_cluster_tags (Optional[pd.DataFrame]): DataFrame with cluster assignments
		selected_cluster (Optional[int]): Currently selected cluster for analysis
		use_cluster_data (bool): Whether to use clustered data in other tabs
		_cluster_settings (Optional[Dict]): User-selected clustering parameters
	"""

	def __init__(self, wound_data_processor: WoundDataProcessor, selected_patient: str):

		# Input Variables
		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)
		self.selected_patient = selected_patient

		# Calculated Variables
		self.df_w_cluster_tags: Optional[pd.DataFrame] = None
		self.selected_cluster: Optional[int] = None
		self._cluster_settings: Optional[Dict[str, Any]] = None
		self.use_cluster_data: bool = False

		# Initialize session state for clusters if not already present
		if 'cluster_tags' not in st.session_state:
			st.session_state.cluster_tags       = None
			st.session_state.df_w_cluster_tags  = None
			st.session_state.feature_importance = None
			st.session_state.selected_cluster   = None
			st.session_state.shap_values        = None
			st.session_state.shap_expected_values = None


	def _display_cluster_analysis(self) -> None:
		cols = st.columns([1, 4])

		with cols[0]:
			self._get_user_selected_cluster()


		tabs = st.tabs(["Cluster Distribution", "Feature Importance", "SHAP Analysis", "Cluster Characteristics"])

		with tabs[0]:
			# Display cluster distribution
			ClusteringTab._display_cluster_distribution(df_w_cluster_tags=st.session_state.df_w_cluster_tags)


		with tabs[1]:
			# Display feature importance
			if st.session_state.feature_importance:
				ClusteringTab._display_feature_importance(feature_importance=st.session_state.feature_importance)

		with tabs[2]:
			if st.session_state.shap_values is not None:
				self._display_shap_analysis()

		with tabs[3]:
			# Display cluster characteristics
			self._display_cluster_characteristics(cluster_df=st.session_state.df_w_cluster_tags)


	def render(self) -> 'ClusteringTab':
		"""
		Render the clustering section and perform clustering if requested.

		Returns:
			ClusteringTab: The current instance for method chaining
		"""

		def _use_cluster_data():

			st.markdown("---")
			btn_use_cluster_data = st.checkbox(f"Use cluster subset in other tabs", value=False)

			if btn_use_cluster_data:
				self.use_cluster_data = False

				if st.session_state.df_w_cluster_tags is None:
					st.error("Please run clustering first")

				elif st.session_state.selected_cluster == "All Data":
					st.error("Please select a cluster first")

				else:
					st.success(f"Using cluster {st.session_state.selected_cluster} subset in other tabs")
					self.use_cluster_data = True


		try:
			self._cluster_settings = self.get_cluster_analysis_settings()

			# Run clustering if requested
			if self._cluster_settings["run_clustering"] and self._cluster_settings["cluster_features"]:

				ca = ClusteringAnalysis(df=self.df.copy(), cluster_settings=self._cluster_settings).render()

				# Calculate SHAP values for the clustering results
				shap_values, expected_values = self._calculate_shap_values( df=ca.df_w_cluster_tags, features=self._cluster_settings["cluster_features"] )

				st.success(f"Successfully clustered data into {self._cluster_settings['n_clusters']} clusters!")

				# Store results in session state
				st.session_state.cluster_tags         = ca.cluster_tags
				st.session_state.df_w_cluster_tags    = ca.df_w_cluster_tags
				st.session_state.feature_importance   = ca.feature_importance
				st.session_state.shap_values          = shap_values
				st.session_state.shap_expected_values = expected_values
				st.session_state.selected_cluster     = None # Reset selected cluster

			if st.session_state.df_w_cluster_tags is not None:
				self._display_cluster_analysis()
				_use_cluster_data()


		except Exception as e:
			st.error(f"Error during clustering: {str(e)}")
			st.error(traceback.format_exc())

		return self


	def get_cluster_analysis_settings(self) -> Dict[str, Union[int, str, List[str], bool]]:
		"""
		Get the cluster analysis settings from the user.
		"""

		with st.expander("Select features for clustering by category:"):
			feature_categories = self.CN.get_clean_names()

			# Combine all feature display names with a dictionary comprehension
			feature_display_names = {key: value for cat in feature_categories.values() for key, value in cat.items()}

			st.markdown("#### Select features for clustering by category:")
			defaults = {
				"Wound Characteristics": [self.CN.WOUND_AREA, self.CN.HEALING_RATE],
				"Impedance"            : [self.CN.HIGHEST_FREQ_ABSOLUTE],
			}

			selected_features = []
			for category, features in feature_categories.items():
				default_selections = defaults.get(category, [])
				selected = st.multiselect(
					category,
					options=list(features.keys()),
					default=default_selections,
					format_func=lambda x: feature_display_names.get(x, x)
				)
				selected_features.extend(selected)

			if not selected_features:
				selected_features = [self.CN.HIGHEST_FREQ_ABSOLUTE, self.CN.WOUND_AREA, self.CN.HEALING_RATE]
				st.info("No features selected. Using default features.")

			cluster_features = selected_features

		cols = st.columns([1, 1, 1], gap="large")

		with cols[0]:

			n_clusters = st.number_input( "Number of Clusters", min_value=2, max_value=10, value=3,
					help="Select the number of clusters to divide patient data into" )

		with cols[1]:
			clustering_method = st.selectbox(
				"Clustering Method",
				options = ["K-Means", "Hierarchical", "DBSCAN"],
				index   = 0,
				help    = "Select the clustering algorithm to use"
			)

		with cols[2]:
			st.markdown("")
			run_clustering = st.button("Run Clustering")


		return {"n_clusters"       : n_clusters,
				"cluster_features" : cluster_features,
				"clustering_method": clustering_method,
				"run_clustering"   : run_clustering}


	@staticmethod
	def _display_cluster_distribution(df_w_cluster_tags: pd.DataFrame) -> None:
		"""
		Display the distribution of data points across clusters.

		Args:
			df_w_cluster_tags: DataFrame with cluster assignments
		"""
		cluster_counts = df_w_cluster_tags['Cluster'].value_counts().sort_index()

		# Filter out noise points (label -1) for visualization
		if -1 in cluster_counts:
			noise_count    = cluster_counts[-1]
			cluster_counts = cluster_counts[cluster_counts.index >= 0]
			st.info(f"Note: {noise_count} points were classified as noise (only applies to DBSCAN)")

		# Create a bar chart for cluster sizes
		fig = px.bar(
			x      = cluster_counts.index,
			y      = cluster_counts.values,
			labels = {'x': 'Cluster', 'y': 'Number of Patients/Visits'},
			title  = "Cluster Distribution",
			color  = cluster_counts.index,
			text   = cluster_counts.values
		)

		fig.update_traces(textposition='outside')
		fig.update_layout(showlegend=False)
		st.plotly_chart(fig, use_container_width=True)


	@staticmethod
	def _display_feature_importance(feature_importance: Dict[str, float]) -> None:
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


	def get_updated_df(self) -> pd.DataFrame:
		"""
		Get the DataFrame for the selected cluster.
		"""

		cluster_id = st.session_state.selected_cluster
		df         = st.session_state.df_w_cluster_tags

		if not self.use_cluster_data:
			return df if (df is not None and not df.empty) else self.df

		elif df is None:
			st.warning("Please run clustering first by selecting features and clicking 'Run Clustering'.")
			return self.df

		return df[df['Cluster'] == cluster_id].copy() if (cluster_id is not None) else df


	def _get_user_selected_cluster(self) -> int | Literal["All Data"]:
		"""
		Render the cluster selection dropdown and display cluster characteristics.
		"""
		if st.session_state.cluster_tags is not None and st.session_state.df_w_cluster_tags is not None:

			# Create selection for which cluster to analyze
			# st.markdown("### Cluster Selection")

			# Create a mapping of display text to actual value
			cluster_options = {"All Data": "All Data"}

			df           = st.session_state.df_w_cluster_tags
			cluster_tags = st.session_state.cluster_tags

			for cluster_id in sorted([c for c in cluster_tags if c >= 0]):
				# Get the count of patients in the cluster
				cluster_count = len(df[df['Cluster'] == cluster_id])

				# Add the cluster to the options
				cluster_options[ f"Cluster {cluster_id} (n={cluster_count})" ] = cluster_id


			# Get display options for the dropdown
			display_options = list(cluster_options.keys())

			# Render cluster selection dropdown with formatted display options
			selected_display = st.radio("Select cluster to analyze:", options=display_options, index=0)

			# Update the selected cluster in session state
			st.session_state.selected_cluster = cluster_options[selected_display]


	def _display_cluster_characteristics(self, cluster_df: pd.DataFrame) -> None:
		"""
		Display characteristics of the selected cluster compared to the overall population.

		Args:
			cluster_df: DataFrame filtered to the selected cluster
		"""

		if cluster_df.empty or cluster_df is None:
			st.warning("No cluster data available. Please run clustering first.")
			return

		full_df          = st.session_state.df_w_cluster_tags
		cluster_features = self._cluster_settings["cluster_features"]


		st.markdown(f"### Characteristics of Cluster {st.session_state.selected_cluster}")

		# Create summary statistics for this cluster vs. overall population
		summary_stats = []

		for feature in cluster_features:
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


	def _calculate_shap_values(
		self,
		df: pd.DataFrame,
		features: List[str]
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Calculate SHAP values for the clustering results.

		Args:
			df: DataFrame containing the features used for clustering
			features: List of feature names used in clustering

		Returns:
			Tuple containing SHAP values and expected values
		"""
		logger.info("Calculating SHAP values for cluster interpretation")

		try:
			# Prepare the feature matrix
			X = df[features].copy()

			# Handle missing values
			X = X.fillna(X.mean())

			# Standardize the features
			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)

			# Calculate number of background samples (kmeans clusters)
			# Use min(n_samples/2, 10) to ensure we don't exceed data size
			n_background = min(len(X) // 2, 10)
			n_background = max(n_background, 1)  # Ensure at least 1 cluster

			# Create a KernelExplainer for the clustering model
			background = shap.kmeans(X_scaled, n_background)
			explainer = shap.KernelExplainer(
				lambda x: x,  # Identity function since we want to explain the features directly
				background
			)

			# Calculate SHAP values
			shap_values = explainer.shap_values(X_scaled)
			expected_values = explainer.expected_value

			logger.info("Successfully calculated SHAP values")
			return shap_values, expected_values

		except Exception as e:
			logger.error(f"Error calculating SHAP values: {str(e)}")
			logger.error(traceback.format_exc())
			raise


	def _ensure_2d_shap(self, shap_values):
		"""
		Ensure SHAP values are 2D (n_samples, n_features).
		Handles cases where SHAP returns a list or 3D array.
		"""
		# If it's a list of arrays (e.g., multiclass), take the first
		if isinstance(shap_values, list):
			shap_values = shap_values[0]
		# If it's 3D, reduce to 2D (e.g., take the first output)
		if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
			shap_values = shap_values[:, :, 0]
		return shap_values


	def _display_shap_analysis(self) -> None:
		"""Display SHAP analysis results using Streamlit."""
		st.markdown("### SHAP Analysis")
		st.markdown("""
		SHAP (SHapley Additive exPlanations) values show how each feature contributes to moving a sample
		from the expected model output to its actual prediction. This helps understand which features are
		most important for each cluster's formation.
		""")

		# try:
		if self._cluster_settings is None:
			st.info("Please run clustering first to view SHAP analysis")
			return

		features = self._cluster_settings["cluster_features"]
		df = st.session_state.df_w_cluster_tags[features]
		shap_values = self._ensure_2d_shap(st.session_state.shap_values)

		if st.session_state.selected_cluster == "All Data":
			# For "All Data", show the overall SHAP value distribution
			st.markdown("#### Overall SHAP Value Distribution")
			st.markdown("""
			This view shows the distribution of SHAP values across all clusters, helping you understand
			how each feature contributes to cluster formation across the entire dataset.
			""")

			# Calculate mean absolute SHAP values for each feature
			mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

			# Create a bar plot of overall feature importance
			fig = go.Figure()
			fig.add_trace(go.Bar(
				y=features,
				x=mean_abs_shap,
				orientation='h'
			))

			fig.update_layout(
				title="Overall Feature Importance (Mean |SHAP|)",
				xaxis_title="Mean |SHAP Value|",
				yaxis_title="Feature"
			)

			st.plotly_chart(fig, use_container_width=True)

			# SHAP summary plot (beeswarm)
			st.markdown("#### SHAP Summary Plot (Beeswarm)")
			buf = io.BytesIO()
			plt.figure(figsize=(8, 4))
			shap.summary_plot(shap_values, df, feature_names=features, show=False)
			plt.tight_layout()
			plt.savefig(buf, format="png")
			plt.close()
			st.image(buf, caption="SHAP Summary Plot (Beeswarm)")

			st.markdown("""
			**Interpretation:**
			- Larger values indicate features that have a stronger overall influence on cluster formation
			- This view helps identify which features are most important for distinguishing between any clusters
			- The absolute values show magnitude of impact, regardless of direction
			- The beeswarm plot shows the distribution of SHAP values for each feature across all samples
			""")

			# Per-sample SHAP value table
			st.markdown("#### Per-Sample SHAP Values (All Data)")
			shap_df = pd.DataFrame(shap_values, columns=features, index=df.index)
			st.dataframe(shap_df)
			st.download_button("Download SHAP Values (CSV)", shap_df.to_csv().encode(), file_name="shap_values_all_data.csv")

		elif st.session_state.selected_cluster is not None:
			# For specific clusters, show cluster-specific SHAP values
			st.markdown(f"#### SHAP Values for Cluster {st.session_state.selected_cluster}")
			st.markdown("""
			This view shows how each feature contributes to forming this specific cluster,
			helping you understand what makes this cluster unique.
			""")

			cluster_mask = st.session_state.df_w_cluster_tags['Cluster'] == st.session_state.selected_cluster
			cluster_shap = self._ensure_2d_shap(shap_values[cluster_mask])
			cluster_df = df[cluster_mask]

			# Calculate mean SHAP values for the selected cluster
			mean_shap = np.mean(cluster_shap, axis=0)

			# Create a bar plot of SHAP values
			fig = go.Figure()
			fig.add_trace(go.Bar(
				y=features,
				x=mean_shap,
				orientation='h'
			))

			fig.update_layout(
				title=f"Mean SHAP Values for Cluster {st.session_state.selected_cluster}",
				xaxis_title="SHAP Value",
				yaxis_title="Feature"
			)

			st.plotly_chart(fig, use_container_width=True)

			# SHAP summary plot (beeswarm) for the cluster
			st.markdown(f"#### SHAP Summary Plot (Beeswarm) for Cluster {st.session_state.selected_cluster}")
			buf = io.BytesIO()
			plt.figure(figsize=(8, 4))
			shap.summary_plot(cluster_shap, cluster_df, feature_names=features, show=False)
			plt.tight_layout()
			plt.savefig(buf, format="png")
			plt.close()
			st.image(buf, caption=f"SHAP Summary Plot (Beeswarm) for Cluster {st.session_state.selected_cluster}")

			st.markdown("""
			**Interpretation:**
			- Positive SHAP values (bars to the right) indicate features that push samples toward this cluster
			- Negative values (bars to the left) indicate features that push samples away from this cluster
			- The magnitude of the bar shows the strength of the feature's influence
			- The beeswarm plot shows the distribution of SHAP values for each feature in this cluster
			""")

			# Per-sample SHAP value table for the cluster
			st.markdown(f"#### Per-Sample SHAP Values (Cluster {st.session_state.selected_cluster})")
			cluster_shap_df = pd.DataFrame(cluster_shap, columns=features, index=cluster_df.index)
			st.dataframe(cluster_shap_df)
			st.download_button(f"Download SHAP Values for Cluster {st.session_state.selected_cluster} (CSV)", cluster_shap_df.to_csv().encode(), file_name=f"shap_values_cluster_{st.session_state.selected_cluster}.csv")

		else:
			st.info("Please select a cluster to view SHAP analysis")

		# except Exception as e:
		#     logger.error(f"Error displaying SHAP analysis: {str(e)}")
		#     logger.error(traceback.format_exc())
		#     st.error("Error displaying SHAP analysis. Please check the logs for details.")


