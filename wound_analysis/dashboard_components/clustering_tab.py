# Standard library imports
import logging
import traceback
from typing import Dict, List, Literal, Union

# Third-party imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
	and individual patient data, including clustering, correlation analysis, and visualization.
	and clinical interpretations.
	"""

	def __init__(self, wound_data_processor: WoundDataProcessor, selected_patient: str):

		# Input Variables
		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)
		self.selected_patient = selected_patient

		# Calculated Variables
		self.df_w_cluster_tags = None
		self.selected_cluster  = None


		# Clustering User Settings
		self._cluster_settings    = None
		self.use_cluster_data     = False

		# Initialize session state for clusters if not already present
		if 'cluster_tags' not in st.session_state:
			st.session_state.cluster_tags       = None
			st.session_state.df_w_cluster_tags  = None
			st.session_state.feature_importance = None
			st.session_state.selected_cluster   = None


	def render(self) -> 'ClusteringTab':
		"""
		Render the clustering section and perform clustering if requested.

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

		def _display_cluster_analysis():
			cols = st.columns([1, 4])

			with cols[0]:
				self._get_user_selected_cluster()


			tabs = st.tabs(["Cluster Distribution", "Feature Importance", "Cluster Characteristics"])

			with tabs[0]:
				# Display cluster distribution
				ClusteringTab._display_cluster_distribution(df_w_cluster_tags=st.session_state.df_w_cluster_tags)


			with tabs[1]:
				# Display feature importance
				if st.session_state.feature_importance:
					ClusteringTab._display_feature_importance(feature_importance=st.session_state.feature_importance)

			with tabs[2]:
				# Display cluster characteristics
				self._display_cluster_characteristics(cluster_df=st.session_state.df_w_cluster_tags)


		try:

			# with st.expander("Patient Data Clustering", expanded=True):

			self._cluster_settings = self.get_cluster_analysis_settings()

			# Run clustering if requested
			if self._cluster_settings["run_clustering"] and self._cluster_settings["cluster_features"]:

				ca = ClusteringTab._perform_clustering(df=self.df, cluster_settings=self._cluster_settings)

				# Store clustering results in session state
				st.session_state.cluster_tags       = ca.cluster_tags
				st.session_state.df_w_cluster_tags  = ca.df_w_cluster_tags
				st.session_state.feature_importance = ca.feature_importance
				st.session_state.selected_cluster   = None # Reset selected cluster

			if st.session_state.df_w_cluster_tags is not None:
				_display_cluster_analysis()
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
	def _perform_clustering(df: pd.DataFrame, cluster_settings: Dict[str, Union[int, str, List[str], bool]]) -> ClusteringAnalysis:
		"""
		Perform clustering on the data using the specified method and features.
		"""

		ca = ClusteringAnalysis(df=df.copy(), cluster_settings=cluster_settings).render()


		# Display success message
		st.success(f"Successfully clustered data into {cluster_settings['n_clusters']} clusters using {cluster_settings['clustering_method']}!")

		return ca


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


