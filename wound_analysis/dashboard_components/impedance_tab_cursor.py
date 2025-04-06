# Standard library imports
import traceback
from typing import List, Dict, Any
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


# Helper functions for styling (can be moved to a utils file if preferred)
def _highlight_significant(row: pd.Series) -> List[str]:
	"""Highlights rows in a DataFrame if the 'Significant' column is True."""
	is_significant = row.get('Significant', False)
	return ['background-color: yellow' if is_significant else '' for _ in range(len(row))]

def _color_visit_change_cells(val: Any) -> str:
	"""Colors cell text based on positive (red) or negative (green) value."""
	try:
		if val is None or val == "N/A":
			return ''
		num_str = str(val).replace('*', '').replace('%', '')
		num_val = float(num_str)
		return 'color: #FF4B4B' if num_val > 0 else 'color: #00CC96'
	except Exception as e:
		logger.error(f"Error in _color_visit_change_cells: {e}")
		return ''

# --- Population Renderer Class ---

class PopulationImpedanceRenderer:
	"""Handles rendering logic for the 'All Patients' population view."""

	def __init__(self, df: pd.DataFrame):
		self.df = df
		self.CN = DColumns(df=df)
		self.analysis_df = df.copy() # Work on a copy

	def render(self) -> None:
		"""Renders the entire population analysis section."""
		cluster_features = self._render_clustering_controls()
		working_df = self._render_cluster_analysis(cluster_features)
		filtered_df = self._render_correlation_analysis(working_df)
		self._render_population_plots(filtered_df, working_df)

	def _render_clustering_controls(self) -> List[str]:
		"""Renders controls for clustering and handles clustering execution."""
		with st.expander("Patient Data Clustering", expanded=True):
			st.markdown("### Cluster Analysis Settings")
			col1, col2, col3 = st.columns([1, 2, 1])

			with col1:
				n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3, key="pop_n_clusters", help="Select the number of clusters")
			with col2:
				available_features = [
					self.CN.HIGHEST_FREQ_Z, self.CN.WOUND_AREA, self.CN.CENTER_TEMP,
					self.CN.OXYGENATION, self.CN.HEMOGLOBIN, self.CN.AGE, self.CN.BMI,
					self.CN.DAYS_SINCE_FIRST_VISIT, self.CN.HEALING_RATE
				]
				valid_options = [f for f in available_features if f in self.df.columns]
				default_selection = [f for f in [self.CN.HIGHEST_FREQ_Z, self.CN.WOUND_AREA, self.CN.HEALING_RATE] if f in valid_options]

				cluster_features = st.multiselect(
					"Features for Clustering", options=valid_options, default=default_selection,
					key="pop_cluster_features", help="Select features for clustering"
				)
			with col3:
				clustering_method = st.selectbox(
					"Clustering Method", options=["K-Means", "Hierarchical", "DBSCAN"], index=0,
					key="pop_clustering_method", help="Select clustering algorithm"
				)
				run_clustering = st.button("Run Clustering", key="pop_run_clustering")

		# Initialize session state if needed
		if 'pop_clusters' not in st.session_state:
			st.session_state.pop_clusters = None
			st.session_state.pop_cluster_df = None
			st.session_state.pop_feature_importance = None

		if run_clustering and len(cluster_features) > 0:
			self._run_clustering(n_clusters, cluster_features, clustering_method)

		return cluster_features # Return selected features for later use

	def _run_clustering(self, n_clusters: int, cluster_features: List[str], clustering_method: str):
		"""Performs the clustering calculation and updates session state."""
		try:
			clustering_df = self.analysis_df[cluster_features].copy()
			clustering_df = clustering_df.fillna(clustering_df.mean()).dropna()

			if len(clustering_df) <= n_clusters:
				st.error("Not enough valid data points for clustering.")
				return

			valid_indices = clustering_df.index
			scaler = StandardScaler()
			scaled_features = scaler.fit_transform(clustering_df)

			cluster_labels, feature_importance = self._perform_clustering_algorithm(
				scaled_features, clustering_method, n_clusters, cluster_features
			)

			cluster_mapping = pd.Series(cluster_labels, index=valid_indices)
			self.analysis_df['Cluster'] = cluster_mapping
			self.analysis_df['Cluster'] = self.analysis_df['Cluster'].fillna(-1).astype(int)

			st.session_state.pop_clusters = sorted(self.analysis_df['Cluster'].unique())
			st.session_state.pop_cluster_df = self.analysis_df.copy() # Store clustered df
			st.session_state.pop_feature_importance = feature_importance
			st.session_state.pop_selected_cluster = None # Reset selected cluster view

			actual_clusters = len([c for c in st.session_state.pop_clusters if c >= 0])
			st.success(f"Successfully clustered data into {actual_clusters} clusters using {clustering_method}!")
			self._display_clustering_results(self.analysis_df, feature_importance)

		except Exception as e:
			st.error(f"Error during clustering: {str(e)}")
			st.error(traceback.format_exc())

	def _perform_clustering_algorithm(self, scaled_features: np.ndarray, method: str, n_clusters: int, features: List[str]) -> (np.ndarray, Dict[str, float]):
		"""Applies the chosen clustering algorithm and calculates feature importance."""
		feature_importance = {}
		cluster_labels = np.array([-1] * len(scaled_features)) # Default to noise

		if method == "K-Means":
			clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
			cluster_labels = clusterer.fit_predict(scaled_features)
			centers = clusterer.cluster_centers_
			for i, feature in enumerate(features):
				variance = np.var([center[i] for center in centers])
				feature_importance[feature] = variance

		elif method == "Hierarchical":
			Z = linkage(scaled_features, 'ward')
			cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
			for i, feature in enumerate(features):
				single_feature = scaled_features[:, i:i+1]
				score = 0.01 # Default
				if len(np.unique(single_feature)) > 1:
					try:
						temp_clusters = fcluster(linkage(single_feature, 'ward'), n_clusters, criterion='maxclust')
						score = max(0, silhouette_score(single_feature, temp_clusters))
					except Exception as e:
						logger.error(f"Error calculating cluster silhouette score: {e}")

				feature_importance[feature] = score

		elif method == "DBSCAN":
			neigh = NearestNeighbors(n_neighbors=3).fit(scaled_features)
			distances, _ = neigh.kneighbors(scaled_features)
			distances = np.sort(distances[:, 2], axis=0)
			epsilon = float(np.percentile(distances, 90)) if len(distances) > 0 else 0.5
			min_samples = max(3, len(scaled_features)//30) if len(scaled_features) > 0 else 3

			clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)
			cluster_labels = clusterer.fit_predict(scaled_features)
			# DBSCAN Feature Importance (using inverse variance within clusters)
			for i, feature in enumerate(features):
				variances = []
				for label in set(cluster_labels):
					if label >= 0:
						cluster_data = scaled_features[cluster_labels == label, i]
						if len(cluster_data) > 1:
							variances.append(np.var(cluster_data))

				if variances and np.var(scaled_features[:, i]) > 1e-6 : # Avoid division by zero
					feature_importance[feature] = 1.0 - min(1.0, np.mean(variances) / np.var(scaled_features[:, i]))
				else:
					feature_importance[feature] = 0.01 # Default

		# Normalize feature importance
		max_imp = max(feature_importance.values()) if feature_importance else 1.0
		if max_imp > 0:
			feature_importance = {k: v / max_imp for k, v in feature_importance.items()}

		return cluster_labels, feature_importance

	def _display_clustering_results(self, clustered_df: pd.DataFrame, feature_importance: Dict[str, float]):
		"""Displays charts related to clustering results."""
		cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
		if -1 in cluster_counts:
			noise_count = cluster_counts[-1]
			cluster_counts = cluster_counts[cluster_counts.index >= 0]
			st.info(f"Note: {noise_count} points classified as noise (DBSCAN).")

		if len(cluster_counts) > 0:
			fig_dist = px.bar(
				x=cluster_counts.index, y=cluster_counts.values,
				labels={'x': 'Cluster', 'y': 'Number of Data Points'}, title="Cluster Distribution",
				color=cluster_counts.index.astype(str), text=cluster_counts.values
			)
			fig_dist.update_traces(textposition='outside')
			fig_dist.update_layout(showlegend=False, coloraxis_showscale=False)
			st.plotly_chart(fig_dist, use_container_width=True)

		if feature_importance:
			fig_imp = go.Figure()
			fig_imp.add_trace(go.Scatterpolar(
				r=list(feature_importance.values()), theta=list(feature_importance.keys()),
				fill='toself', name='Importance'
			))
			fig_imp.update_layout(
				title="Feature Importance in Clustering",
				polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False
			)
			st.plotly_chart(fig_imp, use_container_width=True)

	def _render_cluster_analysis(self, cluster_features: List[str]) -> pd.DataFrame:
		"""Renders cluster selection and characteristics analysis."""
		working_df = self.analysis_df # Default to full df

		if st.session_state.get('pop_clusters') is not None and st.session_state.get('pop_cluster_df') is not None:
			st.markdown("### Cluster Selection and Analysis")
			cluster_df = st.session_state.pop_cluster_df
			cluster_options = ["All Data"]
			for cluster_id in sorted([c for c in st.session_state.pop_clusters if c >= 0]):
				count = len(cluster_df[cluster_df['Cluster'] == cluster_id])
				cluster_options.append(f"Cluster {cluster_id} (n={count})")

			selected_option = st.selectbox(
				"Select data subset to analyze:", options=cluster_options, index=0,
				key="pop_select_cluster_subset"
			)

			if selected_option == "All Data":
				st.session_state.pop_selected_cluster = None
				working_df = self.analysis_df # Use original potentially unclustered df if "All Data" selected before clustering
			else:
				cluster_id = int(selected_option.split(" ")[1])
				st.session_state.pop_selected_cluster = cluster_id
				working_df = cluster_df[cluster_df['Cluster'] == cluster_id].copy()

				st.markdown(f"#### Characteristics of Cluster {cluster_id}")
				summary_stats = []
				base_df_for_stats = self.analysis_df # Compare against original potentially unclustered df

				for feature in cluster_features:
					if feature in working_df.columns and feature in base_df_for_stats.columns:
						try:
							cluster_mean = working_df[feature].mean()
							overall_mean = base_df_for_stats[feature].mean()
							diff_pct = ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
							summary_stats.append({
								"Feature": feature, "Cluster Mean": f"{cluster_mean:.2f}",
								"Population Mean": f"{overall_mean:.2f}", "Difference": f"{diff_pct:+.1f}%",
								"Significant": abs(diff_pct) > 15
							})
						except Exception as e:
							logger.error(f"Error calculating cluster stats for {feature}: {e}")


				if summary_stats:
					summary_df = pd.DataFrame(summary_stats)
					styled_df = summary_df.style.apply(_highlight_significant, axis=1).hide(axis="columns", names=["Significant"])
					st.table(styled_df)
					st.caption("Highlighted rows: Feature mean differs >15% from overall population mean.")
				else:
					st.info("Could not compute cluster characteristics.")

		return working_df

	def _render_correlation_analysis(self, df_subset: pd.DataFrame) -> pd.DataFrame:
		"""Displays correlation controls and statistical analysis."""
		st.markdown("### Correlation and Statistical Analysis")
		col1, col2 = st.columns([1, 3])
		with col1:
			outlier_threshold = st.number_input(
				"Outlier Threshold (Quantile)", min_value=0.0, max_value=0.49, value=0.0, step=0.05,
				key="pop_outlier_threshold", help="0=no removal, 0.1=remove bottom/top 10%"
			)

		features_to_analyze = [
			f for f in [self.CN.HIGHEST_FREQ_Z, self.CN.WOUND_AREA, self.CN.CENTER_TEMP,
						self.CN.OXYGENATION, self.CN.HEMOGLOBIN, self.CN.HEALING_RATE]
			if f in df_subset.columns
		]

		if not features_to_analyze:
			st.warning("No features available for correlation analysis in the selected data subset.")
			return pd.DataFrame()

		corr_df: pd.DataFrame = df_subset[features_to_analyze].copy().dropna()

		# Remove outliers
		if outlier_threshold > 0 and not corr_df.empty:
			for col in corr_df.columns:
				q_low = corr_df[col].quantile(outlier_threshold)
				q_high = corr_df[col].quantile(1 - outlier_threshold)
				corr_df = corr_df[(corr_df[col] >= q_low) & (corr_df[col] <= q_high)]

		if corr_df.empty or len(corr_df) < 2:
			st.warning("Not enough data after outlier removal for correlation analysis.")
			return pd.DataFrame()

		# Correlation and p-values
		corr_matrix = corr_df.corr()
		p_values = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)
		for col1 in corr_df.columns:
			for col2 in corr_df.columns:
				if col1 != col2:
					res = stats.pearsonr(corr_df[col1], corr_df[col2])
					p_values.loc[col1, col2] = res.pvalue

		# --- Display ---
		# Heatmap
		fig_corr = px.imshow(
			corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu",
			labels=dict(color="Correlation"), title="Feature Correlation Matrix"
		)
		st.plotly_chart(fig_corr, use_container_width=True)

		# Stats Tabs
		tab1, tab2, tab3 = st.tabs(["Correlation Details", "Descriptive Stats", "Effect Sizes (vs Impedance)"])
		with tab1:
			st.markdown("#### Significant Correlations (p < 0.05)")
			sig_corrs = []
			for r in range(len(features_to_analyze)):
				for c in range(r + 1, len(features_to_analyze)):
					pval = p_values.iloc[r, c]
					if pval < 0.05:
						feature1 = features_to_analyze[r]
						feature2 = features_to_analyze[c]
						sig_corrs.append({
							"Feature 1": feature1, "Feature 2": feature2,
							"Correlation": f"{corr_matrix.iloc[r, c]:.3f}",
							"p-value": f"{pval:.3e}"
						})
			st.dataframe(pd.DataFrame(sig_corrs) if sig_corrs else "No significant correlations found.")

		with tab2:
			st.markdown("#### Descriptive Statistics")
			desc_stats = corr_df.describe()
			desc_stats.loc["skew"] = corr_df.skew()
			desc_stats.loc["kurtosis"] = corr_df.kurtosis()
			st.dataframe(desc_stats)

		with tab3:
			st.markdown("#### Effect Sizes (Cohen's d) Relative to Impedance")
			impedance_col = self.CN.HIGHEST_FREQ_Z
			if impedance_col in corr_df.columns and len(corr_df) > 1:
				effect_sizes = []
				imp_mean = corr_df[impedance_col].mean()
				imp_std = corr_df[impedance_col].std()
				n = len(corr_df)

				for col in corr_df.columns:
					if col != impedance_col:
						col_mean = corr_df[col].mean()
						col_std = corr_df[col].std()
						pooled_std = np.sqrt(((n-1)*imp_std**2 + (n-1)*col_std**2) / (n+n-2)) if n > 1 else 1
						cohen_d = (col_mean - imp_mean) / pooled_std if pooled_std > 1e-6 else 0
						# Simple CI approximation
						se = np.sqrt((n+n)/(n*n) + cohen_d**2 / (2*(n+n-2))) * 1.96 if n > 1 else 0
						effect_size_label = "Large" if abs(cohen_d) > 0.8 else "Medium" if abs(cohen_d) > 0.5 else "Small"

						effect_sizes.append({
							"Feature": col, "Cohen's d": f"{cohen_d:.3f}",
							"Effect Size": effect_size_label,
							"Approx 95% CI": f"[{cohen_d - se:.3f}, {cohen_d + se:.3f}]"
						})
				st.dataframe(pd.DataFrame(effect_sizes) if effect_sizes else "Could not calculate effect sizes.")
			else:
				st.info(f"'{impedance_col}' not available or insufficient data.")

		return corr_df # Return the filtered dataframe used for analysis

	def _render_population_plots(self, filtered_df: pd.DataFrame, working_df: pd.DataFrame):
		"""Renders the scatter plot and population summary charts."""
		if not filtered_df.empty and self.CN.HIGHEST_FREQ_Z in filtered_df and self.CN.HEALING_RATE in filtered_df:
			self._scatter_plot(filtered_df)
		else:
			st.info("Insufficient data for Impedance vs Healing Rate scatter plot based on current filters.")

		if not working_df.empty:
			self._population_summary_charts(working_df)
		else:
			st.info("Insufficient data for population summary charts based on current filters.")

	def _scatter_plot(self, df: pd.DataFrame):
		"""Renders scatter plot: Impedance vs Healing Rate."""
		st.markdown("---")
		st.subheader("Impedance vs Healing Rate")
		plot_df = df.copy()

		# Ensure required columns exist and handle potential missing size data
		size_col = self.CN.WOUND_AREA if self.CN.WOUND_AREA in plot_df.columns else None
		if size_col:
			mean_area = plot_df[size_col].mean()
			plot_df[size_col] = plot_df[size_col].fillna(mean_area if pd.notnull(mean_area) else 1)

		color_col = self.CN.DIABETES if self.CN.DIABETES in plot_df.columns else None
		hover_cols = [col for col in [self.CN.RECORD_ID, self.CN.EVENT_NAME, self.CN.WOUND_TYPE] if col in plot_df.columns]

		fig = px.scatter(
			plot_df, x=self.CN.HIGHEST_FREQ_Z, y=self.CN.HEALING_RATE,
			color=color_col, size=size_col, size_max=30,
			hover_data=hover_cols, title="Impedance vs Healing Rate (Filtered Data)"
		)
		fig.update_layout(
			xaxis_title=f"{self.CN.HIGHEST_FREQ_Z}",
			yaxis_title=f"{self.CN.HEALING_RATE}"
		)
		st.plotly_chart(fig, use_container_width=True)

	def _population_summary_charts(self, df: pd.DataFrame):
		"""Renders charts for impedance components over time and by wound type."""
		st.markdown("---")
		st.subheader("Population Impedance Trends")
		try:
			avg_impedance, avg_by_type = ImpedanceAnalyzer.prepare_population_stats(df=df)
		except Exception as e:
			st.warning(f"Could not generate population statistics: {e}")
			return

		col1, col2 = st.columns(2)
		with col1:
			if not avg_impedance.empty:
				y_cols = [c for c in [self.CN.HIGHEST_FREQ_Z, self.CN.HIGHEST_FREQ_Z_PRIME, self.CN.HIGHEST_FREQ_Z_DOUBLE_PRIME] if c in avg_impedance.columns]
				if y_cols and self.CN.VISIT_NUMBER in avg_impedance.columns:
					fig1 = px.line(
						avg_impedance, x=self.CN.VISIT_NUMBER, y=y_cols,
						title="Average Impedance Components by Visit", markers=True
					)
					fig1.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)")
					st.plotly_chart(fig1, use_container_width=True)
				else:
					st.caption("Impedance components or visit number data missing.")
			else:
				st.caption("No data for average impedance over time.")
		with col2:
			if not avg_by_type.empty and self.CN.WOUND_TYPE in avg_by_type.columns and self.CN.HIGHEST_FREQ_Z in avg_by_type.columns:
				fig2 = px.bar(
					avg_by_type, x=self.CN.WOUND_TYPE, y=self.CN.HIGHEST_FREQ_Z,
					title="Average Impedance by Wound Type", color=self.CN.WOUND_TYPE
				)
				fig2.update_layout(xaxis_title="Wound Type", yaxis_title=f"Avg {self.CN.HIGHEST_FREQ_Z}")
				st.plotly_chart(fig2, use_container_width=True)
			else:
				st.caption("No data for average impedance by wound type.")


# --- Patient Renderer Class ---

class PatientImpedanceRenderer:
	"""Handles rendering logic for the single patient view."""

	def __init__(self, visits: List[VisitsDataType]):
		self.visits = visits
		self.VISIT_DATE_TAG = WoundDataProcessor.get_visit_date_tag(visits)

	def render(self) -> None:
		"""Renders the tabbed interface for patient analysis."""
		tab_titles = ["Overview", "Clinical Analysis", "Advanced Interpretation"]
		tab1, tab2, tab3 = st.tabs(tab_titles)

		with tab1: self._render_overview()
		with tab2: self._render_clinical_analysis()
		with tab3: self._render_advanced_analysis()

	def _render_overview(self):
		"""Renders the patient overview tab."""
		st.subheader("Impedance Measurements Over Time")
		measurement_mode = st.selectbox(
			"Select Measurement Mode:",
			["Absolute Impedance (|Z|)", "Resistance", "Capacitance"],
			key="patient_impedance_mode"
		)
		fig = self._create_impedance_chart(measurement_mode)
		st.plotly_chart(fig, use_container_width=True)
		self._display_overview_info()

	def _display_overview_info(self):
		"""Displays the informational box for the overview tab."""
		st.markdown("""
		<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
		<p style="margin-top:0; color:#666; font-weight:bold;">ABOUT THIS CHART:</p>
		<strong>Measurement Types:</strong><br>
		• <strong>|Z|</strong>: Total opposition to current flow<br>
		• <strong>Resistance</strong>: Opposition from ionic content (real part)<br>
		• <strong>Capacitance</strong>: Related to cell membrane charge storage (imaginary part)<br><br>
		<strong>Frequency Levels:</strong><br>
		• <strong>Lowest (~100Hz)</strong>: Reflects extracellular fluid properties.<br>
		• <strong>Center (Freq at Max Phase Angle)</strong>: Point of maximum capacitive effect.<br>
		• <strong>Highest (~80kHz)</strong>: Penetrates cell membranes, reflects total tissue properties.
		</div>
		""", unsafe_allow_html=True)

	def _render_clinical_analysis(self):
		"""Renders the patient clinical analysis tab."""
		st.subheader("Bioimpedance Clinical Analysis per Visit")
		if len(self.visits) < 2:
			st.info("At least two visits are needed for comparative clinical analysis.")
			# Optionally, display analysis for the single visit without comparison
			if len(self.visits) == 1:
				with st.expander(f"Visit: {self.visits[0].get(self.VISIT_DATE_TAG, 'N/A')}", expanded=True):
					analysis = ImpedanceAnalyzer.generate_clinical_analysis(self.visits[0], None)
					self._display_clinical_analysis_results(analysis, False)
			return

		visit_tabs = st.tabs([f"{visit.get(self.VISIT_DATE_TAG, f'Visit {i+1}')}" for i, visit in enumerate(self.visits)])
		for visit_idx, visit_tab in enumerate(visit_tabs):
			with visit_tab:
				current_visit = self.visits[visit_idx]
				previous_visit = self.visits[visit_idx-1] if visit_idx > 0 else None
				try:
					analysis = ImpedanceAnalyzer.generate_clinical_analysis(current_visit, previous_visit)
					self._display_clinical_analysis_results(analysis, previous_visit is not None)
				except Exception as e:
					st.error(f"Could not perform clinical analysis for this visit: {e}")
					st.error(traceback.format_exc())

	def _render_advanced_analysis(self):
		"""Renders the patient advanced analysis tab."""
		st.subheader("Advanced Bioelectrical Interpretation (Trends & Insights)")
		if len(self.visits) < 3:
			st.info("At least three visits are needed for robust advanced analysis (e.g., trajectory).")
			# Optionally show available analysis for fewer visits if ImpedanceAnalyzer supports it
			if len(self.visits) >= 1:
				try:
					analysis = ImpedanceAnalyzer.generate_advanced_analysis(self.visits) # Analyzer might handle fewer visits gracefully
					self._display_advanced_analysis_components(analysis)
				except Exception as e:
					st.error(f"Could not perform advanced analysis: {e}")
					st.error(traceback.format_exc())
			return

		try:
			analysis = ImpedanceAnalyzer.generate_advanced_analysis(self.visits)
			self._display_advanced_analysis_components(analysis)
			self._display_advanced_analysis_info() # Show info box only if analysis runs
		except Exception as e:
			st.error(f"Could not perform advanced analysis: {e}")
			st.error(traceback.format_exc())

	def _display_advanced_analysis_components(self, analysis: Dict[str, Any]):
		"""Displays the individual components of the advanced analysis."""
		if 'healing_trajectory' in analysis and analysis['healing_trajectory'].get('status') == 'analyzed':
			self._display_healing_trajectory(analysis['healing_trajectory'])
		else:
			st.caption("Trajectory analysis could not be completed (requires >= 3 visits with data).")

		if 'healing_stage' in analysis:
			self._display_wound_healing_stage(analysis['healing_stage'])
		else:
			st.caption("Wound healing stage classification not available.")

		if 'cole_parameters' in analysis and analysis['cole_parameters']:
			self._display_tissue_electrical_properties(analysis['cole_parameters'])
		else:
			st.caption("Tissue electrical properties (Cole parameters) not available.")

		if 'insights' in analysis:
			self._display_clinical_insights(analysis['insights'])
		else:
			st.caption("Clinical insights not generated.")


	def _display_advanced_analysis_info(self):
		"""Displays the informational box for the advanced analysis tab."""
		st.markdown("""
			<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
			<p style="margin-top:0; color:#666; font-weight:bold;">REFERENCE INFORMATION:</p>
			<p style="font-weight:bold; margin-bottom:5px;">Frequency Significance:</p>
			<ul style="margin-top:0; padding-left:20px;">
				<li><strong>Low Freq (~100Hz):</strong> Primarily reflects extracellular fluid status.</li>
				<li><strong>Center Freq (Max Phase Angle):</strong> Varies, related to cell membrane capacitance peak.</li>
				<li><strong>High Freq (~80kHz):</strong> Penetrates cells, reflects total (intra+extra) cellular status.</li>
			</ul>
			<p style="font-weight:bold; margin-bottom:5px;">General Clinical Correlations (Context Dependent):</p>
			<ul style="margin-top:0; padding-left:20px;">
				<li>↓ High-Freq Impedance: Often linked to improved tissue structure/healing.</li>
				<li>↑ Low/High Freq Ratio: May indicate ↑ extracellular fluid (edema, inflammation).</li>
				<li>↓ Phase Angle: Can suggest compromised cell membrane integrity.</li>
				<li>↑ Alpha (Cole Parameter): May indicate increased tissue heterogeneity.</li>
			</ul>
			</div>
			""", unsafe_allow_html=True)

	# --- Clinical Analysis Display Helpers ---

	def _display_clinical_analysis_results(self, analysis: dict, has_previous_visit: bool):
		"""Displays the structured results for the clinical analysis tab."""
		col1, col2 = st.columns(2)
		with col1:
			self._display_tissue_health_assessment(analysis.get('tissue_health', (None, 'N/A')))
		with col2:
			self._display_infection_risk_assessment(analysis.get('infection_risk', {}))

		st.markdown('---')
		col3, col4 = st.columns(2)
		with col3:
			if has_previous_visit and 'changes' in analysis:
				self._display_visit_changes(analysis.get('changes', {}), analysis.get('significant_changes', {}))
			elif not has_previous_visit:
				st.info("First visit shown. Comparative changes available from the second visit onwards.")
			else: # Has previous visit but no 'changes' key
				st.warning("Change data from previous visit not available.")
		with col4:
			self._display_tissue_composition_analysis(analysis.get('frequency_response', {}))

		self._display_clinical_analysis_info()

	def _display_clinical_analysis_info(self):
		"""Displays the informational box for the clinical analysis tab."""
		st.markdown("""
			<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
			<p style="margin-top:0; color:#666; font-weight:bold;">INTERPRETING CHANGES:</p>
			<p>Color indicates direction of change vs previous visit: <span style="color:#FF4B4B; font-weight:bold;">Red = increase</span>, <span style="color:#00CC96; font-weight:bold;">Green = decrease</span>.<br>
			Asterisk (*) marks changes potentially exceeding clinical thresholds (e.g., Resistance >15%, Capacitance >20%, |Z| >15%). Consult clinical guidelines for specific interpretation.</p>
			</div>
			""", unsafe_allow_html=True)

	def _display_tissue_health_assessment(self, tissue_health):
		st.markdown("#### Tissue Health Assessment", help="Index (0-100) derived from multi-frequency impedance ratios and phase angle, reflecting overall tissue electrical properties. Higher scores generally indicate healthier tissue state.")
		health_score, health_interp = tissue_health
		if health_score is not None:
			color = "red" if health_score < 40 else "orange" if health_score < 60 else "green"
			st.markdown(f"**Index Score:** <span style='color:{color};font-weight:bold'>{health_score:.1f}/100</span>", unsafe_allow_html=True)
			st.markdown(f"**Interpretation:** {health_interp}")
		else:
			st.warning("Insufficient data for tissue health index calculation.")

	def _display_infection_risk_assessment(self, infection_risk):
		st.markdown("#### Infection Risk Assessment", help="Score (0-100) based on factors like impedance ratios, sudden resistance changes, and phase angle. Higher scores indicate elevated risk. Factors are context-dependent.")
		risk_score = infection_risk.get("risk_score")
		risk_level = infection_risk.get("risk_level", "N/A")
		if risk_score is not None:
			risk_color = "green" if risk_score < 30 else "orange" if risk_score < 60 else "red"
			st.markdown(f"**Risk Score:** <span style='color:{risk_color};font-weight:bold'>{risk_score:.1f}/100</span>", unsafe_allow_html=True)
			st.markdown(f"**Risk Level:** {risk_level}")
			factors = infection_risk.get("contributing_factors", [])
			if factors: st.markdown(f"**Potential Factors:** {', '.join(factors)}")
		else:
			st.warning("Insufficient data for infection risk assessment.")

	def _display_tissue_composition_analysis(self, freq_response):
		st.markdown("#### Tissue Composition Analysis", help="Uses Bioelectrical Impedance Analysis (BIA) principles. Alpha dispersion (low-mid freq) relates to extracellular fluid/membranes. Beta dispersion (mid-high freq) relates to intracellular properties/cell structure.")
		alpha = freq_response.get('alpha_dispersion')
		beta = freq_response.get('beta_dispersion')
		interp = freq_response.get('interpretation', 'N/A')
		if alpha is not None: st.markdown(f"**Alpha Dispersion:** {alpha:.3f}")
		if beta is not None: st.markdown(f"**Beta Dispersion:** {beta:.3f}")
		st.markdown(f"**Interpretation:** {interp}")
		if alpha is None and beta is None:
			st.warning("Frequency dispersion data not available.")

	def _display_visit_changes(self, changes: dict, significant_changes: dict):
		st.markdown("#### Changes Since Previous Visit", help="Percentage change in key impedance parameters compared to the last recorded visit. Asterisk (*) indicates a potentially significant change.")
		if not changes:
			st.info("No comparable data from previous visit.")
			return

		data_by_freq = {
			"Low Freq": {"Z": "N/A", "Resistance": "N/A", "Capacitance": "N/A"},
			"Mid Freq": {"Z": "N/A", "Resistance": "N/A", "Capacitance": "N/A"},
			"High Freq": {"Z": "N/A", "Resistance": "N/A", "Capacitance": "N/A"},
		}

		processed_any = False
		for key, change in changes.items():
			parts = key.split('_')
			param_name = parts[0].capitalize()
			freq_type = ' '.join(parts[1:]).replace('_', ' ').lower()

			freq_key = None
			if 'low frequency' in freq_type: freq_key = "Low Freq"
			elif 'center frequency' in freq_type: freq_key = "Mid Freq"
			elif 'high frequency' in freq_type: freq_key = "High Freq"

			if freq_key and param_name in ["Z", "Resistance", "Capacitance"] and change is not None:
				is_significant = significant_changes.get(key, False)
				formatted_change = f"{change*100:+.1f}%{'*' if is_significant else ''}"
				data_by_freq[freq_key][param_name] = formatted_change
				processed_any = True

		if not processed_any:
			st.info("No impedance change data available to display.")
			return

		change_df = pd.DataFrame(data_by_freq).T
		change_df = change_df[["Z", "Resistance", "Capacitance"]] # Ensure order

		styled_df = change_df.style.map(_color_visit_change_cells).set_properties(**{
			'text-align': 'center', 'font-size': '13px', 'border': '1px solid #EEEEEE'
		})
		st.dataframe(styled_df, use_container_width=True)


	# --- Advanced Analysis Display Helpers ---

	def _display_healing_trajectory(self, trajectory):
		st.markdown("#### Healing Trajectory (High-Frequency Impedance)")
		if trajectory.get('status') != 'analyzed' or not trajectory.get('values'):
			st.caption("Trajectory analysis could not be completed (requires >= 3 visits with data).")
			return

		dates = trajectory.get('dates', list(range(len(trajectory.get('values', []))))) # Use index if dates missing
		values = trajectory.get('values', [])
		slope = trajectory.get('slope', 0)
		p_value = trajectory.get('p_value', 1)
		r_squared = trajectory.get('r_squared', 0)
		interpretation = trajectory.get('interpretation', 'N/A')

		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=list(range(len(dates))), # Use simple index for x-axis to avoid date parsing issues
			y=values, mode='lines+markers', name='Impedance',
			line=dict(color='blue'), customdata=dates, # Store actual dates in customdata
			hovertemplate='<b>Visit Date:</b> %{customdata}<br><b>Impedance:</b> %{y:.1f} kOhms<extra></extra>'
		))

		# Add trend line if slope is valid
		if np.isfinite(slope):
			x_trend = np.array(range(len(values)))
			# Calculate intercept based on the mean point
			intercept = np.mean(values) - slope * np.mean(x_trend)
			y_trend = slope * x_trend + intercept
			fig.add_trace(go.Scatter(
				x=x_trend, y=y_trend, mode='lines', name='Trend',
				line=dict(color='red', dash='dash')
			))

		fig.update_layout(
			title="Impedance Trend Over Time", xaxis_title="Visit Sequence",
			yaxis_title="High-Frequency Impedance (kOhms)", hovermode="x unified", height=400
		)
		# Use visit sequence for x-axis ticks for simplicity
		fig.update_xaxes(tickvals=list(range(len(dates))), ticktext=[f"Visit {i+1}" for i in range(len(dates))])

		st.plotly_chart(fig, use_container_width=True)

		col1, col2 = st.columns(2)
		with col1:
			slope_color = "green" if slope < 0 else "red" if slope > 0 else "grey"
			st.markdown(f"**Trend Slope:** <span style='color:{slope_color}'>{slope:.4f}</span> (kOhms/visit)", unsafe_allow_html=True)
			st.markdown(f"**P-value:** {p_value:.4f} {'(Significant)' if p_value < 0.05 else '(Not Significant)'}")
		with col2:
			st.markdown(f"**R² Value:** {r_squared:.4f}")
			st.markdown(f"**Interpretation:** {interpretation}")
		st.markdown("---")

	def _display_wound_healing_stage(self, healing_stage: dict):
		st.markdown("#### Wound Healing Stage Classification")
		stage = healing_stage.get('stage', 'Unknown')
		confidence = healing_stage.get('confidence', 'N/A')
		characteristics = healing_stage.get('characteristics', [])

		stage_colors = {"Inflammatory": "red", "Proliferative": "orange", "Remodeling": "green", "Unknown": "grey"}
		stage_color = stage_colors.get(stage, "blue")

		st.markdown(f"**Estimated Stage:** <span style='color:{stage_color};font-weight:bold'>{stage}</span>", unsafe_allow_html=True)
		st.markdown(f"**Confidence:** {confidence}")
		if characteristics:
			st.markdown("**Associated Characteristics:**")
			for char in characteristics: st.markdown(f"- {char}")
		st.markdown("---")


	def _display_tissue_electrical_properties(self, cole_params: dict):
		st.markdown("#### Tissue Electrical Properties (from Cole Model)")
		if not cole_params:
			st.caption("Cole parameters not available.")
			return

		col1, col2 = st.columns(2)
		with col1:
			r0 = cole_params.get('R0')
			rinf = cole_params.get('Rinf')
			if r0 is not None: st.markdown(f"**Extracellular Resistance (R₀):** {r0:.2f} Ω")
			if rinf is not None: st.markdown(f"**Total Resistance (R∞):** {rinf:.2f} Ω")
		with col2:
			cm = cole_params.get('Cm')
			alpha = cole_params.get('Alpha')
			homogeneity = cole_params.get('tissue_homogeneity', '')
			if cm is not None: st.markdown(f"**Membrane Capacitance (Cm):** {cm:.2e} F")
			if alpha is not None: st.markdown(f"**Tissue Heterogeneity (α):** {alpha:.2f}")
			if homogeneity: st.caption(f"({homogeneity})")

		if all(p is None for p in [r0, rinf, cm, alpha]):
			 st.caption("Cole parameters could not be calculated.")
		st.markdown("---")

	def _display_clinical_insights(self, insights: list):
		st.markdown("#### Clinical Insights Summary")
		if not insights:
			st.info("No specific clinical insights generated based on the current impedance data patterns.")
			return

		for i, insight in enumerate(insights):
			with st.expander(f"Insight {i+1}: {insight.get('insight', 'N/A')[:60]}...", expanded=i==0):
				st.markdown(f"**Insight:** {insight.get('insight', 'N/A')}")
				st.markdown(f"**Confidence:** {insight.get('confidence', 'N/A')}")
				if 'recommendation' in insight: st.markdown(f"**Recommendation:** {insight['recommendation']}")
				if 'supporting_factors' in insight and insight['supporting_factors']:
					st.markdown("**Supporting Factors:**")
					for factor in insight['supporting_factors']: st.markdown(f"- {factor}")
				if 'clinical_meaning' in insight: st.markdown(f"**Clinical Interpretation:** {insight['clinical_meaning']}")

	# --- Chart Creation ---

	def _create_impedance_chart(self, measurement_mode: str = "Absolute Impedance (|Z|)") -> go.Figure:
		"""Creates the impedance overview chart for a single patient."""
		MEASUREMENT_FIELDS = {"Absolute Impedance (|Z|)": "Z", "Resistance": "resistance", "Capacitance": "capacitance"}
		FREQUENCY_LABELS = {"high_frequency": "Highest Freq", "center_frequency": "Center Freq", "low_frequency": "Lowest Freq"}
		LINE_STYLES = {"high_frequency": None, "center_frequency": dict(dash='dot'), "low_frequency": dict(dash='dash')}

		dates = []
		measurements = {freq: {field: [] for field in MEASUREMENT_FIELDS.values()} for freq in FREQUENCY_LABELS}

		for visit in self.visits:
			dates.append(visit.get(self.VISIT_DATE_TAG, 'Unknown Date'))
			impedance_data = visit.get('sensor_data', {}).get('impedance', {})
			for freq in FREQUENCY_LABELS:
				freq_data = impedance_data.get(freq, {})
				for field in MEASUREMENT_FIELDS.values():
					val_str = freq_data.get(field)
					val = None
					if val_str is not None and val_str != '':
						try: val = float(val_str)
						except (ValueError, TypeError): pass
					measurements[freq][field].append(val)

		fig = go.Figure()
		field = MEASUREMENT_FIELDS[measurement_mode]
		y_axis_label = f"{measurement_mode.split()[0]} Values"
		yaxis_type = 'linear' # Default

		# Add traces if data exists
		has_data = False
		for freq in FREQUENCY_LABELS:
			values = measurements[freq][field]
			valid_values = [v for v in values if v is not None and np.isfinite(v)]
			if valid_values:
				has_data = True
				fig.add_trace(go.Scatter(
					x=dates, y=values, name=f"{measurement_mode.split()[0]} ({FREQUENCY_LABELS[freq]})",
					mode='lines+markers', line=LINE_STYLES[freq], connectgaps=False # Don't connect across missing data points
				))
				# Check for capacitance which often requires log scale
				if measurement_mode == "Capacitance" and any(v > 0 for v in valid_values):
					yaxis_type = 'log'
					y_axis_label += " (Log Scale)"


		if not has_data:
			# Add dummy trace if no data to show an empty chart with labels
			fig.add_trace(go.Scatter(x=[], y=[]))

		fig.update_layout(
			title='Impedance Measurements Over Time', xaxis_title=self.VISIT_DATE_TAG,
			yaxis_title=y_axis_label, yaxis_type=yaxis_type,
			hovermode='x unified', showlegend=True, height=500
		)
		return fig

# --- Main Tab Class ---

class ImpedanceTab:
	"""
	Main class to render the impedance analysis tab.
	Delegates rendering to PopulationImpedanceRenderer or PatientImpedanceRenderer.
	"""
	def __init__(self, df: pd.DataFrame, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.wound_data_processor = wound_data_processor
		self.patient_id_str = selected_patient
		self.is_population_view = (selected_patient == "All Patients")
		self.df = df
		# self.CN = DColumns(df=df) # CN is now handled within renderers where needed

	def render(self) -> None:
		"""Renders the appropriate view based on patient selection."""
		st.header("Impedance Analysis")

		if self.is_population_view:
			renderer = PopulationImpedanceRenderer(self.df)
			renderer.render()
		else:
			try:
				patient_id_num = int(self.patient_id_str.split()[1])
				visits_meta_data: VisitsMetadataType = self.wound_data_processor.get_patient_visits(record_id=patient_id_num)
				visits: List[VisitsDataType] = visits_meta_data.get('visits', [])

				if not visits:
					st.warning(f"No visit data found for patient {patient_id_num}.")
					return

				renderer = PatientImpedanceRenderer(visits)
				renderer.render()

			except (IndexError, ValueError):
				st.error(f"Invalid patient identifier format: {self.patient_id_str}")
			except Exception as e:
				st.error(f"Error loading data for patient {self.patient_id_str}: {e}")
				st.error(traceback.format_exc())
