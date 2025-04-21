import pandas as pd
from typing import Dict, Union, Self
import logging
from scipy import stats

from wound_analysis.utils.column_schema import DColumns
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
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationAnalysis:

	def __init__(self, data: pd.DataFrame, x_col: str, y_col: str, outlier_threshold: float = 0.2, REMOVE_OUTLIERS: bool = True):
		self.data  = data
		self.x_col = x_col
		self.y_col = y_col
		self.r, self.p = None, None
		self.outlier_threshold = outlier_threshold
		self.REMOVE_OUTLIERS = REMOVE_OUTLIERS

	def _calculate_correlation(self) -> Self:
		"""
		Calculate Pearson correlation between x and y columns of the data.

		This method:
		1. Drops rows with NaN values in either the x or y columns
		2. Calculates the Pearson correlation coefficient (r) and p-value
		3. Stores the results in self.r and self.p

		Returns:
			Self: The current instance for method chaining

		Raises:
			Exception: Logs any error that occurs during correlation calculation
		"""
		try:
			valid_data = self.data.dropna(subset=[self.x_col, self.y_col])

			if len(valid_data) > 1:
				self.r, self.p = stats.pearsonr(valid_data[self.x_col], valid_data[self.y_col])

		except Exception as e:
			logger.error(f"Error calculating correlation: {str(e)}")

		return self

	@property
	def format_p_value(self) -> str:
		"""
		Format the p-value for display.

		Returns:
			str: A string representation of the p-value. Returns "N/A" if the p-value is None,
					"< 0.001" if the p-value is less than 0.001, or the p-value rounded to three
					decimal places with an equals sign prefix otherwise.
		"""
		if self.p is None:
			return "N/A"
		return "< 0.001" if self.p < 0.001 else f"= {self.p:.3f}"

	def get_correlation_text(self, text: str="Statistical correlation") -> str:
		"""
		Generate a formatted text string describing a statistical correlation.

		This method creates a string that includes the correlation coefficient (r) and p-value
		in a standardized format.

		Parameters
		----------
		text : str, optional
			The descriptive text to prepend to the correlation statistics.
			Default is "Statistical correlation".

		Returns
		-------
		str
			A formatted string containing the correlation coefficient and p-value.
			Returns "N/A" if either r or p values are None.

		Examples
		--------
		>>> obj.r, obj.p = 0.75, 0.03
		>>> obj.get_correlation_text("Pearson correlation")
		'Pearson correlation: r = 0.75 (p < 0.05)'
		"""
		if self.r is None or self.p is None:
			return "N/A"
		return f"{text}: r = {self.r:.2f} (p {self.format_p_value})"

	def calculate_correlation(self):
		"""
		Calculate the correlation between impedance data points after removing outliers.

		This method first calls `_remove_outliers()` to clean the data set and then calculates
		the correlation using `_calculate_correlation()`.

		Returns:
			tuple: A tuple containing three elements:
				- data (pd.DataFrame): The processed data after outlier removal
				- r (float or None): Correlation coefficient if calculation succeeds, None otherwise
				- p (float or None): P-value of the correlation if calculation succeeds, None otherwise

		Raises:
			Exception: The method catches any exceptions that occur during calculation and
						logs the error but returns partial results (data and None values).
		"""

		try:
			self._remove_outliers()
			self._calculate_correlation()

			return self.data, self.r, self.p

		except Exception as e:
			logger.error(f"Error calculating impedance correlation: {str(e)}")
			return self.data, None, None

	def _remove_outliers(self) -> 'CorrelationAnalysis':
		"""
		Removes outliers from the data based on quantile thresholds.

		This method filters out data points whose x_col values fall outside the range
		defined by the lower and upper quantile bounds. The bounds are calculated using
		the outlier_threshold attribute.

		The method will only remove outliers if:
		- REMOVE_OUTLIERS flag is True
		- outlier_threshold is greater than 0
		- data is not empty
		- x_col exists in the data columns

		Returns:
			CorrelationAnalysis: The current instance with outliers removed

		Raises:
			Exceptions are caught and logged but not propagated.
		"""

		try:
			if self.REMOVE_OUTLIERS and (self.outlier_threshold > 0) and (not self.data.empty) and (self.x_col in self.data.columns):

				# Calculate lower and upper bounds
				lower_bound = self.data[self.x_col].quantile(self.outlier_threshold)
				upper_bound = self.data[self.x_col].quantile(1 - self.outlier_threshold)

				# Filter out outliers
				mask = (self.data[self.x_col] >= lower_bound) & (self.data[self.x_col] <= upper_bound)
				self.data = self.data[mask]

		except Exception as e:
			logger.error(f"Error removing outliers: {str(e)}")

		return self



class ClusteringAnalysis:

	def __init__(self,df: pd.DataFrame, cluster_settings: Dict[str, Union[int, str, List[str], bool]]):
		self.df                = df
		self.cluster_features  = cluster_settings["cluster_features"]
		self.n_clusters        = cluster_settings["n_clusters"]
		self.clustering_method = cluster_settings["clustering_method"]

		# Calculated Variables
		self.df_w_cluster_tags  = None
		self.feature_importance = None
		self.cluster_tags       = None


	def render(self) -> 'ClusteringAnalysis':
		"""
		Perform clustering on the data using the specified method and features.
		"""
		df2 = self.df.copy()

		# Create a feature dataframe for clustering
		clustering_df = df2[self.cluster_features].copy()

		# Handle missing values
		clustering_df = clustering_df.fillna(clustering_df.mean())

		# Drop rows with any remaining NaN values
		clustering_df = clustering_df.dropna()

		if len(clustering_df) <= self.n_clusters:
			st.error("Not enough valid data points for clustering. Try selecting different features or reducing the number of clusters.")
			return

		# Get indices of valid rows to map back to original dataframe
		valid_indices = clustering_df.index

		# Standardize the data
		scaler = StandardScaler()
		scaled_features = scaler.fit_transform(clustering_df)

		# Perform clustering based on selected method
		cluster_labels, self.feature_importance = self._apply_clustering_algorithm(scaled_features)

		# Create a new column in the original dataframe with cluster labels
		cluster_mapping = pd.Series(cluster_labels, index=valid_indices)
		df2.loc[valid_indices, 'Cluster'] = cluster_mapping

		# Handle any NaN in cluster column (rows that were dropped during clustering)
		df2['Cluster'] = df2['Cluster'].fillna(-1).astype(int)

		self.df_w_cluster_tags = df2
		self.cluster_tags      = sorted(self.df_w_cluster_tags['Cluster'].unique())

		return self



	def _apply_clustering_algorithm(self, scaled_features: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
		"""
			Apply the specified clustering algorithm to the data.

			Args:
				scaled_features: Standardized features to cluster

			Returns:
				Tuple of (cluster_labels, feature_importance)
		"""
		feature_importance = {}

		if self.clustering_method == "K-Means":

			clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
			cluster_labels = clusterer.fit_predict(scaled_features)

			# Calculate feature importance for K-Means
			centers = clusterer.cluster_centers_
			for i, feature in enumerate(self.cluster_features):

				# Calculate the variance of this feature across cluster centers
				variance = np.var([center[i] for center in centers])
				feature_importance[feature] = variance

		elif self.clustering_method == "Hierarchical":

			# Perform hierarchical clustering
			Z = linkage(scaled_features, 'ward')
			cluster_labels = fcluster(Z, self.n_clusters, criterion='maxclust') - 1  # Adjust to 0-based

			# For hierarchical clustering, use silhouette coefficients for feature importance
			for i, feature in enumerate(self.cluster_features):

				# Create single-feature clustering and measure its quality
				single_feature = scaled_features[:, i:i+1]
				if len(np.unique(single_feature)) > 1:  # Only if feature has variation
					temp_clusters = fcluster(linkage(single_feature, 'ward'), self.n_clusters, criterion='maxclust')

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
			for i, feature in enumerate(self.cluster_features):

				variances = []
				for label in set(cluster_labels):

					if label >= 0:  # Exclude noise points
						cluster_data = scaled_features[cluster_labels == label, i]

						if len(cluster_data) > 1:
							variances.append(np.var(cluster_data))

				if variances:
					# Calculate importance based on ratio of within-cluster variance to total variance
					# Higher ratio means less importance (features with high variance within clusters are less useful)
					within_cluster_variance = np.mean(variances)
					total_variance          = np.var(scaled_features[:, i]) if np.var(scaled_features[:, i]) > 0 else 1e-10
					feature_importance[feature] = 1.0 - min(1.0, within_cluster_variance/total_variance)

				else:
					# Assign minimal importance if no variance information is available
					feature_importance[feature] = 0.01

		# Normalize feature importance
		if feature_importance and max(feature_importance.values()) > 0:

			max_importance     = max(feature_importance.values())
			feature_importance = {k: v/max_importance for k, v in feature_importance.items()}

		return cluster_labels, feature_importance
