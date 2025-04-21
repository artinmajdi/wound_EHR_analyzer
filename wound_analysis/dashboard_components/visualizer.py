from typing import Optional
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from wound_analysis.utils.column_schema import DColumns


class Visualizer:
	"""A class that provides visualization methods for wound analysis data.

	The Visualizer class contains static methods for creating various plots and charts
	to visualize wound healing metrics over time. These visualizations help in monitoring
	patient progress and comparing trends across different patients.

	Methods
	create_wound_area_plot(df, patient_id=None)
		Create a wound area progression plot for one or all patients.

	create_temperature_chart(df)
		Create a chart showing wound temperature measurements and gradients.

	create_impedance_chart(visits, measurement_mode="Absolute Impedance (|Z|)")
		Create an interactive chart showing impedance measurements over time.

	create_oxygenation_chart(patient_data, visits)
		Create charts showing oxygenation and hemoglobin measurements over time.

	create_exudate_chart(visits)
		Create a chart showing exudate characteristics over time.

	Private Methods
	--------------
	_remove_outliers(df, column, quantile_threshold=0.1)
		Remove outliers from a DataFrame column using IQR and z-score methods.

	_create_all_patients_plot(df, variable_column)
		Create an interactive plot showing a variable's progression for all patients over time.

	_create_single_patient_plot(df, patient_id)
		Create a detailed wound area and dimensions plot for a single patient.
	"""

	@staticmethod
	def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None, variable_column: Optional[str] = None) -> go.Figure:
		"""
		Create a plot showing either wound area or another variable's progression over time.

		This function generates a visualization of a selected variable's changes over time.
		If a patient_id is provided, it creates a plot specific to that patient.
		Otherwise, it creates a comparative plot for all patients in the DataFrame.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing measurements and dates
		patient_id : Optional[int], default=None
			The ID of a specific patient to plot. If None, plots data for all patients
		variable_column : Optional[str], default=None
			The column name of the variable to plot. If None, defaults to wound area.

		Returns
		-------
		go.Figure
			A Plotly figure object containing the progression plot
		"""
		# Default to wound area if no variable specified
		CN = DColumns(df=df)
		if variable_column is None:
			variable_column = CN.WOUND_AREA

		if patient_id is None or patient_id == "All Patients":
			return Visualizer._create_all_patients_plot(df, variable_column)

		return Visualizer._create_single_patient_plot(df, patient_id)

	@staticmethod
	def _remove_outliers(df: pd.DataFrame, column: str, quantile_threshold: float = 0.1) -> pd.DataFrame:
		"""
		Remove outliers from a DataFrame column using a combination of IQR and z-score methods.

		This function identifies and filters out outlier values in the specified column
		using both the Interquartile Range (IQR) method and z-scores. It's designed to be
		conservative in outlier removal, requiring that values pass both tests to be retained.

		Parameters
		----------
		df : pd.DataFrame
			The input DataFrame containing the column to be filtered
		column : str
			The name of the column to remove outliers from
		quantile_threshold : float, default=0.1
			The quantile threshold used to calculate Q1 and Q3
			Lower values result in more aggressive filtering
			Value must be > 0; if ≤ 0, no filtering is performed

		Returns
		-------
		pd.DataFrame
			A filtered copy of the input DataFrame with outliers removed

		Notes
		-----
		- Values less than 0 are considered outliers (forced to lower_bound of 0)
		- If all values are the same (IQR=0) or there are fewer than 3 data points, the original DataFrame is returned unchanged
		- The function combines two outlier detection methods:
			1. IQR method: filters values outside [Q1-1.5*IQR, Q3+1.5*IQR]
			2. Z-score method: filters values with z-score > 3
		"""
		CN = DColumns(df=df)

		try:
			if quantile_threshold <= 0 or len(df) < 3:  # Not enough data points or no filtering requested
				return df

			Q1 = df[column].quantile(quantile_threshold)
			Q3 = df[column].quantile(1 - quantile_threshold)
			IQR = Q3 - Q1

			if IQR == 0:  # All values are the same
				return df

			lower_bound = max(0, Q1 - 1.5 * IQR)  # Ensure non-negative values
			upper_bound = Q3 + 1.5 * IQR

			# Calculate z-scores for additional validation
			z_scores = abs((df[column] - df[column].mean()) / df[column].std())

			# Combine IQR and z-score methods
			mask = (df[column] >= lower_bound) & (df[column] <= upper_bound) & (z_scores <= 3)

			return df[mask].copy()

		except Exception as e:
			print(f"------- Error removing outliers: {e} ----- ")
			print(df[CN.OXYGENATION].to_frame().describe())
			print(df[CN.OXYGENATION].to_frame())
			return df

	@staticmethod
	def _create_all_patients_plot(df: pd.DataFrame, variable_column: str) -> go.Figure:
		"""
		Create an interactive plot showing a variable's progression for all patients over time.

		This function generates a Plotly figure where each patient's selected variable is plotted against
		days since their first visit. The plot includes interactive features such as hovering for
		patient details and an outlier threshold control to filter extreme values.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing patient data with columns:
			- CN.RECORD_ID: Patient identifier
			- CN.VISIT_DATE: Date of the assessment
			- variable_column: The measurements to be plotted
		variable_column : str
			The column name of the variable to plot

		Returns
		-------
		go.Figure
			A Plotly figure object containing the progression plot for all patients.
			The figure includes:
			- Individual patient lines with distinct colors
			- Interactive hover information with patient ID and the variable value
			- Y-axis automatically scaled based on outlier threshold setting
			- Annotation explaining outlier removal status

		Notes
		-----
		The function adds a Streamlit number input widget for controlling outlier removal threshold.
		Patient progression lines are colored based on their rates.
		"""
		# Get the variable's display name for labels
		CN = DColumns(df=df)
		# If it's in wound_area or another common column, get a nice display name
		variable_display = variable_column
		for attr_name, column_name in CN.__dict__.items():
			if column_name == variable_column and attr_name.isupper():
				# Convert SNAKE_CASE to Title Case with spaces
				variable_display = " ".join(attr_name.split("_")).title()
				break

		group_by = "None"

		# Add controls for plot customization
		col1, col2, col3 = st.columns([2, 2, 1])

		with col1:
			plot_type = st.selectbox(
				"Plot Type",
				options=["Line Plot", "Box Plot", "Histogram", "Violin Plot", "Strip Plot"],
				index=0,
				help="Select the type of visualization to display"
			)

		with col2:
			if plot_type == "Histogram":
				bin_count = st.slider("Number of Bins", min_value=5, max_value=50, value=20, step=5)
			elif plot_type == "Box Plot":
				group_by = st.selectbox(
					"Group By",
					options=["None", "Week", "Month"],
					index=0,
					help="Select how to group data for box plot"
				)

		with col3:
			outlier_threshold = st.number_input(
				f"Outlier Threshold",
				min_value=0.0,
				max_value=0.5,
				value=0.0,
				step=0.01,
				help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		# Get the filtered data for y-axis limits
		if variable_column in df.columns:
			all_values_df = pd.DataFrame({variable_column: df[variable_column].dropna()})
			filtered_df = Visualizer._remove_outliers(all_values_df, variable_column, outlier_threshold)

			# Set y-axis limits based on filtered data
			if not filtered_df.empty:
				lower_bound = (filtered_df[variable_column].min() if outlier_threshold > 0 else all_values_df[variable_column].min()) * 0.95
				upper_bound = (filtered_df[variable_column].max() if outlier_threshold > 0 else all_values_df[variable_column].max()) * 1.05

				# Ensure non-negative lower bound for variables that can't be negative
				if variable_column in [CN.WOUND_AREA, CN.LENGTH, CN.WIDTH, CN.DEPTH, CN.OXYGENATION, CN.HEMOGLOBIN,
									  CN.CENTER_TEMP, CN.EDGE_TEMP, CN.PERI_TEMP]:
					lower_bound = max(0, lower_bound)
			else:
				lower_bound = 0
				upper_bound = 1  # Default if no data
		else:
			st.warning(f"Column '{variable_column}' not found in the dataframe")
			return go.Figure()  # Return empty figure

		# Create different plot types
		fig = go.Figure()

		if plot_type == "Histogram":
			# For histograms, we just use the variable directly
			valid_values = df[variable_column].dropna()
			if len(valid_values) > 0:
				fig = go.Figure(data=[go.Histogram(
					x=valid_values,
					nbinsx=bin_count,
					marker_color='royalblue',
					opacity=0.75
				)])

				fig.update_layout(
					title=f"{variable_display} Distribution - All Patients",
					xaxis_title=variable_display,
					yaxis_title="Count",
					bargap=0.1,
				)

		elif plot_type == "Box Plot":
			# For box plots, we can group by time periods if requested
			if group_by == "None":
				# Simple box plot without grouping
				fig = go.Figure()
				fig.add_trace(go.Box(
					y=df[variable_column].dropna(),
					name=variable_display,
					boxpoints='all',
					jitter=0.3,
					pointpos=-1.8,
					marker_color='royalblue',
					line_color='darkblue'
				))

				fig.update_layout(
					title=f"{variable_display} Distribution - All Patients",
					yaxis_title=variable_display
				)
			else:
				# Group by time periods
				df_copy = df.copy()
				if group_by == "Week":
					# Add a week number column based on days since first visit
					df_copy['Week'] = (df_copy[CN.DAYS_SINCE_FIRST_VISIT] // 7) + 1
					group_column = 'Week'
					xaxis_title = "Week Number"
				else:  # Month
					# Add a month number column based on days since first visit
					df_copy['Month'] = (df_copy[CN.DAYS_SINCE_FIRST_VISIT] // 30) + 1
					group_column = 'Month'
					xaxis_title = "Month Number"

				# Get unique groups with at least one valid measurement
				valid_groups = []
				for group in sorted(df_copy[group_column].unique()):
					if not df_copy[df_copy[group_column] == group][variable_column].isna().all():
						valid_groups.append(group)

				# Create box plot for each time period
				fig = go.Figure()
				for group in valid_groups:
					group_data = df_copy[df_copy[group_column] == group][variable_column].dropna()
					if len(group_data) > 0:
						fig.add_trace(go.Box(
							y=group_data,
							name=str(group),
							boxpoints='all',
							jitter=0.3,
							pointpos=-1.8,
							marker_color='royalblue',
							line_color='darkblue'
						))

				fig.update_layout(
					title=f"{variable_display} Distribution by {group_by} - All Patients",
					xaxis_title=xaxis_title,
					yaxis_title=variable_display
				)

		elif plot_type == "Violin Plot":
			fig = go.Figure()

			# Group patients by outcome (improving vs not)
			df_copy = df.copy()

			# Try to categorize by healing status if available
			if CN.OVERALL_IMPROVEMENT in df.columns:
				df_copy['Outcome'] = df_copy[CN.OVERALL_IMPROVEMENT].fillna('Unknown')

				for outcome in df_copy['Outcome'].unique():
					subset = df_copy[df_copy['Outcome'] == outcome]
					if not subset[variable_column].isna().all():
						fig.add_trace(go.Violin(
							y=subset[variable_column].dropna(),
							name=outcome,
							box_visible=True,
							meanline_visible=True,
							points='all'
						))

				title_suffix = "by Wound Improvement Status"
				xaxis_title = "Wound Improvement Status"
			else:
				# If no outcome data, just show one violin plot
				fig.add_trace(go.Violin(
					y=df_copy[variable_column].dropna(),
					name=variable_display,
					box_visible=True,
					meanline_visible=True,
					points='all'
				))

				title_suffix = "Distribution"
				xaxis_title = ""

			fig.update_layout(
				title=f"{variable_display} {title_suffix} - All Patients",
				yaxis_title=variable_display,
				xaxis_title=xaxis_title
			)

		elif plot_type == "Strip Plot":
			# Create a strip/swarm plot - this is a simplified version that just shows all points
			fig = go.Figure()

			# Add a small jitter to points to avoid overlap
			all_values = df[variable_column].dropna().values
			if len(all_values) > 0:
				jittered_x = np.random.normal(0, 0.1, size=len(all_values))

				fig.add_trace(go.Scatter(
					x=jittered_x,
					y=all_values,
					mode='markers',
					marker=dict(
						color='royalblue',
						size=8,
						opacity=0.7
					),
					hovertemplate=f"{variable_display}: %{{y:.2f}}<extra></extra>"
				))

				fig.update_layout(
					title=f"{variable_display} Strip Plot - All Patients",
					yaxis_title=variable_display,
					xaxis_title="Individual Data Points (jittered for visibility)",
					xaxis=dict(
						showticklabels=False,
						zeroline=True,
						zerolinewidth=2,
						zerolinecolor='black'
					)
				)

		else:  # Default: Line Plot
			# Store patient statistics for coloring
			patient_stats = []

			for pid in df[CN.RECORD_ID].unique():
				patient_df = df[df[CN.RECORD_ID] == pid].copy()
				patient_df[CN.DAYS_SINCE_FIRST_VISIT] = (pd.to_datetime(patient_df[CN.VISIT_DATE]) - pd.to_datetime(patient_df[CN.VISIT_DATE]).min()).dt.days

				# Remove NaN values for this variable
				patient_df = patient_df.dropna(subset=[CN.DAYS_SINCE_FIRST_VISIT, variable_column])

				if not patient_df.empty:
					# Calculate rate of change for this patient
					if len(patient_df) >= 2:
						first_value = patient_df.loc[patient_df[CN.DAYS_SINCE_FIRST_VISIT].idxmin(), variable_column]
						last_value = patient_df.loc[patient_df[CN.DAYS_SINCE_FIRST_VISIT].idxmax(), variable_column]
						total_days = patient_df[CN.DAYS_SINCE_FIRST_VISIT].max()

						if total_days > 0:
							change_rate = (first_value - last_value) / total_days
							patient_stats.append({
								'pid': pid,
								'change_rate': change_rate,
								'initial_value': first_value
							})

					fig.add_trace(go.Scatter(
						x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
						y=patient_df[variable_column],
						mode='lines+markers',
						name=f'Patient {pid}',
						hovertemplate=(
							'Day: %{x}<br>'
							f'{variable_display}: %{{y:.1f}}<br>'
							'<extra>Patient %{text}</extra>'
						),
						text=[str(pid)] * len(patient_df),
						line=dict(width=2),
						marker=dict(size=8)
					))

		# Update layout with improved styling
		if plot_type == "Line Plot":
			plot_title = f"{variable_display} Progression - All Patients"
			x_title = CN.DAYS_SINCE_FIRST_VISIT
			y_title = f"{variable_display}" if variable_display != CN.WOUND_AREA else "Wound Area (cm²)"
		elif plot_type in ["Box Plot", "Violin Plot"] and 'xaxis_title' not in locals():
			plot_title = f"{variable_display} Distribution - All Patients"
			x_title = "Wound Improvement Status" if plot_type == "Violin Plot" else ""
			y_title = f"{variable_display}" if variable_display != CN.WOUND_AREA else "Wound Area (cm²)"
		elif 'title' not in locals():
			plot_title = f"{variable_display} - All Patients"
			x_title = variable_display if plot_type == "Histogram" else "Individual Data Points (jittered)" if plot_type == "Strip Plot" else ""
			y_title = "Count" if plot_type == "Histogram" else variable_display
		else:
			# Use titles already set above
			plot_title = fig.layout.title.text if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text') else f"{variable_display} - All Patients"
			x_title = fig.layout.xaxis.title.text if hasattr(fig.layout, 'xaxis') and hasattr(fig.layout.xaxis, 'title') else ""
			y_title = fig.layout.yaxis.title.text if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title') else variable_display

		fig.update_layout(
			title=dict(
				text=plot_title,
				font=dict(size=20)
			),
			xaxis=dict(
				title=x_title,
				title_font=dict(size=14),
				gridcolor='lightgray',
				showgrid=True
			),
			yaxis=dict(
				title=y_title,
				title_font=dict(size=14),
				range=[lower_bound, upper_bound] if plot_type not in ["Histogram"] else None,
				gridcolor='lightgray',
				showgrid=True
			),
			hovermode='closest',
			showlegend=plot_type == "Line Plot" or (plot_type == "Violin Plot" and CN.OVERALL_IMPROVEMENT in df.columns),
			legend=dict(
				yanchor="top",
				y=1,
				xanchor="left",
				x=1.02,
				bgcolor="rgba(255, 255, 255, 0.8)",
				bordercolor="lightgray",
				borderwidth=1
			),
			margin=dict(l=60, r=120, t=50, b=50),
			plot_bgcolor='white'
		)

		# Add annotation about outlier filtering if applicable
		if plot_type != "Histogram":  # No outlier filtering for histograms
			annotation_text = (
				"Note: No outliers removed" if outlier_threshold == 0 else
				f"Note: Outliers removed using combined IQR and z-score methods<br>"
				f"Threshold: {outlier_threshold:.2f} quantile"
			)

			fig.add_annotation(
				text=annotation_text,
				xref="paper", yref="paper",
				x=0.99, y=0.02,
				showarrow=False,
				font=dict(size=10, color="gray"),
				xanchor="right",
				yanchor="bottom",
				align="right"
			)

		# Display the plot
		st.plotly_chart(fig, use_container_width=True)
		Visualizer.plot_interpretation(plot_type=plot_type, group_by=group_by, variable_display=variable_display, df=df, CN=CN)

		return fig


	@staticmethod
	def plot_interpretation(plot_type: str, group_by: str, variable_display: str, df: pd.DataFrame, CN: DColumns) -> None:

		with st.expander("Interpretation"):
			# Add interpretation text based on the selected plot type
			if plot_type == "Line Plot":
				st.markdown("""
				### How to Interpret This Line Plot

				**Line Plot Interpretation:**
				* Each line represents an individual patient's progression over time
				* The x-axis shows days since the patient's first recorded visit
				* Upward trends indicate increasing measurements, while downward trends indicate decreasing measurements
				* For wound area, decreasing trends generally indicate healing (positive progress)
				* Steeper slopes indicate more rapid change
				* Comparing slopes between patients helps identify different healing rates
				* Plateaus may indicate stalled healing or treatment effects

				**Action Items:**
				* Investigate patients with unusual trajectories or sudden changes in slope
				* Consider treatment adjustments for patients with static or worsening trends
				* Look for commonalities among patients with similar trajectories
				""")

			elif plot_type == "Box Plot":
				group_text = f" across {group_by.lower()} periods" if group_by != "None" else ""
				st.markdown(f"""
				### How to Interpret This Box Plot

				**Box Plot Components{group_text}:**
				* **Box**: The interquartile range (IQR) containing the middle 50% of values
				* **Line inside box**: The median (middle value)
				* **Bottom of box**: 25th percentile (Q1)
				* **Top of box**: 75th percentile (Q3)
				* **Whiskers**: Extend to the most extreme data points within 1.5 × IQR from the box
				* **Individual points**: Potential outliers beyond the whiskers

				**Clinical Insights:**
				* The median (middle line) shows the central tendency of {variable_display.lower()}
				* Box height indicates variability/spread of the values
				* Skewed boxes suggest uneven distribution of values
				* Outliers may represent patients with unusual responses or measurement errors
				* {f"Progression across {group_by.lower()} periods reveals temporal patterns in healing" if group_by != "None" else ""}

				**For Further Investigation:**
				* Focus on outliers to identify patients with atypical values
				* Compare box height to understand variability in the cohort
				* {f"Track median changes across {group_by.lower()} periods to assess overall population trends" if group_by != "None" else ""}
				""")

			elif plot_type == "Histogram":
				st.markdown(f"""
				### How to Interpret This Histogram

				**Histogram Components:**
				* Each bar represents the count of observations falling within a specific range (bin)
				* The x-axis shows the values of {variable_display.lower()}
				* The y-axis shows the frequency (count) of observations in each bin
				* The total area of all bars equals the total number of observations

				**Distribution Characteristics:**
				* **Peak(s)**: Most common value range(s)
				* **Spread**: Width of the distribution
				* **Shape**: Normal (bell-shaped), skewed, bimodal, or multimodal

				**Clinical Significance:**
				* Single peak (unimodal): Suggests a homogeneous patient population
				* Multiple peaks: May indicate distinct subgroups of patients
				* Skewed right (tail extends right): Common for wound area measurements
				* Skewed left (tail extends left): Less common, may indicate unusual distribution

				**For Analysis:**
				* Adjust bin count to reveal patterns that might be hidden
				* Look for natural breakpoints or clusters that might suggest patient subgroups
				* Consider whether the distribution supports using mean vs. median for summary statistics
				""")

			elif plot_type == "Violin Plot":
				outcome_text = "by healing outcome status" if CN.OVERALL_IMPROVEMENT in df.columns else ""
				st.markdown(f"""
				### How to Interpret This Violin Plot

				**Violin Plot Components {outcome_text}:**
				* The width at each y-value shows the probability density at that value
				* Wider sections represent more common values
				* The internal box plot shows:
				* Median (white dot or line in center)
				* Interquartile range (IQR) box
				* Whiskers extending to 1.5 × IQR
				* Individual dots show actual data points

				**Key Insights:**
				* Shape indicates the distribution characteristics:
				* Symmetrical: evenly distributed around the median
				* Skewed: more values concentrated at top or bottom
				* Bimodal: two distinct peaks suggesting subgroups
				* Width at any point shows relative frequency of that value
				* {f"Comparing shapes between outcome groups reveals differences in {variable_display.lower()} distribution" if CN.OVERALL_IMPROVEMENT in df.columns else ""}

				**Clinical Applications:**
				* Identify value ranges most associated with positive outcomes
				* Compare distribution shapes to detect population differences
				* Assess whether {variable_display.lower()} shows different patterns for different healing outcomes
				""")

			elif plot_type == "Strip Plot":
				st.markdown(f"""
				### How to Interpret This Strip Plot

				**Strip Plot Features:**
				* Each dot represents a single measurement from a patient visit
				* Dots are jittered horizontally to prevent overlap
				* The vertical position shows the exact {variable_display.lower()} value
				* All data points are shown without aggregation

				**Advantages of This View:**
				* Shows the complete raw dataset without summarization
				* Reveals the actual distribution and density of values
				* Identifies clusters, gaps, and potential outliers
				* Provides a sense of sample size through dot density

				**For Analysis:**
				* Look for natural clustering of values that might indicate subgroups
				* Identify sparse or dense regions in the value range
				* Detect outliers that might warrant further investigation
				* Assess the overall range and spread of {variable_display.lower()} values in the population
				""")


	@staticmethod
	def _create_single_patient_plot(df: pd.DataFrame, patient_id: int) -> go.Figure:
		"""
		Create a detailed plot showing wound healing progression for a single patient.

		This function generates a Plotly figure with multiple traces showing the wound area,
		dimensions (length, width, depth), and a trend line for the wound area. It also
		calculates and displays the healing rate if sufficient data is available.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing wound measurements data with columns:
			CN.RECORD_ID, 'Days_Since_First_Visit', 'Calculated Wound Area',
			'Length (cm)', 'Width (cm)', 'Depth (cm)'

		patient_id : int
			The patient identifier to filter data for

		Returns
		-------
		go.Figure
			A Plotly figure object containing the wound progression plot with:
			- Wound area measurements (blue line)
			- Wound length measurements (green line)
			- Wound width measurements (red line)
			- Wound depth measurements (brown line)
			- Trend line for wound area (dashed red line, if sufficient data points)
			- Annotation showing healing rate and status (if sufficient time elapsed)

		Notes
		-----
		- The trend line is calculated using linear regression (numpy.polyfit)
		- Healing rate is calculated as (first_area - last_area) / total_days
		- The plot includes hover information and unified hover mode
		"""
		CN = DColumns(df=df)
		# Access columns directly with uppercase attributes


		patient_df = df[df[CN.RECORD_ID] == patient_id].sort_values(CN.DAYS_SINCE_FIRST_VISIT)

		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
			y=patient_df[CN.WOUND_AREA],
			mode='lines+markers',
			name='Wound Area',
			line=dict(color='blue'),
			hovertemplate='%{y:.1f} cm²'
		))

		fig.add_trace(go.Scatter(
			x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
			y=patient_df[CN.LENGTH],
			mode='lines+markers',
			name='Length (cm)',
			line=dict(color='green'),
			hovertemplate='%{y:.1f} cm'
		))

		fig.add_trace(go.Scatter(
			x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
			y=patient_df[CN.WIDTH],
			mode='lines+markers',
			name='Width (cm)',
			line=dict(color='red'),
			hovertemplate='%{y:.1f} cm'
		))

		fig.add_trace(go.Scatter(
			x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
			y=patient_df[CN.DEPTH],
			mode='lines+markers',
			name='Depth (cm)',
			line=dict(color='brown'),
			hovertemplate='%{y:.1f} cm'
		))

		if len(patient_df) >= 2:

			x = patient_df[CN.DAYS_SINCE_FIRST_VISIT].values
			y = patient_df[CN.WOUND_AREA].values
			mask = np.isfinite(x) & np.isfinite(y)

			# Add trendline
			if np.sum(mask) >= 2:
				z = np.polyfit(x[mask], y[mask], 1)
				p = np.poly1d(z)

				# Add trend line
				fig.add_trace(go.Scatter(
					x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
					y=p(patient_df[CN.DAYS_SINCE_FIRST_VISIT]),
					mode='lines',
					name='Trend',
					line=dict(color='red', dash='dash'),
					hovertemplate='Day %{x}<br>Trend: %{y:.1f} cm²'
				))

			# Calculate and display healing rate
			total_days = patient_df[CN.DAYS_SINCE_FIRST_VISIT].max()
			if total_days > 0:
				first_area        = patient_df.loc[patient_df[CN.DAYS_SINCE_FIRST_VISIT].idxmin(), CN.WOUND_AREA]
				last_area         = patient_df.loc[patient_df[CN.DAYS_SINCE_FIRST_VISIT].idxmax(), CN.WOUND_AREA]
				healing_rate      = (first_area - last_area) / total_days
				healing_status    = "Improving" if healing_rate > 0 else "Worsening"
				healing_rate_text = f"Healing Rate: {healing_rate:.2f} cm²/day<br> {healing_status}"
			else:
				healing_rate_text = "Insufficient time between measurements for healing rate calculation"

			fig.add_annotation(
				x=0.02,
				y=0.98,
				xref="paper",
				yref="paper",
				text=healing_rate_text,
				showarrow=False,
				font=dict(size=12),
				bgcolor="rgba(255, 255, 255, 0.8)",
				bordercolor="black",
				borderwidth=1
			)

		fig.update_layout(
			title=f"Wound Area Progression - Patient {patient_id}",
			xaxis_title=f"{CN.DAYS_SINCE_FIRST_VISIT}",
			yaxis_title="Wound Area (cm²)",
			hovermode='x unified',
			showlegend=True
		)



		return fig

	@staticmethod
	def create_temperature_chart(df):
		"""
		Creates an interactive temperature chart for wound analysis using Plotly.

		The function generates line charts for available temperature measurements (Center, Edge, Peri-wound)
		and bar charts for temperature gradients when all three measurements are available.

		Parameters:
		-----------
		df : pandas.DataFrame
			DataFrame containing wound temperature data with columns for visit date, temperature
			measurements at different locations, and visit status

		Returns:
		--------
		plotly.graph_objs._figure.Figure
			A Plotly figure object with temperature measurements as line charts on the primary y-axis
			and temperature gradients as bar charts on the secondary y-axis (if all temperature types available).

		Notes:
		------
		- Skipped visits are excluded from visualization
		- Derived temperature gradients are calculated when all three temperature measurements are available
		- Color coding: Center (red), Edge (orange), Peri-wound (blue)
		- Temperature gradients are displayed as semi-transparent bars
		"""
		# Initialize DataColumns and update with the dataframe
		CN = DColumns(df=df)
		# Access columns directly with uppercase attributes

		# Check which temperature columns have data
		temp_cols = {
			'Center': CN.CENTER_TEMP,
			'Edge': CN.EDGE_TEMP,
			'Peri': CN.PERI_TEMP
		}

		# Remove skipped visits
		df = df[df[CN.SKIPPED_VISIT] != 'Yes']

		# Create derived variables for temperature if they exist
		if all(col in df.columns for col in [CN.CENTER_TEMP, CN.EDGE_TEMP, CN.PERI_TEMP]):
			df['Center-Edge Temp Gradient'] = df[CN.CENTER_TEMP] - df[CN.EDGE_TEMP]
			df['Edge-Peri Temp Gradient']   = df[CN.EDGE_TEMP]   - df[CN.PERI_TEMP]
			df['Total Temp Gradient']       = df[CN.CENTER_TEMP] - df[CN.PERI_TEMP]

		available_temps = {k: v for k, v in temp_cols.items()
							if v in df.columns and not df[v].isna().all()}

		fig = make_subplots(specs=[[{"secondary_y": len(available_temps) == 3}]])

		# Color mapping for temperature lines
		colors = {'Center': 'red', 'Edge': 'orange', 'Peri': 'blue'}

		# Add available temperature lines
		for temp_name, col_name in available_temps.items():
			fig.add_trace(
				go.Scatter(
					x=df[CN.VISIT_DATE],
					y=df[col_name],
					name=f"{temp_name} Temp",
					line=dict(color=colors[temp_name]),
					mode='lines+markers'
				)
			)

		# Only add gradients if all three temperatures are available
		if len(available_temps) == 3:
			# Calculate temperature gradients
			df['Center-Edge'] = df[CN.CENTER_TEMP] - df[CN.EDGE_TEMP]
			df['Edge-Peri']   = df[CN.EDGE_TEMP]   - df[CN.PERI_TEMP]

			# Add gradient bars on secondary y-axis
			fig.add_trace(
				go.Bar(
					x=df[CN.VISIT_DATE],
					y=df['Center-Edge'],
					name="Center-Edge Gradient",
					opacity=0.5,
					marker_color='lightpink'
				),
				secondary_y=True
			)
			fig.add_trace(
				go.Bar(
					x=df[CN.VISIT_DATE],
					y=df['Edge-Peri'],
					name="Edge-Peri Gradient",
					opacity=0.5,
					marker_color='lightblue'
				),
				secondary_y=True
			)

			# Add secondary y-axis title only if showing gradients
			fig.update_yaxes(title_text="Temperature Gradient (°F)", secondary_y=True)

		return fig


	@staticmethod
	def create_exudate_chart(visits: list, VISIT_DATE_TAG: str) -> go.Figure:
		"""
		Create a chart showing exudate characteristics over time.

		This function processes a series of visit data to visualize changes in wound exudate
		characteristics across multiple visits. It displays exudate volume as a line graph,
		and exudate type and viscosity as text markers on separate horizontal lines.

		Parameters:
		-----------
		visits : list
			A list of dictionaries, where each dictionary represents a visit and contains:
			- VISIT_DATE_TAG: datetime or string, the date of the visit
			- 'wound_info': dict, containing wound information including an 'exudate' key with:
				- 'volume': numeric, the volume of exudate
				- 'type': string, the type of exudate (e.g., serous, sanguineous)
				- 'viscosity': string, the viscosity of exudate (e.g., thin, thick)

		Returns:
		--------
		go.Figure
			A plotly figure object containing the exudate characteristics chart with
			three potential traces: volume as a line graph, and type and viscosity as
			text markers on fixed y-positions.

		Note:
		-----
		The function handles missing data gracefully, only plotting traces if data exists.
		"""

		dates = []
		volumes = []
		types = []
		viscosities = []

		for visit in visits:
			date       = visit[VISIT_DATE_TAG]
			wound_info = visit['wound_info']
			exudate    = wound_info.get('exudate', {})

			dates.append(date)
			volumes.append(exudate.get('volume', None))
			types.append(exudate.get('type', None))
			viscosities.append(exudate.get('viscosity', None))

		# Create figure for categorical data
		fig = go.Figure()

		# Add volume as lines
		if any(volumes):
			fig.add_trace(go.Scatter(x=dates, y=volumes, name='Volume', mode='lines+markers'))

		# Add types and viscosities as markers with text
		if any(types):
			fig.add_trace(go.Scatter(
				x=dates,
				y=[1]*len(dates),
				text=types,
				name='Type',
				mode='markers+text',
				textposition='bottom center'
			))

		if any(viscosities):
			fig.add_trace(go.Scatter(
				x=dates,
				y=[0]*len(dates),
				text=viscosities,
				name='Viscosity',
				mode='markers+text',
				textposition='top center'
			))

		fig.update_layout(
			# title='Exudate Characteristics Over Time',
			xaxis_title='Visit Date',
			yaxis_title='Properties',
			hovermode='x unified'
		)
		return fig
