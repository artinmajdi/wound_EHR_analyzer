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

	_create_all_patients_plot(df)
		Create an interactive line plot showing wound area progression for all patients.

	_create_single_patient_plot(df, patient_id)
		Create a detailed wound area and dimensions plot for a single patient.
	"""

	@staticmethod
	def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
		"""
		Create a wound area progression plot for either a single patient or all patients.

		This function generates a visualization of wound area changes over time.
		If a patient_id is provided, it creates a plot specific to that patient.
		Otherwise, it creates a comparative plot for all patients in the DataFrame.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing wound area measurements and dates
		patient_id : Optional[int], default=None
			The ID of a specific patient to plot. If None, plots data for all patients

		Returns
		-------
		go.Figure
			A Plotly figure object containing the wound area progression plot
		"""
		if patient_id is None or patient_id == "All Patients":
			return Visualizer._create_all_patients_plot(df)

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
	def _create_all_patients_plot(df: pd.DataFrame) -> go.Figure:
		"""
		Create an interactive line plot showing wound area progression for all patients over time.

		This function generates a Plotly figure where each patient's wound area is plotted against
		days since their first visit. The plot includes interactive features such as hovering for
		patient details and an outlier threshold control to filter extreme values.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing patient wound data with columns:
			- CN.RECORD_ID: Patient identifier
			- CN.VISIT_DATE: Date of the wound assessment
			- 'Calculated Wound Area': Wound area measurements in square centimeters

		Returns
		-------
		go.Figure
			A Plotly figure object containing the wound area progression plot for all patients.
			The figure includes:
			- Individual patient lines with distinct colors
			- Interactive hover information with patient ID and wound area
			- Y-axis automatically scaled based on outlier threshold setting
			- Annotation explaining outlier removal status

		Notes
		-----
		The function adds a Streamlit number input widget for controlling outlier removal threshold.
		Patient progression lines are colored based on their healing rates.
		"""
		# Add outlier threshold control
		col1, col2 = st.columns([4, 1])
		with col2:
			outlier_threshold = st.number_input(
				"Temperature Outlier Threshold",
				min_value=0.0,
				max_value=0.5,
				value=0.0,
				step=0.01,
				help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		fig = go.Figure()

		# Initialize DataColumns and update with the dataframe
		CN = DColumns(df=df)
		# Access columns directly with uppercase attributes

		# Get the filtered data for y-axis limits
		all_wound_areas_df = pd.DataFrame({CN.WOUND_AREA: df[CN.WOUND_AREA].dropna()})
		filtered_df = Visualizer._remove_outliers(all_wound_areas_df, CN.WOUND_AREA, outlier_threshold)

		# Set y-axis limits based on filtered data
		lower_bound = 0
		upper_bound = (filtered_df[CN.WOUND_AREA].max() if outlier_threshold > 0 else all_wound_areas_df[CN.WOUND_AREA].max()) * 1.05

		# Store patient statistics for coloring
		patient_stats = []

		for pid in df[CN.RECORD_ID].unique():
			patient_df = df[df[CN.RECORD_ID] == pid].copy()
			patient_df[CN.DAYS_SINCE_FIRST_VISIT] = (pd.to_datetime(patient_df[CN.VISIT_DATE]) - pd.to_datetime(patient_df[CN.VISIT_DATE]).min()).dt.days

			# Remove NaN values
			patient_df = patient_df.dropna(subset=[CN.DAYS_SINCE_FIRST_VISIT, CN.WOUND_AREA])

			if not patient_df.empty:
				# Calculate healing rate for this patient
				if len(patient_df) >= 2:
					first_area = patient_df.loc[patient_df[CN.DAYS_SINCE_FIRST_VISIT].idxmin(), CN.WOUND_AREA]
					last_area = patient_df.loc[patient_df[CN.DAYS_SINCE_FIRST_VISIT].idxmax(), CN.WOUND_AREA]
					total_days = patient_df[CN.DAYS_SINCE_FIRST_VISIT].max()

					if total_days > 0:
						healing_rate = (first_area - last_area) / total_days
						patient_stats.append({
							'pid': pid,
							'healing_rate': healing_rate,
							'initial_area': first_area
						})

				fig.add_trace(go.Scatter(
					x=patient_df[CN.DAYS_SINCE_FIRST_VISIT],
					y=patient_df[CN.WOUND_AREA],
					mode='lines+markers',
					name=f'Patient {pid}',
					hovertemplate=(
						'Day: %{x}<br>'
						'Area: %{y:.1f} cm²<br>'
						'<extra>Patient %{text}</extra>'
					),
					text=[str(pid)] * len(patient_df),
					line=dict(width=2),
					marker=dict(size=8)
				))

		# Update layout with improved styling
		fig.update_layout(
			title=dict(
				text="Wound Area Progression - All Patients",
				font=dict(size=20)
			),
			xaxis=dict(
				title=CN.DAYS_SINCE_FIRST_VISIT,
				title_font=dict(size=14),
				gridcolor='lightgray',
				showgrid=True
			),
			yaxis=dict(
				title="Wound Area (cm²)",
				title_font=dict(size=14),
				range=[lower_bound, upper_bound],
				gridcolor='lightgray',
				showgrid=True
			),
			hovermode='closest',
			showlegend=True,
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

		# Update annotation text based on outlier threshold
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

		return fig

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
