from typing import Optional
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from wound_analysis.utils.column_schema import DataColumns

data_columns = DataColumns()

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
			print(df['Oxygenation (%)'].to_frame().describe())
			print(df['Oxygenation (%)'].to_frame())
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
			- 'Record ID': Patient identifier
			- 'Visit date': Date of the wound assessment
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

		# Get the filtered data for y-axis limits
		all_wound_areas_df = pd.DataFrame({'wound_area': df['Calculated Wound Area'].dropna()})
		filtered_df = Visualizer._remove_outliers(all_wound_areas_df, 'wound_area', outlier_threshold)

		# Set y-axis limits based on filtered data
		lower_bound = 0
		upper_bound = (filtered_df['wound_area'].max() if outlier_threshold > 0 else all_wound_areas_df['wound_area'].max()) * 1.05

		# Store patient statistics for coloring
		patient_stats = []

		for pid in df['Record ID'].unique():
			patient_df = df[df['Record ID'] == pid].copy()
			patient_df['Days_Since_First_Visit'] = (pd.to_datetime(patient_df['Visit date']) - pd.to_datetime(patient_df['Visit date']).min()).dt.days

			# Remove NaN values
			patient_df = patient_df.dropna(subset=['Days_Since_First_Visit', 'Calculated Wound Area'])

			if not patient_df.empty:
				# Calculate healing rate for this patient
				if len(patient_df) >= 2:
					first_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmin(), 'Calculated Wound Area']
					last_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmax(), 'Calculated Wound Area']
					total_days = patient_df['Days_Since_First_Visit'].max()

					if total_days > 0:
						healing_rate = (first_area - last_area) / total_days
						patient_stats.append({
							'pid': pid,
							'healing_rate': healing_rate,
							'initial_area': first_area
						})

				fig.add_trace(go.Scatter(
					x=patient_df['Days_Since_First_Visit'],
					y=patient_df['Calculated Wound Area'],
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
				title="Days Since First Visit",
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
			'Record ID', 'Days_Since_First_Visit', 'Calculated Wound Area',
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
		record_id_str              = data_columns.patient_identifiers.record_id
		days_since_first_visit_str = data_columns.visit_info.days_since_first_visit
		patient_df = df[df[record_id_str] == patient_id].sort_values(days_since_first_visit_str)

		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=patient_df['Days_Since_First_Visit'],
			y=patient_df['Calculated Wound Area'],
			mode='lines+markers',
			name='Wound Area',
			line=dict(color='blue'),
			hovertemplate='%{y:.1f} cm²'
		))

		fig.add_trace(go.Scatter(
			x=patient_df['Days_Since_First_Visit'],
			y=patient_df['Length (cm)'],
			mode='lines+markers',
			name='Length (cm)',
			line=dict(color='green'),
			hovertemplate='%{y:.1f} cm'
		))

		fig.add_trace(go.Scatter(
			x=patient_df['Days_Since_First_Visit'],
			y=patient_df['Width (cm)'],
			mode='lines+markers',
			name='Width (cm)',
			line=dict(color='red'),
			hovertemplate='%{y:.1f} cm'
		))

		fig.add_trace(go.Scatter(
			x=patient_df['Days_Since_First_Visit'],
			y=patient_df['Depth (cm)'],
			mode='lines+markers',
			name='Depth (cm)',
			line=dict(color='brown'),
			hovertemplate='%{y:.1f} cm'
		))

		if len(patient_df) >= 2:

			x = patient_df['Days_Since_First_Visit'].values
			y = patient_df['Calculated Wound Area'].values
			mask = np.isfinite(x) & np.isfinite(y)

			# Add trendline
			if np.sum(mask) >= 2:
				z = np.polyfit(x[mask], y[mask], 1)
				p = np.poly1d(z)

				# Add trend line
				fig.add_trace(go.Scatter(
					x=patient_df['Days_Since_First_Visit'],
					y=p(patient_df['Days_Since_First_Visit']),
					mode='lines',
					name='Trend',
					line=dict(color='red', dash='dash'),
					hovertemplate='Day %{x}<br>Trend: %{y:.1f} cm²'
				))

			# Calculate and display healing rate
			total_days = patient_df['Days_Since_First_Visit'].max()
			if total_days > 0:
				first_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmin(), 'Calculated Wound Area']
				last_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmax(), 'Calculated Wound Area']
				healing_rate = (first_area - last_area) / total_days
				healing_status = "Improving" if healing_rate > 0 else "Worsening"
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
			xaxis_title="Days Since First Visit",
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
			DataFrame containing wound temperature data with columns:
			- 'Visit date': Dates of wound assessment visits
			- 'Center of Wound Temperature (Fahrenheit)': Temperature at wound center (optional)
			- 'Edge of Wound Temperature (Fahrenheit)': Temperature at wound edge (optional)
			- 'Peri-wound Temperature (Fahrenheit)': Temperature of surrounding tissue (optional)
			- 'Skipped Visit?': Indicator for skipped visits

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
		# Check which temperature columns have data
		temp_cols = {
			'Center': 'Center of Wound Temperature (Fahrenheit)',
			'Edge': 'Edge of Wound Temperature (Fahrenheit)',
			'Peri': 'Peri-wound Temperature (Fahrenheit)'
		}

		# Remove rows where all three temperature values are NaN
		# df_temp = df_temp.dropna(subset=list(temp_cols.values()))
		# Remove skipped visits
		df = df[df['Skipped Visit?'] != 'Yes']

		# Create derived variables for temperature if they exist
		if all(col in df.columns for col in ['Center of Wound Temperature (Fahrenheit)', 'Edge of Wound Temperature (Fahrenheit)', 'Peri-wound Temperature (Fahrenheit)']):
			df['Center-Edge Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Edge of Wound Temperature (Fahrenheit)']
			df['Edge-Peri Temp Gradient']   = df['Edge of Wound Temperature (Fahrenheit)']   - df['Peri-wound Temperature (Fahrenheit)']
			df['Total Temp Gradient']       = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']

		available_temps = {k: v for k, v in temp_cols.items()
							if v in df.columns and not df[v].isna().all()}

		fig = make_subplots(specs=[[{"secondary_y": len(available_temps) == 3}]])

		# Color mapping for temperature lines
		colors = {'Center': 'red', 'Edge': 'orange', 'Peri': 'blue'}

		# Add available temperature lines
		for temp_name, col_name in available_temps.items():
			fig.add_trace(
				go.Scatter(
					x=df['Visit date'],
					y=df[col_name],
					name=f"{temp_name} Temp",
					line=dict(color=colors[temp_name]),
					mode='lines+markers'
				)
			)

		# Only add gradients if all three temperatures are available
		if len(available_temps) == 3:
			# Calculate temperature gradients
			df['Center-Edge'] = df[temp_cols['Center']] - df[temp_cols['Edge']]
			df['Edge-Peri']   = df[temp_cols['Edge']] - df[temp_cols['Peri']]

			# Add gradient bars on secondary y-axis
			fig.add_trace(
				go.Bar(
					x=df['Visit date'],
					y=df['Center-Edge'],
					name="Center-Edge Gradient",
					opacity=0.5,
					marker_color='lightpink'
				),
				secondary_y=True
			)
			fig.add_trace(
				go.Bar(
					x=df['Visit date'],
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
	def create_impedance_chart(visits, measurement_mode: str = "Absolute Impedance (|Z|)"):
		"""
		Create an interactive chart displaying impedance measurements over time for different frequencies.

		This function processes visit data that contains sensor impedance measurements and generates
		a plotly figure showing the selected impedance parameter (absolute impedance, resistance, or capacitance)
		across visits on a logarithmic scale.

		Parameters:
		-----------
		visits : list
			List of visit dictionaries, each containing visit date and sensor data with impedance measurements.
			Each visit should have the structure:
			{
				'visit_date': datetime,
				'sensor_data': {
					'impedance': {
						'high_frequency'  : {'Z': float, 'resistance': float, 'capacitance': float, 'frequency': float},
						'center_frequency': {'Z': float, 'resistance': float, 'capacitance': float, 'frequency': float},
						'low_frequency'   : {'Z': float, 'resistance': float, 'capacitance': float, 'frequency': float}
					}
				}
			}

		measurement_mode : str, optional
			Type of impedance measurement to display. Options are:
			- "Absolute Impedance (|Z|)" (default)
			- "Resistance"
			- "Capacitance"

		Returns:
		--------
		plotly.graph_objects.Figure
			Interactive plotly figure showing the selected impedance parameter over time with
			separate traces for each frequency (high, center, low).
			The y-axis is displayed on a logarithmic scale for better visualization.
		"""

		dates = []
		high_freq_z  , high_freq_r,   high_freq_c   = [], [], []
		center_freq_z, center_freq_r, center_freq_c = [], [], []
		low_freq_z   , low_freq_r,    low_freq_c    = [], [], []

		high_freq_val, center_freq_val, low_freq_val = None, None, None

		for visit in visits:
			date = visit['visit_date']
			sensor_data = visit.get('sensor_data', {})
			impedance_data = sensor_data.get('impedance', {})

			dates.append(date)

			# Process high frequency data
			high_freq = impedance_data.get('high_frequency', {})
			try:
				z_val = float(high_freq.get('Z')) 			if high_freq.get('Z') 			not in (None, '') else None
				r_val = float(high_freq.get('resistance'))  if high_freq.get('resistance')  not in (None, '') else None
				c_val = float(high_freq.get('capacitance')) if high_freq.get('capacitance') not in (None, '') else None

				if high_freq.get('frequency'):
					high_freq_val = high_freq.get('frequency')

				high_freq_z.append(z_val)
				high_freq_r.append(r_val)
				high_freq_c.append(c_val)
			except (ValueError, TypeError):
				high_freq_z.append(None)
				high_freq_r.append(None)
				high_freq_c.append(None)

			# Process center frequency data
			center_freq = impedance_data.get('center_frequency', {})
			try:
				z_val = float(center_freq.get('Z')) 		 if center_freq.get('Z') 			not in (None, '') else None
				r_val = float(center_freq.get('resistance'))  if center_freq.get('resistance')  not in (None, '') else None
				c_val = float(center_freq.get('capacitance')) if center_freq.get('capacitance') not in (None, '') else None

				if center_freq.get('frequency'):
					center_freq_val = center_freq.get('frequency')

				center_freq_z.append(z_val)
				center_freq_r.append(r_val)
				center_freq_c.append(c_val)
			except (ValueError, TypeError):
				center_freq_z.append(None)
				center_freq_r.append(None)
				center_freq_c.append(None)

			# Process low frequency data
			low_freq = impedance_data.get('low_frequency', {})
			try:
				z_val = float(low_freq.get('Z')) 		   if low_freq.get('Z') 		  not in (None, '') else None
				r_val = float(low_freq.get('resistance'))  if low_freq.get('resistance')  not in (None, '') else None
				c_val = float(low_freq.get('capacitance')) if low_freq.get('capacitance') not in (None, '') else None

				if low_freq.get('frequency'):
					low_freq_val = low_freq.get('frequency')

				low_freq_z.append(z_val)
				low_freq_r.append(r_val)
				low_freq_c.append(c_val)

			except (ValueError, TypeError):
				low_freq_z.append(None)
				low_freq_r.append(None)
				low_freq_c.append(None)

		fig = go.Figure()

		# Format frequency labels
		# high_freq_label   = f"{float(high_freq_val):.0f}Hz"   if high_freq_val 	 else "High Freq"
		# center_freq_label = f"{float(center_freq_val):.0f}Hz" if center_freq_val else "Center Freq"
		# low_freq_label    = f"{float(low_freq_val):.0f}Hz" 	  if low_freq_val	 else "Low Freq"

		high_freq_label   = "Highest Freq"
		center_freq_label = "Center Freq (Max Phase Angle)"
		low_freq_label    = "Lowest Freq"

		# Add high frequency traces
		if measurement_mode == "Absolute Impedance (|Z|)":
			if any(x is not None for x in high_freq_z):
				fig.add_trace(go.Scatter(
					x=dates, y=high_freq_z,
					name=f'|Z| ({high_freq_label})',
					mode='lines+markers'
				))
		elif measurement_mode == "Resistance":
			if any(x is not None for x in high_freq_r):
				fig.add_trace(go.Scatter(
					x=dates, y=high_freq_r,
					name=f'Resistance ({high_freq_label})',
					mode='lines+markers'
				))
		elif measurement_mode == "Capacitance":
			if any(x is not None for x in high_freq_c):
				fig.add_trace(go.Scatter(
					x=dates, y=high_freq_c,
					name=f'Capacitance ({high_freq_label})',
					mode='lines+markers'
				))

		# Add center frequency traces
		if measurement_mode == "Absolute Impedance (|Z|)":
			if any(x is not None for x in center_freq_z):
				fig.add_trace(go.Scatter(
					x=dates, y=center_freq_z,
					name=f'|Z| ({center_freq_label})',
					mode='lines+markers',
					line=dict(dash='dot')
				))
		elif measurement_mode == "Resistance":
			if any(x is not None for x in center_freq_r):
				fig.add_trace(go.Scatter(
					x=dates, y=center_freq_r,
					name=f'Resistance ({center_freq_label})',
					mode='lines+markers',
					line=dict(dash='dot')
				))
		elif measurement_mode == "Capacitance":
			if any(x is not None for x in center_freq_c):
				fig.add_trace(go.Scatter(
					x=dates, y=center_freq_c,
					name=f'Capacitance ({center_freq_label})',
					mode='lines+markers',
					line=dict(dash='dot')
				))

		# Add low frequency traces
		if measurement_mode == "Absolute Impedance (|Z|)":
			if any(x is not None for x in low_freq_z):
				fig.add_trace(go.Scatter(
					x=dates, y=low_freq_z,
					name=f'|Z| ({low_freq_label})',
					mode='lines+markers',
					line=dict(dash='dash')
				))
		elif measurement_mode == "Resistance":
			if any(x is not None for x in low_freq_r):
				fig.add_trace(go.Scatter(
					x=dates, y=low_freq_r,
					name=f'Resistance ({low_freq_label})',
					mode='lines+markers',
					line=dict(dash='dash')
				))
		elif measurement_mode == "Capacitance":
			if any(x is not None for x in low_freq_c):
				fig.add_trace(go.Scatter(
					x=dates, y=low_freq_c,
					name=f'Capacitance ({low_freq_label})',
					mode='lines+markers',
					line=dict(dash='dash')
				))

		fig.update_layout(
			title='Impedance Measurements Over Time',
			xaxis_title='Visit Date',
			yaxis_title='Impedance Values',
			hovermode='x unified',
			showlegend=True,
			height=600,
			yaxis=dict(type='log'),  # Use log scale for better visualization
		)

		return fig

	@staticmethod
	def create_oxygenation_chart(patient_data, visits):
		"""
		Creates two charts for visualizing oxygenation and hemoglobin data.

		Parameters:
		-----------
		patient_data : pandas.DataFrame
			DataFrame containing patient visit data with columns 'Visit date',
			'Oxyhemoglobin Level', and 'Deoxyhemoglobin Level'.
		visits : list
			List of dictionaries, each containing visit data with keys 'visit_date'
			and 'sensor_data'. The 'sensor_data' dictionary should contain
			'oxygenation' and 'hemoglobin' measurements.

		Returns:
		--------
		tuple
			A tuple containing two plotly figures:
			- fig_bar: A stacked bar chart showing Oxyhemoglobin and Deoxyhemoglobin levels
			- fig_line: A line chart showing Oxygenation percentage and Hemoglobin levels over time

		Notes:
		------
		The hemoglobin values are multiplied by 100 for visualization purposes in the line chart.
		"""
		fig_bar = go.Figure()
		fig_bar.add_trace(go.Bar(
			x=patient_data['Visit date'],
			y=patient_data['Oxyhemoglobin Level'],
			name="Oxyhemoglobin",
			marker_color='red'
		))
		fig_bar.add_trace(go.Bar(
			x=patient_data['Visit date'],
			y=patient_data['Deoxyhemoglobin Level'],
			name="Deoxyhemoglobin",
			marker_color='purple'
		))
		fig_bar.update_layout(
			title=f"Hemoglobin Components",
			xaxis_title="Visit Date",
			yaxis_title="Level (g/dL)",
			barmode='stack',
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
		)

		# Create an interactive chart showing oxygenation and hemoglobin measurements over time.
		dates = []
		oxygenation = []
		hemoglobin = []

		for visit in visits:
			date = visit['visit_date']
			sensor_data = visit['sensor_data']
			dates.append(date)
			oxygenation.append(sensor_data.get('oxygenation'))

			# Handle None values for hemoglobin measurements
			hb = sensor_data.get('hemoglobin')
			hemoglobin.append(100 * hb if hb is not None else None)

		fig_line = go.Figure()
		fig_line.add_trace(go.Scatter(x=dates, y=oxygenation, name='Oxygenation (%)', mode='lines+markers'))
		fig_line.add_trace(go.Scatter(x=dates, y=hemoglobin, name='Hemoglobin', mode='lines+markers'))

		fig_line.update_layout(
			title='Oxygenation and Hemoglobin Measurements Over Time',
			xaxis_title='Visit Date',
			yaxis_title='Value',
			hovermode='x unified'
		)
		return fig_bar, fig_line

	@staticmethod
	def create_exudate_chart(visits):
		"""
		Create a chart showing exudate characteristics over time.

		This function processes a series of visit data to visualize changes in wound exudate
		characteristics across multiple visits. It displays exudate volume as a line graph,
		and exudate type and viscosity as text markers on separate horizontal lines.

		Parameters:
		-----------
		visits : list
			A list of dictionaries, where each dictionary represents a visit and contains:
			- 'visit_date': datetime or string, the date of the visit
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
			date       = visit['visit_date']
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
