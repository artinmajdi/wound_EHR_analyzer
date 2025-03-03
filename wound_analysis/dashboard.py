# Standard library imports
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Local application imports
from utils.column_schema import DataColumns
from utils.data_processor import DataManager, ImpedanceAnalyzer, WoundDataProcessor
from utils.llm_interface import WoundAnalysisLLM
from utils.statistical_analysis import CorrelationAnalysis

# Debug mode disabled
st.set_option('client.showErrorDetails', True)

@dataclass
class Config:
	"""Application configuration settings."""
	PAGE_TITLE: str = "Wound Care Management & Interpreter Dashboard"
	PAGE_ICON : str = "🩹"
	LAYOUT    : str = "wide"
	DATA_PATH    : Optional[pathlib.Path] = pathlib.Path('/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset')
	COLUMN_SCHEMA: DataColumns            = field(default_factory=DataColumns)

	# Constants for bioimpedance analysis
	# ANOMALY_THRESHOLD                       : float = 2.5  # Z-score threshold for anomaly detection
	# MIN_VISITS_FOR_ANALYSIS                 : int   = 3  # Minimum visits needed for trend analysis
	# SIGNIFICANT_CHANGE_THRESHOLD            : float = 15.0  # Percentage change threshold
	# INFECTION_RISK_RATIO_THRESHOLD          : float = 15.0  # Low/high frequency ratio threshold
	# SIGNIFICANT_CHANGE_THRESHOLD_IMPEDANCE  : float = 15.0  # Percentage change threshold for resistance/absolute impedance
	# SIGNIFICANT_CHANGE_THRESHOLD_CAPACITANCE: float = 20.0  # Percentage change threshold for capacitance
	# INFLAMMATORY_INCREASE_THRESHOLD         : float = 30.0  # Percentage increase threshold for low-frequency resistance

	# Exudate analysis information
	EXUDATE_TYPE_INFO = {
		'Serous': {
			'description': 'Straw-colored, clear, thin',
			'indication': 'Normal healing process',
			'severity': 'info'
		},
		'Serosanguineous': {
			'description': 'Pink or light red, thin',
			'indication': 'Presence of blood cells in early healing',
			'severity': 'info'
		},
		'Sanguineous': {
			'description': 'Red, thin',
			'indication': 'Active bleeding or trauma',
			'severity': 'warning'
		},
		'Seropurulent': {
			'description': 'Cloudy, milky, or creamy',
			'indication': 'Possible early infection or inflammation',
			'severity': 'warning'
		},
		'Purulent': {
			'description': 'Yellow, tan, or green, thick',
			'indication': 'Active infection present',
			'severity': 'error'
		}
	}

	@staticmethod
	def get_exudate_analysis(volume: str, viscosity: str, exudate_types: str = None) -> dict:
		"""
			Provides clinical interpretation and treatment implications for exudate characteristics.

			Args:
				volume: The exudate volume level ('High', 'Medium', or 'Low')
				viscosity: The exudate viscosity level ('High', 'Medium', or 'Low')
				exudate_type: The type of exudate (e.g., 'Serous', 'Purulent')

			Returns:
				A dictionary containing:
				- volume_analysis: Interpretation of volume level
				- viscosity_analysis: Interpretation of viscosity level
				- type_info: Information about the exudate type
				- treatment_implications: List of treatment recommendations
		"""
		result = {
			"volume_analysis": "",
			"viscosity_analysis": "",
			"type_info": {},
			"treatment_implications": []
		}

		# Volume interpretation
		if volume == 'High':
			result["volume_analysis"] = """
			**High volume exudate** is common in:
			- Chronic venous leg ulcers
			- Dehisced surgical wounds
			- Inflammatory ulcers
			- Burns

			This may indicate active inflammation or healing processes.
			"""
		elif volume == 'Low':
			result["volume_analysis"] = """
			**Low volume exudate** is typical in:
			- Necrotic wounds
			- Ischaemic/arterial wounds
			- Neuropathic diabetic foot ulcers

			Monitor for signs of insufficient moisture.
			"""

		# Viscosity interpretation
		if viscosity == 'High':
			result["viscosity_analysis"] = """
			**High viscosity** (thick) exudate may indicate:
			- High protein content
			- Possible infection
			- Inflammatory processes
			- Presence of necrotic material

			Consider reassessing treatment approach.
			"""
		elif viscosity == 'Low':
			result["viscosity_analysis"] = """
			**Low viscosity** (thin) exudate may suggest:
			- Low protein content
			- Possible venous condition
			- Potential malnutrition
			- Presence of fistulas

			Monitor fluid balance and nutrition.
			"""
		for exudate_type in [t.strip() for t in exudate_types.split(',')]:
			if exudate_type and exudate_type in Config.EXUDATE_TYPE_INFO:
				result["type_info"][exudate_type] = Config.EXUDATE_TYPE_INFO[exudate_type]

		# Treatment implications based on volume and viscosity
		if volume == 'High' and viscosity == 'High':
			result["treatment_implications"] = [
				"- Consider highly absorbent dressings",
				"- More frequent dressing changes may be needed",
				"- Monitor for maceration of surrounding tissue"
			]
		elif volume == 'Low' and viscosity == 'Low':
			result["treatment_implications"] = [
				"- Use moisture-retentive dressings",
				"- Protect wound bed from desiccation",
				"- Consider hydrating dressings"
			]

		return result


	@staticmethod
	def initialize() -> None:
		"""
		Initializes the Streamlit session state with necessary variables.

		This function sets up the following session state variables if they don't exist:
		- processor: Stores the wound image processor instance
		- analysis_complete: Boolean flag indicating if analysis has been completed
		- analysis_results: Stores the results of wound analysis
		- report_path: Stores the file path to the generated report

		Returns:
			None
		"""

		if 'processor' not in st.session_state:
			st.session_state.processor = None

		if 'analysis_complete' not in st.session_state:
			st.session_state.analysis_complete = False

		if 'analysis_results' not in st.session_state:
			st.session_state.analysis_results = None

		if 'report_path' not in st.session_state:
			st.session_state.report_path = None

class SessionStateManager:
	pass

class Visualizer:
	"""Handles data visualization."""

	@staticmethod
	def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
		"""Create wound area progression plot."""
		if patient_id:
			return Visualizer._create_single_patient_plot(df, patient_id)
		return Visualizer._create_all_patients_plot(df)

	@staticmethod
	def _remove_outliers(df: pd.DataFrame, column: str, quantile_threshold: float = 0.1) -> pd.DataFrame:
		"""Remove outliers using quantile thresholds and z-score validation.

		Args:
			df: DataFrame containing the data
			column: Column name to check for outliers
			quantile_threshold: Value between 0 and 0.5 for quantile-based filtering (0 = no filtering)
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
		"""Create wound area plot for all patients."""
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
		"""Create wound area plot for a single patient."""
		patient_df = df[df['Record ID'] == patient_id].sort_values('Days_Since_First_Visit')

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
			df['Edge-Peri Temp Gradient'] = df['Edge of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']
			df['Total Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']

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
			df['Edge-Peri'] = df[temp_cols['Edge']] - df[temp_cols['Peri']]

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
		"""Create an interactive chart showing impedance measurements over time."""
		dates = []
		high_freq_z, high_freq_r, high_freq_c = [], [], []
		center_freq_z, center_freq_r, center_freq_c = [], [], []
		low_freq_z, low_freq_r, low_freq_c = [], [], []

		high_freq_val, center_freq_val, low_freq_val = None, None, None

		for visit in visits:
			date = visit['visit_date']
			sensor_data = visit.get('sensor_data', {})
			impedance_data = sensor_data.get('impedance', {})

			dates.append(date)

			# Process high frequency data
			high_freq = impedance_data.get('high_frequency', {})
			try:
				z_val = float(high_freq.get('Z')) if high_freq.get('Z') not in (None, '') else None
				r_val = float(high_freq.get('resistance')) if high_freq.get('resistance') not in (None, '') else None
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
				z_val = float(center_freq.get('Z')) if center_freq.get('Z') not in (None, '') else None
				r_val = float(center_freq.get('resistance')) if center_freq.get('resistance') not in (None, '') else None
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
				z_val = float(low_freq.get('Z')) if low_freq.get('Z') not in (None, '') else None
				r_val = float(low_freq.get('resistance')) if low_freq.get('resistance') not in (None, '') else None
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
		high_freq_label = f"{float(high_freq_val):.0f}Hz" if high_freq_val else "High Freq"
		center_freq_label = f"{float(center_freq_val):.0f}Hz" if center_freq_val else "Center Freq"
		low_freq_label = f"{float(low_freq_val):.0f}Hz" if low_freq_val else "Low Freq"

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
		"""Create a chart showing exudate characteristics over time."""
		dates = []
		volumes = []
		types = []
		viscosities = []

		for visit in visits:
			date = visit['visit_date']
			wound_info = visit['wound_info']
			exudate = wound_info.get('exudate', {})
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

class Dashboard:
	"""Main dashboard application."""

	def __init__(self):
		"""Initialize the dashboard."""
		self.config             = Config()
		self.data_manager       = DataManager()
		self.visualizer         = Visualizer()
		self.impedance_analyzer = ImpedanceAnalyzer()

		# LLM configuration placeholders
		self.llm_platform   = None
		self.llm_model      = None
		self.uploaded_file  = None
		self.data_processor = None

	def setup(self) -> None:
		"""Set up the dashboard configuration."""
		st.set_page_config(
			page_title = self.config.PAGE_TITLE,
			page_icon  = self.config.PAGE_ICON,
			layout     = self.config.LAYOUT
		)
		Config.initialize()
		self._create_left_sidebar()

	@staticmethod
	@st.cache_data
	def load_data(uploaded_file) -> Optional[pd.DataFrame]:
		df = DataManager.load_data(uploaded_file)
		return df

	def run(self) -> None:
		"""Run the main dashboard application."""
		self.setup()
		if not self.uploaded_file:
			st.info("Please upload a CSV file to proceed.")
			return

		df = self.load_data(self.uploaded_file)
		self.data_processor = WoundDataProcessor(df=df, dataset_path=Config.DATA_PATH)

		if df is None:
			st.error("Failed to load data. Please check the CSV file.")
			return

		# Header
		st.title(self.config.PAGE_TITLE)

		# Patient selection
		patient_ids = sorted(df['Record ID'].unique())
		patient_options = ["All Patients"] + [f"Patient {id:d}" for id in patient_ids]
		selected_patient = st.selectbox("Select Patient", patient_options)

		# Create tabs
		self._create_dashboard_tabs(df, selected_patient)

	def _create_dashboard_tabs(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Create and manage dashboard tabs."""
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
		Render impedance analysis for the entire patient population.

		Args:
			df: DataFrame containing all patient data
		"""
		# Create a copy of the dataframe for analysis
		analysis_df = df.copy()

		# Add outlier threshold control and calculate correlation
		filtered_df = self._display_impedance_correlation_controls(analysis_df)

		# Create scatter plot if we have valid data
		if not filtered_df.empty:
			self._render_impedance_scatter_plot(filtered_df)
		else:
			st.warning("No valid data available for the scatter plot.")

		# Create additional visualizations in a two-column layout
		self._render_population_impedance_charts(df)

	def _display_impedance_correlation_controls(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Display controls for impedance correlation analysis and perform correlation calculation.

		Args:
			df: DataFrame containing impedance data

		Returns:
			Filtered DataFrame with outlier treatment applied
		"""
		col1, _, col3 = st.columns([2, 3, 3])

		with col1:
			outlier_threshold = st.number_input(
				"Impedance Outlier Threshold",
				min_value=0.0,
				max_value=0.9,
				value=0.2,
				step=0.05,
				help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
			)

		# Calculate correlation with outlier handling
		stats_analyzer = CorrelationAnalysis(data=df, x_col='Skin Impedance (kOhms) - Z', y_col='Healing Rate (%)', outlier_threshold=outlier_threshold)
		valid_df, r, p = stats_analyzer.calculate_correlation()

		# Display correlation statistics
		with col3:
			st.info(stats_analyzer.get_correlation_text())

		# Prepare data for visualization
		valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

		# Add consistent diabetes status for each patient
		first_diabetes_status = valid_df.groupby('Record ID')['Diabetes?'].first()
		valid_df['Diabetes?'] = valid_df['Record ID'].map(first_diabetes_status)

		return valid_df

	def _render_impedance_scatter_plot(self, df: pd.DataFrame) -> None:
		"""
		Render scatter plot showing relationship between impedance and healing rate.

		Args:
			df: DataFrame containing impedance and healing rate data
		"""
		fig = px.scatter(
			df,
			x='Skin Impedance (kOhms) - Z',
			y='Healing Rate (%)',
			color='Diabetes?',
			size='Calculated Wound Area',
			size_max=30,
			hover_data=['Record ID', 'Event Name', 'Wound Type'],
			title="Impedance vs Healing Rate Correlation"
		)

		fig.update_layout(
			xaxis_title="Impedance Z (kOhms)",
			yaxis_title="Healing Rate (% reduction per visit)"
		)

		st.plotly_chart(fig, use_container_width=True)

	def _render_population_impedance_charts(self, df: pd.DataFrame) -> None:
		"""
		Render additional impedance charts for population analysis.

		Args:
			df: DataFrame containing impedance data
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
			None: This method renders UI elements directly to the Streamlit app

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
				None

			Note:
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
				A tuple containing (health_score, health_interpretation) where:
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
				This method renders UI components directly to the Streamlit UI but does not return any value.
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
					indicating clinical significance
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
		styled_df = change_df.style.applymap(color_cells).set_properties(**{
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

			Returns
			-------
			None
				The method updates the Streamlit UI directly

			Notes
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
					exudate_analysis = Config.get_exudate_analysis(volume=volume, viscosity=viscosity, exudate_types=exudate_type_str)

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
							if exudate_type in Config.EXUDATE_TYPE_INFO:
								info = Config.EXUDATE_TYPE_INFO[exudate_type]
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
							f"(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
							f"SD={stats_data[('Healing Rate (%)', 'std')]}, "
							f"Improvement Rate={improvement_rate:.1f}%)")

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
							f"(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
							f"SD={stats_data[('Healing Rate (%)', 'std')]}, "
							f"Improvement Rate={improvement_rate:.1f}%)")

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
							f"(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
							f"SD={stats_data[('Healing Rate (%)', 'std')]}, "
							f"Improvement Rate={improvement_rate:.1f}%)")

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

		Returns
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

		st.header("LLM-Powered Wound Analysis")

		# Initialize reports dictionary in session state if it doesn't exist
		if 'llm_reports' not in st.session_state:
			st.session_state.llm_reports = {}

		if selected_patient == "All Patients":
			if self.uploaded_file is not None:
				if st.button("Run Analysis", key="run_analysis"):
					# Initialize LLM with platform and model
					llm = WoundAnalysisLLM(platform=self.llm_platform, model_name=self.llm_model)
					patient_data = self.data_processor.get_population_statistics()
					prompt = llm._format_population_prompt(patient_data)
					analysis = llm.analyze_population_data(patient_data)

					# Store analysis in session state for "All Patients"
					st.session_state.llm_reports['all_patients'] = dict(analysis_results=analysis, patient_metadata=None, prompt=prompt)

				# Display analysis if it exists
				if 'all_patients' in st.session_state.llm_reports:
					tab1, tab2 = st.tabs(["Analysis", "Prompt"])
					with tab1:
						st.markdown(st.session_state.llm_reports['all_patients']['analysis_results'])
					with tab2:
						st.markdown(st.session_state.llm_reports['all_patients']['prompt'])


					# Add download button for the report
					report_doc = DataManager.create_and_save_report(**st.session_state.llm_reports['all_patients'])

					DataManager.download_word_report(st=st, report_path=report_doc)

		else:
			patient_id = selected_patient.split(' ')[1]
			st.subheader(f"Patient {patient_id}")

			if self.uploaded_file is not None:
				if st.button("Run Analysis", key="run_analysis"):
					# Initialize LLM with platform and model
					llm = WoundAnalysisLLM(platform=self.llm_platform, model_name=self.llm_model)
					patient_data = self.data_processor.get_patient_visits(int(patient_id))
					prompt = llm._format_per_patient_prompt(patient_data)
					analysis = llm.analyze_patient_data(patient_data)

					# Store analysis in session state for this patient
					st.session_state.llm_reports[patient_id] = dict(analysis_results=analysis, patient_metadata=patient_data['patient_metadata'], prompt=prompt)

				# Display analysis if it exists for this patient
				if patient_id in st.session_state.llm_reports:
					tab1, tab2 = st.tabs(["Analysis", "Prompt"])
					with tab1:
						st.markdown(st.session_state.llm_reports[patient_id]['analysis_results'])
					with tab2:
						st.markdown(st.session_state.llm_reports[patient_id]['prompt'])

					# Add download button for the report
					report_doc = DataManager.create_and_save_report(**st.session_state.llm_reports[patient_id])

					DataManager.download_word_report(st=st, report_path=report_doc)
			else:
				st.warning("Please upload a patient data file from the sidebar to enable LLM analysis.")

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
			st.subheader("Model Configuration")

			self.uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=['csv'])
			# print('uploaded file type', type(self.uploaded_file))
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

			with st.expander("Advanced Model Settings"):

				api_key = st.text_input("API Key", type="password")
				if api_key:
					os.environ["OPENAI_API_KEY"] = api_key

				if self.llm_platform == "ai-verde":
					base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", ""))


			st.header("About This Dashboard")
			st.write("""
			This Streamlit dashboard visualizes wound healing data collected with smart bandage technology.

			The analysis focuses on key metrics:
			- Impedance measurements (Z, Z', Z'')
			- Temperature gradients
			- Oxygenation levels
			- Patient risk factors
			""")

			st.header("Statistical Methods")
			st.write("""
			The visualization is supported by these statistical approaches:
			- Longitudinal analysis of healing trajectories
			- Risk factor significance assessment
			- Comparative analysis across wound types
			""")

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

		Notes:
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
						f"WIfI Classification: {get_metric_value(infection.get('wifi_classification'))}"
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
