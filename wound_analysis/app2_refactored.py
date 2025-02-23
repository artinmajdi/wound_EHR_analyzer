"""
VSCODE - O3 Smart Bandage Wound Healing Analytics Dashboard
A Streamlit app for visualizing and analyzing wound healing data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import pathlib
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from docx import Document
import base64
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, create_and_save_report, download_word_report
import pdb  # Debug mode disabled

# Debug mode disabled
st.set_option('client.showErrorDetails', True)

@dataclass
class Config:
	"""Application configuration settings."""
	PAGE_TITLE: str = "VSCODE - O3"
	PAGE_ICON: str = "🩹"
	LAYOUT: str = "wide"
	DATA_PATH: Optional[pathlib.Path] = pathlib.Path('/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset')

@dataclass
class DataManager:
	"""Handles data loading, processing and manipulation."""
	data_processor: WoundDataProcessor = field(default_factory=lambda: None)
	df: pd.DataFrame = field(default_factory=lambda: None)

	@staticmethod
	@st.cache_data
	def load_data(uploaded_file):
		"""Load and preprocess the wound healing data from an uploaded CSV file. Returns None if no file is provided."""

		if uploaded_file is None:
			return None

		df = pd.read_csv(uploaded_file)
		df = DataManager._preprocess_data(df)
		return df

	@staticmethod
	def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
		"""Clean and preprocess the loaded data."""

		# Make a copy to avoid modifying the cached data
		# df = df.copy()



		df.columns = df.columns.str.strip()

		# Filter out skipped visits
		df = df[df['Skipped Visit?'] != 'Yes']

		# Fill missing Visit Number with 1 before converting to int
		df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

		df['Visit date'] = pd.to_datetime(df['Visit date']).dt.strftime('%m-%d-%Y')

		# Convert Wound Type to categorical with specified categories
		df['Wound Type'] = pd.Categorical(df['Wound Type'].fillna('Unknown'), categories=df['Wound Type'].dropna().unique())

		# 2. Calculate wound area if not present but dimensions are available
		if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
			df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

		df = DataManager._create_derived_features(df)
		df = DataManager._calculate_healing_rates(df)
		return df

	@staticmethod
	def _calculate_healing_rates(df: pd.DataFrame) -> pd.DataFrame:
		"""Calculate healing rates for each patient visit."""
		# Constants
		MAX_TREATMENT_DAYS = 730  # 2 years in days
		MIN_WOUND_AREA = 0


		def calculate_patient_healing_metrics(patient_data: pd.DataFrame) -> tuple[list, bool, float]:
			"""Calculate healing rate and estimated days for a patient.

			Returns:
				tuple: (healing_rates, is_improving, estimated_days_to_heal)
			"""
			if len(patient_data) < 2:
				return [0.0], False, np.nan

			# 3. Calculate healing rate (% change in wound area per visit)
			healing_rates = []
			for i, row in patient_data.iterrows():
				if row['Visit Number'] == 1 or len(patient_data[patient_data['Visit Number'] < row['Visit Number']]) == 0:
					healing_rates.append(0)  # No healing rate for first visit
				else:
					# Find the most recent previous visit
					prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
					prev_visit  = prev_visits[prev_visits['Visit Number'] == prev_visits['Visit Number'].max()]

					if len(prev_visit) > 0 and 'Calculated Wound Area' in patient_data.columns:
						prev_area = prev_visit['Calculated Wound Area'].values[0]
						curr_area = row['Calculated Wound Area']

						# Check for valid values to avoid division by zero
						if prev_area > 0 and not pd.isna(prev_area) and not pd.isna(curr_area):
							healing_rate = (prev_area - curr_area) / prev_area * 100  # Percentage decrease
							healing_rates.append(healing_rate)
						else:
							healing_rates.append(0)
					else:
						healing_rates.append(0)

			# Calculate average healing rate and determine if improving
			valid_rates = [rate for rate in healing_rates if rate > 0]
			avg_healing_rate = np.mean(valid_rates) if valid_rates else 0
			is_improving = avg_healing_rate > 0

			# Calculate estimated days to heal based on the latest wound area and average healing rate
			estimated_days = np.nan
			if is_improving and len(patient_data) > 0:
				last_visit = patient_data.iloc[-1]
				current_area = last_visit['Calculated Wound Area']

				if current_area > MIN_WOUND_AREA and avg_healing_rate > 0:
					# Convert percentage rate to area change per day
					daily_healing_rate = (avg_healing_rate / 100) * current_area
					if daily_healing_rate > 0:
						days_to_heal = current_area / daily_healing_rate
						total_days = last_visit['Days_Since_First_Visit'] + days_to_heal
						if 0 < total_days < MAX_TREATMENT_DAYS:
							estimated_days = float(total_days)

			return healing_rates, is_improving, estimated_days

		# Process each patient's data
		for patient_id in df['Record ID'].unique():
			patient_data = df[df['Record ID'] == patient_id].sort_values('Days_Since_First_Visit')

			healing_rates, is_improving, estimated_days = calculate_patient_healing_metrics(patient_data)

			# Update patient records with healing rates
			for i, (idx, row) in enumerate(patient_data.iterrows()):
				if i < len(healing_rates):
					df.loc[idx, 'Healing Rate (%)'] = healing_rates[i]

			# Update the last visit with overall improvement status
			df.loc[patient_data.iloc[-1].name, 'Overall_Improvement'] = 'Yes' if is_improving else 'No'

			if not np.isnan(estimated_days):
				df.loc[patient_data.index, 'Estimated_Days_To_Heal'] = estimated_days

		# Calculate and store average healing rates
		df['Average Healing Rate (%)'] = df.groupby('Record ID')['Healing Rate (%)'].transform('mean')

		# Ensure estimated days column exists
		if 'Estimated_Days_To_Heal' not in df.columns:
			df['Estimated_Days_To_Heal'] = pd.Series(np.nan, index=df.index, dtype=float)

		return df

	@staticmethod
	def _create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
		"""Create additional derived features from the data."""
		import numpy as np

		# # Make a copy to avoid modifying the cached data
		# df = df.copy()

		# Temperature gradients
		center = 'Center of Wound Temperature (Fahrenheit)'
		edge   = 'Edge of Wound Temperature (Fahrenheit)'
		peri   = 'Peri-wound Temperature (Fahrenheit)'
		if all(col in df.columns for col in [center, edge, peri]):
			df['Center-Edge Temp Gradient'] = df[center] - df[edge]
			df['Edge-Peri Temp Gradient']   = df[edge]   - df[peri]
			df['Total Temp Gradient']       = df[center] - df[peri]

		# BMI categories
		if 'BMI' in df.columns:
			df['BMI Category'] = pd.cut(
				df['BMI'],
				bins=[0, 18.5, 25, 30, 35, 100],
				labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II-III']
			)

		if df is not None and not df.empty:
			# Convert Visit date to datetime if not already
			df['Visit date'] = pd.to_datetime(df['Visit date'])#.dt.strftime('%m-%d-%Y')

			# Calculate days since first visit for each patient
			df['Days_Since_First_Visit'] = df.groupby('Record ID')['Visit date'].transform(
				lambda x: (x - x.min()).dt.days
			)

			# Initialize columns with explicit dtypes
			df['Healing Rate (%)'] = pd.Series(0.0, index=df.index, dtype=float)
			df['Estimated_Days_To_Heal'] = pd.Series(np.nan, index=df.index, dtype=float)
			df['Overall_Improvement'] = pd.Series(np.nan, index=df.index, dtype=str)

		return df

	@staticmethod
	def get_patient_data(df: pd.DataFrame, patient_id: int) -> pd.DataFrame:
		"""Get data for a specific patient."""
		return df[df['Record ID'] == patient_id].sort_values('Visit Number')

	@staticmethod
	def _extract_patient_metadata(patient_data) -> Dict:
		"""Extract relevant patient metadata from a single row."""

		metadata = {
			'age': patient_data['Calculated Age at Enrollment'] if not pd.isna(patient_data.get('Calculated Age at Enrollment')) else None,

			'sex': patient_data['Sex'] if not pd.isna(patient_data.get('Sex')) else None,

			'race': patient_data['Race'] if not pd.isna(patient_data.get('Race')) else None,

			'ethnicity': patient_data['Ethnicity'] if not pd.isna(patient_data.get('Ethnicity')) else None,

			'weight': patient_data['Weight'] if not pd.isna(patient_data.get('Weight')) else None,
			'height': patient_data['Height'] if not pd.isna(patient_data.get('Height')) else None,
			'bmi': patient_data['BMI'] if not pd.isna(patient_data.get('BMI')) else None,

			'study_cohort': patient_data['Study Cohort'] if not pd.isna(patient_data.get('Study Cohort')) else None,

			'smoking_status': patient_data['Smoking status'] if not pd.isna(patient_data.get('Smoking status')) else None,

			'packs_per_day': patient_data['Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes'] if not pd.isna(patient_data.get('Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes')) else None,

			'years_smoking': patient_data['Number of Years smoked/has been smoking cigarettes'] if not pd.isna(patient_data.get('Number of Years smoked/has been smoking cigarettes')) else None,

			'alcohol_use': patient_data['Alcohol Use Status'] if not pd.isna(patient_data.get('Alcohol Use Status')) else None,

			'alcohol_frequency': patient_data['Number of alcohol drinks consumed/has been consuming'] if not pd.isna(patient_data.get('Number of alcohol drinks consumed/has been consuming')) else None
		}

		# Medical history from individual columns
		medical_conditions = [
			'Respiratory', 'Cardiovascular', 'Gastrointestinal', 'Musculoskeletal',
			'Endocrine/ Metabolic', 'Hematopoietic', 'Hepatic/Renal', 'Neurologic', 'Immune'
		]
		# Get medical history from standard columns
		metadata['medical_history'] = {
			condition: patient_data[condition]
			for condition in medical_conditions if not pd.isna(patient_data.get(condition))
		}

		# Check additional medical history from free text field
		other_history = patient_data.get('Medical History (select all that apply)')
		if not pd.isna(other_history):
			existing_conditions = set(medical_conditions)
			other_conditions = [cond.strip() for cond in str(other_history).split(',')]
			other_conditions = [cond for cond in other_conditions if cond and cond not in existing_conditions]
			if other_conditions:
				metadata['medical_history']['other'] = ', '.join(other_conditions)


		# Diabetes information
		metadata['diabetes'] = {
			'status': patient_data.get('Diabetes?'),
			'hemoglobin_a1c': patient_data.get('Hemoglobin A1c (%)'),
			'a1c_available': patient_data.get('A1c  available within the last 3 months?')
		}

		return metadata


	# Add a new static method _generate_mock_data to handle fallback data generation
	@staticmethod
	def _generate_mock_data() -> pd.DataFrame:
		"""Generate mock data in case the CSV file cannot be loaded."""
		np.random.seed(42)
		n_patients = 10
		n_visits = 3
		rows = []
		for p in range(1, n_patients + 1):
			initial_area = np.random.uniform(10, 20)
			for v in range(1, n_visits + 1):
				area = initial_area * (0.9 ** (v - 1))
				rows.append({
					'Record ID': p,
					'Event Name': f'Visit {v}',
					'Calculated Wound Area': area,
					'Center of Wound Temperature (Fahrenheit)': np.random.uniform(97, 102),
					'Edge of Wound Temperature (Fahrenheit)': np.random.uniform(96, 101),
					'Peri-wound Temperature (Fahrenheit)': np.random.uniform(95, 100),
					'BMI': np.random.uniform(18, 35),
					'Diabetes?': np.random.choice(['Yes', 'No']),
					'Smoking status': np.random.choice(['Never', 'Current', 'Former'])
				})
		df = pd.DataFrame(rows)
		df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)
		df = DataManager._create_derived_features(df)
		df = DataManager._calculate_healing_rates(df)
		return df

class SessionStateManager:
	"""Manages Streamlit session state."""

	@staticmethod
	def initialize() -> None:
		"""Initialize session state variables."""
		if 'processor' not in st.session_state:
			st.session_state.processor = None
		if 'analysis_complete' not in st.session_state:
			st.session_state.analysis_complete = False
		if 'analysis_results' not in st.session_state:
			st.session_state.analysis_results = None
		if 'report_path' not in st.session_state:
			st.session_state.report_path = None

# class DocumentHandler:
# 	"""Handles document generation and downloads."""

# 	@staticmethod
# 	def create_and_save_report(patient_data: dict, analysis_results: str) -> str:
# 		"""Create and save analysis report."""
# 		doc = Document()
# 		report_path = format_word_document(doc, patient_data, analysis_results)
# 		return report_path

# 	@staticmethod
# 	def download_word_report(report_path: str) -> str:
# 		"""Create a download link for the Word report."""
# 		try:
# 			with open(report_path, 'rb') as f:
# 				bytes_data = f.read()
# 				st.download_button(
# 					label="Download Full Report (DOCX)",
# 					data=bytes_data,
# 					file_name=os.path.basename(report_path),
# 					mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
# 				)
# 		except Exception as e:
# 			st.error(f"Error preparing report download: {str(e)}")

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
	def create_impedance_chart(visits):
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
		if any(x is not None for x in high_freq_z):
			fig.add_trace(go.Scatter(
				x=dates, y=high_freq_z,
				name=f'|Z| ({high_freq_label})',
				mode='lines+markers'
			))
		if any(x is not None for x in high_freq_r):
			fig.add_trace(go.Scatter(
				x=dates, y=high_freq_r,
				name=f'Resistance ({high_freq_label})',
				mode='lines+markers'
			))
		if any(x is not None for x in high_freq_c):
			fig.add_trace(go.Scatter(
				x=dates, y=high_freq_c,
				name=f'Capacitance ({high_freq_label})',
				mode='lines+markers'
			))

		# Add center frequency traces
		if any(x is not None for x in center_freq_z):
			fig.add_trace(go.Scatter(
				x=dates, y=center_freq_z,
				name=f'|Z| ({center_freq_label})',
				mode='lines+markers',
				line=dict(dash='dot')
			))
		if any(x is not None for x in center_freq_r):
			fig.add_trace(go.Scatter(
				x=dates, y=center_freq_r,
				name=f'Resistance ({center_freq_label})',
				mode='lines+markers',
				line=dict(dash='dot')
			))
		if any(x is not None for x in center_freq_c):
			fig.add_trace(go.Scatter(
				x=dates, y=center_freq_c,
				name=f'Capacitance ({center_freq_label})',
				mode='lines+markers',
				line=dict(dash='dot')
			))

		# Add low frequency traces
		if any(x is not None for x in low_freq_z):
			fig.add_trace(go.Scatter(
				x=dates, y=low_freq_z,
				name=f'|Z| ({low_freq_label})',
				mode='lines+markers',
				line=dict(dash='dash')
			))
		if any(x is not None for x in low_freq_r):
			fig.add_trace(go.Scatter(
				x=dates, y=low_freq_r,
				name=f'Resistance ({low_freq_label})',
				mode='lines+markers',
				line=dict(dash='dash')
			))
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

class RiskAnalyzer:
	"""Handles risk analysis and assessment."""

	@staticmethod
	def calculate_risk_score(patient_data: pd.Series) -> Tuple[int, List[str], str]:
		"""Calculate risk score and identify risk factors."""
		risk_score = 0
		risk_factors = []

		# Check diabetes
		if patient_data['Diabetes?'] == 'Yes':
			risk_factors.append("Diabetes")
			risk_score += 3

		# Check smoking status
		if patient_data['Smoking status'] == 'Current':
			risk_factors.append("Current smoker")
			risk_score += 2
		elif patient_data['Smoking status'] == 'Former':
			risk_factors.append("Former smoker")
			risk_score += 1

		# Check BMI
		if patient_data['BMI'] >= 30:
			risk_factors.append("Obesity")
			risk_score += 2
		elif patient_data['BMI'] >= 25:
			risk_factors.append("Overweight")
			risk_score += 1

		# Determine risk category
		if risk_score >= 6:
			risk_category = "High"
		elif risk_score >= 3:
			risk_category = "Moderate"
		else:
			risk_category = "Low"

		return risk_score, risk_factors, risk_category

class Dashboard:
	"""Main dashboard application."""

	def __init__(self):
		"""Initialize the dashboard."""
		self.config = Config()
		self.data_manager = DataManager()
		self.visualizer = Visualizer()
		# self.doc_handler = DocumentHandler()
		self.risk_analyzer = RiskAnalyzer()
		# LLM configuration placeholders
		self.llm_platform = None
		self.llm_model = None
		self.uploaded_file = None
		self.data_processor = None

	def setup(self) -> None:
		"""Set up the dashboard configuration."""
		st.set_page_config(
			page_title=self.config.PAGE_TITLE,
			page_icon=self.config.PAGE_ICON,
			layout=self.config.LAYOUT
		)
		SessionStateManager.initialize()
		self._create_left_sidebar()

	def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
		"""Load and prepare data for the dashboard."""
		df = self.data_manager.load_data(uploaded_file)
		self.data_processor = WoundDataProcessor(df=df, dataset_path=Config.DATA_PATH)
		return df

	def run(self) -> None:
		"""Run the main dashboard application."""
		self.setup()
		if not self.uploaded_file:
			st.info("Please upload a CSV file to proceed.")
			return

		df = self.load_data(self.uploaded_file)
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

		# Footer
		self._add_footer()

		# Additional sidebar info
		self._create_sidebar()

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
			self._exudate_tab(selected_patient)
		with tabs[5]:
			self._risk_factors_tab(df, selected_patient)
		with tabs[6]:
			self._llm_analysis_tab(selected_patient)

	def _overview_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the overview tab."""
		st.header("Overview")

		if selected_patient == "All Patients":
			self._render_tab_all_patients(df)
		else:
			patient_id = int(selected_patient.split(" ")[1])
			self._render_patient_overview(df, patient_id)
			st.subheader("Wound Area Over Time")
			fig = self.visualizer.create_wound_area_plot(DataManager.get_patient_data(df, patient_id), patient_id)
			st.plotly_chart(fig, use_container_width=True)

	def _impedance_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the impedance analysis tab."""
		st.header("Impedance Analysis")

		if selected_patient == "All Patients":

			# Scatter plot: Impedance vs Healing Rate
			valid_df = df[df['Healing Rate (%)'] > 0].copy()

			# Add outlier threshold control
			col1, _, col3 = st.columns([1, 1, 3])

			with col1:
				outlier_threshold = st.number_input(
					"Impedance Outlier Threshold",
					min_value=0.0,
					max_value=0.9,
					value=0.2,
					step=0.05,
					help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
				)

			# Get the filtered data for y-axis limits
			valid_df = Visualizer._remove_outliers(valid_df, 'Skin Impedance (kOhms) - Z', outlier_threshold)

			with col3:
				if not valid_df.empty:
					# Calculate correlation
					r, p = stats.pearsonr(valid_df['Skin Impedance (kOhms) - Z'], valid_df['Healing Rate (%)'])
					p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
					st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")
					st.write("Higher impedance values correlate with slower healing rates, especially in diabetic patients")

			# Add consistent diabetes status for each patient
			first_diabetes_status = valid_df.groupby('Record ID')['Diabetes?'].first()
			valid_df['Diabetes?'] = valid_df['Record ID'].map(first_diabetes_status)

			if not valid_df.empty:
				fig = px.scatter(
					valid_df,
					x='Skin Impedance (kOhms) - Z',
					y='Healing Rate (%)',
					color='Diabetes?',
					size='Calculated Wound Area',
					size_max=30,  # Maximum marker size
					hover_data=['Record ID', 'Event Name', 'Wound Type'],
					title="Impedance vs Healing Rate Correlation"
				)
				fig.update_layout(xaxis_title="Impedance Z (kOhms)", yaxis_title="Healing Rate (% reduction per visit)")
				st.plotly_chart(fig, use_container_width=True)
			else:
				st.warning("No valid data available for the scatter plot.")

			# Rest of the impedance tab code...
			col1, col2 = st.columns(2)
			with col1:
				st.subheader("Impedance Components Over Time")
				avg_impedance = df.groupby('Visit Number')[["Skin Impedance (kOhms) - Z", "Skin Impedance (kOhms) - Z'", "Skin Impedance (kOhms) - Z''"]].mean().reset_index()

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
				avg_by_type = df.groupby('Wound Type')["Skin Impedance (kOhms) - Z"].mean().reset_index()
				fig2 = px.bar(
					avg_by_type,
					x='Wound Type',
					y="Skin Impedance (kOhms) - Z",
					title="Average Impedance by Wound Type",
					color='Wound Type'
				)
				fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Average Impedance Z (kOhms)")
				st.plotly_chart(fig2, use_container_width=True)
		else:
			# For individual patient
			patient_data = self.data_processor.get_patient_visits(record_id=int(selected_patient.split(" ")[1]))
			visits = patient_data['visits']
			fig = Visualizer.create_impedance_chart(visits)

			# patient_data = DataManager.get_patient_data(df, int(selected_patient.split(" ")[1])).sort_values('Visit Number')
			# fig = px.line(
			# 	patient_data,
			# 	x='Visit Number',
			# 	y=['Skin Impedance (kOhms) - Z', "Skin Impedance (kOhms) - Z'", "Skin Impedance (kOhms) - Z''"],
			# 	title=f"Impedance Measurements Over Time for {selected_patient}",
			# 	markers=True
			# )
			fig.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)", legend_title="Measurement")
			st.plotly_chart(fig, use_container_width=True)

	def _temperature_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the temperature analysis tab."""
		st.header("Temperature Gradient Analysis")

		if selected_patient == "All Patients":
			# Create a temperature gradient dataframe
			temp_df = df.copy()

			temp_df['Visit date'] = pd.to_datetime(temp_df['Visit date']).dt.strftime('%m-%d-%Y')

			# Remove skipped visits
			# temp_df = temp_df[temp_df['Skipped Visit?'] != 'Yes']

			temp_df['Calculated Wound Area'] = temp_df['Calculated Wound Area'].fillna(temp_df['Calculated Wound Area'].mean())

			# Define temperature column names
			temp_cols = [	'Center of Wound Temperature (Fahrenheit)',
							'Edge of Wound Temperature (Fahrenheit)',
							'Peri-wound Temperature (Fahrenheit)']

			# Drop rows with missing temperature data
			temp_df = temp_df.dropna(subset=temp_cols)

			# Calculate temperature gradients if all required columns exist
			if all(col in temp_df.columns for col in temp_cols):
				temp_df['Center-Edge Gradient'] = temp_df[temp_cols[0]] - temp_df[temp_cols[1]]
				temp_df['Edge-Peri Gradient'] = temp_df[temp_cols[1]] - temp_df[temp_cols[2]]
				temp_df['Total Gradient'] = temp_df[temp_cols[0]] - temp_df[temp_cols[2]]

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
			fig = px.scatter(
				temp_df[temp_df['Healing Rate (%)'] > 0],  # Exclude first visits
				x='Total Gradient',
				y='Healing Rate (%)',
				color='Wound Type',
				size='Calculated Wound Area',
				hover_data=['Record ID', 'Event Name'],
				title="Temperature Gradient vs. Healing Rate"
			)

			fig.update_layout(
				xaxis_title="Temperature Gradient (Center to Peri-wound, °F)",
				yaxis_title="Healing Rate (% reduction per visit)"
			)
			st.plotly_chart(fig, use_container_width=True)

		else:
			# For individual patient
			df_temp = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
			df_temp['Visit date'] = pd.to_datetime(df_temp['Visit date']).dt.strftime('%m-%d-%Y')
			fig = Visualizer.create_temperature_chart(df=df_temp)

			fig.update_layout(
				title=f"Temperature Measurements for {selected_patient}",
				xaxis_title="Visit Date",  # Changed from "Visit Number"
				legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
			)

			fig.update_yaxes(title_text="Temperature (°F)", secondary_y=False)

			st.plotly_chart(fig, use_container_width=True)

	def _oxygenation_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the oxygenation analysis tab."""
		st.header("Oxygenation Analysis")

		if selected_patient == "All Patients":
			# Prepare data for scatter plot
			valid_df = df[df['Healing Rate (%)'] > 0].copy()

			# Handle NaN values
			valid_df['Hemoglobin Level'] = valid_df['Hemoglobin Level'].fillna(valid_df['Hemoglobin Level'].mean())
			valid_df = valid_df.dropna(subset=['Oxygenation (%)', 'Healing Rate (%)', 'Hemoglobin Level'])

			if not valid_df.empty:
				fig1 = px.scatter(
					valid_df,
					x='Oxygenation (%)',
					y='Healing Rate (%)',
					color='Diabetes?',
					size='Hemoglobin Level',
					size_max=30,
					hover_data=['Record ID', 'Event Name', 'Wound Type'],
					title="Relationship Between Oxygenation and Healing Rate"
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

	def _exudate_tab(self, selected_patient: str) -> None:
		"""Render the exudate analysis tab."""
		st.header("Exudate Characteristics Over Time")

		if selected_patient == "All Patients":
			pass

		else:
			visits = self.data_processor.get_patient_visits(record_id=int(selected_patient.split(" ")[1]))['visits']

			fig = Visualizer.create_exudate_chart(visits)
			st.plotly_chart(fig, use_container_width=True)

	def _risk_factors_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the risk factors analysis tab."""
		st.header("Risk Factors Analysis")

		if selected_patient == "All Patients":
			risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

			with risk_tab1:
				# Compare healing rates by diabetes status
				diab_healing = df.groupby(['Diabetes?', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
				diab_healing = diab_healing[diab_healing['Visit Number'] > 1]  # Exclude first visit

				fig = px.line(
					diab_healing,
					x='Visit Number',
					y='Healing Rate (%)',
					color='Diabetes?',
					title="Average Healing Rate by Diabetes Status",
					markers=True
				)
				fig.update_layout( xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)" )
				st.plotly_chart(fig, use_container_width=True)

				# Compare impedance by diabetes status
				diab_imp = df.groupby('Diabetes?')['Skin Impedance (kOhms) - Z'].mean().reset_index()

				fig = px.bar(
					diab_imp,
					x='Diabetes?',
					y='Skin Impedance (kOhms) - Z',
					color='Diabetes?',
					title="Average Impedance by Diabetes Status"
				)
				fig.update_layout(xaxis_title="Diabetes Status", yaxis_title="Average Impedance Z (kOhms)")
				st.plotly_chart(fig, use_container_width=True)

			with risk_tab2:
				smoke_healing = df.groupby(['Smoking status', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
				smoke_healing = smoke_healing[smoke_healing['Visit Number'] > 1]
				fig_line = px.line(
					smoke_healing,
					x='Visit Number',
					y='Healing Rate (%)',
					color='Smoking status',
					title="Average Healing Rate by Smoking Status",
					markers=True
				)
				fig_line.update_layout(xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)")
				st.plotly_chart(fig_line, use_container_width=True)

				smoke_wound = pd.crosstab(df['Smoking status'], df['Wound Type'])
				smoke_wound_pct = smoke_wound.div(smoke_wound.sum(axis=1), axis=0) * 100
				fig_bar2 = px.bar(
					smoke_wound_pct.reset_index().melt(id_vars='Smoking status', var_name='Wound Type', value_name='Percentage'),
					x='Smoking status',
					y='Percentage',
					color='Wound Type',
					title="Distribution of Wound Types by Smoking Status",
					barmode='stack'
				)
				fig_bar2.update_layout(xaxis_title="Smoking Status", yaxis_title="Percentage (%)")
				st.plotly_chart(fig_bar2, use_container_width=True)

			with risk_tab3:
				df_temp = df.copy()
				# Create BMI categories
				df_temp['BMI Category'] = pd.cut(
					df_temp['BMI'],
					bins=[0, 18.5, 25, 30, 35, 100],
					labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II-III']
				)

				# Compare healing rates by BMI category
				bmi_healing = df_temp.groupby(['BMI Category', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
				bmi_healing = bmi_healing[bmi_healing['Visit Number'] > 1]  # Exclude first visit

				fig = px.line(
					bmi_healing,
					x='Visit Number',
					y='Healing Rate (%)',
					color='BMI Category',
					title="Average Healing Rate by BMI Category",
					markers=True
				)
				fig.update_layout(
					xaxis_title="Visit Number",
					yaxis_title="Average Healing Rate (%)"
				)
				st.plotly_chart(fig, use_container_width=True)

				# Scatter plot of BMI vs. healing rate
				fig = px.scatter(
					df_temp[df_temp['Healing Rate (%)'] > 0],  # Exclude first visits
					x='BMI',
					y='Healing Rate (%)',
					color='Diabetes?',
					size='Calculated Wound Area',
					hover_data=['Record ID', 'Event Name', 'Wound Type'],
					title="BMI vs. Healing Rate"
				)
				fig.update_layout(
					xaxis_title="BMI",
					yaxis_title="Healing Rate (% reduction per visit)"
				)
				st.plotly_chart(fig, use_container_width=True)
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
		"""Render the LLM analysis tab."""
		st.header("LLM Analysis")
		if selected_patient == "All Patients":
			st.warning("Please select a specific patient to view their analysis.")
		else:
			st.subheader("LLM-Powered Wound Analysis")
			if self.uploaded_file is not None:
				if st.button("Run Analysis", key="run_analysis"):
					# Initialize LLM with platform and model
					llm = WoundAnalysisLLM(platform=self.llm_platform, model_name=self.llm_model)
					patient_data = self.data_processor.get_patient_visits(record_id=int(selected_patient.split(" ")[1]))
					analysis = llm.analyze_patient_data(patient_data)

					# Store analysis results and patient data in session state
					st.session_state.analysis_results = analysis
					st.session_state.current_patient_data = patient_data
					st.session_state.report_path = create_and_save_report(patient_data=patient_data, analysis_results=analysis)

					st.success("LLM analysis complete.")

				# Display analysis results if they exist in session state
				if st.session_state.analysis_results is not None:
					st.header("Analysis Results")
					st.markdown(st.session_state.analysis_results)

					# Show download button if report path exists
					if 'report_path' in st.session_state:
						download_word_report(st=st, report_path=st.session_state.report_path)
			else:
				st.warning("Please upload a patient data file from the sidebar to enable LLM analysis.")

	def _add_footer(self) -> None:
		"""Add footer information."""
		st.markdown("---")
		st.markdown("""
		**Note:** This dashboard loads data from a user-uploaded CSV file.
		""")

	def _create_sidebar(self) -> None:
		"""Create the sidebar with additional information."""
		with st.sidebar:
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
			# - Correlation analysis between measurements and outcomes


	def _create_left_sidebar(self) -> None:
		"""Create sidebar components specific to LLM configuration."""

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

				api_key = st.text_input("API Key", value="sk-h8JtQkCCJUOy-TAdDxCLGw", type="password")

				if api_key:
					os.environ["OPENAI_API_KEY"] = api_key

				if self.llm_platform == "ai-verde":
					base_url = st.text_input("Base URL", value="https://llm-api.cyverse.ai")

	def _render_tab_all_patients(self, df: pd.DataFrame) -> None:
		"""Render statistics for all patients."""
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

		"""Render overview for a specific patient."""

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


		# wound_info = {
		# 	'location'       : clean_field(visit_data, 'Describe the wound location'),
		# 	'type'           : clean_field(visit_data, 'Wound Type'),
		# 	'current_care'   : clean_field(visit_data, 'Current wound care'),
		# 	'clinical_events': clean_field(visit_data, 'Clinical events'),
		# 	'undermining': {
		# 		'present'  : None if present is None else present == 'Yes',
		# 		'location' : visit_data.get('Undermining Location Description'),
		# 		'tunneling': visit_data.get('Tunneling Location Description')
		# 	},
		# 	'infection': {
		# 		'status'             : clean_field(visit_data, 'Infection'),
		# 		'wifi_classification': visit_data.get('Diabetic Foot Wound - WIfI Classification: foot Infection (fI)')
		# 	},
		# 	'granulation': {
		# 		'coverage': clean_field(visit_data, 'Granulation'),
		# 		'quality' : clean_field(visit_data, 'Granulation Quality')
		# 	},

def main():
	"""Main application entry point."""
	# try:
	dashboard = Dashboard()
	dashboard.run()
	# except Exception as e:
	#     st.error(f"Application error: {str(e)}")
	#     st.error("Please check the data source and configuration.")

if __name__ == "__main__":
	main()
