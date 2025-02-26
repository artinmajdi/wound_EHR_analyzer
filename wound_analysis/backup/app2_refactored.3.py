import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
import pathlib
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, create_and_save_report, download_word_report
import pdb  # Debug mode disabled

# Debug mode disabled
st.set_option('client.showErrorDetails', True)

@dataclass
class Config:
	"""Application configuration settings."""
	PAGE_TITLE: str = "Wound Care Management & Interpreter Dashboard"
	PAGE_ICON: str = "🩹"
	LAYOUT: str = "wide"
	DATA_PATH: Optional[pathlib.Path] = pathlib.Path('/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset')

	# Constants for bioimpedance analysis
	ANOMALY_THRESHOLD: float = 2.5  # Z-score threshold for anomaly detection
	MIN_VISITS_FOR_ANALYSIS: int = 3  # Minimum visits needed for trend analysis
	SIGNIFICANT_CHANGE_THRESHOLD: float = 15.0  # Percentage change threshold
	INFECTION_RISK_RATIO_THRESHOLD: float = 15.0  # Low/high frequency ratio threshold
	SIGNIFICANT_CHANGE_THRESHOLD_IMPEDANCE: float = 15.0  # Percentage change threshold for resistance/absolute impedance
	SIGNIFICANT_CHANGE_THRESHOLD_CAPACITANCE: float = 20.0  # Percentage change threshold for capacitance
	INFLAMMATORY_INCREASE_THRESHOLD: float = 30.0  # Percentage increase threshold for low-frequency resistance

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

class ImpedanceAnalyzer:
	"""Handles advanced bioimpedance analysis and clinical interpretation."""

	@staticmethod
	def calculate_visit_changes(current_visit, previous_visit):
		"""
		Calculate percentage changes between consecutive visits for all impedance parameters.

		Returns:
			Dict: Percentage changes with clinical significance flags
		"""
		changes = {}
		clinically_significant = {}

		# Define significance thresholds based on clinical literature
		significance_thresholds = {
			'resistance': 0.15,  # 15% change is clinically significant
			'capacitance': 0.20, # 20% change is clinically significant
			'Z': 0.15  # 15% change is clinically significant for absolute impedance
		}

		freq_types = ['low_frequency', 'center_frequency', 'high_frequency']
		params = ['Z', 'resistance', 'capacitance']

		for freq_type in freq_types:
			current_freq_data = current_visit.get('sensor_data', {}).get('impedance', {}).get(freq_type, {})
			previous_freq_data = previous_visit.get('sensor_data', {}).get('impedance', {}).get(freq_type, {})

			for param in params:
				try:
					current_val = float(current_freq_data.get(param, 0))
					previous_val = float(previous_freq_data.get(param, 0))

					if previous_val != 0 and current_val != 0:
						percent_change = (current_val - previous_val) / previous_val
						key = f"{param}_{freq_type}"
						changes[key] = percent_change

						# Determine clinical significance
						is_significant = abs(percent_change) > significance_thresholds.get(param, 0.15)
						clinically_significant[key] = is_significant
				except (ValueError, TypeError, ZeroDivisionError):
					continue

		return changes, clinically_significant

	@staticmethod
	def calculate_tissue_health_index(visit):
		"""
		Calculate tissue health index based on multi-frequency impedance ratios.

		Returns:
			Tuple: (health_score, interpretation)
		"""
		sensor_data = visit.get('sensor_data', {})
		impedance_data = sensor_data.get('impedance', {})

		# Extract absolute impedance at different frequencies
		low_freq = impedance_data.get('low_frequency', {})
		high_freq = impedance_data.get('high_frequency', {})

		try:
			low_z = float(low_freq.get('Z', 0))
			high_z = float(high_freq.get('Z', 0))

			if low_z > 0 and high_z > 0:
				# Calculate low/high frequency ratio
				lf_hf_ratio = low_z / high_z

				# Calculate phase angle if resistance and reactance available
				phase_angle = None
				if 'resistance' in high_freq and 'capacitance' in high_freq:
					r = float(high_freq.get('resistance', 0))
					c = float(high_freq.get('capacitance', 0))
					if r > 0 and c > 0:
						# Approximate phase angle calculation
						import math
						# Using arctan(1/(2πfRC))
						f = float(high_freq.get('frequency', 80000))
						phase_angle = math.atan(1/(2 * math.pi * f * r * c)) * (180/math.pi)

				# Normalize scores to 0-100 scale
				# Typical healthy ratio range: 5-12
				ratio_score = max(0, min(100, (1 - (lf_hf_ratio - 5) / 7) * 100)) if 5 <= lf_hf_ratio <= 12 else max(0, 50 - abs(lf_hf_ratio - 8.5) * 5)

				if phase_angle:
					# Typical healthy phase angle range: 5-7 degrees
					phase_score = max(0, min(100, (phase_angle / 7) * 100))
					health_score = (ratio_score * 0.6) + (phase_score * 0.4)  # Weighted average
				else:
					health_score = ratio_score

				# Interpretation
				if health_score >= 80:
					interpretation = "Excellent tissue health"
				elif health_score >= 60:
					interpretation = "Good tissue health"
				elif health_score >= 40:
					interpretation = "Moderate tissue health"
				elif health_score >= 20:
					interpretation = "Poor tissue health"
				else:
					interpretation = "Very poor tissue health"

				return health_score, interpretation

		except (ValueError, TypeError, ZeroDivisionError):
			pass

		return None, "Insufficient data for tissue health calculation"

	@staticmethod
	def analyze_healing_trajectory(visits):
		"""
		Analyze the healing trajectory using regression analysis of impedance values over time.

		Returns:
			Dict: Analysis results including slope, significance, and interpretation
		"""
		if len(visits) < 3:
			return {"status": "insufficient_data"}

		import numpy as np
		from scipy import stats

		# Extract high frequency impedance values
		dates = []
		z_values = []

		for visit in visits:
			try:
				high_freq = visit.get('sensor_data', {}).get('impedance', {}).get('high_frequency', {})
				z_val = float(high_freq.get('Z', 0))
				if z_val > 0:
					z_values.append(z_val)
					dates.append(visit.get('visit_date'))
			except (ValueError, TypeError):
				continue

		if len(z_values) < 3:
			return {"status": "insufficient_data"}

		# Convert dates to numerical values for regression
		x = np.arange(len(z_values))
		y = np.array(z_values)

		# Perform linear regression
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
		r_squared = r_value ** 2

		# Determine clinical significance of slope
		result = {
			"slope": slope,
			"r_squared": r_squared,
			"p_value": p_value,
			"dates": dates,
			"values": z_values,
			"status": "analyzed"
		}

		# Interpret the slope
		if slope < -0.5 and p_value < 0.05:
			result["interpretation"] = "Strong evidence of healing progression"
		elif slope < -0.2 and p_value < 0.10:
			result["interpretation"] = "Moderate evidence of healing progression"
		elif slope > 0.5 and p_value < 0.05:
			result["interpretation"] = "Potential deterioration detected"
		else:
			result["interpretation"] = "No significant trend detected"

		return result

	@staticmethod
	def analyze_frequency_response(visit):
		"""
		Analyze the pattern of impedance across different frequencies to assess tissue composition.

		Returns:
			Dict: Dispersion characteristics and interpretation
		"""
		results = {}
		sensor_data = visit.get('sensor_data', {})
		impedance_data = sensor_data.get('impedance', {})

		# Extract absolute impedance at different frequencies
		low_freq = impedance_data.get('low_frequency', {})
		center_freq = impedance_data.get('center_frequency', {})
		high_freq = impedance_data.get('high_frequency', {})

		try:
			z_low = float(low_freq.get('Z', 0))
			z_center = float(center_freq.get('Z', 0))
			z_high = float(high_freq.get('Z', 0))

			if z_low > 0 and z_center > 0 and z_high > 0:
				# Calculate alpha dispersion (low-to-center frequency drop)
				alpha_dispersion = (z_low - z_center) / z_low

				# Calculate beta dispersion (center-to-high frequency drop)
				beta_dispersion = (z_center - z_high) / z_center

				results['alpha_dispersion'] = alpha_dispersion
				results['beta_dispersion'] = beta_dispersion

				# Interpret dispersion patterns
				if alpha_dispersion > 0.4 and beta_dispersion < 0.2:
					results['interpretation'] = "High extracellular fluid content, possible edema"
				elif alpha_dispersion < 0.2 and beta_dispersion > 0.3:
					results['interpretation'] = "High cellular density, possible granulation"
				elif alpha_dispersion > 0.3 and beta_dispersion > 0.3:
					results['interpretation'] = "Mixed tissue composition, active remodeling"
				else:
					results['interpretation'] = "Normal tissue composition pattern"
			else:
				results['interpretation'] = "Insufficient frequency data for analysis"
		except (ValueError, TypeError, ZeroDivisionError) as e:
			error_message = f"Error processing frequency response data: {type(e).__name__}: {str(e)}"
			print(error_message)  # For console debugging
			import traceback
			traceback.print_exc()  # Print the full traceback
			results['interpretation'] = error_message  # Or keep the generic message if preferred

		return results

	@staticmethod
	def detect_impedance_anomalies(previous_visits, current_visit, z_score_threshold=2.5):
		"""
		Detect statistically significant anomalies in impedance measurements.

		Returns:
			Dict: Alerts with clinical interpretations
		"""
		if len(previous_visits) < 3:
			return {}

		import numpy as np

		alerts = {}

		# Parameters to monitor
		params = [
			('high_frequency', 'Z', 'High-frequency impedance'),
			('low_frequency', 'resistance', 'Low-frequency resistance'),
			('high_frequency', 'capacitance', 'High-frequency capacitance')
		]

		current_impedance = current_visit.get('sensor_data', {}).get('impedance', {})

		for freq_type, param_name, display_name in params:
			# Collect historical values
			historical_values = []

			for visit in previous_visits:
				visit_impedance = visit.get('sensor_data', {}).get('impedance', {})
				freq_data = visit_impedance.get(freq_type, {})
				try:
					value = float(freq_data.get(param_name, 0))
					if value > 0:
						historical_values.append(value)
				except (ValueError, TypeError):
					continue

			if len(historical_values) >= 3:
				# Calculate historical statistics
				mean = np.mean(historical_values)
				std = np.std(historical_values)

				# Get current value
				current_freq_data = current_impedance.get(freq_type, {})
				try:
					current_value = float(current_freq_data.get(param_name, 0))
					if current_value > 0 and std > 0:
						z_score = (current_value - mean) / std

						if abs(z_score) > z_score_threshold:
							direction = "increase" if z_score > 0 else "decrease"

							# Clinical interpretation
							if freq_type == 'high_frequency' and param_name == 'Z':
								if direction == 'increase':
									clinical_meaning = "Possible deterioration in tissue quality or increased inflammation"
								else:
									clinical_meaning = "Possible improvement in cellular integrity or reduction in edema"
							elif freq_type == 'low_frequency' and param_name == 'resistance':
								if direction == 'increase':
									clinical_meaning = "Possible decrease in extracellular fluid or improved barrier function"
								else:
									clinical_meaning = "Possible increase in extracellular fluid or breakdown of tissue barriers"
							elif freq_type == 'high_frequency' and param_name == 'capacitance':
								if direction == 'increase':
									clinical_meaning = "Possible increase in cellular density or membrane integrity"
								else:
									clinical_meaning = "Possible decrease in viable cell count or membrane dysfunction"
							else:
								clinical_meaning = "Significant change detected, clinical correlation advised"

							key = f"{freq_type}_{param_name}"
							alerts[key] = {
								"parameter": display_name,
								"z_score": z_score,
								"direction": direction,
								"interpretation": clinical_meaning
							}
				except (ValueError, TypeError):
					continue

		return alerts

	@staticmethod
	def assess_infection_risk(current_visit, previous_visit=None):
		"""
		Assess the risk of wound infection based on impedance parameters.

		Returns:
			Dict: Risk score, level, and contributing factors
		"""
		risk_score = 0
		factors = []

		current_impedance = current_visit.get('sensor_data', {}).get('impedance', {})

		# Factor 1: Low/high frequency impedance ratio
		low_freq = current_impedance.get('low_frequency', {})
		high_freq = current_impedance.get('high_frequency', {})

		try:
			low_z = float(low_freq.get('Z', 0))
			high_z = float(high_freq.get('Z', 0))

			if low_z > 0 and high_z > 0:
				ratio = low_z / high_z
				# Ratios > 15 are associated with increased infection risk in literature
				if ratio > 20:
					risk_score += 40
					factors.append("Very high low/high frequency impedance ratio")
				elif ratio > 15:
					risk_score += 25
					factors.append("Elevated low/high frequency impedance ratio")
		except (ValueError, TypeError, ZeroDivisionError):
			pass

		# Factor 2: Sudden increase in low-frequency resistance (inflammatory response)
		if previous_visit:
			prev_impedance = previous_visit.get('sensor_data', {}).get('impedance', {})
			prev_low_freq = prev_impedance.get('low_frequency', {})

			try:
				current_r = float(low_freq.get('resistance', 0))
				prev_r = float(prev_low_freq.get('resistance', 0))

				if prev_r > 0 and current_r > 0:
					pct_change = (current_r - prev_r) / prev_r
					if pct_change > 0.30:  # >30% increase
						risk_score += 30
						factors.append("Significant increase in low-frequency resistance")
			except (ValueError, TypeError, ZeroDivisionError):
				pass

		# Factor 3: Phase angle calculation (if resistance and capacitance available)
		try:
			r = float(high_freq.get('resistance', 0))
			c = float(high_freq.get('capacitance', 0))
			f = float(high_freq.get('frequency', 80000))

			if r > 0 and c > 0:
				import math
				phase_angle = math.atan(1/(2 * math.pi * f * r * c)) * (180/math.pi)

				if phase_angle < 2:  # Low phase angles are associated with poor tissue health
					risk_score += 30
					factors.append("Very low phase angle")
				elif phase_angle < 3:
					risk_score += 15
					factors.append("Low phase angle")
		except (ValueError, TypeError, ZeroDivisionError):
			pass

		# Limit score to 0-100 range
		risk_score = min(100, max(0, risk_score))

		# Interpret risk level
		if risk_score >= 60:
			interpretation = "High infection risk"
		elif risk_score >= 30:
			interpretation = "Moderate infection risk"
		else:
			interpretation = "Low infection risk"

		return {
			"risk_score": risk_score,
			"risk_level": interpretation,
			"contributing_factors": factors
		}

	@staticmethod
	def calculate_cole_parameters(visit):
		"""
		Calculate bioimpedance Cole-Cole model parameters from multi-frequency measurements.

		Returns:
			Dictionary containing R0, R∞, Fc (characteristic frequency), and alpha
		"""
		import math

		results = {}
		impedance_data = visit.get('sensor_data', {}).get('impedance', {})

		# Extract resistance at different frequencies
		low_freq = impedance_data.get('low_frequency', {})
		center_freq = impedance_data.get('center_frequency', {})
		high_freq = impedance_data.get('high_frequency', {})

		try:
			# Get resistance values
			r_low = float(low_freq.get('resistance', 0))
			r_center = float(center_freq.get('resistance', 0))
			r_high = float(high_freq.get('resistance', 0))

			# Get capacitance values
			c_low = float(low_freq.get('capacitance', 0))
			c_center = float(center_freq.get('capacitance', 0))
			c_high = float(high_freq.get('capacitance', 0))

			# Get frequency values
			f_low = float(low_freq.get('frequency', 100))
			f_center = float(center_freq.get('frequency', 7499))
			f_high = float(high_freq.get('frequency', 80000))

			if r_low > 0 and r_high > 0:
				# Approximate R0 and R∞
				results['R0'] = r_low  # Low frequency resistance approximates R0
				results['Rinf'] = r_high  # High frequency resistance approximates R∞

				# Calculate membrane capacitance (Cm)
				if r_center > 0 and c_center > 0:
					results['Fc'] = f_center

					# Calculate time constant
					tau = 1 / (2 * math.pi * f_center)
					results['Tau'] = tau

					# Membrane capacitance estimation
					if (r_low - r_high) > 0 and r_high > 0:
						cm = tau / ((r_low - r_high) * r_high)
						results['Cm'] = cm

				# Calculate alpha (tissue heterogeneity)
				# Using resistance values to estimate alpha
				if r_low > 0 and r_center > 0 and r_high > 0:
					# Simplified alpha estimation
					alpha_est = 1 - (r_center / math.sqrt(r_low * r_high))
					results['Alpha'] = max(0, min(1, abs(alpha_est)))

					# Interpret alpha value
					if results['Alpha'] < 0.6:
						results['tissue_homogeneity'] = "High tissue homogeneity"
					elif results['Alpha'] < 0.8:
						results['tissue_homogeneity'] = "Moderate tissue homogeneity"
					else:
						results['tissue_homogeneity'] = "Low tissue homogeneity (heterogeneous tissue)"
		except (ValueError, TypeError, ZeroDivisionError, KeyError):
			pass

		return results

	@staticmethod
	def generate_clinical_insights(analyses):
		"""
		Generate clinical insights based on comprehensive impedance analysis.

		Returns:
			List of clinical insights with confidence levels
		"""
		insights = []

		# Healing trajectory insights
		if 'healing_trajectory' in analyses:
			trajectory = analyses['healing_trajectory']
			if trajectory.get('status') == 'analyzed':
				if trajectory.get('slope', 0) < -0.3 and trajectory.get('p_value', 1) < 0.05:
					insights.append({
						"insight": "Strong evidence of consistent wound healing progression based on impedance trends",
						"confidence": "High",
						"recommendation": "Continue current treatment protocol"
					})
				elif trajectory.get('slope', 0) > 0.3 and trajectory.get('p_value', 1) < 0.1:
					insights.append({
						"insight": "Potential stalling or deterioration in wound healing process",
						"confidence": "Moderate",
						"recommendation": "Consider reassessment of treatment approach"
					})

		# Infection risk insights
		if 'infection_risk' in analyses:
			risk = analyses['infection_risk']
			if risk.get('risk_score', 0) > 50:
				insights.append({
					"insight": f"Elevated infection risk detected ({risk.get('risk_score')}%)",
					"confidence": "Moderate to High",
					"recommendation": "Consider microbiological assessment and prophylactic measures",
					"supporting_factors": risk.get('contributing_factors', [])
				})

		# Tissue composition insights
		if 'frequency_response' in analyses:
			freq_response = analyses['frequency_response']
			if 'interpretation' in freq_response:
				insights.append({
					"insight": freq_response['interpretation'],
					"confidence": "Moderate",
					"recommendation": "Correlate with clinical assessment of wound bed"
				})

		# Anomaly detection insights
		if 'anomalies' in analyses and analyses['anomalies']:
			for param, anomaly in analyses['anomalies'].items():
				insights.append({
					"insight": f"Significant {anomaly.get('direction')} in {anomaly.get('parameter')} detected (z-score: {anomaly.get('z_score', 0):.2f})",
					"confidence": "High" if abs(anomaly.get('z_score', 0)) > 3 else "Moderate",
					"clinical_meaning": anomaly.get('interpretation', '')
				})

		return insights

	@staticmethod
	def classify_wound_healing_stage(analyses):
		"""
		Classify the wound healing stage based on impedance patterns.

		Returns:
			Dict: Healing stage classification and characteristics
		"""
		# Default to inflammatory if we don't have enough data
		stage = "Inflammatory"
		characteristics = []
		confidence = "Low"

		# Get tissue health index
		tissue_health = analyses.get('tissue_health', (None, ""))
		health_score = tissue_health[0] if tissue_health else None

		# Get frequency response
		freq_response = analyses.get('frequency_response', {})
		alpha = freq_response.get('alpha_dispersion', 0)
		beta = freq_response.get('beta_dispersion', 0)

		# Get Cole parameters
		cole_params = analyses.get('cole_parameters', {})

		# Stage classification logic
		if health_score is not None and freq_response and cole_params:
			confidence = "Moderate"

			# Inflammatory phase characteristics:
			# - High alpha dispersion (high extracellular fluid)
			# - Low tissue health score
			# - High low/high frequency ratio
			if alpha > 0.4 and health_score < 40:
				stage = "Inflammatory"
				characteristics = [
					"High extracellular fluid content",
					"Low tissue health score",
					"Elevated cellular permeability"
				]
				confidence = "High" if alpha > 0.5 and health_score < 30 else "Moderate"

			# Proliferative phase characteristics:
			# - High beta dispersion (cellular proliferation)
			# - Moderate tissue health score
			# - Moderate alpha dispersion
			elif beta > 0.3 and 40 <= health_score <= 70 and 0.2 <= alpha <= 0.4:
				stage = "Proliferative"
				characteristics = [
					"Active cellular proliferation",
					"Increasing tissue organization",
					"Moderate extracellular fluid"
				]
				confidence = "High" if beta > 0.4 and health_score > 50 else "Moderate"

			# Remodeling phase characteristics:
			# - Low alpha dispersion (reduced extracellular fluid)
			# - High tissue health score
			# - Low variability in impedance
			elif alpha < 0.2 and health_score > 70:
				stage = "Remodeling"
				characteristics = [
					"Reduced extracellular fluid",
					"Improved tissue organization",
					"Enhanced barrier function"
				]
				confidence = "High" if alpha < 0.15 and health_score > 80 else "Moderate"

		return {
			"stage": stage,
			"characteristics": characteristics,
			"confidence": confidence
		}

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
			self._exudate_tab(df, selected_patient)
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
			# Existing code for all patients view remains the same
			# Scatter plot: Impedance vs Healing Rate
			valid_df = df.copy()
			valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)
			# Add outlier threshold control
			col1, _, col3 = st.columns([2,3,3])

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
			# For individual patient - Enhanced with advanced bioimpedance analysis
			patient_id = int(selected_patient.split(" ")[1])
			patient_data = self.data_processor.get_patient_visits(record_id=patient_id)
			visits = patient_data['visits']

			# Create tabs for different analysis views
			tab1, tab2, tab3 = st.tabs([
				"Impedance Measurements",
				"Clinical Analysis",
				"Advanced Interpretation"
			])

			with tab1:
				# Basic impedance visualization with mode selection dropdown
				st.subheader("Impedance Measurements Over Time")
				
				# Add measurement mode selector
				measurement_mode = st.selectbox(
					"Select Measurement Mode:",
					["Absolute Impedance (|Z|)", "Resistance", "Capacitance"],
					key="impedance_mode_selector"
				)
				
				# Create impedance chart with selected mode
				fig = Visualizer.create_impedance_chart(visits, measurement_mode=measurement_mode)
				st.plotly_chart(fig, use_container_width=True)

				# Add basic interpretation
				st.markdown("### Measurement Interpretation")
				st.info("""
				**Interpreting Impedance Measurements:**
				- **Absolute Impedance (|Z|)**: Total opposition to electrical current flow
				- **Resistance**: Opposition from the tissue's ionic content
				- **Capacitance**: Opposition from cell membranes

				Lower frequencies (100Hz) primarily measure extracellular fluid, while higher frequencies (80000Hz)
				penetrate cell membranes to measure both intracellular and extracellular properties.
				""")

			with tab2:
				st.subheader("Bioimpedance Clinical Analysis")

				# Only analyze if we have at least two visits
				if len(visits) >= 2:
					# Pick most recent visit for current analysis
					current_visit = visits[-1]
					previous_visit = visits[-2]

					# Perform analyses
					tissue_health = ImpedanceAnalyzer.calculate_tissue_health_index(current_visit)
					infection_risk = ImpedanceAnalyzer.assess_infection_risk(current_visit, previous_visit)
					freq_response = ImpedanceAnalyzer.analyze_frequency_response(current_visit)
					changes, significant = ImpedanceAnalyzer.calculate_visit_changes(current_visit, previous_visit)

					# Display Tissue Health Index
					col1, col2 = st.columns(2)
					with col1:
						st.markdown("### Tissue Health Assessment")
						health_score, health_interp = tissue_health

						if health_score is not None:
							# Create a color scale for the health score
							color = "red" if health_score < 40 else "orange" if health_score < 60 else "green"
							st.markdown(f"**Tissue Health Index:** <span style='color:{color};font-weight:bold'>{health_score:.1f}/100</span>", unsafe_allow_html=True)
							st.markdown(f"**Interpretation:** {health_interp}")
						else:
							st.warning("Insufficient data for tissue health calculation")

					with col2:
						st.markdown("### Infection Risk Assessment")
						risk_score = infection_risk["risk_score"]
						risk_level = infection_risk["risk_level"]

						# Create a color scale for the risk score
						risk_color = "green" if risk_score < 30 else "orange" if risk_score < 60 else "red"
						st.markdown(f"**Infection Risk Score:** <span style='color:{risk_color};font-weight:bold'>{risk_score:.1f}/100</span>", unsafe_allow_html=True)
						st.markdown(f"**Risk Level:** {risk_level}")

						# Display contributing factors if any
						factors = infection_risk["contributing_factors"]
						if factors:
							st.markdown("**Contributing Factors:**")
							for factor in factors:
								st.markdown(f"- {factor}")

					# Display Tissue Composition Analysis
					st.markdown("### Tissue Composition Analysis")
					st.info(freq_response['interpretation'])

					# Display changes since last visit
					st.markdown("### Changes Since Previous Visit")

					if changes:
						change_cols = st.columns(3)
						col_idx = 0

						for key, change in changes.items():
							with change_cols[col_idx % 3]:
								# Format parameter name for display
								param_parts = key.split('_')
								param_name = param_parts[0].capitalize()
								freq_name = ' '.join(param_parts[1:]).replace('_', ' ').capitalize()

								# Format the value with correct sign and style
								direction = "increase" if change > 0 else "decrease"
								arrow = "↑" if change > 0 else "↓"
								color = "red" if change > 0 else "green"  # In most cases, decreasing is good

								# Check if clinically significant
								sig_badge = ""
								if significant.get(key, False):
									sig_badge = "🔔 "

								st.markdown(f"**{sig_badge}{param_name} ({freq_name}):**")
								st.markdown(f"<span style='color:{color};font-weight:bold'>{arrow} {abs(change)*100:.1f}%</span> {direction}", unsafe_allow_html=True)

							col_idx += 1
					else:
						st.warning("No comparable data from previous visit")
				else:
					st.warning("At least two visits are required for clinical analysis")

			with tab3:
				st.subheader("Advanced Bioelectrical Interpretation")

				if len(visits) >= 3:
					# Perform advanced analyses
					healing_trajectory = ImpedanceAnalyzer.analyze_healing_trajectory(visits)
					anomalies = ImpedanceAnalyzer.detect_impedance_anomalies(visits[:-1], visits[-1])
					cole_params = ImpedanceAnalyzer.calculate_cole_parameters(visits[-1])

					# Consolidate all analyses for insight generation
					all_analyses = {
						'healing_trajectory': healing_trajectory,
						'tissue_health': ImpedanceAnalyzer.calculate_tissue_health_index(visits[-1]),
						'infection_risk': ImpedanceAnalyzer.assess_infection_risk(visits[-1], visits[-2] if len(visits) > 1 else None),
						'frequency_response': ImpedanceAnalyzer.analyze_frequency_response(visits[-1]),
						'anomalies': anomalies,
						'cole_parameters': cole_params
					}

					# Generate comprehensive clinical insights
					insights = ImpedanceAnalyzer.generate_clinical_insights(all_analyses)
					healing_stage = ImpedanceAnalyzer.classify_wound_healing_stage(all_analyses)

					# Display healing trajectory
					if healing_trajectory['status'] == 'analyzed':
						st.markdown("### Healing Trajectory Analysis")

						# Create simple line chart with trend line
						import plotly.graph_objects as go
						import numpy as np

						dates = healing_trajectory['dates']
						values = healing_trajectory['values']

						fig = go.Figure()
						fig.add_trace(go.Scatter(
							x=list(range(len(dates))),
							y=values,
							mode='lines+markers',
							name='Impedance',
							hovertext=dates
						))

						# Add trend line
						x = np.array(range(len(values)))
						y = healing_trajectory['slope'] * x + np.mean(values)
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
							yaxis_title="High-Frequency Impedance (Z)",
							hovermode="x unified"
						)

						st.plotly_chart(fig, use_container_width=True)

						# Display statistical results
						col1, col2 = st.columns(2)
						with col1:
							slope_color = "green" if healing_trajectory['slope'] < 0 else "red"
							st.markdown(f"**Trend Slope:** <span style='color:{slope_color}'>{healing_trajectory['slope']:.4f}</span>", unsafe_allow_html=True)
							st.markdown(f"**Statistical Significance:** p = {healing_trajectory['p_value']:.4f}")

						with col2:
							st.markdown(f"**R² Value:** {healing_trajectory['r_squared']:.4f}")
							st.info(healing_trajectory['interpretation'])

					# Display Wound Healing Stage Classification
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

					# Display Cole-Cole parameters if available
					if cole_params:
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

					# Display comprehensive clinical insights
					st.markdown("### Clinical Insights")

					if insights:
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
					else:
						st.info("No significant clinical insights generated from current data.")
				else:
					st.warning("At least three visits are required for advanced analysis")

				# Add clinical reference information
				with st.expander("Bioimpedance Reference Information", expanded=False):
					st.markdown("""
					### Interpreting Impedance Parameters

					**Frequency Significance:**
					- **Low Frequency (100Hz):** Primarily reflects extracellular fluid and tissue properties
					- **Center Frequency:** Reflects the maximum reactance point, varies based on tissue composition
					- **High Frequency (80000Hz):** Penetrates cell membranes, reflects total tissue properties

					**Clinical Correlations:**
					- **Decreasing High-Frequency Impedance:** Often associated with improved healing
					- **Increasing Low-to-High Frequency Ratio:** May indicate inflammation or infection
					- **Decreasing Phase Angle:** May indicate deterioration in cellular health
					- **Increasing Alpha Parameter:** Often indicates increasing tissue heterogeneity

					**Reference Ranges:**
					- **Healthy Tissue Low/High Ratio:** 5-12
					- **Optimal Phase Angle:** 5-7 degrees
					- **Typical Alpha Range:** 0.6-0.8
					""")

	def _temperature_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the temperature analysis tab."""

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
				temp_df['Edge-Peri Gradient'] = temp_df[temp_cols[1]] - temp_df[temp_cols[2]]
				temp_df['Total Gradient'] = temp_df[temp_cols[0]] - temp_df[temp_cols[2]]


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

			# Get the filtered data for y-axis limits
			temp_df = Visualizer._remove_outliers(temp_df, 'Center of Wound Temperature (Fahrenheit)', outlier_threshold)

			with col3:
				if not temp_df.empty:
					# Calculate correlation
					r, p = stats.pearsonr(temp_df['Center of Wound Temperature (Fahrenheit)'], temp_df['Healing Rate (%)'])
					p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
					st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")

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
		"""Render the oxygenation analysis tab."""
		st.header("Oxygenation Analysis")

		if selected_patient == "All Patients":
			# Prepare data for scatter plot
			# valid_df = df[df['Healing Rate (%)'] < 0].copy()
			valid_df = df.copy()
			# valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

			# Handle NaN values and convert columns to float
			valid_df['Hemoglobin Level'] = pd.to_numeric(valid_df['Hemoglobin Level'], errors='coerce')#.fillna(valid_df['Hemoglobin Level'].astype(float).mean())
			valid_df['Oxygenation (%)'] = pd.to_numeric(valid_df['Oxygenation (%)'], errors='coerce')
			valid_df['Healing Rate (%)'] = pd.to_numeric(valid_df['Healing Rate (%)'], errors='coerce')
			# valid_df['Calculated Wound Area'] = pd.to_numeric(valid_df['Calculated Wound Area'], errors='coerce')

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

			# Get the filtered data for y-axis limits
			valid_df = Visualizer._remove_outliers(df=valid_df, column='Oxygenation (%)', quantile_threshold=outlier_threshold)

			with col3:
				if not valid_df.empty:
					# Calculate correlation
					r, p = stats.pearsonr(valid_df['Oxygenation (%)'], valid_df['Healing Rate (%)'])
					p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
					st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")

			# # Add consistent diabetes status for each patient
			# first_diabetes_status = valid_df.groupby('Record ID')['Diabetes?'].first()
			# valid_df['Diabetes?'] = valid_df['Record ID'].map(first_diabetes_status)
			valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)
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
		"""Render the exudate analysis tab."""
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
			exudate_df['Volume_Numeric'] = exudate_df['Exudate Volume'].map(level_mapping)
			exudate_df['Viscosity_Numeric'] = exudate_df['Exudate Viscosity'].map(level_mapping)

			if not exudate_df.empty:
				# Create two columns for volume and viscosity analysis
				col1, col2 = st.columns(2)

				with col1:
					st.subheader("Volume Analysis")
					# Calculate correlation between volume and healing rate
					valid_df = exudate_df.dropna(subset=['Volume_Numeric', 'Healing Rate (%)'])
					if len(valid_df) > 1:
						r, p = stats.pearsonr(valid_df['Volume_Numeric'], valid_df['Healing Rate (%)'])
						p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
						st.info(f"Volume correlation vs Healing Rate: r = {r:.2f} (p {p_formatted})")

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
						showlegend=True
					)
					st.plotly_chart(fig_vol, use_container_width=True)

				with col2:
					st.subheader("Viscosity Analysis")
					# Calculate correlation between viscosity and healing rate
					valid_df = exudate_df.dropna(subset=['Viscosity_Numeric', 'Healing Rate (%)'])
					if len(valid_df) > 1:
						r, p = stats.pearsonr(valid_df['Viscosity_Numeric'], valid_df['Healing Rate (%)'])
						p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
						st.info(f"Viscosity correlation vs Healing Rate: r = {r:.2f} (p {p_formatted})")

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
						showlegend=True
					)
					st.plotly_chart(fig_visc, use_container_width=True)

				# Create scatter plot matrix for volume, viscosity, and healing rate
				st.subheader("Relationship Analysis")

				exudate_df['Healing Rate (%)'] = exudate_df['Healing Rate (%)'].clip(-100, 100)
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
						type_by_wound,
						title="Exudate Types by Wound Category",
						barmode='group'
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

			# df_temp = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()

			# Clinical interpretation section
			st.subheader("Clinical Interpretation of Exudate Characteristics")

			# Create tabs for each visit
			visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits])

			# Process each visit in its corresponding tab
			for tab, visit in zip(visit_tabs, visits):
				with tab:
					col1, col2 = st.columns(2)

					with col1:
						st.markdown("### Volume Analysis")
						volume = visit['wound_info']['exudate'].get('volume', 'N/A')
						st.write(f"**Current Level:** {volume}")

						# Volume interpretation
						if volume == 'High':
							st.info("""
							**High volume exudate** is common in:
							- Chronic venous leg ulcers
							- Dehisced surgical wounds
							- Inflammatory ulcers
							- Burns

							This may indicate active inflammation or healing processes.
							""")
						elif volume == 'Low':
							st.info("""
							**Low volume exudate** is typical in:
							- Necrotic wounds
							- Ischaemic/arterial wounds
							- Neuropathic diabetic foot ulcers

							Monitor for signs of insufficient moisture.
							""")

					with col2:
						st.markdown("### Viscosity Analysis")
						viscosity = visit['wound_info']['exudate'].get('viscosity', 'N/A')
						st.write(f"**Current Level:** {viscosity}")

						# Viscosity interpretation
						if viscosity == 'High':
							st.warning("""
							**High viscosity** (thick) exudate may indicate:
							- High protein content
							- Possible infection
							- Inflammatory processes
							- Presence of necrotic material

							Consider reassessing treatment approach.
							""")
						elif viscosity == 'Low':
							st.info("""
							**Low viscosity** (thin) exudate may suggest:
							- Low protein content
							- Possible venous condition
							- Potential malnutrition
							- Presence of fistulas

							Monitor fluid balance and nutrition.
							""")

					# Exudate Type Analysis
					st.markdown("### Type Analysis")
					# st.markdown(visit['wound_info']['exudate'])
					exudate_type_str = str(visit['wound_info']['exudate'].get('type', 'N/A'))

					if exudate_type_str != 'N/A':
						# Split types and strip whitespace
						exudate_types = [t.strip() for t in exudate_type_str.split(',')]
						st.write(f"**Current Types:** {exudate_type_str}")
					else:
						exudate_types = ['N/A']
						st.write("**Current Type:** N/A")

					# Type interpretation
					type_info = {
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

					# Process each exudate type
					highest_severity = 'info'  # Track highest severity for overall implications
					for exudate_type in exudate_types:
						if exudate_type in type_info:
							info = type_info[exudate_type]
							message = f"""
							**Type: {exudate_type}**
							**Description:** {info['description']}
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

					# Treatment Implications
					st.markdown("### Treatment Implications")
					implications = []

					# Combine volume and viscosity for treatment recommendations
					if volume == 'High' and viscosity == 'High':
						implications.append("- Consider highly absorbent dressings")
						implications.append("- More frequent dressing changes may be needed")
						implications.append("- Monitor for maceration of surrounding tissue")
					elif volume == 'Low' and viscosity == 'Low':
						implications.append("- Use moisture-retentive dressings")
						implications.append("- Protect wound bed from desiccation")
						implications.append("- Consider hydrating dressings")

					if implications:
						st.write("**Recommended Actions:**")
						for imp in implications:
							st.write(imp)

	def _risk_factors_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
		"""Render the risk factors analysis tab."""
		st.header("Risk Factors Analysis")

		if selected_patient == "All Patients":
			risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

			valid_df = df.dropna(subset=['Healing Rate (%)', 'Visit Number']).copy()

			# Add detailed warning for outliers with statistics
			# outliers = valid_df[abs(valid_df['Healing Rate (%)']) > 100]
			# if not outliers.empty:
			# 	st.warning(
			# 		f"⚠️ Data Quality Alert:\n\n"
			# 		f"Found {len(outliers)} measurements ({(len/outliers)/len(valid_df)*100):.1f}% of data) "
			# 		f"with healing rates outside the expected range (-100% to 100%).\n\n"
			# 		f"Statistics:\n"
			# 		f"- Minimum value: {outliers['Healing Rate (%)'].min():.1f}%\n"
			# 		f"- Maximum value: {outliers['Healing Rate (%)'].max():.1f}%\n"
			# 		f"- Mean value: {outliers['Healing Rate (%)'].mean():.1f}%\n"
			# 		f"- Number of unique patients affected: {len(outliers['Record ID'].unique())}\n\n"
			# 		"These values will be clipped to [-100%, 100%] range for visualization purposes."
			# 	)

			# Clip healing rates to reasonable range
			valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

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
				diab_stats = valid_df.groupby('Diabetes?').agg({
					'Healing Rate (%)': ['mean', 'count', 'std']
				}).round(2)

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
					stats = diab_stats.loc[status]
					improvement_rate = (valid_df[valid_df['Diabetes?'] == status]['Healing Rate (%)'] < 0).mean() * 100
					st.write(f"- {status}: Average Healing Rate = {stats[('Healing Rate (%)', 'mean')]}% "
							f"(n={int(stats[('Healing Rate (%)', 'count')])}, "
							f"SD={stats[('Healing Rate (%)', 'std')]}, "
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
					stats = smoke_stats.loc[status]
					improvement_rate = (valid_df[valid_df['Smoking status'] == status]['Healing Rate (%)'] < 0).mean() * 100
					st.write(f"- {status}: Average Healing Rate = {stats[('Healing Rate (%)', 'mean')]}% "
							f"(n={int(stats[('Healing Rate (%)', 'count')])}, "
							f"SD={stats[('Healing Rate (%)', 'std')]}, "
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
				bmi_stats = valid_df.groupby('BMI Category').agg({
					'Healing Rate (%)': ['mean', 'count', 'std']
				}).round(2)

				st.write("**Statistical Summary:**")
				for category in bmi_stats.index:
					stats = bmi_stats.loc[category]
					improvement_rate = (valid_df[valid_df['BMI Category'] == category]['Healing Rate (%)'] < 0).mean() * 100
					st.write(f"- {category}: Average Healing Rate = {stats[('Healing Rate (%)', 'mean')]}% "
							f"(n={int(stats[('Healing Rate (%)', 'count')])}, "
							f"SD={stats[('Healing Rate (%)', 'std')]}, "
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
		"""Render the LLM analysis tab."""
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
					report_doc = create_and_save_report(**st.session_state.llm_reports['all_patients'])
					download_word_report(st=st, report_path=report_doc)

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
					report_doc = create_and_save_report(**st.session_state.llm_reports[patient_id])
					download_word_report(st=st, report_path=report_doc)
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

				api_key = st.text_input("API Key", type="password")
				if api_key:
					os.environ["OPENAI_API_KEY"] = api_key

				if self.llm_platform == "ai-verde":
					base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", ""))

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

def main():
	"""Main application entry point."""
	dashboard = Dashboard()
	dashboard.run()

if __name__ == "__main__":
	main()
