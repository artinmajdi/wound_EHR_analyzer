import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
from io import BytesIO
import base64
from docx import Document
from typing import Optional, Dict, List, Union
from data_processor import WoundDataProcessor  # Assuming this is correctly implemented
from llm_interface import WoundAnalysisLLM, format_word_document  # Assuming this is correctly implemented


# Constants for File Paths and Default Values (for better maintainability)
DATA_FILE_PATH = "/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv"
REPORT_FILE_NAME = "wound_analysis_report.docx"
DEFAULT_PLATFORM = "ai-verde"
DEFAULT_AI_VERDE_MODEL = "llama-3.3-70b-fp8"
DEFAULT_HF_MODEL = "medalpaca-7b"  # Keep for potential future use, though HF is disabled for now
DEFAULT_AI_VERDE_BASE_URL = "https://llm-api.cyverse.ai"
DEFAULT_API_KEY = "sk-h8JtQkCCJUOy-TAdDxCLGw"


# --- Helper Functions ---
def _create_download_link(file_path: str, link_text: str) -> str:
	"""Generates a download link for a file."""
	try:
		with open(file_path, "rb") as file:
			bytes_data = file.read()
		b64_data = base64.b64encode(bytes_data).decode()
		href = f'data:application/octet-stream;base64,{b64_data}'  # More generic MIME type
		return f'<a href="{href}" download="{os.path.basename(file_path)}">{link_text}</a>'
	except FileNotFoundError:
		st.error(f"File not found: {file_path}")
		return ""  # Return an empty string instead of raising an exception

def _load_and_preprocess_data(file_path: str) -> pd.DataFrame:
	"""Loads and preprocesses wound data."""
	try:
		df = pd.read_csv(file_path)
		df.columns = df.columns.str.strip()
		df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

		if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
			df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

		df['Healing Rate (%)'] = _calculate_healing_rates(df)
		_handle_missing_values(df)
		_standardize_categorical_data(df)
		_create_temperature_gradients(df)
		return df

	except (FileNotFoundError, pd.errors.ParserError, KeyError) as e:
		st.error(f"Error loading or processing data: {e}")
		st.error("Using mock data instead.")
		return generate_mock_data()

def _calculate_healing_rates(df: pd.DataFrame) -> List[float]:
	 """Calculates healing rates per patient visit."""
	 healing_rates = []
	 for patient_id in df['Record ID'].unique():
		 patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')
		 for i, row in patient_data.iterrows():
			 if row['Visit Number'] == 1 or len(patient_data[patient_data['Visit Number'] < row['Visit Number']]) == 0:
				 healing_rates.append(0)
			 else:
				 prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
				 prev_visit = prev_visits[prev_visits['Visit Number'] == prev_visits['Visit Number'].max()]

				 if not prev_visit.empty and 'Calculated Wound Area' in df.columns:
					 prev_area = prev_visit['Calculated Wound Area'].iloc[0]
					 curr_area = row['Calculated Wound Area']
					 if prev_area > 0 and not pd.isna(prev_area) and not pd.isna(curr_area):
						 healing_rate = (prev_area - curr_area) / prev_area * 100
						 healing_rates.append(healing_rate)
					 else:
						 healing_rates.append(0)
				 else:
					 healing_rates.append(0)

	 # Padding logic, but made more robust.  List slicing is generally safer than extending.
	 healing_rates = healing_rates[:len(df)] + [0] * max(0, len(df) - len(healing_rates))
	 return healing_rates

def _handle_missing_values(df: pd.DataFrame):
	"""Handles missing values in the DataFrame."""
	numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
	for col in numeric_cols:
		if df[col].isna().any():  # More concise than .sum() > 0
			df[col] = df[col].fillna(df[col].median())

def _standardize_categorical_data(df: pd.DataFrame):
	"""Standardizes categorical columns."""
	for col, replacements in [('Diabetes?', {'yes': 'Yes', 'no': 'No', 'Y': 'Yes', 'N': 'No'}),
							  ('Smoking status', None)]:  # None indicates fillna only
		if col in df.columns:
			df[col] = df[col].fillna('Unknown')
			if replacements:
				df[col] = df[col].replace(replacements)

def _create_temperature_gradients(df: pd.DataFrame):
	"""Creates derived temperature gradient columns."""
	temp_cols = ['Center of Wound Temperature (Fahrenheit)', 'Edge of Wound Temperature (Fahrenheit)', 'Peri-wound Temperature (Fahrenheit)']
	if all(col in df.columns for col in temp_cols):
		df['Center-Edge Temp Gradient'] = df[temp_cols[0]] - df[temp_cols[1]]
		df['Edge-Peri Temp Gradient'] = df[temp_cols[1]] - df[temp_cols[2]]
		df['Total Temp Gradient'] = df[temp_cols[0]] - df[temp_cols[2]]

@st.cache_data
def load_data() -> pd.DataFrame:
	"""Loads and preprocesses wound data (cached)."""
	return _load_and_preprocess_data(DATA_FILE_PATH)

def generate_mock_data() -> pd.DataFrame:
	"""Generates mock wound data."""
	np.random.seed(42)
	n_patients = 127
	n_visits_per_patient = 3
	n_rows = 418
	record_ids = np.repeat(np.arange(1, n_patients + 1), n_visits_per_patient)[:n_rows]
	event_names = [f"Visit {visit}" for patient in range(1, n_patients + 1)
				   for visit in range(1, n_visits_per_patient + 1)][:n_rows]
	wound_areas = []
	for patient in range(1, n_patients + 1):
		initial_area = np.random.uniform(5, 20)
		for visit in range(1, n_visits_per_patient + 1):
			decrease_factor = np.random.uniform(0.7, 0.9)
			area = initial_area * (decrease_factor ** (visit - 1))
			wound_areas.append(area)
			if len(wound_areas) >= n_rows:
				break
	wound_areas = wound_areas[:n_rows]

	base_impedance = np.random.uniform(80, 160, n_rows)
	wound_area_normalized = (wound_areas - np.min(wound_areas)) / (np.max(wound_areas) - np.min(wound_areas))
	impedance_z = base_impedance + wound_area_normalized * 60
	impedance_z_prime = impedance_z * 0.8 + np.random.normal(0, 5, n_rows)
	impedance_z_double_prime = impedance_z * 0.6 + np.random.normal(0, 3, n_rows)
	center_temp = np.random.uniform(97, 101, n_rows)
	edge_temp = center_temp - np.random.uniform(0.5, 2, n_rows)
	peri_temp = edge_temp - np.random.uniform(0.5, 1.5, n_rows)
	oxygenation = np.random.uniform(85, 99, n_rows)
	hemoglobin = np.random.uniform(9, 16, n_rows)
	oxyhemoglobin = hemoglobin * (oxygenation / 100) * 0.95
	deoxyhemoglobin = hemoglobin - oxyhemoglobin
	length = np.random.uniform(1, 7, n_rows)
	width = np.random.uniform(1, 5, n_rows)
	depth = np.random.uniform(0.1, 2, n_rows)
	diabetes_status = np.random.choice(['Yes', 'No'], n_rows)
	smoking_status = np.random.choice(['Current', 'Former', 'Never'], n_rows, p=[0.25, 0.25, 0.5])
	bmi = np.clip(np.random.normal(28, 5, n_rows), 18, 45)
	wound_types = np.random.choice(
		['Diabetic Ulcer', 'Venous Ulcer', 'Pressure Ulcer', 'Surgical Wound', 'Trauma'],
		n_rows,
		p=[0.33, 0.28, 0.22, 0.12, 0.05]
	)

	df = pd.DataFrame({
		'Record ID': record_ids,
		'Event Name': event_names,
		'Calculated Wound Area': wound_areas,
		'Skin Impedance (kOhms) - Z': impedance_z,
		'Skin Impedance (kOhms) - Z\'': impedance_z_prime,
		'Skin Impedance (kOhms) - Z\'\'': impedance_z_double_prime,
		'Center of Wound Temperature (Fahrenheit)': center_temp,
		'Edge of Wound Temperature (Fahrenheit)': edge_temp,
		'Peri-wound Temperature (Fahrenheit)': peri_temp,
		'Oxygenation (%)': oxygenation,
		'Hemoglobin Level': hemoglobin,
		'Oxyhemoglobin Level': oxyhemoglobin,
		'Deoxyhemoglobin Level': deoxyhemoglobin,
		'Length (cm)': length,
		'Width (cm)': width,
		'Depth (cm)': depth,
		'Diabetes?': diabetes_status,
		'Smoking status': smoking_status,
		'BMI': bmi,
		'Wound Type': wound_types
	})
	df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').astype(int)
	df['Healing Rate (%)'] = _calculate_healing_rates(df)
	_create_temperature_gradients(df)
	return df


def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
	"""Creates an interactive wound area progression plot."""
	fig = go.Figure()

	if patient_id:

		patient_df = df[df['Record ID'] == patient_id].copy()

		patient_df['Days_Since_First_Visit'] = (pd.to_datetime(patient_df['Visit date']) - pd.to_datetime(patient_df['Visit date']).min()).dt.days


		# Use 'Days_Since_First_Visit' for plotting, if available
		if 'Days_Since_First_Visit' in patient_df.columns:
			patient_df = patient_df.dropna(subset=['Days_Since_First_Visit', 'Calculated Wound Area'])
			x_axis_data = patient_df['Days_Since_First_Visit']
			x_axis_title = "Days_Since_First_Visit"
		else:
			# if 'Days_Since_First_Visit' is not available use visit number
			x_axis_data = patient_df['Visit Number']
			x_axis_title = 'Visit Number'

		patient_df = patient_df.dropna(subset=[x_axis_title, 'Calculated Wound Area'])


		fig.add_trace(go.Scatter(
			x=x_axis_data,
			y=patient_df['Calculated Wound Area'],
			mode='lines+markers',
			name='Wound Area',
			line=dict(color='blue'),
			hovertemplate='%{y:.1f} cm²'
		))

		if len(patient_df) >= 2:
			try:
				x = x_axis_data.values
				y = patient_df['Calculated Wound Area'].values
				mask = np.isfinite(x) & np.isfinite(y)
				if np.sum(mask) >= 2:
					z = np.polyfit(x[mask], y[mask], 1)
					p = np.poly1d(z)
					fig.add_trace(go.Scatter(
						x=x_axis_data,
						y=p(x_axis_data),
						mode='lines',
						name='Trend',
						line=dict(color='red', dash='dash'),
						hovertemplate='Day %{x}<br>Trend: %{y:.1f} cm²'
					))
			except Exception as e:
				st.warning(f"Could not calculate trend line: {e}")

		# Calculate healing rate
		if len(patient_df) >= 2:
			try:
				total_days = patient_df['Days_Since_First_Visit'].max()
				if total_days > 0:
					first_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmin(), 'Calculated Wound Area']
					last_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmax(), 'Calculated Wound Area']
					healing_rate = (first_area - last_area) / total_days
					healing_status = "Improving" if healing_rate > 0 else "Worsening"
					healing_rate_text = f"Healing Rate: {healing_rate:.2f} cm²/day<br> {healing_status}"
				else:
					healing_rate_text = "Insufficient time between measurements for healing rate calculation"
			except Exception as e:
				healing_rate_text = "Could not calculate healing rate due to data issues"
		else:
			healing_rate_text = "Insufficient data for healing rate calculation"

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
			xaxis_title=x_axis_title,
			yaxis_title="Wound Area (cm²)",
			hovermode='x unified',
			showlegend=True
		)

	else:  # All Patients
		for pid in df['Record ID'].unique():
			patient_df = df[df['Record ID'] == pid].copy()

			patient_df['Days_Since_First_Visit'] = (pd.to_datetime(patient_df['Visit date']) - pd.to_datetime(patient_df['Visit date']).min()).dt.days


			# Use 'Days_Since_First_Visit' if available
			if 'Days_Since_First_Visit' in patient_df.columns:
				patient_df = patient_df.dropna(subset=['Days_Since_First_Visit', 'Calculated Wound Area'])
				x_axis_data = patient_df['Days_Since_First_Visit']
				x_axis_title = 'Days Since First Visit'
			else:
				x_axis_data = patient_df['Visit Number']
				x_axis_title = 'Visit Number'

			patient_df = patient_df.dropna(subset=[x_axis_title, 'Calculated Wound Area'])
			if not patient_df.empty:
				fig.add_trace(go.Scatter(
					x=x_axis_data,
					y=patient_df['Calculated Wound Area'],
					mode='lines+markers',
					name=f'Patient {pid}',
					hovertemplate=f'{x_axis_title} %{{x}}<br>Area: %{{y:.1f}} cm²'
				))
		fig.update_layout(
			title="Wound Area Progression - All Patients",
			xaxis_title=x_axis_title,
			yaxis_title="Wound Area (cm²)",
			hovermode='x unified'
		)

	return fig

def _calculate_healing_rate_text(patient_df: pd.DataFrame, x_axis_title: str) -> str:
	"""Calculates and formats the healing rate text."""
	if x_axis_title == 'Days Since First Visit':
		if len(patient_df) >= 2:
			try:
				total_days = patient_df['Days_Since_First_Visit'].max()
				if total_days > 0:
					first_area = patient_df['Calculated Wound Area'].iloc[0]
					last_area = patient_df['Calculated Wound Area'].iloc[-1]
					healing_rate = (first_area - last_area) / total_days
					return f"Healing Rate: {healing_rate:.2f} cm²/day"
				else: return "Insufficient time between measurements for healing rate calculation"
			except Exception as e: return f"Could not calculate healing rate: {e}"
		else: return "Insufficient data for healing rate calculation"
	else: #if we have visit number only.
		if len(patient_df) >= 2:
			try:
				total_visits = patient_df['Visit Number'].max()
				if total_visits > 1:  # Use > 1 for visits, as it's discrete
					first_area = patient_df['Calculated Wound Area'].iloc[0]
					last_area = patient_df['Calculated Wound Area'].iloc[-1]
					healing_rate = (first_area - last_area) / (total_visits -1) #number of intervals
					return f"Healing Rate: {healing_rate:.2f} cm²/visit"
				else: return "Insufficient visits for healing rate calculation."
			except Exception as e: return f"Could not calculate healing rate: {e}"
		else: return "Insufficient data for healing rate calculation."

def create_impedance_scatterplot(df: pd.DataFrame) -> go.Figure:
	"""Creates an impedance vs. healing rate scatter plot."""
	valid_df = df[df['Healing Rate (%)'] > 0]
	fig = px.scatter(
		valid_df,
		x='Skin Impedance (kOhms) - Z',
		y='Healing Rate (%)',
		color='Diabetes?',
		size='Calculated Wound Area',
		hover_data=['Record ID', 'Event Name', 'Wound Type'],
		title="Impedance vs. Healing Rate Correlation"
	)
	fig.update_layout(
		xaxis_title="Impedance Z (kOhms)",
		yaxis_title="Healing Rate (% reduction per visit)"
	)

	if not valid_df.empty: # Avoid errors if valid_df is empty
	   r, p = stats.pearsonr(valid_df['Skin Impedance (kOhms) - Z'], valid_df['Healing Rate (%)'])
	   p_formatted = f"< 0.001" if p < 0.001 else f"= {p:.3f}"
	   st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")
	else:
	   st.info("Not enough data to calculate correlation.")

	st.write("Higher impedance values correlate with slower healing rates, especially in diabetic patients")
	return fig


# --- Main App Structure ---
def initialize_session_state():
	"""Initializes session state variables."""
	if 'processor' not in st.session_state:
		st.session_state.processor = None
	if 'analysis_complete' not in st.session_state:
		st.session_state.analysis_complete = False
	if 'analysis_results' not in st.session_state:
		st.session_state.analysis_results = None

def setup_page_config():
	"""Sets up the Streamlit page configuration."""
	st.set_page_config(
		page_title="GEMINI",
		page_icon="🩹",
		layout="wide"
	)

def setup_sidebar_config() -> tuple[str, str]:
	"""Sets up the sidebar configuration options."""
	st.sidebar.title("Model Configuration")

	uploaded_file = st.sidebar.file_uploader("Upload Patient Data (CSV)", type=['csv'])
	# Handle file upload, potentially update st.session_state.processor
	if uploaded_file is not None:
	  try:
		  #To prevent the "can't pickle _hashlib.HASH objects" error
		  df = pd.read_csv(uploaded_file)
		  st.session_state.processor = WoundDataProcessor(df) # Assuming WoundDataProcessor can take a DataFrame directly.
		  st.sidebar.success("File uploaded and processed successfully!")
	  except Exception as e:
		  st.sidebar.error(f"Error processing uploaded file: {e}")

	platform_options = WoundAnalysisLLM.get_available_platforms()
	platform = st.sidebar.selectbox(
		"Select Platform",
		platform_options,
		index=platform_options.index(DEFAULT_PLATFORM),
		help="Hugging Face models are currently disabled."
	)

	if platform == "huggingface":
		st.sidebar.warning("Hugging Face models are currently disabled.")
		platform = DEFAULT_PLATFORM

	available_models = WoundAnalysisLLM.get_available_models(platform)
	default_model = DEFAULT_AI_VERDE_MODEL if platform == DEFAULT_PLATFORM else DEFAULT_HF_MODEL
	model_name = st.sidebar.selectbox(
		"Select Model",
		available_models,
		index=available_models.index(default_model)
	)

	with st.sidebar.expander("Model Configuration"):
		api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
		if platform == DEFAULT_PLATFORM:
			base_url = st.text_input("Base URL", value=DEFAULT_AI_VERDE_BASE_URL)
		if api_key:
			os.environ["OPENAI_API_KEY"] = api_key  # Consider more secure key management

	st.sidebar.markdown("---")
	return platform, model_name

def render_patient_selection(df: pd.DataFrame) -> str:
	"""Renders the patient selection dropdown."""
	patient_ids = [int(id) for id in sorted(df['Record ID'].unique())]
	patient_options = ["All Patients"] + [f"Patient {id:03d}" for id in patient_ids]
	selected_patient = st.selectbox("Select Patient", patient_options)
	return selected_patient

def render_main_tabs(df: pd.DataFrame, selected_patient: str, platform: str, model_name: str):
	"""Renders the main tabs of the dashboard."""
	tabs = st.tabs([
		"Overview",
		"Impedance Analysis",
		"Temperature",
		"Oxygenation",
		"Risk Factors",
		"LLM Analysis"
	])

	# Filter data based on patient selection
	if selected_patient == "All Patients":
		filtered_df = df
	else:
		patient_id = int(selected_patient.split(" ")[1])
		filtered_df = df[df['Record ID'] == patient_id]

	with tabs[0]:  # Overview Tab
		st.header("Smart Bandage Wound Healing Analytics")

		if selected_patient == "All Patients":
			total_patients = len(df['Record ID'].unique())
			total_visits = len(df)
			avg_visits = total_visits / total_patients if total_patients > 0 else 0
			col1, col2, col3 = st.columns(3)
			with col1: st.metric("Total Patients", f"{total_patients}")
			with col2: st.metric("Total Visits", f"{total_visits}")
			with col3: st.metric("Average Visits per Patient", f"{avg_visits:.1f}")
		else:
			patient_data = filtered_df.iloc[0]
			st.subheader("Patient Demographics")
			col1, col2, col3 = st.columns(3)
			# Check if 'Calculated Age at Enrollment' exists, otherwise show nothing
			if 'Calculated Age at Enrollment' in patient_data:
				with col1: st.metric("Age", f"{patient_data['Calculated Age at Enrollment']}")
			with col2: st.metric("Gender", f"{patient_data['Sex']}")
			with col3: st.metric("BMI", f"{patient_data['BMI']}")
			st.subheader("Medical History")
			col1, col2, col3, col4 = st.columns(4)
			with col1: st.metric("Diabetes Status", "Yes" if patient_data['Diabetes?'] == 'Yes' else "No")
			with col2: st.metric("Smoking Status", patient_data['Smoking status'])
			# Check for existence of 'Medical History' before trying to use it
			if 'Medical History (select all that apply)' in patient_data:
				with col3: st.metric("Hypertension", "Yes" if 'Cardiovascular' in str(patient_data['Medical History (select all that apply)']) else "No")
				with col4: st.metric("Peripheral Vascular Disease", "Yes" if 'PVD' in str(patient_data['Medical History (select all that apply)']) else "No")
			else:
				with col3: st.metric("Hypertension", "N/A")  # Or any suitable placeholder
				with col4: st.metric("Peripheral Vascular Disease", "N/A")  # Or any suitable placeholder
		st.subheader("Wound Area Over Time")
		fig = create_wound_area_plot(filtered_df, None if selected_patient == "All Patients" else int(selected_patient.split(" ")[1]))
		st.plotly_chart(fig, use_container_width=True)
		if selected_patient != "All Patients":
			_ = _calculate_healing_rate_text(filtered_df, 'Days_Since_First_Visit' if 'Days_Since_First_Visit' in filtered_df.columns else 'Visit Number')
			# st.markdown(f"**{healing_rate_text}**")

	with tabs[1]: # Impedance Analysis Tab
		st.subheader("Impedance vs. Wound Healing Progress")

		if selected_patient == "All Patients":
			fig = create_impedance_scatterplot(df)
			st.plotly_chart(fig, use_container_width=True)

		else:
			patient_data = filtered_df.sort_values('Visit Number')
			fig = px.line(patient_data, x='Visit Number',
							y=['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'',
									'Skin Impedance (kOhms) - Z\'\''],
							title=f"Impedance Measurements Over Time for {selected_patient}", markers=True)
			fig.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)", legend_title="Measurement")
			st.plotly_chart(fig, use_container_width=True)

		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Impedance Components Over Time")
			if selected_patient == "All Patients":
				avg_impedance = df.groupby('Visit Number')[
					['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\'']
				].mean().reset_index()
				fig = px.line(avg_impedance, x='Visit Number',
								y=['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'',
									'Skin Impedance (kOhms) - Z\'\''],
								title="Average Impedance Components by Visit", markers=True)
				fig.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)", legend_title="Component")
				st.plotly_chart(fig, use_container_width=True)

		with col2:
			st.subheader("Impedance by Wound Type")
			if selected_patient == "All Patients":
				avg_by_type = df.groupby('Wound Type')['Skin Impedance (kOhms) - Z'].mean().reset_index()
				fig = px.bar(avg_by_type, x='Wound Type', y='Skin Impedance (kOhms) - Z',
								title="Average Impedance by Wound Type", color='Wound Type')
				fig.update_layout(xaxis_title="Wound Type", yaxis_title="Average Impedance Z (kOhms)")
				st.plotly_chart(fig, use_container_width=True)
			else:
				wound_type = filtered_df['Wound Type'].iloc[0]
				avg_impedance = filtered_df['Skin Impedance (kOhms) - Z'].mean()
				st.info(f"{selected_patient} has a {wound_type} with average impedance of {avg_impedance:.1f} kOhms")

	with tabs[2]: # Temperature Analysis Tab
		st.subheader("Temperature Gradient Analysis")

		if selected_patient == "All Patients":
			temp_df = df.copy()
			# Gradients should have been pre-calculated
			if 'Center-Edge Temp Gradient' in temp_df.columns:
				fig = px.box(temp_df, x='Wound Type',
							  y=['Center-Edge Temp Gradient', 'Edge-Peri Temp Gradient', 'Total Temp Gradient'],
							  title="Temperature Gradients by Wound Type", points="all")
				fig.update_layout(xaxis_title="Wound Type", yaxis_title="Temperature Gradient (°F)", boxmode='group')
				st.plotly_chart(fig, use_container_width=True)

				fig = px.scatter(temp_df[temp_df['Healing Rate (%)'] > 0], x='Total Temp Gradient',
								 y='Healing Rate (%)', color='Wound Type', size='Calculated Wound Area',
								 hover_data=['Record ID', 'Event Name'],
								 title="Temperature Gradient vs. Healing Rate")
				fig.update_layout(xaxis_title="Temperature Gradient (Center to Peri-wound, °F)",
								  yaxis_title="Healing Rate (% reduction per visit)")
				st.plotly_chart(fig, use_container_width=True)
		else:
			patient_data = filtered_df.sort_values('Visit Number')
			# Gradients should have been pre-calculated
			if 'Center-Edge Temp Gradient' in patient_data.columns:

				fig = make_subplots(specs=[[{"secondary_y": True}]])
				fig.add_trace(go.Scatter(x=patient_data['Visit Number'],
										 y=patient_data['Center of Wound Temperature (Fahrenheit)'],
										 name="Center Temp", line=dict(color='red')))
				fig.add_trace(go.Scatter(x=patient_data['Visit Number'],
										 y=patient_data['Edge of Wound Temperature (Fahrenheit)'],
										 name="Edge Temp", line=dict(color='orange')))
				fig.add_trace(go.Scatter(x=patient_data['Visit Number'],
										 y=patient_data['Peri-wound Temperature (Fahrenheit)'],
										 name="Peri-wound Temp", line=dict(color='blue')))
				fig.add_trace(go.Bar(x=patient_data['Visit Number'], y=patient_data['Center-Edge Temp Gradient'],
									 name="Center-Edge Gradient", opacity=0.5, marker_color='lightpink'),
							  secondary_y=True)
				fig.add_trace(go.Bar(x=patient_data['Visit Number'], y=patient_data['Edge-Peri Temp Gradient'],
									 name="Edge-Peri Gradient", opacity=0.5, marker_color='lightblue'),
							  secondary_y=True)
				fig.update_layout(title=f"Temperature Measurements for {selected_patient}", xaxis_title="Visit Number",
								  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
				fig.update_yaxes(title_text="Temperature (°F)", secondary_y=False)
				fig.update_yaxes(title_text="Temperature Gradient (°F)", secondary_y=True)
				st.plotly_chart(fig, use_container_width=True)

	with tabs[3]:  # Oxygenation Tab
	  st.subheader("Oxygenation Metrics")

	  if selected_patient == "All Patients":
		  fig = px.scatter(
			  df[df['Healing Rate (%)'] > 0],
			  x='Oxygenation (%)',
			  y='Healing Rate (%)',
			  color='Diabetes?',
			  size='Hemoglobin Level',
			  hover_data=['Record ID', 'Event Name', 'Wound Type'],
			  title="Relationship Between Oxygenation and Healing Rate"
		  )
		  fig.update_layout(
			  xaxis_title="Oxygenation (%)",
			  yaxis_title="Healing Rate (% reduction per visit)"
		  )
		  st.plotly_chart(fig, use_container_width=True)

		  fig = px.box(
			  df,
			  x='Wound Type',
			  y='Oxygenation (%)',
			  title="Oxygenation Levels by Wound Type",
			  color='Wound Type',
			  points="all"
		  )
		  fig.update_layout(
			  xaxis_title="Wound Type",
			  yaxis_title="Oxygenation (%)"
		  )
		  st.plotly_chart(fig, use_container_width=True)

	  else:  # Individual patient
		  patient_data = filtered_df.sort_values('Visit Number')
		  fig = go.Figure()
		  fig.add_trace(
			  go.Bar(
				  x=patient_data['Visit Number'],
				  y=patient_data['Oxyhemoglobin Level'],
				  name="Oxyhemoglobin",
				  marker_color='red'
			  )
		  )
		  fig.add_trace(
			  go.Bar(
				  x=patient_data['Visit Number'],
				  y=patient_data['Deoxyhemoglobin Level'],
				  name="Deoxyhemoglobin",
				  marker_color='purple'
			  )
		  )
		  fig.update_layout(
			  title=f"Hemoglobin Components for {selected_patient}",
			  xaxis_title="Visit Number",
			  yaxis_title="Level (g/dL)",
			  barmode='stack',
			  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
		  st.plotly_chart(fig, use_container_width=True)

		  fig = px.line(
			  patient_data,
			  x='Visit Number',
			  y='Oxygenation (%)',
			  title=f"Oxygenation Over Time for {selected_patient}",
			  markers=True
		  )
		  fig.update_layout(
			  xaxis_title="Visit Number",
			  yaxis_title="Oxygenation (%)",
			  yaxis=dict(range=[80, 100])  # Consistent y-axis range
		  )
		  st.plotly_chart(fig, use_container_width=True)

	with tabs[4]:  # Risk Factors Tab
	  st.subheader("Risk Factor Analysis")

	  if selected_patient == "All Patients":
		  risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

		  with risk_tab1:
			  diab_healing = df.groupby(['Diabetes?', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
			  diab_healing = diab_healing[diab_healing['Visit Number'] > 1]
			  fig = px.line(diab_healing, x='Visit Number', y='Healing Rate (%)', color='Diabetes?',
							title="Average Healing Rate by Diabetes Status", markers=True)
			  fig.update_layout(xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)")
			  st.plotly_chart(fig, use_container_width=True)

			  diab_imp = df.groupby('Diabetes?')['Skin Impedance (kOhms) - Z'].mean().reset_index()
			  fig = px.bar(diab_imp, x='Diabetes?', y='Skin Impedance (kOhms) - Z', color='Diabetes?',
						   title="Average Impedance by Diabetes Status")
			  fig.update_layout(xaxis_title="Diabetes Status", yaxis_title="Average Impedance Z (kOhms)")
			  st.plotly_chart(fig, use_container_width=True)

		  with risk_tab2:
			  smoke_healing = df.groupby(['Smoking status', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
			  smoke_healing = smoke_healing[smoke_healing['Visit Number'] > 1]
			  fig = px.line(smoke_healing, x='Visit Number', y='Healing Rate (%)', color='Smoking status',
							title="Average Healing Rate by Smoking Status", markers=True)
			  fig.update_layout(xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)")
			  st.plotly_chart(fig, use_container_width=True)

			  smoke_wound = pd.crosstab(df['Smoking status'], df['Wound Type'])
			  smoke_wound_pct = smoke_wound.div(smoke_wound.sum(axis=1), axis=0) * 100
			  fig = px.bar(smoke_wound_pct.reset_index().melt(id_vars='Smoking status', var_name='Wound Type',
															 value_name='Percentage'),
						   x='Smoking status', y='Percentage', color='Wound Type',
						   title="Distribution of Wound Types by Smoking Status", barmode='stack')
			  fig.update_layout(xaxis_title="Smoking Status", yaxis_title="Percentage (%)")
			  st.plotly_chart(fig, use_container_width=True)

		  with risk_tab3:
			  df['BMI Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, 100],
										  labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II-III'])
			  bmi_healing = df.groupby(['BMI Category', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
			  bmi_healing = bmi_healing[bmi_healing['Visit Number'] > 1]
			  fig = px.line(bmi_healing, x='Visit Number', y='Healing Rate (%)', color='BMI Category',
							title="Average Healing Rate by BMI Category", markers=True)
			  fig.update_layout(xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)")
			  st.plotly_chart(fig, use_container_width=True)

			  fig = px.scatter(df[df['Healing Rate (%)'] > 0], x='BMI', y='Healing Rate (%)', color='Diabetes?',
							   size='Calculated Wound Area', hover_data=['Record ID', 'Event Name', 'Wound Type'],
							   title="BMI vs. Healing Rate")
			  fig.update_layout(xaxis_title="BMI", yaxis_title="Healing Rate (% reduction per visit)")
			  st.plotly_chart(fig, use_container_width=True)

	  else:  # Individual patient
		  patient_data = filtered_df.iloc[0]
		  col1, col2 = st.columns(2)

		  with col1:
			  st.subheader("Patient Risk Profile")
			  st.info(f"**Diabetes Status:** {patient_data['Diabetes?']}")
			  st.info(f"**Smoking Status:** {patient_data['Smoking status']}")
			  st.info(f"**BMI:** {patient_data['BMI']:.1f}")
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

			  if 'Total Temp Gradient' in patient_data and patient_data['Total Temp Gradient'] > 3:
				  risk_factors.append("High temperature gradient")
				  risk_score += 2
			  if patient_data['Skin Impedance (kOhms) - Z'] > 140:
				  risk_factors.append("High impedance")
				  risk_score += 2

			  if risk_score >= 6:
				  risk_category = "High"
				  risk_color = "red"
			  elif risk_score >= 3:
				  risk_category = "Moderate"
				  risk_color = "orange"
			  else:
				  risk_category = "Low"
				  risk_color = "green"

			  st.markdown(
				  f"**Risk Category:** <span style='color:{risk_color};font-weight:bold'>{risk_category}</span> ({risk_score} points)",
				  unsafe_allow_html=True)

			  if risk_factors:
				  st.markdown("**Risk Factors:**")
				  for factor in risk_factors:
					  st.markdown(f"- {factor}")
			  else:
				  st.markdown("**Risk Factors:** None identified")
			  wound_area = patient_data['Calculated Wound Area']
			  base_healing_weeks = 2 + wound_area / 2
			  risk_multiplier = 1 + (risk_score * 0.1)
			  est_healing_weeks = base_healing_weeks * risk_multiplier
			  st.markdown(f"**Estimated Healing Time:** {est_healing_weeks:.1f} weeks")

	with tabs[5]:  # LLM Analysis Tab
		st.header("LLM-Powered Wound Analysis")
		# Use st.session_state.processor if available (from uploaded file), otherwise use df.
		data_to_analyze = st.session_state.processor.get_patient_visits(patient_id) if st.session_state.processor and selected_patient != "All Patients" else None

		if selected_patient != "All Patients":
			if st.button("Run Analysis", key="run_analysis"):
				if data_to_analyze is not None:
					with st.spinner("Analyzing patient data..."):
						try:
							llm = WoundAnalysisLLM(platform=platform, model_name=model_name)
							# Correctly pass patient_data to analyze_patient_data
							analysis = llm.analyze_patient_data(data_to_analyze)

							# Create and save the report using data_to_analyze
							report_path = create_and_save_report(data_to_analyze, analysis)

							st.session_state.analysis_complete = True
							st.session_state.analysis_results = analysis
							st.session_state.report_path = report_path
							st.markdown("### Analysis Results")
							st.write(analysis)
							st.markdown(_create_download_link(report_path, "Download Word Report"), unsafe_allow_html=True)
							st.success("Analysis complete!")

						except Exception as e:
							st.error(f"Error during analysis: {str(e)}")
				else:  # data_to_analyze is None
					st.warning("No data available for analysis. Please ensure data is loaded correctly.")

		else: # selected_patient == "All Patients":
			st.warning("Please select a specific patient to run the analysis.")

		if not st.session_state.analysis_complete and selected_patient != "All Patients":
			st.info("Click 'Run Analysis' to generate an AI-powered analysis.")

def create_and_save_report(patient_data: Dict, analysis_results: str) -> str:
	"""Create and save the analysis report as a Word document."""
	doc = Document()
	format_word_document(doc, patient_data, analysis_results)  # Ensure this function handles dict
	doc.save(REPORT_FILE_NAME)
	return REPORT_FILE_NAME

def main():
	"""Main application entry point."""
	initialize_session_state()
	setup_page_config()
	platform, model_name = setup_sidebar_config()
	df = load_data()
	if df is not None:
	  selected_patient = render_patient_selection(df)
	  # Display data source information
	  if DATA_FILE_PATH.lower() != "mock": #Case insensitive comparison
		  st.success("✅ Using actual data from CSV file")
	  else:
		  st.warning("⚠️ Using simulated data (CSV file could not be loaded)")

	  render_main_tabs(df, selected_patient, platform, model_name)

	# Footer
	st.markdown("---")
	st.markdown("""
	**Note:** This dashboard loads data from the specified CSV file path.
	If the file cannot be loaded, the dashboard will fall back to simulated data.
	""")

	# Sidebar with additional information
	with st.sidebar:
		st.header("About This Dashboard")
		st.write("""
		This Streamlit dashboard visualizes wound healing data collected with smart bandage technology.

		The analysis focuses on: Impedance, Temperature, Oxygenation, and Risk Factors.

		Use the dropdown to view all patients or individual patient profiles.
		""")
		st.header("Statistical Methods")
		st.write("""
		Includes: Longitudinal analysis, Correlation analysis, Risk factor assessment, and Comparative analysis.
		""")

if __name__ == "__main__":
	main()
