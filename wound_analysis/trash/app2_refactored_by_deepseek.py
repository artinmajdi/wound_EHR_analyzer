"""Smart Bandage Wound Healing Analytics Dashboard"""

import base64
from io import BytesIO
from typing import Optional, Dict, Any
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from docx import Document

from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document

# Constants
DATA_PATH = "/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv"
DEFAULT_MODEL_MAP = {"ai-verde": "llama-3.3-70b-fp8", "huggingface": "medalpaca-7b"}

class DataLoader:
	"""Handles data loading and preprocessing"""

	@staticmethod
	@st.cache_data
	def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
		"""Load and preprocess wound healing data"""
		try:
			df = pd.read_csv(file_path)
			df.columns = df.columns.str.strip()
			return DataProcessor.preprocess_data(df)
		except Exception as e:
			st.error(f"Error loading data: {e}\nUsing mock data instead")
			return DataLoader.generate_mock_data()

	@staticmethod
	def generate_mock_data() -> pd.DataFrame:
		"""Generate simulated wound healing data"""
		np.random.seed(42)
		n_rows = 418
		patient_ids = np.repeat(np.arange(1, 128), 3)[:n_rows]

		data = {
			'Record ID': patient_ids,
			'Event Name': [f"Visit {v}" for v in np.tile([1, 2, 3], 127)[:n_rows]],
			'Calculated Wound Area': np.random.uniform(5, 20, n_rows),
			'Skin Impedance (kOhms) - Z': np.random.uniform(80, 160, n_rows),
			'Skin Impedance (kOhms) - Z\'': np.random.uniform(60, 140, n_rows),
			'Skin Impedance (kOhms) - Z\'\'': np.random.uniform(40, 120, n_rows),
			'Center of Wound Temperature (Fahrenheit)': np.random.uniform(97, 101, n_rows),
			'Edge of Wound Temperature (Fahrenheit)': np.random.uniform(96, 100, n_rows),
			'Peri-wound Temperature (Fahrenheit)': np.random.uniform(95, 99, n_rows),
			'Oxygenation (%)': np.random.uniform(85, 99, n_rows),
			'Hemoglobin Level': np.random.uniform(9, 16, n_rows),
			'Diabetes?': np.random.choice(['Yes', 'No'], n_rows),
			'Smoking status': np.random.choice(['Current', 'Former', 'Never'], n_rows),
			'BMI': np.clip(np.random.normal(28, 5, n_rows), 18, 45),
			'Wound Type': np.random.choice(
				['Diabetic Ulcer', 'Venous Ulcer', 'Pressure Ulcer', 'Surgical Wound', 'Trauma'],
				n_rows
			)
		}

		df = pd.DataFrame(data)
		return DataProcessor.preprocess_data(df)

class DataProcessor:
	"""Contains data preprocessing utilities"""

	@staticmethod
	def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
		"""Apply all preprocessing steps to raw data"""
		df = DataProcessor._extract_visit_numbers(df)
		df = DataProcessor._calculate_derived_features(df)
		df = DataProcessor._handle_missing_values(df)
		df = DataProcessor._add_temperature_gradients(df)
		return df

	@staticmethod
	def _extract_visit_numbers(df: pd.DataFrame) -> pd.DataFrame:
		df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)
		return df

	@staticmethod
	def _calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
		if 'Calculated Wound Area' not in df.columns:
			df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

		df['Healing Rate (%)'] = DataProcessor._calculate_healing_rates(df)
		return df

	@staticmethod
	def _calculate_healing_rates(df: pd.DataFrame) -> list:
		rates = []
		for pid in df['Record ID'].unique():
			patient_data = df[df['Record ID'] == pid].sort_values('Visit Number')
			for i, row in patient_data.iterrows():
				if row['Visit Number'] == 1:
					rates.append(0)
				else:
					prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
					if not prev_visits.empty:
						prev_area = prev_visits.iloc[-1]['Calculated Wound Area']
						curr_area = row['Calculated Wound Area']
						if prev_area > 0:
							rates.append(((prev_area - curr_area) / prev_area) * 100)
						else:
							rates.append(0)
					else:
						rates.append(0)
		return rates[:len(df)]

	@staticmethod
	def _add_temperature_gradients(df: pd.DataFrame) -> pd.DataFrame:
		if all(col in df.columns for col in ['Center of Wound Temperature (Fahrenheit)',
										   'Edge of Wound Temperature (Fahrenheit)',
										   'Peri-wound Temperature (Fahrenheit)']):
			df['Center-Edge Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Edge of Wound Temperature (Fahrenheit)']
			df['Edge-Peri Temp Gradient'] = df['Edge of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']
			df['Total Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']
		return df

	@staticmethod
	def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
		numeric_cols = df.select_dtypes(include=['number']).columns
		for col in numeric_cols:
			df[col] = df[col].fillna(df[col].median())

		if 'Diabetes?' in df.columns:
			df['Diabetes?'] = df['Diabetes?'].fillna('Unknown').replace({'yes': 'Yes', 'no': 'No', 'Y': 'Yes', 'N': 'No'})

		if 'Smoking status' in df.columns:
			df['Smoking status'] = df['Smoking status'].fillna('Unknown')

		return df

class VisualizationManager:
	"""Manages creation of all visualizations"""

	@staticmethod
	def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
		"""Create interactive wound area progression plot"""
		fig = go.Figure()

		if patient_id:
			patient_df = df[df['Record ID'] == patient_id].sort_values('Visit Number')
			patient_df = patient_df.dropna(subset=['Calculated Wound Area'])

			fig.add_trace(go.Scatter(
				x=patient_df['Visit Number'],
				y=patient_df['Calculated Wound Area'],
				mode='lines+markers',
				name='Wound Area',
				line=dict(color='blue')
			))  # Fixed missing parenthesis

			if len(patient_df) >= 2:
				try:
					x = patient_df['Visit Number'].values
					y = patient_df['Calculated Wound Area'].values
					z = np.polyfit(x, y, 1)
					p = np.poly1d(z)
					fig.add_trace(go.Scatter(
						x=x,
						y=p(x),
						mode='lines',
						name='Trend',
						line=dict(color='red', dash='dash')
					))
				except Exception as e:
					st.warning(f"Could not calculate trend line: {e}")

			fig.update_layout(
				title=f"Wound Area Progression - Patient {patient_id}",
				xaxis_title="Visit Number",
				yaxis_title="Wound Area (cm²)",
				height=600
			)
		else:
			for pid in df['Record ID'].unique()[:10]:  # Limit to first 10 for performance
				patient_df = df[df['Record ID'] == pid].sort_values('Visit Number')
				fig.add_trace(go.Scatter(
					x=patient_df['Visit Number'],
					y=patient_df['Calculated Wound Area'],
					mode='lines+markers',
					name=f'Patient {pid}'
				))

			fig.update_layout(
				title="Wound Area Progression - All Patients",
				xaxis_title="Visit Number",
				yaxis_title="Wound Area (cm²)",
				height=600,
				showlegend=True
			)

		return fig

	@staticmethod
	def create_impedance_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
		"""Create impedance-related visualizations"""
		if patient_id:
			patient_df = df[df['Record ID'] == patient_id].sort_values('Visit Number')
			fig = px.line(
				patient_df,
				x='Visit Number',
				y=['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\''],
				title="Impedance Over Time",
				markers=True
			)
		else:
			fig = px.scatter(
				df[df['Healing Rate (%)'] > 0],
				x='Skin Impedance (kOhms) - Z',
				y='Healing Rate (%)',
				color='Diabetes?',
				title="Impedance vs Healing Rate"
			)
		return fig

class ReportGenerator:
	"""Handles report generation and download functionality"""

	@staticmethod
	def create_word_report(patient_data: dict, analysis_results: str) -> str:
		"""Generate and save Word report"""
		doc = Document()
		format_word_document(doc, patient_data, analysis_results)
		report_path = "wound_analysis_report.docx"
		doc.save(report_path)
		return report_path

	@staticmethod
	def create_download_link(report_path: str) -> str:
		"""Generate download link for generated report"""
		with open(report_path, "rb") as f:
			b64 = base64.b64encode(f.read()).decode()
		return f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}'

class StreamlitUI:
	"""Streamlit user interface class"""
	def __init__(self):
		"""Initialize StreamlitUI instance"""
		self.df = None

	def _create_sidebar(self, df: pd.DataFrame) -> tuple:
		"""Create sidebar components and return model config"""
		st.sidebar.title("Smart Bandage Analytics")

		# Patient Selection
		selected_patient = st.sidebar.selectbox(
			"Select Patient",
			self._get_patient_options(df),
			key='selected_patient'
		)

		# Date Range Filter
		if 'Visit date' in df.columns:
			min_date = df['Visit date'].min()
			max_date = df['Visit date'].max()
			date_range = st.sidebar.date_input(
				"Date Range",
				value=(min_date, max_date),
				min_value=min_date,
				max_value=max_date
			)

		# Model Configuration
		st.sidebar.markdown("---")
		st.sidebar.subheader("Model Configuration")

		platform = st.sidebar.selectbox(
			"Select Platform",
			WoundAnalysisLLM.get_available_platforms(),
			index=0,
			help="Choose the AI platform for analysis"
		)

		model_name = st.sidebar.selectbox(
			"Select Model",
			WoundAnalysisLLM.get_available_models(platform),
			index=0
		)

		with st.sidebar.expander("Advanced Configuration"):
			api_key = st.text_input("API Key", type='password', value="sk-h8JtQkCCJUOy-TAdDxCLGw")
			if platform == "ai-verde":
				st.text_input("Base URL", value="https://llm-api.cyverse.ai")

		return platform, model_name, selected_patient

	def _get_patient_options(self, df: pd.DataFrame) -> list:
		"""Generate patient selection options"""
		return ["All Patients"] + [f"Patient {int(pid):03d}" for pid in df['Record ID'].unique()]

	def _show_population_stats(self, df: pd.DataFrame):
		"""Display population-level statistics"""
		col1, col2, col3 = st.columns(3)
		col1.metric("Total Patients", df['Record ID'].nunique())
		col2.metric("Total Visits", len(df))
		col3.metric("Avg Visits/Patient", f"{len(df)/df['Record ID'].nunique():.1f}")

	def _show_patient_details(self, df: pd.DataFrame, selected_patient: str):
		"""Display individual patient details"""
		patient_id = self._get_patient_id(selected_patient)
		patient_data = df[df['Record ID'] == patient_id].iloc[0]

		cols = st.columns(4)
		cols[0].metric("Diabetes Status", patient_data['Diabetes?'])
		cols[1].metric("Smoking Status", patient_data['Smoking status'])
		cols[2].metric("BMI", f"{patient_data['BMI']:.1f}")
		cols[3].metric("Wound Type", patient_data['Wound Type'])

	def _render_impedance_tab(self, df: pd.DataFrame):
		"""Render impedance analysis tab"""
		st.header("Impedance Analysis")
		selected_patient = st.session_state.get('selected_patient', "All Patients")

		fig = VisualizationManager.create_impedance_plot(df, self._get_patient_id(selected_patient))
		st.plotly_chart(fig, use_container_width=True)

		if selected_patient == "All Patients":
			r, p = stats.pearsonr(
				df[df['Healing Rate (%)'] > 0]['Skin Impedance (kOhms) - Z'],
				df[df['Healing Rate (%)'] > 0]['Healing Rate (%)']
			)
			st.info(f"Correlation: r = {r:.2f}, p = {p:.3f}")

	def _render_temperature_tab(self, df: pd.DataFrame):
		"""Render temperature analysis tab"""
		st.header("Temperature Analysis")
		selected_patient = st.session_state.get('selected_patient', "All Patients")

		if selected_patient == "All Patients":
			fig = px.box(
				df,
				x='Wound Type',
				y=['Center-Edge Temp Gradient', 'Edge-Peri Temp Gradient'],
				title="Temperature Gradients by Wound Type"
			)
		else:
			patient_df = df[df['Record ID'] == self._get_patient_id(selected_patient)]
			fig = make_subplots(specs=[[{"secondary_y": True}]])
			fig.add_trace(go.Scatter(
				x=patient_df['Visit Number'],
				y=patient_df['Center of Wound Temperature (Fahrenheit)'],
				name="Center Temp"
			))
			fig.add_trace(go.Scatter(
				x=patient_df['Visit Number'],
				y=patient_df['Edge of Wound Temperature (Fahrenheit)'],
				name="Edge Temp"
			))

		st.plotly_chart(fig, use_container_width=True)

	def _render_oxygenation_tab(self, df: pd.DataFrame):
		"""Render oxygenation analysis tab"""
		st.header("Oxygenation Analysis")
		selected_patient = st.session_state.get('selected_patient', "All Patients")

		if selected_patient == "All Patients":
			fig = px.scatter(
				df,
				x='Oxygenation (%)',
				y='Healing Rate (%)',
				color='Wound Type',
				title="Oxygenation vs Healing Rate"
			)
		else:
			patient_df = df[df['Record ID'] == self._get_patient_id(selected_patient)]
			fig = px.line(
				patient_df,
				x='Visit Number',
				y='Oxygenation (%)',
				title="Oxygenation Over Time"
			)

		st.plotly_chart(fig, use_container_width=True)

	def _render_risk_factors_tab(self, df: pd.DataFrame):
		"""Render risk factors analysis tab"""
		st.header("Risk Factor Analysis")
		selected_patient = st.session_state.get('selected_patient', "All Patients")

		if selected_patient == "All Patients":
			fig = px.box(
				df,
				x='Diabetes?',
				y='Healing Rate (%)',
				title="Healing Rate by Diabetes Status"
			)
		else:
			patient_data = df[df['Record ID'] == self._get_patient_id(selected_patient)].iloc[0]
			risk_score = self._calculate_risk_score(patient_data)

			cols = st.columns(2)
			cols[0].metric("Risk Score", risk_score)
			cols[1].metric("Estimated Healing Time", f"{risk_score*2} weeks")

		st.plotly_chart(fig, use_container_width=True) if selected_patient == "All Patients" else None

	def _render_llm_tab(self, df: pd.DataFrame, platform: str, model_name: str):
		"""Render LLM analysis tab"""
		st.header("AI Analysis")
		selected_patient = st.session_state.get('selected_patient', "All Patients")

		if selected_patient != "All Patients" and st.button("Run Analysis"):
			with st.spinner("Analyzing..."):
				try:
					llm = WoundAnalysisLLM(platform=platform, model_name=model_name)
					patient_id = self._get_patient_id(selected_patient)
					analysis = llm.analyze_patient_data(df[df['Record ID'] == patient_id])
					st.session_state.analysis_results = analysis
					st.session_state.report_path = ReportGenerator.create_word_report(
						df[df['Record ID'] == patient_id].to_dict(),
						analysis
					)
					st.success("Analysis complete!")
				except Exception as e:
					st.error(f"Analysis failed: {str(e)}")

		if st.session_state.analysis_results:
			st.markdown(st.session_state.analysis_results)
			href = ReportGenerator.create_download_link(st.session_state.report_path)
			st.markdown(f'<a href="{href}" download>Download Report</a>', unsafe_allow_html=True)

	def _get_selected_patient(self) -> str:
		"""Get currently selected patient from state"""
		return st.session_state.get('selected_patient', "All Patients")

	def _get_patient_id(self, selected_patient: str) -> Optional[int]:
		"""Extract patient ID from selection string"""
		return int(selected_patient.split()[-1]) if selected_patient != "All Patients" else None

	def _calculate_risk_score(self, patient_data: pd.Series) -> int:
		"""Calculate simple risk score based on patient data"""
		score = 0
		if patient_data['Diabetes?'] == 'Yes':
			score += 2
		if patient_data['Smoking status'] in ['Current', 'Former']:
			score += 1
		if patient_data['BMI'] > 30:
			score += 1
		return score

	def _render_overview_tab(self, df: pd.DataFrame):
		"""Render content for overview tab"""
		st.header("Wound Healing Overview")
		selected_patient = st.session_state.get('selected_patient', "All Patients")

		if selected_patient == "All Patients":
			self._show_population_stats(df)
		else:
			self._show_patient_details(df, selected_patient)

		st.plotly_chart(
			VisualizationManager.create_wound_area_plot(df, self._get_patient_id(selected_patient)),
			use_container_width=True
		)

	def _create_main_tabs(self, df: pd.DataFrame, platform: str, model_name: str):
		"""Create and populate main interface tabs"""
		tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
			"Overview", "Impedance Analysis", "Temperature",
			"Oxygenation", "Risk Factors", "LLM Analysis"
		])

		with tab1:
			self._render_overview_tab(df)
		with tab2:
			self._render_impedance_tab(df)
		with tab3:
			self._render_temperature_tab(df)
		with tab4:
			self._render_oxygenation_tab(df)
		with tab5:
			self._render_risk_factors_tab(df)
		with tab6:
			self._render_llm_tab(df, platform, model_name)

	def _render_footer(self):
		"""Render application footer"""
		st.markdown("---")
		st.markdown("**Note:** Data sourced from 'dataset/SmartBandage-Data_for_llm.csv'")
		st.sidebar.markdown("---")
		st.sidebar.download_button(
			label="Download Sample Report",
			data=b"Sample content",
			file_name="wound_report.pdf",
			mime="application/pdf"
		)

	def run(self):
		"""Main application entry point"""
		self.df = DataLoader.load_data()
		
		if self.df is not None:
			platform, model_name, selected_patient = self._create_sidebar(self.df)
			self._create_main_tabs(self.df, platform, model_name)
		else:
			st.error("Failed to load data. Please check your data source.")

if __name__ == "__main__":
	ui = StreamlitUI()
	ui.run()
