import pandas as pd
import plotly.express as px
import streamlit as st

from wound_analysis.dashboard_components.visualizer import Visualizer
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.statistical_analysis import CorrelationAnalysis

class OxygenationTab:
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

	def __init__(self, df: pd.DataFrame, selected_patient: str, data_processor: WoundDataProcessor):
		self.data_processor = data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = df

	def render(self) -> None:
		st.header("Oxygenation Analysis")

		if self.patient_id == "All Patients":
			OxygenationTab._render_population(df=self.df)
		else:
			visits     = self.data_processor.get_patient_visits(record_id=self.patient_id)['visits']
			df_patient = self.data_processor.get_patient_dataframe(record_id=self.patient_id)
			OxygenationTab._render_patient(df_patient=df_patient, visits=visits)

	@staticmethod
	def _render_population(df: pd.DataFrame) -> None:
		"""
		Renders oxygenation analysis for the entire patient population.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing oxygenation data for all patients.
		"""
		valid_df = df.copy()
		# valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

		valid_df['Hemoglobin Level'] = pd.to_numeric(valid_df['Hemoglobin Level'], errors='coerce')
		valid_df['Oxygenation (%)']  = pd.to_numeric(valid_df['Oxygenation (%)'], errors='coerce')
		valid_df['Healing Rate (%)'] = pd.to_numeric(valid_df['Healing Rate (%)'], errors='coerce')

		valid_df = valid_df.dropna(subset=['Oxygenation (%)', 'Healing Rate (%)', 'Hemoglobin Level'])

		# Add outlier threshold control
		col1, _, col3 = st.columns([2, 3, 3])

		with col1:
			outlier_threshold = st.number_input(
				"Oxygenation Outlier Threshold",
				min_value = 0.0,
				max_value = 0.9,
				value     = 0.2,
				step      = 0.05,
				help      = "Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
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

	@staticmethod
	def _render_patient(df_patient: pd.DataFrame, visits: list) -> None:
		"""
		Renders oxygenation analysis for an individual patient.

		Parameters
		----------
		df_patient : pd.DataFrame
			The dataframe containing oxygenation data.
		visits : list
			The list of visits for the patient.
		"""

		# Convert Visit date to string for display
		df_patient['Visit date'] = pd.to_datetime(df_patient['Visit date']).dt.strftime('%m-%d-%Y')


		fig_bar, fig_line = Visualizer.create_oxygenation_chart(patient_data=df_patient, visits=visits)

		st.plotly_chart(fig_bar, use_container_width=True)
		st.plotly_chart(fig_line, use_container_width=True)

		# Display interpretation guidance
		st.markdown("### Oxygenation Interpretation")
		st.markdown("""
		<div style="margin-top:10px; padding:15px; background-color:#f8f9fa; border-left:4px solid #6c757d; font-size:0.9em;">
		<p style="margin-top:0; color:#666; font-weight:bold;">ABOUT THIS ANALYSIS:</p>
		<strong>Oxygenation levels indicate:</strong><br>
		• <strong>Below 90%</strong>: Hypoxic conditions - impaired healing<br>
		• <strong>90-95%</strong>: Borderline - monitor closely<br>
		• <strong>Above 95%</strong>: Adequate oxygen - favorable for healing<br><br>
		<strong>Clinical Significance:</strong><br>
		• Oxygenation trends over time are key indicators of healing progress<br>
		• Consistently improving oxygenation suggests better perfusion and healing potential
		</div>
		""", unsafe_allow_html=True)
