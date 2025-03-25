import pandas as pd
import plotly.express as px
import streamlit as st

from wound_analysis.dashboard_components.visualizer import Visualizer
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.statistical_analysis import CorrelationAnalysis


class TemperatureTab:
	"""
	A class for managing and rendering the Temperature tab in the wound analysis dashboard.

	This class contains methods to display temperature analysis for both population-level data
	and individual patient data, including temperature trends, gradient analysis, and clinical
	interpretations.
	"""

	def __init__(self, df: pd.DataFrame, selected_patient: str, data_processor: WoundDataProcessor):
		self.data_processor = data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = df

	def render(self) -> None:
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
		st.header("Temperature Analysis")

		if self.patient_id == "All Patients":
			self._render_population()
		else:
			visits     = self.data_processor.get_patient_visits(record_id=self.patient_id)['visits']
			df_patient = self.data_processor.get_patient_dataframe(record_id=self.patient_id)
			self._render_patient(df_patient=df_patient, visits=visits, patient_id=self.patient_id)


	def _render_population(self) -> None:
		"""
		Renders temperature analysis for the entire patient population.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing temperature data for all patients.
		"""
		st.header("Temperature Gradient Analysis")

		# Create a temperature gradient dataframe
		temp_df = self.df.copy()

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


	@staticmethod
	def _render_patient(df_patient: pd.DataFrame, visits: list, patient_id: int) -> None:
		"""
		Renders temperature analysis for an individual patient.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing temperature data.
		patient_id : int
			The ID of the patient to analyze.
		data_processor : object
			An object that processes data to retrieve patient visits.
		"""

		df_temp = df_patient.copy()
		df_temp['Visit date'] = pd.to_datetime(df_temp['Visit date']).dt.strftime('%m-%d-%Y')
		st.header(f"Temperature Gradient Analysis for Patient {str(patient_id)}")

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
				for visit in visits
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
			visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits])

			for tab, visit in zip(visit_tabs, visits):
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

