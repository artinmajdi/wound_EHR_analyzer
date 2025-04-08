import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.statistical_analysis import CorrelationAnalysis
from wound_analysis.utils.column_schema import DColumns

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
		selected_patient : str
			The selected patient (either "All Patients" or a specific patient identifier)
		wound_data_processor : WoundDataProcessor
			The data processor instance containing the filtered DataFrame and processing methods.

		Returns:
		--------
		None
			This method renders Streamlit components directly to the app
	"""

	def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])

		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)

		if self.patient_id != "All Patients":
			self.visits        = wound_data_processor.get_patient_visits(record_id=self.patient_id)['visits']
			self.patient_data  = wound_data_processor.get_patient_dataframe(record_id=self.patient_id)

	def render(self) -> None:
		st.header("Oxygenation Analysis")

		if self.patient_id == "All Patients":
			self._render_population()
		else:
			self._render_patient()

	def _render_population(self) -> None:
		"""
		Renders oxygenation analysis for the entire patient population.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing oxygenation data for all patients.
		"""
		valid_df = self.df.copy()

		valid_df[self.CN.HEMOGLOBIN]   = pd.to_numeric(valid_df[self.CN.HEMOGLOBIN], errors='coerce')
		valid_df[self.CN.OXYGENATION]  = pd.to_numeric(valid_df[self.CN.OXYGENATION], errors='coerce')
		valid_df[self.CN.HEALING_RATE] = pd.to_numeric(valid_df[self.CN.HEALING_RATE], errors='coerce')

		valid_df = valid_df.dropna(subset=[self.CN.OXYGENATION, self.CN.HEALING_RATE, self.CN.HEMOGLOBIN])

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
		stats_analyzer = CorrelationAnalysis(data=valid_df, x_col=self.CN.OXYGENATION, y_col=self.CN.HEALING_RATE, outlier_threshold=outlier_threshold)
		valid_df, r, p = stats_analyzer.calculate_correlation()

		# Display correlation statistics
		with col3:
			st.info(stats_analyzer.get_correlation_text())

		# Add consistent diabetes status for each patient
		# first_diabetes_status = valid_df.groupby(CN.RECORD_ID)['Diabetes?'].first()
		# valid_df['Diabetes?'] = valid_df[CN.RECORD_ID].map(first_diabetes_status)
		valid_df[self.CN.HEALING_RATE] = valid_df[self.CN.HEALING_RATE].clip(-100, 100)
		valid_df[self.CN.WOUND_AREA]   = valid_df[self.CN.WOUND_AREA].fillna(valid_df[self.CN.WOUND_AREA].mean())

		if not valid_df.empty:
			fig1 = px.scatter(
				valid_df,
				x=self.CN.OXYGENATION,
				y=self.CN.HEALING_RATE,
				# color=CN.DIABETES,
				size=self.CN.WOUND_AREA, # CN.HEMOGLOBIN,
				size_max=30,
				hover_data=[self.CN.RECORD_ID, self.CN.EVENT_NAME, self.CN.WOUND_TYPE],
				title="Relationship Between Oxygenation and Healing Rate (size=Hemoglobin Level)"
			)
			fig1.update_layout(xaxis_title="Oxygenation (%)", yaxis_title="Healing Rate (% reduction per visit)")
			st.plotly_chart(fig1, use_container_width=True)
		else:
			st.warning("Insufficient data for oxygenation analysis.")

		# Create boxplot for oxygenation levels
		valid_box_df = self.df.dropna(subset=[self.CN.OXYGENATION, self.CN.WOUND_TYPE])
		if not valid_box_df.empty:
			fig2 = px.box(
				valid_box_df,
				x=self.CN.WOUND_TYPE,
				y=self.CN.OXYGENATION,
				title="Oxygenation Levels by Wound Type",
				color=self.CN.WOUND_TYPE,
				points="all"
			)
			fig2.update_layout(xaxis_title=self.CN.WOUND_TYPE, yaxis_title=self.CN.OXYGENATION)
			st.plotly_chart(fig2, use_container_width=True)
		else:
			st.warning("Insufficient data for wound type comparison.")


	def _render_patient(self) -> None:
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
		self.patient_data[self.CN.VISIT_DATE] = pd.to_datetime(self.patient_data[self.CN.VISIT_DATE]).dt.strftime('%m-%d-%Y')

		fig_bar, fig_line = self.create_oxygenation_chart()

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


	def create_oxygenation_chart(self) -> go.Figure:
		"""
		Creates two charts for visualizing oxygenation and hemoglobin data.

		Parameters:
		-----------
		patient_data : pandas.DataFrame
			DataFrame containing patient visit data with columns for visit date,
			oxyhemoglobin level, and deoxyhemoglobin level.
		visits : list
			List of dictionaries, each containing visit data with keys self.CN.VISIT_DATE
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
			x=self.patient_data[self.CN.VISIT_DATE],
			y=self.patient_data[self.CN.OXYHEMOGLOBIN],
			name="Oxyhemoglobin",
			marker_color='red'
		))
		fig_bar.add_trace(go.Bar(
			x=self.patient_data[self.CN.VISIT_DATE],
			y=self.patient_data[self.CN.DEOXYHEMOGLOBIN],
			name="Deoxyhemoglobin",
			marker_color='purple'
		))
		fig_bar.update_layout(
			title="Hemoglobin Components",
			xaxis_title="Visit Date",
			yaxis_title="Level (g/dL)",
			barmode='stack',
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
		)

		# Create an interactive chart showing oxygenation and hemoglobin measurements over time.
		dates = []
		oxygenation = []
		hemoglobin = []

		for visit in self.visits:
			date = visit[self.CN.VISIT_DATE]
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
			xaxis_title=self.CN.VISIT_DATE,
			yaxis_title='Value',
			hovermode='x unified'
		)
		return fig_bar, fig_line
