import numpy as np
import streamlit as st
import pandas as pd

from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.dashboard_components.visualizer import Visualizer


class OverviewTab:
	"""
	Renders the Overview tab in the Streamlit application.

	This method displays different content based on whether all patients or a specific patient is selected.
	For all patients, it renders a summary overview of all patients' data.
	For a specific patient, it renders that patient's individual overview and a wound area over time plot.

	Parameters:
	----------
	selected_patient : str
		The currently selected patient from the sidebar dropdown. Could be "All Patients"
		or a specific patient name in the format "Patient X" where X is the patient ID.
	wound_data_processor : WoundDataProcessor
		The data processor instance containing the filtered DataFrame and processing methods.

	Returns:
	-------
	None
		This method directly renders content to the Streamlit UI and doesn't return any value.
	"""

	def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.wound_data_processor = wound_data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)

	def render(self) -> None:

		st.header("Overview")

		if self.patient_id == "All Patients":
			self._render_all_patients_overview(df=self.df)
		else:
			df_patient = self.wound_data_processor.get_patient_dataframe(record_id=self.patient_id)
			visits 	   = self.wound_data_processor.get_patient_visits(record_id=self.patient_id)['visits']
			self._render_patient_overview(df_patient=df_patient, visits=visits)

			st.subheader("Wound Area Over Time")
			fig = Visualizer.create_wound_area_plot(df = self.df, patient_id=self.patient_id)
			st.plotly_chart(fig, use_container_width=True)


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
			- CN.RECORD_ID: Patient identifier
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
			Visualizer.create_wound_area_plot(df=df, patient_id=None),
			use_container_width=True
		)

		col1, col2, col3, col4 = st.columns(4)

		with col1:
			# Calculate average days in study (actual duration so far)
			avg_days_in_study = df.groupby(self.CN.RECORD_ID)[self.CN.DAYS_SINCE_FIRST_VISIT].max().mean()
			st.metric("Average Days in Study", f"{avg_days_in_study:.1f} days")

		with col2:
			try:
				# Calculate average estimated treatment duration for improving wounds
				estimated_days = df.groupby(self.CN.RECORD_ID)[self.CN.ESTIMATED_DAYS_TO_HEAL].mean()
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
			healing_rates = df[self.CN.HEALING_RATE]
			valid_rates = healing_rates[(healing_rates != 0) & (np.isfinite(healing_rates))]
			avg_healing_rate = np.mean(valid_rates) if len(valid_rates) > 0 else 0
			st.metric("Average Healing Rate", f"{abs(avg_healing_rate):.2f} cm²/day")

		with col4:
			try:
				# Calculate improvement rate using only the last visit for each patient
				if self.CN.OVERALL_IMPROVEMENT not in df.columns:
					df[self.CN.OVERALL_IMPROVEMENT] = np.nan

				# Get the last visit for each patient and calculate improvement rate
				last_visits = df.groupby(self.CN.RECORD_ID).agg({
					self.CN.OVERALL_IMPROVEMENT: 'last',
					self.CN.HEALING_RATE: 'last'
				})

				# Calculate improvement rate from patients with valid improvement status
				valid_improvements = last_visits[self.CN.OVERALL_IMPROVEMENT].dropna()
				if len(valid_improvements) > 0:
					improvement_rate = (valid_improvements == 'Yes').mean() * 100
					st.metric("Improvement Rate", f"{improvement_rate:.1f}%")
				else:
					st.metric("Improvement Rate", "N/A")

			except Exception as e:
				st.metric("Improvement Rate", "N/A")
				print(f"Error calculating improvement rate: {e}")

	def _render_patient_overview(self, df_patient: pd.DataFrame, visits: list) -> None:
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

		Notes
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

		if df_patient.empty:
			st.error("No data available for this patient.")
			return

		patient_data = df_patient.iloc[0]

		metadata = self.wound_data_processor._extract_patient_metadata(patient_data)

		col1, col2, col3 = st.columns(3)

		with col1:
			st.subheader("Patient Demographics")
			st.write(f"Age: {metadata.get(self.CN.AGE)} years")
			st.write(f"Sex: {metadata.get(self.CN.SEX)}")
			st.write(f"BMI: {metadata.get(self.CN.BMI)}")
			st.write(f"Race: {metadata.get(self.CN.RACE)}")
			st.write(f"Ethnicity: {metadata.get(self.CN.ETHNICITY)}")

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

			smoking     = get_metric_value(patient_data.get(self.CN.SMOKING_STATUS))
			med_history = get_metric_value(patient_data.get(self.CN.MEDICAL_HISTORY))

			st.write("Smoking Status:", "Yes" if smoking == "Current" else "No")
			st.write("Hypertension:", "Yes" if "Cardiovascular" in str(med_history) else "No")
			st.write("Peripheral Vascular Disease:", "Yes" if "PVD" in str(med_history) else "No")

		with col3:
			st.subheader("Diabetes Status")
			st.write(f"Status: {get_metric_value(patient_data.get(self.CN.DIABETES))}")
			st.write(f"HbA1c: {get_metric_value(patient_data.get(self.CN.A1C))}%")
			st.write(f"A1c available: {get_metric_value(patient_data.get(self.CN.A1C_AVAILABLE))}")


		st.title("Wound Details (present at 1st visit)")
		wound_info = visits[0]['wound_info']

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
						"WiFi Classification: {get_metric_value(infection.get('wifi_classification'))}"
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

