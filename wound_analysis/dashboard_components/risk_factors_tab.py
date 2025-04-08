import streamlit as st
import pandas as pd
import plotly.express as px
from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor

class RiskFactorsTab:
	"""
	A class for managing and rendering the Risk Factors tab in the wound analysis dashboard.

	This class contains methods to display risk factor analysis for both population-level
	data and individual patient data, focusing on factors that influence wound healing such as
	diabetes, smoking status, and BMI.
	"""
	def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.wound_data_processor = wound_data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)

	def render(self) -> None:
		st.header("Risk Factors Analysis")

		if self.patient_id == "All Patients":
			RiskFactorsTab._render_population(df=self.df)
		else:
			df_patient = self.wound_data_processor.get_patient_dataframe(record_id=self.patient_id)
			RiskFactorsTab._render_patient(df_patient=df_patient)



	@staticmethod
	def _render_population(df: pd.DataFrame) -> None:
		"""
		Renders risk factor analysis for the entire patient population.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing risk factor data for all patients.
		"""
		CN = DColumns(df=df)
		risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

		valid_df = df.dropna(subset=[CN.HEALING_RATE, CN.VISIT_NUMBER]).copy()

		# Clip healing rates to reasonable range
		for col in [CN.DIABETES, CN.SMOKING_STATUS, CN.BMI]:
			# Add consistent status for each patient
			first_status = valid_df.groupby(CN.RECORD_ID)[col].first()
			valid_df[col] = valid_df[CN.RECORD_ID].map(first_status)

		# Create a color column for healing rate direction
		valid_df['Healing_Color'] = valid_df[CN.HEALING_RATE].apply(
			lambda x: 'green' if x < 0 else 'red'
		)

		with risk_tab1:
			st.subheader("Impact of Diabetes on Wound Healing")

			# Ensure diabetes status is consistent for each patient
			valid_df[CN.DIABETES] = valid_df[CN.DIABETES].fillna('No')

			# Compare average healing rates by diabetes status
			diab_stats = valid_df.groupby(CN.DIABETES).agg({ CN.HEALING_RATE: ['mean', 'count', 'std'] }).round(2)

			# Create a box plot for healing rates with color coding
			fig1 = px.box(
				valid_df,
				x=CN.DIABETES,
				y=CN.HEALING_RATE,
				title="Healing Rate Distribution by Diabetes Status",
				color='Healing_Color',
				color_discrete_map={'green': 'green', 'red': 'red'},
				points='all'
			)
			fig1.update_layout(
				xaxis_title="Diabetes Status",
				yaxis_title=CN.HEALING_RATE,
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
				improvement_rate = (valid_df[valid_df[CN.DIABETES] == status][CN.HEALING_RATE] < 0).mean() * 100
				st.write(f"- {status}: Average Healing Rate = {stats_data[(CN.HEALING_RATE, 'mean')]}% "
						f"(n={int(stats_data[(CN.HEALING_RATE, 'count')])}, "
						f"SD={stats_data[(CN.HEALING_RATE, 'std')]}, "
						f"Improvement Rate={improvement_rate:.1f}%)")

			# Compare wound types distribution
			wound_diab = pd.crosstab(index=valid_df[CN.DIABETES].astype(str), columns=valid_df[CN.WOUND_TYPE].astype(str), normalize='index') * 100

			fig2 = px.bar(
				wound_diab.reset_index().melt(id_vars=CN.DIABETES, var_name=CN.WOUND_TYPE, value_name='Percentage'),
				x=CN.DIABETES,
				y='Percentage',
				color=CN.WOUND_TYPE,
				title="Wound Type Distribution by Diabetes Status",
				labels={'Percentage': 'Percentage of Wounds (%)'}
			)
			st.plotly_chart(fig2, use_container_width=True)

		with risk_tab2:
			st.subheader("Impact of Smoking on Wound Healing")

			# Clean smoking status
			valid_df[CN.SMOKING_STATUS] = valid_df[CN.SMOKING_STATUS].fillna('Never')

			# Create healing rate distribution by smoking status with color coding
			fig1 = px.box(
				valid_df,
				x=CN.SMOKING_STATUS,
				y=CN.HEALING_RATE,
				title="Healing Rate Distribution by Smoking Status",
				color='Healing_Color',
				color_discrete_map={'green': 'green', 'red': 'red'},
				points='all'
			)
			fig1.update_layout(
				xaxis_title=CN.SMOKING_STATUS,
				yaxis_title=CN.HEALING_RATE,
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
			smoke_stats = valid_df.groupby(CN.SMOKING_STATUS).agg({ CN.HEALING_RATE: ['mean', 'count', 'std'] }).round(2)

			st.write("**Statistical Summary:**")
			for status in smoke_stats.index:
				stats_data = smoke_stats.loc[status]
				improvement_rate = (valid_df[valid_df[CN.SMOKING_STATUS] == status][CN.HEALING_RATE] < 0).mean() * 100
				st.write(f"- {status}: Average Healing Rate = {stats_data[(CN.HEALING_RATE, 'mean')]}% "
						f"(n={int(stats_data[(CN.HEALING_RATE, 'count')])}, "
						f"SD={stats_data[(CN.HEALING_RATE, 'std')]}, "
						f"Improvement Rate={improvement_rate:.1f}%)")

			# Wound type distribution by smoking status
			wound_smoke = pd.crosstab(valid_df[CN.SMOKING_STATUS].astype(str), valid_df[CN.WOUND_TYPE].astype(str), normalize='index') * 100
			fig2 = px.bar(
				wound_smoke.reset_index().melt(id_vars=CN.SMOKING_STATUS, var_name=CN.WOUND_TYPE, value_name='Percentage'),
				x=CN.SMOKING_STATUS,
				y='Percentage',
				color=CN.WOUND_TYPE,
				title="Wound Type Distribution by Smoking Status",
				labels={'Percentage': 'Percentage of Wounds (%)'}
			)
			st.plotly_chart(fig2, use_container_width=True)

		with risk_tab3:
			st.subheader("Impact of BMI on Wound Healing")

			# Create BMI categories
			bins = [0, 18.5, 24.9, 29.9, float('inf')]
			labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
			valid_df[CN.BMI_CATEGORY] = pd.cut(valid_df[CN.BMI], bins=bins, labels=labels)

			# Create healing rate distribution by BMI category with color coding
			fig1 = px.box(
				valid_df,
				x=CN.BMI_CATEGORY,
				y=CN.HEALING_RATE,
				title="Healing Rate Distribution by BMI Category",
				color='Healing_Color',
				color_discrete_map={'green': 'green', 'red': 'red'},
				points='all'
			)
			fig1.update_layout(
				xaxis_title=CN.BMI_CATEGORY,
				yaxis_title=CN.HEALING_RATE,
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
			bmi_stats = valid_df.groupby(CN.BMI_CATEGORY, observed=False).agg({
				CN.HEALING_RATE: ['mean', 'count', 'std']
			}).round(2)

			st.write("**Statistical Summary:**")
			for category in bmi_stats.index:
				stats_data = bmi_stats.loc[category]
				improvement_rate = (valid_df[valid_df[CN.BMI_CATEGORY] == category][CN.HEALING_RATE] < 0).mean() * 100
				st.write(f"- {category}: Average Healing Rate = {stats_data[(CN.HEALING_RATE, 'mean')]}% "
						f"(n={int(stats_data[(CN.HEALING_RATE, 'count')])}, "
						f"SD={stats_data[(CN.HEALING_RATE, 'std')]}, "
						f"Improvement Rate={improvement_rate:.1f}%)")

			# Wound type distribution by BMI category
			wound_bmi = pd.crosstab(valid_df[CN.BMI_CATEGORY], valid_df[CN.WOUND_TYPE], normalize='index') * 100
			fig2 = px.bar(
				wound_bmi.reset_index().melt(id_vars=CN.BMI_CATEGORY, var_name=CN.WOUND_TYPE, value_name='Percentage'),
				x=CN.BMI_CATEGORY,
				y='Percentage',
				color=CN.WOUND_TYPE,
				title="Wound Type Distribution by BMI Category",
				labels={'Percentage': 'Percentage of Wounds (%)'}
			)
			st.plotly_chart(fig2, use_container_width=True)

	@staticmethod
	def _render_patient(df_patient: pd.DataFrame) -> None:
		"""
		Renders risk factor analysis for an individual patient.

		Parameters
		----------
		df_patient : pd.DataFrame
			The dataframe containing risk factor data.
		"""
		# For individual patient
		CN = DColumns(df=df_patient)
		df_temp = df_patient.copy()
		patient_data = df_temp.iloc[0]

		# Create columns for the metrics
		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Patient Risk Profile")

			# Display key risk factors
			st.info(f"**Diabetes Status:** {patient_data[CN.DIABETES]}")
			st.info(f"**Smoking Status:** {patient_data[CN.SMOKING_STATUS]}")
			st.info(f"**BMI:** {patient_data[CN.BMI]:.1f}")

			# BMI category
			bmi = patient_data[CN.BMI]
			if pd.isna(bmi):
				bmi_category = "Unknown"
			elif bmi < 18.5:
				bmi_category = "Underweight"
			elif bmi < 25:
				bmi_category = "Normal"
			elif bmi < 30:
				bmi_category = "Overweight"
			else:
				bmi_category = "Obese"

			st.info(f"**BMI Category:** {bmi_category}")

		with col2:
			st.subheader("Risk Assessment")

			# Calculate risk score based on risk factors
			risk_score = 0
			risk_factors = []

			# Diabetes risk
			if patient_data[CN.DIABETES] == 'Yes':
				risk_factors.append("Diabetes")
				risk_score += 2

			# Smoking risk
			if patient_data[CN.SMOKING_STATUS] == 'Current':
				risk_factors.append("Current smoker")
				risk_score += 3
			elif patient_data[CN.SMOKING_STATUS] == 'Former':
				risk_factors.append("Former smoker")
				risk_score += 1

			# BMI risk
			if bmi_category == "Obese":
				risk_factors.append("Obesity")
				risk_score += 2
			elif bmi_category == "Overweight":
				risk_factors.append("Overweight")
				risk_score += 1

			# Temperature gradient risk
			try:
				temp_gradient = patient_data[CN.CENTER_TEMP] - patient_data[CN.PERI_TEMP]
				if temp_gradient > 3:
					risk_factors.append("High temperature gradient")
					risk_score += 2
			except Exception as e:
				st.error(f"Error calculating temperature gradient risk: {e}")
				pass

			# Impedance risk
			try:
				if patient_data[CN.HIGHEST_FREQ_ABSOLUTE] > 140:
					risk_factors.append("High impedance")
					risk_score += 2
			except Exception as e:
				st.error(f"Error calculating impedance risk: {e}")
				pass

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
			try:
				wound_area         = patient_data[CN.WOUND_AREA]
				base_healing_weeks = 2 + wound_area/2  # Simple formula: 2 weeks + 0.5 weeks per cm²
				risk_multiplier    = 1 + (risk_score * 0.1)  # Each risk point adds 10% to healing time
				est_healing_weeks  = base_healing_weeks * risk_multiplier

				st.markdown(f"**Estimated Healing Time:** {est_healing_weeks:.1f} weeks")
			except Exception as e:
				st.error(f"Error calculating estimated healing time: {e}")
				st.markdown("**Estimated Healing Time:** Unable to calculate (missing wound area data)")
