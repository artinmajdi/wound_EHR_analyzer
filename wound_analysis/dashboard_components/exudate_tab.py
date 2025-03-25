import streamlit as st
import pandas as pd
import plotly.express as px

from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.statistical_analysis import CorrelationAnalysis
from wound_analysis.dashboard_components.visualizer import Visualizer
from wound_analysis.dashboard_components.settings import DashboardSettings


class ExudateTab:
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
	data_processor : object, optional
		An object that processes data to retrieve patient visits.

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

	def __init__(self, df: pd.DataFrame, selected_patient: str, data_processor: WoundDataProcessor):
		self.data_processor = data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = df

	def render(self) -> None:

		st.header("Exudate Analysis")

		if self.patient_id == "All Patients":
			ExudateTab._render_population(df=self.df)
		else:
			visits = self.data_processor.get_patient_visits(record_id=self.patient_id)['visits']
			ExudateTab._render_patient(visits=visits)



	@staticmethod
	def _render_population(df: pd.DataFrame) -> None:
		"""
		Renders exudate analysis for the entire patient population.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing exudate data for all patients.
		"""
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

	@staticmethod
	def _render_patient(visits: list) -> None:
		"""
		Renders exudate analysis for an individual patient.

		Parameters
		----------
		visits : list
			The list of visits for the patient.
		"""


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
				volume = visit['wound_info']['exudate'].get('volume', 'N/A')
				viscosity = visit['wound_info']['exudate'].get('viscosity', 'N/A')
				exudate_type_str = str(visit['wound_info']['exudate'].get('type', 'N/A'))
				exudate_analysis = DashboardSettings.get_exudate_analysis(volume=volume, viscosity=viscosity, exudate_types=exudate_type_str)

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
						if exudate_type in DashboardSettings.EXUDATE_TYPE_INFO:
							info = DashboardSettings.EXUDATE_TYPE_INFO[exudate_type]
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
