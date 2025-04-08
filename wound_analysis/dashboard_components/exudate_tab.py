from termios import VINTR
import streamlit as st
import pandas as pd
import plotly.express as px

from wound_analysis.utils.column_schema import DColumns
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
	selected_patient : str
		The currently selected patient ID or "All Patients"
	wound_data_processor : WoundDataProcessor
		The data processor instance containing the filtered DataFrame and processing methods.

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

	def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor):
		self.wound_data_processor = wound_data_processor
		self.patient_id = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])
		self.df = wound_data_processor.df
		self.CN = DColumns(df=self.df)

	def render(self) -> None:
		st.header("Exudate Analysis")

		if self.patient_id == "All Patients":
			ExudateTab._render_population(df=self.df)
		else:
			visits = self.wound_data_processor.get_patient_visits(record_id=self.patient_id)['visits']
			ExudateTab._render_patient(visits=visits, VISIT_DATE_TAG=self.CN.VISIT_DATE)

	@staticmethod
	def _render_population(df: pd.DataFrame) -> None:
		"""
		Renders the population-level exudate analysis section.

		This method creates visualizations and statistical analyses for exudate characteristics
		across the entire patient population, including volume, viscosity, and type distributions.

		Parameters
		----------
		df : pd.DataFrame
			The dataframe containing wound data for all patients
		"""
		CN = DColumns(df=df)

		# Create a copy of the dataframe for analysis
		exudate_df = df.copy()

		# Convert exudate volume and viscosity to numeric values
		volume_map    = {'Low': 1, 'Medium': 2, 'High': 3}
		viscosity_map = {'Low': 1, 'Medium': 2, 'High': 3}

		exudate_df['Volume_Numeric']    = exudate_df[CN.EXUDATE_VOLUME].map(volume_map)
		exudate_df['Viscosity_Numeric'] = exudate_df[CN.EXUDATE_VISCOSITY].map(viscosity_map)

		if not exudate_df.empty:
			# Create two columns for volume and viscosity analysis
			col1, col2 = st.columns(2)

			with col1:
				st.subheader("Volume Analysis")
				# Calculate correlation between volume and healing rate
				valid_df = exudate_df.dropna(subset=['Volume_Numeric', CN.HEALING_RATE])

				if len(valid_df) > 1:
					stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Volume_Numeric', y_col=CN.HEALING_RATE, REMOVE_OUTLIERS=False)
					_, _, _ = stats_analyzer.calculate_correlation()
					st.info(stats_analyzer.get_correlation_text(text="Volume correlation vs Healing Rate"))

				# Boxplot of exudate volume by wound type
				fig_vol = px.box(
					exudate_df,
					x      = CN.WOUND_TYPE,
					y      = CN.EXUDATE_VOLUME,
					title  = "Exudate Volume by Wound Type",
					points = "all"
				)

				fig_vol.update_layout(
					xaxis_title = CN.WOUND_TYPE,
					yaxis_title = CN.EXUDATE_VOLUME,
					boxmode     = 'group'
				)
				st.plotly_chart(fig_vol, use_container_width=True)

			with col2:
				st.subheader("Viscosity Analysis")
				# Calculate correlation between viscosity and healing rate
				valid_df = exudate_df.dropna(subset=['Viscosity_Numeric', CN.HEALING_RATE])

				if len(valid_df) > 1:
					stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Viscosity_Numeric', y_col=CN.HEALING_RATE, REMOVE_OUTLIERS=False)
					_, _, _ = stats_analyzer.calculate_correlation()
					st.info(stats_analyzer.get_correlation_text(text="Viscosity correlation vs Healing Rate"))

				# Boxplot of exudate viscosity by wound type
				fig_visc = px.box(
					exudate_df,
					x=CN.WOUND_TYPE,
					y=CN.EXUDATE_VISCOSITY,
					title="Exudate Viscosity by Wound Type",
					points="all"
				)
				fig_visc.update_layout(
					xaxis_title=CN.WOUND_TYPE,
					yaxis_title=CN.EXUDATE_VISCOSITY,
					boxmode='group'
				)
				st.plotly_chart(fig_visc, use_container_width=True)



			# Create scatter plot matrix for volume, viscosity, and healing rate
			st.subheader("Relationship Analysis")

			exudate_df[CN.HEALING_RATE] = exudate_df[CN.HEALING_RATE].clip(-100, 100)
			exudate_df[CN.WOUND_AREA]   = exudate_df[CN.WOUND_AREA].fillna(exudate_df[CN.WOUND_AREA].mean())

			fig_scatter = px.scatter(
				exudate_df,
				x='Volume_Numeric',
				y=CN.HEALING_RATE,
				color=CN.WOUND_TYPE,
				size=CN.WOUND_AREA,
				hover_data=[CN.RECORD_ID, CN.EVENT_NAME, CN.EXUDATE_VOLUME, CN.EXUDATE_VISCOSITY, CN.EXUDATE_TYPE],
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
				type_by_wound = pd.crosstab(exudate_df[CN.WOUND_TYPE], exudate_df[CN.EXUDATE_TYPE])
				fig_type = px.bar(
					type_by_wound.reset_index().melt(id_vars=CN.WOUND_TYPE, var_name=CN.EXUDATE_TYPE, value_name='Percentage'),
					x=CN.WOUND_TYPE,
					y='Percentage',
					color=CN.EXUDATE_TYPE,
				)

				st.plotly_chart(fig_type, use_container_width=True)

			with col2:
				# Overall distribution
				type_counts = exudate_df[CN.EXUDATE_TYPE].value_counts()
				fig_pie = px.pie(
					values=type_counts.values,
					names=type_counts.index,
					title="Overall Distribution of Exudate Types"
				)
				st.plotly_chart(fig_pie, use_container_width=True)

		else:
			st.warning("No valid exudate data available for analysis.")

	@staticmethod
	def _render_patient(visits: list, VISIT_DATE_TAG: str) -> None:
		"""
		Renders the exudate analysis section for a specific patient.

		This method displays a timeline of exudate changes and provides clinical interpretations
		for each visit.

		Parameters
		----------
		visits : list
			List of visit data dictionaries containing exudate information
		"""
		# Display the exudate chart
		fig = Visualizer.create_exudate_chart(visits=visits, VISIT_DATE_TAG=VISIT_DATE_TAG)
		st.plotly_chart(fig, use_container_width=True)

		# Clinical interpretation section
		st.subheader("Clinical Interpretation of Exudate Characteristics")

		# Create tabs for each visit
		visit_tabs = st.tabs([visit.get(VISIT_DATE_TAG, 'N/A') for visit in visits])

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
					ExudateTab.get_exudate_severity(exudate_type_str)

				with col2:
					st.markdown("### Treatment Implications")
					if exudate_analysis['treatment_implications']:
						st.write("**Recommended Actions:**")
						st.success("\n".join(exudate_analysis['treatment_implications']))

	@staticmethod
	def get_exudate_severity(exudate_type_str: str) -> str:

		st.markdown("#### Current Types")
		exudate_types = exudate_type_str.split(',') if exudate_type_str != 'N/A' else []
		exudate_types = [t.strip() for t in exudate_types if t.strip()]

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
