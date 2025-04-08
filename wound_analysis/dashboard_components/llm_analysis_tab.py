"""
Module for LLM Analysis tab component in the wound analysis dashboard.

This module provides the LLMAnalysisTab class which handles the rendering and functionality
of the LLM Analysis tab, allowing users to analyze wound data using LLM services.
"""

import streamlit as st

from wound_analysis.utils.data_processor import WoundDataProcessor, DataManager
from wound_analysis.utils.llm_interface import WoundAnalysisLLM


class LLMAnalysisTab:
	"""
	Creates and manages the LLM analysis tab in the Streamlit application.

	This method sets up the interface for running LLM-powered wound analysis
	on either all patients collectively or on a single selected patient.
	It handles the initialization of the LLM, retrieval of patient data,
	generation of analysis, and presentation of results.

	Parameters
	----------
	data_processor : WoundDataProcessor
		The data processor object to use.
	llm_platform : str
		The LLM platform to use.
	llm_model : str
		The LLM model to use.
	selected_patient : str
		The currently selected patient identifier (e.g., "Patient 1") or "All Patients"

	Returns:
	-------
	None
		The method updates the Streamlit UI directly

	Notes:
	-----
	- Analysis results are stored in session state to persist between reruns
	- The method supports two analysis modes:
		1. Population analysis (when "All Patients" is selected)
		2. Individual patient analysis (when a specific patient is selected)
	- Analysis results and prompts are displayed in separate tabs
	- A download button is provided for exporting reports as Word documents
	"""

	def __init__(self, selected_patient: str, wound_data_processor: WoundDataProcessor=None, llm_platform: str='ai-verde', llm_model: str='llama-3.3-70b-fp8'):
		self.wound_data_processor = wound_data_processor
		self.llm_platform         = llm_platform
		self.llm_model            = llm_model
		self.patient_id           = "All Patients" if selected_patient == "All Patients" else int(selected_patient.split()[1])

	def render(self) -> None:

		def _display_llm_analysis(llm_reports: dict):
			if 'thinking_process' in llm_reports and llm_reports['thinking_process'] is not None:
				tab1, tab2, tab3 = st.tabs(["Analysis", "Prompt", "Thinking Process"])
			else:
				tab1, tab2 = st.tabs(["Analysis", "Prompt"])
				tab3 = None

			with tab1:
				st.markdown(llm_reports['analysis_results'])
			with tab2:
				st.markdown(llm_reports['prompt'])

			if tab3 is not None:
				st.markdown("### Model's Thinking Process")
				st.markdown(llm_reports['thinking_process'])

			# Add download button for the report
			DataManager.download_word_report(st=st, report_path=llm_reports['report_path'])


		def _run_llm_analysis(prompt: str, patient_metadata: dict=None, analysis_results: dict=None) -> dict:

			if self.llm_model == "deepseek-r1":

				# Create a container for the analysis
				analysis_container = st.empty()
				analysis_container.markdown("### Analysis\n\n*Generating analysis...*")

				# Create a container for the thinking process

				thinking_container.markdown("### Thinking Process\n\n*Thinking...*")

				# Update the analysis container with final results
				analysis_container.markdown(f"### Analysis\n\n{analysis}")

				thinking_process = llm.get_thinking_process()

			else:
				thinking_process = None


			# Store analysis in session state for this patient
			report_path = DataManager.create_and_save_report(patient_metadata=patient_metadata, analysis_results=analysis_results, report_path=None)

			# Return data dictionary
			return dict(
				analysis_results = analysis,
				patient_metadata = patient_metadata,
				prompt           = prompt,
				report_path      = report_path,
				thinking_process = thinking_process
			)

		def stream_callback(data):
			if data["type"] == "thinking":
				thinking_container.markdown(f"### Thinking Process\n\n{data['content']}")

		st.header("LLM-Powered Wound Analysis")

		thinking_container = st.empty()

		# Initialize reports dictionary in session state if it doesn't exist
		if 'llm_reports' not in st.session_state:
			st.session_state.llm_reports = {}

		if self.wound_data_processor is not None:

			llm = WoundAnalysisLLM(platform=self.llm_platform, model_name=self.llm_model)

			callback = stream_callback if self.llm_model == "deepseek-r1" else None

			if self.patient_id == "All Patients":

				# Check if the model is llama-3.3-70b-fp8 and show warning
				# if self.llm_model == "llama-3.3-70b-fp8":
				# 	st.warning("The llama-3.3-70b-fp8 model currently doesn't work with All Patients view due to issues with the AI-Verde server. Please select a different model.")
				# else:

				if st.button("Run Analysis", key="run_analysis"):
					population_data = self.wound_data_processor.get_population_statistics()
					prompt = llm._format_population_prompt(population_data=population_data)
					analysis = llm.analyze_population_data(population_data=population_data, callback=callback)
					st.session_state.llm_reports['all_patients'] = _run_llm_analysis(prompt=prompt, analysis_results=analysis, patient_metadata=population_data)

				# Display analysis if it exists for this patient
				if 'all_patients' in st.session_state.llm_reports:
					_display_llm_analysis(st.session_state.llm_reports['all_patients'])

			else:
				st.subheader(f"Patient {self.patient_id}")

				if st.button("Run Analysis", key="run_analysis"):
					patient_data = self.wound_data_processor.get_patient_visits(int(self.patient_id))
					prompt = llm._format_per_patient_prompt(patient_data=patient_data)
					analysis = llm.analyze_patient_data(patient_data=patient_data, callback=callback)
					st.session_state.llm_reports[self.patient_id] = _run_llm_analysis(prompt=prompt, analysis_results=analysis, patient_metadata=patient_data['patient_metadata'])

				# Display analysis if it exists for this patient
				if self.patient_id in st.session_state.llm_reports:
					_display_llm_analysis(st.session_state.llm_reports[self.patient_id])
		else:
			st.warning("Please upload a patient data file from the sidebar to enable LLM analysis.")
