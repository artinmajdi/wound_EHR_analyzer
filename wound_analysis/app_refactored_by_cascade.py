"""
Smart Bandage Wound Analysis Dashboard
A Streamlit application for analyzing and visualizing wound healing data.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pathlib
import os
from typing import Dict, List, Optional, Tuple, Union
import base64
from io import BytesIO
from docx import Document

from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document

class ChartManager:
    """Manages creation of all visualization charts."""

    @staticmethod
    def create_wound_measurements_chart(visits: List[Dict]) -> go.Figure:
        """Create an interactive chart showing wound measurements over time."""
        dates, areas, lengths, widths, depths = [], [], [], [], []

        for visit in visits:
            date = visit['visit_date']
            measurements = visit['wound_measurements']
            dates.append(date)
            areas.append(measurements.get('area'))
            lengths.append(measurements.get('length'))
            widths.append(measurements.get('width'))
            depths.append(measurements.get('depth'))

        fig = go.Figure()
        metrics = {
            'Area (cm²)': areas,
            'Length (cm)': lengths,
            'Width (cm)': widths,
            'Depth (cm)': depths
        }

        for name, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                name=name,
                mode='lines+markers'
            ))

        fig.update_layout(
            title='Wound Measurements Over Time',
            xaxis_title='Visit Date',
            yaxis_title='Measurement',
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def create_temperature_chart(visits: List[Dict]) -> go.Figure:
        """Create an interactive chart showing temperature measurements over time."""
        dates = []
        temps = {'Center': [], 'Edge': [], 'Peri-wound': []}

        for visit in visits:
            date = visit['visit_date']
            temp_data = visit['sensor_data']['temperature']
            dates.append(date)
            temps['Center'].append(temp_data.get('center'))
            temps['Edge'].append(temp_data.get('edge'))
            temps['Peri-wound'].append(temp_data.get('peri'))

        fig = go.Figure()
        for location, values in temps.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                name=location,
                mode='lines+markers'
            ))

        fig.update_layout(
            title='Temperature Measurements Over Time',
            xaxis_title='Visit Date',
            yaxis_title='Temperature (°F)',
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def create_impedance_chart(visits: List[Dict]) -> go.Figure:
        """Create an interactive chart showing impedance measurements over time."""
        dates = []
        freq_data = {
            'high': {'Z': [], 'resistance': [], 'capacitance': [], 'freq': None},
            'center': {'Z': [], 'resistance': [], 'capacitance': [], 'freq': None},
            'low': {'Z': [], 'resistance': [], 'capacitance': [], 'freq': None}
        }

        for visit in visits:
            date = visit['visit_date']
            impedance_data = visit.get('sensor_data', {}).get('impedance', {})
            dates.append(date)

            for freq_type in freq_data:
                freq_key = f'{freq_type}_frequency'
                freq_measurements = impedance_data.get(freq_key, {})

                if freq_measurements.get('frequency'):
                    freq_data[freq_type]['freq'] = freq_measurements['frequency']

                for measure in ['Z', 'resistance', 'capacitance']:
                    try:
                        val = float(freq_measurements.get(measure)) if freq_measurements.get(measure) not in (None, '') else None
                        freq_data[freq_type][measure].append(val)
                    except (ValueError, TypeError):
                        freq_data[freq_type][measure].append(None)

        fig = go.Figure()
        line_styles = {'high': 'solid', 'center': 'dot', 'low': 'dash'}

        for freq_type, measurements in freq_data.items():
            freq_label = f"{float(measurements['freq']):.0f}Hz" if measurements['freq'] else f"{freq_type.title()} Freq"

            for measure, values in measurements.items():
                if measure != 'freq' and any(v is not None for v in values):
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        name=f'{measure.title()} ({freq_label})',
                        mode='lines+markers',
                        line=dict(dash=line_styles[freq_type])
                    ))

        fig.update_layout(
            title='Impedance Measurements Over Time',
            xaxis_title='Visit Date',
            yaxis_title='Measurement',
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def create_oxygenation_chart(visits: List[Dict]) -> go.Figure:
        """Create an interactive chart showing oxygenation and hemoglobin measurements over time."""
        dates = []
        metrics = {
            'Oxygenation (%)': [],
            'Total Hemoglobin': [],
            'Oxygenated Hemoglobin': [],
            'Deoxygenated Hemoglobin': []
        }

        for visit in visits:
            date = visit['visit_date']
            oxy_data = visit['sensor_data'].get('oxygenation', {})
            dates.append(date)

            metrics['Oxygenation (%)'].append(oxy_data.get('oxygenation'))
            metrics['Total Hemoglobin'].append(oxy_data.get('total_hemoglobin'))
            metrics['Oxygenated Hemoglobin'].append(oxy_data.get('oxygenated_hemoglobin'))
            metrics['Deoxygenated Hemoglobin'].append(oxy_data.get('deoxygenated_hemoglobin'))

        fig = go.Figure()
        for metric, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                name=metric,
                mode='lines+markers'
            ))

        fig.update_layout(
            title='Oxygenation Measurements Over Time',
            xaxis_title='Visit Date',
            yaxis_title='Measurement',
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def create_exudate_chart(visits: List[Dict]) -> go.Figure:
        """Create a chart showing exudate characteristics over time."""
        dates = []
        characteristics = {
            'Amount': [],
            'Type': [],
            'Color': [],
            'Consistency': [],
            'Odor': []
        }

        for visit in visits:
            date = visit['visit_date']
            exudate_data = visit.get('exudate_characteristics', {})
            dates.append(date)

            for char in characteristics:
                characteristics[char].append(exudate_data.get(char.lower()))

        fig = go.Figure()
        for char, values in characteristics.items():
            if any(v is not None for v in values):
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    name=char,
                    mode='lines+markers'
                ))

        fig.update_layout(
            title='Exudate Characteristics Over Time',
            xaxis_title='Visit Date',
            yaxis_title='Characteristic',
            hovermode='x unified'
        )
        return fig

class DocumentHandler:
    """Handles document creation and download functionality."""

    @staticmethod
    def create_and_save_report(patient_data: Dict, analysis_results: str) -> str:
        """Create and save the analysis report as a Word document."""
        doc = format_word_document(patient_data, analysis_results)
        report_path = "wound_analysis_report.docx"
        doc.save(report_path)
        return report_path

    @staticmethod
    def get_binary_file_downloader_html(report_path: str) -> str:
        """Create a download link for the Word report."""
        with open(report_path, 'rb') as file:
            bytes_data = file.read()
        b64 = base64.b64encode(bytes_data).decode()
        return f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}'

class WoundAnalysisDashboard:
    """Main dashboard class managing the Streamlit interface."""

    def __init__(self):
        """Initialize the dashboard with default settings."""
        self.setup_page_config()
        self.initialize_session_state()
        self.chart_manager = ChartManager()
        self.doc_handler = DocumentHandler()

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Wound Analysis Dashboard",
            layout="wide",
            page_icon="🏥"
        )

    def initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'processor': None,
            'analysis_complete': False,
            'analysis_results': None,
            'active_tab': "Dashboard",
            'report_path': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def create_sidebar(self) -> None:
        """Create and manage the sidebar components."""
        st.sidebar.title("Data Input")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Patient Data (JSON)",
            type=['json']
        )

        if uploaded_file:
            st.session_state.processor = WoundDataProcessor(uploaded_file)
            st.session_state.analysis_complete = False
            st.session_state.analysis_results = None

    def create_main_content(self) -> None:
        """Create and manage the main content area."""
        if not st.session_state.processor:
            st.info("Please upload patient data to begin analysis.")
            return

        patient_data = st.session_state.processor.get_patient_data()
        visits = patient_data.get('visits', [])

        # Create tabs
        tabs = st.tabs([
            "Wound Measurements",
            "Temperature",
            "Impedance",
            "Oxygenation",
            "Exudate",
            "Analysis"
        ])

        # Populate tabs with charts
        with tabs[0]:
            st.plotly_chart(
                self.chart_manager.create_wound_measurements_chart(visits),
                use_container_width=True
            )

        with tabs[1]:
            st.plotly_chart(
                self.chart_manager.create_temperature_chart(visits),
                use_container_width=True
            )

        with tabs[2]:
            st.plotly_chart(
                self.chart_manager.create_impedance_chart(visits),
                use_container_width=True
            )

        with tabs[3]:
            st.plotly_chart(
                self.chart_manager.create_oxygenation_chart(visits),
                use_container_width=True
            )

        with tabs[4]:
            st.plotly_chart(
                self.chart_manager.create_exudate_chart(visits),
                use_container_width=True
            )

        with tabs[5]:
            self.render_analysis_tab(patient_data)

    def render_analysis_tab(self, patient_data: Dict) -> None:
        """Render the analysis tab content."""
        st.header("AI-Powered Wound Analysis")

        if not st.session_state.analysis_complete:
            if st.button("Generate Analysis"):
                with st.spinner("Analyzing wound data..."):
                    llm = WoundAnalysisLLM()
                    analysis = llm.analyze_wound_data(patient_data)
                    st.session_state.analysis_results = analysis
                    st.session_state.analysis_complete = True

        if st.session_state.analysis_complete:
            st.write(st.session_state.analysis_results)

            if st.button("Generate Report"):
                with st.spinner("Creating report..."):
                    report_path = self.doc_handler.create_and_save_report(
                        patient_data,
                        st.session_state.analysis_results
                    )
                    st.session_state.report_path = report_path
                    st.success("Report generated successfully!")

            if st.session_state.report_path:
                st.download_button(
                    label="Download Report",
                    data=open(st.session_state.report_path, "rb"),
                    file_name="wound_analysis_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    def run(self):
        """Main application entry point."""
        self.create_sidebar()
        self.create_main_content()

if __name__ == "__main__":
    dashboard = WoundAnalysisDashboard()
    dashboard.run()
