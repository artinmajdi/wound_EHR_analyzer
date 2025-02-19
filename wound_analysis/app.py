import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pathlib
import os
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM
import base64
from io import BytesIO
from docx import Document
from llm_interface import format_word_document

st.set_page_config(page_title="Wound Analysis Dashboard", layout="wide", page_icon="🏥")

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def create_wound_measurements_chart(visits):
    """Create an interactive chart showing wound measurements over time."""
    dates = []
    areas = []
    lengths = []
    widths = []
    depths = []

    for visit in visits:
        date = visit['visit_date']
        measurements = visit['wound_measurements']
        dates.append(date)
        areas.append(measurements.get('area', None))
        lengths.append(measurements.get('length', None))
        widths.append(measurements.get('width', None))
        depths.append(measurements.get('depth', None))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=areas, name='Area (cm²)', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=lengths, name='Length (cm)', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=widths, name='Width (cm)', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=depths, name='Depth (cm)', mode='lines+markers'))

    fig.update_layout(
        title='Wound Measurements Over Time',
        xaxis_title='Visit Date',
        yaxis_title='Measurement',
        hovermode='x unified'
    )
    return fig

def create_temperature_chart(visits):
    """Create an interactive chart showing temperature measurements over time."""
    dates = []
    center_temps = []
    edge_temps = []
    peri_temps = []

    for visit in visits:
        date = visit['visit_date']
        temp_data = visit['sensor_data']['temperature']
        dates.append(date)
        center_temps.append(temp_data.get('center', None))
        edge_temps.append(temp_data.get('edge', None))
        peri_temps.append(temp_data.get('peri', None))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=center_temps, name='Center', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=edge_temps, name='Edge', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=peri_temps, name='Peri-wound', mode='lines+markers'))

    fig.update_layout(
        title='Temperature Measurements Over Time',
        xaxis_title='Visit Date',
        yaxis_title='Temperature (°F)',
        hovermode='x unified'
    )
    return fig

def create_impedance_chart(visits):
    """Create an interactive chart showing impedance measurements over time."""
    dates = []
    high_freq_z = []
    high_freq_r = []
    high_freq_c = []

    for visit in visits:
        date = visit['visit_date']
        imp_data = visit['sensor_data']['impedance']['high_frequency']
        dates.append(date)
        high_freq_z.append(imp_data.get('Z', None))
        high_freq_r.append(imp_data.get('resistance', None))
        high_freq_c.append(imp_data.get('capacitance', None))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=high_freq_z, name='|Z| (kΩ)', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=high_freq_r, name="Resistance (kΩ)", mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=high_freq_c, name="Capacitance", mode='lines+markers'))

    fig.update_layout(
        title='Impedance Measurements Over Time',
        xaxis_title='Visit Date',
        yaxis_title='Impedance',
        hovermode='x unified'
    )
    return fig

def create_oxygenation_chart(visits):
    """Create an interactive chart showing oxygenation and hemoglobin measurements over time."""
    dates = []
    oxygenation = []
    hemoglobin = []
    oxyhemoglobin = []
    deoxyhemoglobin = []

    for visit in visits:
        date = visit['visit_date']
        sensor_data = visit['sensor_data']
        dates.append(date)
        oxygenation.append(sensor_data.get('oxygenation', None))
        hemoglobin.append(100 * sensor_data.get('hemoglobin', None))
        oxyhemoglobin.append(100 * sensor_data.get('oxyhemoglobin', None))
        deoxyhemoglobin.append(100 * sensor_data.get('deoxyhemoglobin', None))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=oxygenation, name='Oxygenation (%)', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=hemoglobin, name='Hemoglobin', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=oxyhemoglobin, name='Oxyhemoglobin', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates, y=deoxyhemoglobin, name='Deoxyhemoglobin', mode='lines+markers'))

    fig.update_layout(
        title='Oxygenation and Hemoglobin Measurements Over Time',
        xaxis_title='Visit Date',
        yaxis_title='Value',
        hovermode='x unified'
    )
    return fig

def create_exudate_chart(visits):
    """Create a chart showing exudate characteristics over time."""
    dates = []
    volumes = []
    types = []
    viscosities = []

    for visit in visits:
        date = visit['visit_date']
        wound_info = visit['wound_info']
        exudate = wound_info.get('exudate', {})
        dates.append(date)
        volumes.append(exudate.get('volume', None))
        types.append(exudate.get('type', None))
        viscosities.append(exudate.get('viscosity', None))

    # Create figure for categorical data
    fig = go.Figure()

    # Add volume as lines
    if any(volumes):
        fig.add_trace(go.Scatter(x=dates, y=volumes, name='Volume', mode='lines+markers'))

    # Add types and viscosities as markers with text
    if any(types):
        fig.add_trace(go.Scatter(
            x=dates,
            y=[1]*len(dates),
            text=types,
            name='Type',
            mode='markers+text',
            textposition='bottom center'
        ))

    if any(viscosities):
        fig.add_trace(go.Scatter(
            x=dates,
            y=[0]*len(dates),
            text=viscosities,
            name='Viscosity',
            mode='markers+text',
            textposition='top center'
        ))

    fig.update_layout(
        title='Exudate Characteristics Over Time',
        xaxis_title='Visit Date',
        yaxis_title='Properties',
        hovermode='x unified'
    )
    return fig

def create_and_save_report(patient_data: dict, analysis_results: str) -> str:
    """Create and save the analysis report as a Word document."""
    doc = Document()
    report_path = format_word_document(doc, analysis_results, patient_data)
    return report_path

def download_word_report(report_path: str):
    """Create a download link for the Word report."""
    try:
        with open(report_path, 'rb') as f:
            bytes_data = f.read()
            st.download_button(
                label="Download Full Report (DOCX)",
                data=bytes_data,
                file_name=os.path.basename(report_path),
                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
    except Exception as e:
        st.error(f"Error preparing report download: {str(e)}")

def main():
    # Sidebar configuration
    st.sidebar.title("Configuration")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Patient Data (CSV)", type=['csv'])

    # Model selection and configuration
    model_name = st.sidebar.selectbox(
        "Select Analysis Model",
        ["ai-verde", "falcon-7b-medical", "biogpt", "clinical-bert"]
    )

    # Model-specific configuration
    with st.sidebar.expander("Model Configuration"):
        api_key = st.text_input("API Key", value="sk-h8JtQkCCJUOy-TAdDxCLGw", type="password")
        if model_name == "ai-verde":
            base_url = st.text_input("Base URL", value="https://llm-api.cyverse.ai")

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    # Patient selection (only shown after file upload)
    patient_id = None
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = pathlib.Path("temp_dataset")
        temp_path.mkdir(exist_ok=True)
        temp_file = temp_path / uploaded_file.name
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Initialize processor with temporary file
        st.session_state.processor = WoundDataProcessor(temp_path)

        # Get unique patient IDs from the data
        df = pd.read_csv(temp_file)
        patient_ids = df['Record ID'].unique()
        patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)

        # Add Run Analysis button to sidebar
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Analyzing patient data..."):  # Fixed spinner location
                try:
                    # Initialize LLM and run analysis
                    llm = WoundAnalysisLLM(model_name=model_name)
                    patient_data = st.session_state.processor.get_patient_visits(patient_id)
                    analysis = llm.analyze_patient_data(patient_data)
                    # Save the report path in session state
                    report_path = create_and_save_report(patient_data, analysis)
                    st.session_state.analysis_complete = True
                    st.session_state.analysis_results = analysis
                    st.session_state.report_path = report_path
                    st.sidebar.success("Analysis complete! View results in the Analysis tab.")
                except Exception as e:
                    st.sidebar.error(f"Error during analysis: {str(e)}")

    # Main content area
    st.title("Wound Care Analysis Dashboard")

    if patient_id and st.session_state.processor:
        try:
            # Get patient data
            patient_data = st.session_state.processor.get_patient_visits(patient_id)

            # Create tabs for different sections
            dashboard_tab, analysis_tab = st.tabs(["Dashboard", "Analysis Results"])

            with dashboard_tab:
                # Display patient information in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Patient Demographics")
                    metadata = patient_data['patient_metadata']
                    st.write(f"Age: {metadata.get('age')} years")
                    st.write(f"Sex: {metadata.get('sex')}")
                    st.write(f"BMI: {metadata.get('bmi')}")
                    st.write(f"Race: {metadata.get('race')}")
                    st.write(f"Ethnicity: {metadata.get('ethnicity')}")

                with col2:
                    st.subheader("Medical History")
                    if metadata.get('medical_history'):
                        for condition, status in metadata['medical_history'].items():
                            if status and status != 'None':
                                st.write(f"- {condition}")

                with col3:
                    st.subheader("Diabetes Status")
                    diabetes = metadata.get('diabetes', {})
                    st.write(f"Status: {diabetes.get('status')}")
                    st.write(f"HbA1c: {diabetes.get('hemoglobin_a1c')}%")

                # Charts section with dropdown
                st.header("Wound Analysis Visualizations")

                plot_type = st.selectbox(
                    "Select Visualization",
                    [
                        "Wound Measurements",
                        "Temperature",
                        "Impedance",
                        "Oxygenation & Hemoglobin",
                        "Exudate Characteristics"
                    ]
                )

                if plot_type == "Wound Measurements":
                    measurements_chart = create_wound_measurements_chart(patient_data['visits'])
                    st.plotly_chart(measurements_chart, use_container_width=True)
                elif plot_type == "Temperature":
                    temperature_chart = create_temperature_chart(patient_data['visits'])
                    st.plotly_chart(temperature_chart, use_container_width=True)
                elif plot_type == "Impedance":
                    impedance_chart = create_impedance_chart(patient_data['visits'])
                    st.plotly_chart(impedance_chart, use_container_width=True)
                elif plot_type == "Oxygenation & Hemoglobin":
                    oxygenation_chart = create_oxygenation_chart(patient_data['visits'])
                    st.plotly_chart(oxygenation_chart, use_container_width=True)
                elif plot_type == "Exudate Characteristics":
                    exudate_chart = create_exudate_chart(patient_data['visits'])
                    st.plotly_chart(exudate_chart, use_container_width=True)

                # Display visit history
                with st.expander("Visit History"):
                    for visit in patient_data['visits']:
                        st.subheader(f"Visit Date: {visit['visit_date']}")

                        # Create three columns for visit details
                        v_col1, v_col2, v_col3 = st.columns(3)

                        with v_col1:
                            st.write("**Wound Measurements**")
                            measurements = visit['wound_measurements']
                            st.write(f"Length: {measurements.get('length')} cm")
                            st.write(f"Width: {measurements.get('width')} cm")
                            st.write(f"Depth: {measurements.get('depth')} cm")
                            st.write(f"Area: {measurements.get('area')} cm²")

                        with v_col2:
                            st.write("**Wound Characteristics**")
                            wound_info = visit['wound_info']
                            st.write(f"Location: {wound_info.get('location')}")
                            st.write(f"Type: {wound_info.get('type')}")

                            exudate = wound_info.get('exudate', {})
                            if exudate:
                                st.write(f"Exudate Volume: {exudate.get('volume')}")
                                st.write(f"Exudate Type: {exudate.get('type')}")

                        with v_col3:
                            st.write("**Sensor Data**")
                            sensor = visit['sensor_data']
                            st.write(f"Oxygenation: {sensor.get('oxygenation')}%")
                            temp = sensor.get('temperature', {})
                            st.write(f"Temperature (Center): {temp.get('center')}°F")
                            st.write(f"Temperature (Edge): {temp.get('edge')}°F")

            with analysis_tab:
                if st.session_state.analysis_complete and st.session_state.analysis_results:
                    st.header("Analysis Results")
                    st.markdown(st.session_state.analysis_results)

                    # Add download button for the report using the saved path
                    if hasattr(st.session_state, 'report_path'):
                        st.subheader("Download Report")
                        download_word_report(st.session_state.report_path)
                else:
                    st.info("Run the analysis from the sidebar to view results here.")

        except Exception as e:
            st.error(f"Error processing patient data: {str(e)}")

    else:
        st.info("Please upload a CSV file and select a patient ID to begin analysis.")

if __name__ == "__main__":
    main()
