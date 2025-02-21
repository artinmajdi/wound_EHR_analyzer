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
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Dashboard"

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
    high_freq_z, high_freq_r, high_freq_c = [], [], []
    center_freq_z, center_freq_r, center_freq_c = [], [], []
    low_freq_z, low_freq_r, low_freq_c = [], [], []

    high_freq_val, center_freq_val, low_freq_val = None, None, None

    for visit in visits:
        date = visit['visit_date']
        sensor_data = visit.get('sensor_data', {})
        impedance_data = sensor_data.get('impedance', {})

        dates.append(date)

        # Process high frequency data
        high_freq = impedance_data.get('high_frequency', {})
        try:
            z_val = float(high_freq.get('Z')) if high_freq.get('Z') not in (None, '') else None
            r_val = float(high_freq.get('resistance')) if high_freq.get('resistance') not in (None, '') else None
            c_val = float(high_freq.get('capacitance')) if high_freq.get('capacitance') not in (None, '') else None

            if high_freq.get('frequency'):
                high_freq_val = high_freq.get('frequency')

            high_freq_z.append(z_val)
            high_freq_r.append(r_val)
            high_freq_c.append(c_val)
        except (ValueError, TypeError):
            high_freq_z.append(None)
            high_freq_r.append(None)
            high_freq_c.append(None)

        # Process center frequency data
        center_freq = impedance_data.get('center_frequency', {})
        try:
            z_val = float(center_freq.get('Z')) if center_freq.get('Z') not in (None, '') else None
            r_val = float(center_freq.get('resistance')) if center_freq.get('resistance') not in (None, '') else None
            c_val = float(center_freq.get('capacitance')) if center_freq.get('capacitance') not in (None, '') else None

            if center_freq.get('frequency'):
                center_freq_val = center_freq.get('frequency')

            center_freq_z.append(z_val)
            center_freq_r.append(r_val)
            center_freq_c.append(c_val)
        except (ValueError, TypeError):
            center_freq_z.append(None)
            center_freq_r.append(None)
            center_freq_c.append(None)

        # Process low frequency data
        low_freq = impedance_data.get('low_frequency', {})
        try:
            z_val = float(low_freq.get('Z')) if low_freq.get('Z') not in (None, '') else None
            r_val = float(low_freq.get('resistance')) if low_freq.get('resistance') not in (None, '') else None
            c_val = float(low_freq.get('capacitance')) if low_freq.get('capacitance') not in (None, '') else None

            if low_freq.get('frequency'):
                low_freq_val = low_freq.get('frequency')

            low_freq_z.append(z_val)
            low_freq_r.append(r_val)
            low_freq_c.append(c_val)
        except (ValueError, TypeError):
            low_freq_z.append(None)
            low_freq_r.append(None)
            low_freq_c.append(None)

    fig = go.Figure()

    # Format frequency labels
    high_freq_label = f"{float(high_freq_val):.0f}Hz" if high_freq_val else "High Freq"
    center_freq_label = f"{float(center_freq_val):.0f}Hz" if center_freq_val else "Center Freq"
    low_freq_label = f"{float(low_freq_val):.0f}Hz" if low_freq_val else "Low Freq"

    # Add high frequency traces
    if any(x is not None for x in high_freq_z):
        fig.add_trace(go.Scatter(
            x=dates, y=high_freq_z,
            name=f'|Z| ({high_freq_label})',
            mode='lines+markers'
        ))
    if any(x is not None for x in high_freq_r):
        fig.add_trace(go.Scatter(
            x=dates, y=high_freq_r,
            name=f'Resistance ({high_freq_label})',
            mode='lines+markers'
        ))
    if any(x is not None for x in high_freq_c):
        fig.add_trace(go.Scatter(
            x=dates, y=high_freq_c,
            name=f'Capacitance ({high_freq_label})',
            mode='lines+markers'
        ))

    # Add center frequency traces
    if any(x is not None for x in center_freq_z):
        fig.add_trace(go.Scatter(
            x=dates, y=center_freq_z,
            name=f'|Z| ({center_freq_label})',
            mode='lines+markers',
            line=dict(dash='dot')
        ))
    if any(x is not None for x in center_freq_r):
        fig.add_trace(go.Scatter(
            x=dates, y=center_freq_r,
            name=f'Resistance ({center_freq_label})',
            mode='lines+markers',
            line=dict(dash='dot')
        ))
    if any(x is not None for x in center_freq_c):
        fig.add_trace(go.Scatter(
            x=dates, y=center_freq_c,
            name=f'Capacitance ({center_freq_label})',
            mode='lines+markers',
            line=dict(dash='dot')
        ))

    # Add low frequency traces
    if any(x is not None for x in low_freq_z):
        fig.add_trace(go.Scatter(
            x=dates, y=low_freq_z,
            name=f'|Z| ({low_freq_label})',
            mode='lines+markers',
            line=dict(dash='dash')
        ))
    if any(x is not None for x in low_freq_r):
        fig.add_trace(go.Scatter(
            x=dates, y=low_freq_r,
            name=f'Resistance ({low_freq_label})',
            mode='lines+markers',
            line=dict(dash='dash')
        ))
    if any(x is not None for x in low_freq_c):
        fig.add_trace(go.Scatter(
            x=dates, y=low_freq_c,
            name=f'Capacitance ({low_freq_label})',
            mode='lines+markers',
            line=dict(dash='dash')
        ))

    fig.update_layout(
        title='Impedance Measurements Over Time',
        xaxis_title='Visit Date',
        yaxis_title='Impedance Values',
        hovermode='x unified',
        showlegend=True,
        height=600,
        yaxis=dict(type='log'),  # Use log scale for better visualization
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
        oxygenation.append(sensor_data.get('oxygenation'))

        # Handle None values for hemoglobin measurements
        hb = sensor_data.get('hemoglobin')
        oxyhb = sensor_data.get('oxyhemoglobin')
        deoxyhb = sensor_data.get('deoxyhemoglobin')

        hemoglobin.append(100 * hb if hb is not None else None)
        oxyhemoglobin.append(100 * oxyhb if oxyhb is not None else None)
        deoxyhemoglobin.append(100 * deoxyhb if deoxyhb is not None else None)

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
    # Wrap the entire main function in a try-except block to catch Streamlit file watcher errors
    try:
        # Sidebar configuration
        st.sidebar.title("Configuration")

        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload Patient Data (CSV)", type=['csv'])

        # Platform and Model selection
        platform_options = WoundAnalysisLLM.get_available_platforms()
        platform = st.sidebar.selectbox(
            "Select Platform",
            platform_options,
            index=platform_options.index("ai-verde"),  # Default to ai-verde
            help="Hugging Face models are currently disabled. Please use AI Verde models."
        )

        # If huggingface is selected, show warning and force ai-verde
        if platform == "huggingface":
            st.sidebar.warning("Hugging Face models are currently disabled. Please use AI Verde models.")
            platform = "ai-verde"

        # Get available models for the selected platform
        available_models = WoundAnalysisLLM.get_available_models(platform)
        default_model = "llama-3.3-70b-fp8" if platform == "ai-verde" else "medalpaca-7b"
        model_name = st.sidebar.selectbox(
            "Select Model",
            available_models,
            index=available_models.index(default_model)
        )

        # Model-specific configuration
        with st.sidebar.expander("Model Configuration"):
            api_key = st.text_input("API Key", value="sk-h8JtQkCCJUOy-TAdDxCLGw", type="password")
            if platform == "ai-verde":
                base_url = st.text_input("Base URL", value="https://llm-api.cyverse.ai")

            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        # Patient selection (only shown after file upload)
        patient_id = None
        if uploaded_file is not None:
            try:
                # Create a temporary dataset directory using absolute path
                dataset_path = pathlib.Path(__file__).resolve().parent / "dataset"
                dataset_path.mkdir(parents=True, exist_ok=True)

                # Save the uploaded file using absolute path
                temp_csv_path = dataset_path / "SmartBandage-Data_for_llm.csv"
                temp_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_csv_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Initialize processor with the absolute dataset path
                st.session_state.processor = WoundDataProcessor(dataset_path)

                # Use the uploaded file directly for patient IDs
                df = pd.read_csv(uploaded_file)
                patient_ids = sorted(df['Record ID'].unique())
                patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)

                # Add Run Analysis button to sidebar
                if st.sidebar.button("Run Analysis"):
                    with st.spinner("Analyzing patient data..."):
                        try:
                            # Initialize LLM with platform and model
                            llm = WoundAnalysisLLM(platform=platform, model_name=model_name)
                            patient_data = st.session_state.processor.get_patient_visits(patient_id)
                            analysis = llm.analyze_patient_data(patient_data)
                            # Save the report path in session state
                            report_path = create_and_save_report(patient_data, analysis)
                            st.session_state.analysis_complete = True
                            st.session_state.analysis_results = analysis
                            st.session_state.report_path = report_path
                            st.session_state.active_tab = "Analysis Results"  # Set active tab
                            st.sidebar.success("Analysis complete! View results in the Analysis tab.")
                        except Exception as e:
                            st.sidebar.error(f"Error during analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")

        # Main content area
        st.title("Wound Care Analysis Dashboard")

        if patient_id and st.session_state.processor:
            try:
                # Get patient data
                patient_data = st.session_state.processor.get_patient_visits(patient_id)

                # Create tabs for different sections
                dashboard_tab, analysis_tab = st.tabs(["Dashboard", "Analysis Results"])

                # Set the active tab based on session state
                if st.session_state.active_tab == "Analysis Results":
                    analysis_tab.active = True

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
                                    st.write(f"- {condition}: {status}")

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
    except Exception as e:
        if isinstance(e, RuntimeError) and any(x in str(e) for x in ["torch.classes", "__path__._path", "torch::class_"]):
            # Silently ignore PyTorch class path errors from Streamlit file watcher
            # These errors are benign and don't affect functionality
            pass
        elif "StreamlitAPIException" in str(type(e)):
            # Handle Streamlit-specific exceptions
            st.error("A Streamlit error occurred. Please refresh the page and try again.")
        else:
            # Handle all other exceptions
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())

if __name__ == "__main__":
    main()
