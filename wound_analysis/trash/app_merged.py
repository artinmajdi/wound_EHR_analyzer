import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime
import pathlib
import os
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document
import base64
from io import BytesIO
from docx import Document

# Set page configuration
st.set_page_config(
    page_title="Smart Bandage Wound Analysis Dashboard",
    page_icon="🩹",
    layout="wide"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Dashboard"

# Function to load data
@st.cache_data
def load_data(file_path):
    """
    Load and preprocess the wound healing data from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()

        # Extract visit number from Event Name
        df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

        # Calculate wound area if not present
        if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
            df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

        # Calculate healing rate
        healing_rates = []
        for patient_id in df['Record ID'].unique():
            patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')
            for i, row in patient_data.iterrows():
                if row['Visit Number'] == 1:
                    healing_rates.append(0)
                else:
                    prev_visit = patient_data[patient_data['Visit Number'] == row['Visit Number'] - 1]
                    if len(prev_visit) > 0:
                        prev_area = prev_visit['Calculated Wound Area'].values[0]
                        curr_area = row['Calculated Wound Area']
                        if prev_area > 0:
                            healing_rate = (prev_area - curr_area) / prev_area * 100
                            healing_rates.append(healing_rate)
                        else:
                            healing_rates.append(0)
                    else:
                        healing_rates.append(0)

        df['Healing Rate (%)'] = healing_rates

        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # Standardize categorical columns
        if 'Diabetes?' in df.columns:
            df['Diabetes?'] = df['Diabetes?'].fillna('Unknown')
            df['Diabetes?'] = df['Diabetes?'].replace({'yes': 'Yes', 'no': 'No', 'Y': 'Yes', 'N': 'No'})

        if 'Smoking status' in df.columns:
            df['Smoking status'] = df['Smoking status'].fillna('Unknown')

        # Create temperature gradients
        temp_cols = [
            'Center of Wound Temperature (Fahrenheit)',
            'Edge of Wound Temperature (Fahrenheit)',
            'Peri-wound Temperature (Fahrenheit)'
        ]
        if all(col in df.columns for col in temp_cols):
            df['Center-Edge Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Edge of Wound Temperature (Fahrenheit)']
            df['Edge-Peri Temp Gradient'] = df['Edge of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']
            df['Total Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_and_save_report(patient_data: dict, analysis_results: str) -> str:
    """Create and save the analysis report as a Word document."""
    doc = Document()
    format_word_document(doc, patient_data, analysis_results)
    
    # Save the document
    report_path = "wound_analysis_report.docx"
    doc.save(report_path)
    return report_path

def download_word_report(report_path: str):
    """Create a download link for the Word report."""
    with open(report_path, "rb") as file:
        bytes_data = file.read()
    
    b64_data = base64.b64encode(bytes_data).decode()
    href = f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_data}'
    
    return href

def main():
    try:
        # Top Sidebar - Model Configuration
        st.sidebar.title("Model Configuration")

        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload Patient Data (CSV)", type=['csv'])

        # Platform and Model selection
        platform_options = WoundAnalysisLLM.get_available_platforms()
        platform = st.sidebar.selectbox(
            "Select Platform",
            platform_options,
            index=platform_options.index("ai-verde"),
            help="Hugging Face models are currently disabled. Please use AI Verde models."
        )

        if platform == "huggingface":
            st.sidebar.warning("Hugging Face models are currently disabled. Please use AI Verde models.")
            platform = "ai-verde"

        available_models = WoundAnalysisLLM.get_available_models(platform)
        default_model = "llama-3.3-70b-fp8" if platform == "ai-verde" else "medalpaca-7b"
        model_name = st.sidebar.selectbox(
            "Select Model",
            available_models,
            index=available_models.index(default_model)
        )

        # Model configuration
        with st.sidebar.expander("Model Configuration"):
            api_key = st.text_input("API Key", value="sk-h8JtQkCCJUOy-TAdDxCLGw", type="password")
            if platform == "ai-verde":
                base_url = st.text_input("Base URL", value="https://llm-api.cyverse.ai")

            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        st.sidebar.markdown("---")

        # Bottom Sidebar - Data Analysis
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                # Patient selection
                patient_ids = sorted(df['Record ID'].unique())
                patient_options = ["All Patients"] + [f"Patient {id:03d}" for id in patient_ids]
                selected_patient = st.sidebar.selectbox("Select Patient", patient_options)

                # Filter data based on patient selection
                if selected_patient == "All Patients":
                    filtered_df = df
                else:
                    patient_id = int(selected_patient.split(" ")[1])
                    filtered_df = df[df['Record ID'] == patient_id]

                # Run Analysis button
                if st.sidebar.button("Run Analysis"):
                    with st.spinner("Analyzing patient data..."):
                        try:
                            # Initialize LLM
                            llm = WoundAnalysisLLM(platform=platform, model_name=model_name)
                            
                            # Create temporary dataset
                            dataset_path = pathlib.Path(__file__).resolve().parent / "dataset"
                            dataset_path.mkdir(parents=True, exist_ok=True)
                            temp_csv_path = dataset_path / "SmartBandage-Data_for_llm.csv"
                            
                            with open(temp_csv_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            
                            st.session_state.processor = WoundDataProcessor(dataset_path)
                            patient_data = st.session_state.processor.get_patient_visits(patient_id)
                            analysis = llm.analyze_patient_data(patient_data)
                            
                            report_path = create_and_save_report(patient_data, analysis)
                            st.session_state.analysis_complete = True
                            st.session_state.analysis_results = analysis
                            st.session_state.report_path = report_path
                            st.session_state.active_tab = "Analysis Results"
                            st.sidebar.success("Analysis complete! View results in the Analysis tab.")
                        except Exception as e:
                            st.sidebar.error(f"Error during analysis: {str(e)}")

                # Main content area
                st.title("Wound Care Analysis Dashboard")

                # Create tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Dashboard & Overview",
                    "Analysis Results",
                    "Impedance Analysis",
                    "Temperature Analysis",
                    "Risk Factors"
                ])

                with tab1:
                    # Original Dashboard content
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Patients",
                            len(df['Record ID'].unique()),
                            help="Number of unique patients in the dataset"
                        )
                    with col2:
                        avg_visits = df.groupby('Record ID').size().mean()
                        st.metric(
                            "Average Visits",
                            f"{avg_visits:.1f}",
                            help="Average number of visits per patient"
                        )
                    with col3:
                        avg_healing = df[df['Healing Rate (%)'] > 0]['Healing Rate (%)'].mean()
                        st.metric(
                            "Average Healing Rate",
                            f"{avg_healing:.1f}%",
                            help="Average wound area reduction per visit"
                        )

                    # Wound measurements over time
                    st.subheader("Wound Measurements Over Time")
                    if selected_patient != "All Patients":
                        patient_data = filtered_df.sort_values('Visit Number')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=patient_data['Visit Number'],
                            y=patient_data['Calculated Wound Area'],
                            mode='lines+markers',
                            name='Wound Area'
                        ))
                        st.plotly_chart(fig, use_container_width=True)

                    # Wound Type Distribution
                    st.subheader("Wound Type Distribution")
                    wound_counts = df.groupby('Wound Type').size().reset_index(name='Count')
                    fig = px.bar(
                        wound_counts,
                        x='Wound Type',
                        y='Count',
                        title='Distribution of Wound Types'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Healing Progress
                    st.subheader("Healing Progress Analysis")
                    healing_data = df[df['Healing Rate (%)'] > 0]
                    fig = px.box(
                        healing_data,
                        x='Wound Type',
                        y='Healing Rate (%)',
                        title='Healing Rates by Wound Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    if st.session_state.analysis_complete:
                        st.markdown("### Analysis Results")
                        st.write(st.session_state.analysis_results)
                        
                        # Download report button
                        report_path = st.session_state.report_path
                        href = download_word_report(report_path)
                        st.markdown(
                            f'<a href="{href}" download="wound_analysis_report.docx">Download Word Report</a>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("Run the analysis from the sidebar to view results here.")

                with tab3:
                    st.subheader("Impedance vs. Wound Healing Progress")
                    if selected_patient == "All Patients":
                        # Scatter plot of impedance vs healing rate
                        fig = px.scatter(
                            df,
                            x='Skin Impedance (kOhms) - Z',
                            y='Healing Rate (%)',
                            color='Diabetes?',
                            size='Calculated Wound Area',
                            hover_data=['Record ID', 'Event Name', 'Wound Type'],
                            title="Relationship Between Skin Impedance and Healing Rate"
                        )
                        
                        # Calculate correlation
                        valid_data = df[df['Healing Rate (%)'] > 0]
                        r, p = stats.pearsonr(valid_data['Skin Impedance (kOhms) - Z'], valid_data['Healing Rate (%)'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Format p-value
                        p_formatted = f"< 0.001" if p < 0.001 else f"= {p:.3f}"
                        st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")
                        
                        # Show impedance by wound type
                        avg_by_type = df.groupby('Wound Type')['Skin Impedance (kOhms) - Z'].mean().reset_index()
                        fig = px.bar(
                            avg_by_type,
                            x='Wound Type',
                            y='Skin Impedance (kOhms) - Z',
                            title='Average Impedance by Wound Type'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        patient_data = filtered_df.sort_values('Visit Number')
                        fig = px.line(
                            patient_data,
                            x='Visit Number',
                            y=['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\''],
                            title=f"Impedance Measurements Over Time for {selected_patient}",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab4:
                    st.subheader("Temperature Gradient Analysis")
                    if selected_patient == "All Patients":
                        # Temperature gradients overview
                        temp_data = pd.melt(
                            df,
                            value_vars=['Center-Edge Temp Gradient', 'Edge-Peri Temp Gradient', 'Total Temp Gradient'],
                            var_name='Gradient Type',
                            value_name='Temperature Difference (°F)'
                        )
                        fig = px.box(
                            temp_data,
                            x='Gradient Type',
                            y='Temperature Difference (°F)',
                            title='Temperature Gradients Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation with healing
                        fig = px.scatter(
                            df,
                            x='Total Temp Gradient',
                            y='Healing Rate (%)',
                            color='Wound Type',
                            title='Temperature Gradient vs. Healing Rate'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Individual patient temperature trends
                        patient_data = filtered_df.sort_values('Visit Number')
                        fig = px.line(
                            patient_data,
                            x='Visit Number',
                            y=['Center of Wound Temperature (Fahrenheit)',
                               'Edge of Wound Temperature (Fahrenheit)',
                               'Peri-wound Temperature (Fahrenheit)'],
                            title=f"Temperature Measurements Over Time for {selected_patient}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab5:
                    st.subheader("Risk Factor Analysis")
                    if selected_patient == "All Patients":
                        # Create tabs for different risk factors
                        risk_tab1, risk_tab2 = st.tabs(["Diabetes Impact", "Smoking Status"])
                        
                        with risk_tab1:
                            # Diabetes analysis
                            diabetes_healing = df.groupby('Diabetes?')['Healing Rate (%)'].mean().reset_index()
                            fig = px.bar(
                                diabetes_healing,
                                x='Diabetes?',
                                y='Healing Rate (%)',
                                title='Average Healing Rate by Diabetes Status'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with risk_tab2:
                            # Smoking status analysis
                            smoke_wound = pd.crosstab(df['Smoking status'], df['Wound Type'])
                            smoke_wound_pct = smoke_wound.div(smoke_wound.sum(axis=1), axis=0) * 100
                            
                            fig = px.bar(
                                smoke_wound_pct.reset_index().melt(
                                    id_vars='Smoking status',
                                    var_name='Wound Type',
                                    value_name='Percentage'
                                ),
                                x='Smoking status',
                                y='Percentage',
                                color='Wound Type',
                                title='Wound Type Distribution by Smoking Status'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Individual patient risk analysis
                        patient_data = filtered_df.iloc[0]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Diabetes Status", patient_data['Diabetes?'])
                            st.metric("Smoking Status", patient_data['Smoking status'])
                        
                        with col2:
                            if 'BMI' in patient_data:
                                st.metric("BMI", f"{patient_data['BMI']:.1f}")
                            st.metric("Wound Type", patient_data['Wound Type'])

        else:
            st.info("Please upload a CSV file to begin analysis.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
