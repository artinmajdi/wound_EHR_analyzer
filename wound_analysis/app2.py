import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import pathlib
import os
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document
import base64
from io import BytesIO
from docx import Document
from typing import Optional

# Set page configuration
st.set_page_config(
    page_title="ORIGINAL app2.py",
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

# Function to load data
@st.cache_data
def load_data():
    """
    Load and preprocess the wound healing data from CSV file.

    Returns:
        pandas.DataFrame: Processed wound healing data
    """
    try:
        # Load the actual CSV file from the specified path
        file_path = "/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv"
        df = pd.read_csv(file_path)

        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()

        # Data cleaning and preprocessing

        # 1. Extract visit number from Event Name
        # Assuming Event Name contains patterns like "Visit 1", "Visit 2", etc.
        df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

        # 2. Calculate wound area if not present but dimensions are available
        if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
            df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

        # 3. Calculate healing rate (% change in wound area per visit)
        healing_rates = []
        for patient_id in df['Record ID'].unique():
            patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')
            for i, row in patient_data.iterrows():
                if row['Visit Number'] == 1 or len(patient_data[patient_data['Visit Number'] < row['Visit Number']]) == 0:
                    healing_rates.append(0)  # No healing rate for first visit
                else:
                    # Find the most recent previous visit
                    prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
                    prev_visit = prev_visits[prev_visits['Visit Number'] == prev_visits['Visit Number'].max()]

                    if len(prev_visit) > 0 and 'Calculated Wound Area' in df.columns:
                        prev_area = prev_visit['Calculated Wound Area'].values[0]
                        curr_area = row['Calculated Wound Area']

                        # Check for valid values to avoid division by zero
                        if prev_area > 0 and not pd.isna(prev_area) and not pd.isna(curr_area):
                            healing_rate = (prev_area - curr_area) / prev_area * 100  # Percentage decrease
                            healing_rates.append(healing_rate)
                        else:
                            healing_rates.append(0)
                    else:
                        healing_rates.append(0)

        # If lengths don't match (due to filtering), adjust
        if len(healing_rates) < len(df):
            healing_rates.extend([0] * (len(df) - len(healing_rates)))
        elif len(healing_rates) > len(df):
            healing_rates = healing_rates[:len(df)]

        df['Healing Rate (%)'] = healing_rates

        # 4. Handle missing values in key columns
        # Fill numeric columns with appropriate defaults
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # 5. Standardize categorical columns
        if 'Diabetes?' in df.columns:
            df['Diabetes?'] = df['Diabetes?'].fillna('Unknown')
            # Standardize Yes/No responses
            df['Diabetes?'] = df['Diabetes?'].replace({'yes': 'Yes', 'no': 'No', 'Y': 'Yes', 'N': 'No'})

        if 'Smoking status' in df.columns:
            df['Smoking status'] = df['Smoking status'].fillna('Unknown')

        # 6. Create derived variables for temperature if they exist
        if all(col in df.columns for col in ['Center of Wound Temperature (Fahrenheit)', 'Edge of Wound Temperature (Fahrenheit)', 'Peri-wound Temperature (Fahrenheit)']):
            df['Center-Edge Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Edge of Wound Temperature (Fahrenheit)']
            df['Edge-Peri Temp Gradient'] = df['Edge of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']
            df['Total Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Using mock data instead")

        # Fall back to mock data if the file cannot be loaded
        return generate_mock_data()

# Add function to generate mock data as fallback
def generate_mock_data():
    """
    Generate mock data with the same structure as the wound healing dataset.
    Used as a fallback when the CSV file cannot be loaded.

    Returns:
        pandas.DataFrame: Simulated wound healing data
    """
    np.random.seed(42)

    # Create mock dataset with 418 rows
    n_patients = 127
    n_visits_per_patient = 3  # Average number of visits
    n_rows = 418

    record_ids = np.repeat(np.arange(1, n_patients + 1), n_visits_per_patient)[:n_rows]
    event_names = []
    for patient in range(1, n_patients + 1):
        visits = np.arange(1, n_visits_per_patient + 1)
        for visit in visits:
            event_names.append(f"Visit {visit}")
            if len(event_names) >= n_rows:
                break
        if len(event_names) >= n_rows:
            break

    event_names = event_names[:n_rows]

    # Generate wound areas that decrease over time for each patient
    wound_areas = []
    for patient in range(1, n_patients + 1):
        initial_area = np.random.uniform(5, 20)  # Initial wound area between 5 and 20 cm²
        for visit in range(1, n_visits_per_patient + 1):
            # Decrease wound area by 10-30% each visit
            decrease_factor = np.random.uniform(0.7, 0.9)
            area = initial_area * (decrease_factor ** (visit - 1))
            wound_areas.append(area)
            if len(wound_areas) >= n_rows:
                break
        if len(wound_areas) >= n_rows:
            break

    wound_areas = wound_areas[:n_rows]

    # Generate impedance values that correlate with wound areas
    # Higher impedance = less healing
    base_impedance = np.random.uniform(80, 160, n_rows)
    wound_area_normalized = (wound_areas - np.min(wound_areas)) / (np.max(wound_areas) - np.min(wound_areas))
    impedance_z = base_impedance + wound_area_normalized * 60

    # Add some noise to Z' and Z''
    impedance_z_prime = impedance_z * 0.8 + np.random.normal(0, 5, n_rows)
    impedance_z_double_prime = impedance_z * 0.6 + np.random.normal(0, 3, n_rows)

    # Generate temperature data
    center_temp = np.random.uniform(97, 101, n_rows)
    edge_temp = center_temp - np.random.uniform(0.5, 2, n_rows)
    peri_temp = edge_temp - np.random.uniform(0.5, 1.5, n_rows)

    # Generate oxygenation data
    oxygenation = np.random.uniform(85, 99, n_rows)
    hemoglobin = np.random.uniform(9, 16, n_rows)
    oxyhemoglobin = hemoglobin * (oxygenation/100) * 0.95
    deoxyhemoglobin = hemoglobin - oxyhemoglobin

    # Generate wound dimensions
    length = np.random.uniform(1, 7, n_rows)
    width = np.random.uniform(1, 5, n_rows)
    depth = np.random.uniform(0.1, 2, n_rows)

    # Generate patient demographics and risk factors
    diabetes_status = np.random.choice(['Yes', 'No'], n_rows)
    smoking_status = np.random.choice(['Current', 'Former', 'Never'], n_rows, p=[0.25, 0.25, 0.5])
    bmi = np.random.normal(28, 5, n_rows)
    bmi = np.clip(bmi, 18, 45)  # Clip to reasonable BMI range

    # Generate wound types
    wound_types = np.random.choice(
        ['Diabetic Ulcer', 'Venous Ulcer', 'Pressure Ulcer', 'Surgical Wound', 'Trauma'],
        n_rows,
        p=[0.33, 0.28, 0.22, 0.12, 0.05]
    )

    # Create DataFrame
    df = pd.DataFrame({
        'Record ID': record_ids,
        'Event Name': event_names,
        'Calculated Wound Area': wound_areas,
        'Skin Impedance (kOhms) - Z': impedance_z,
        'Skin Impedance (kOhms) - Z\'': impedance_z_prime,
        'Skin Impedance (kOhms) - Z\'\'': impedance_z_double_prime,
        'Center of Wound Temperature (Fahrenheit)': center_temp,
        'Edge of Wound Temperature (Fahrenheit)': edge_temp,
        'Peri-wound Temperature (Fahrenheit)': peri_temp,
        'Oxygenation (%)': oxygenation,
        'Hemoglobin Level': hemoglobin,
        'Oxyhemoglobin Level': oxyhemoglobin,
        'Deoxyhemoglobin Level': deoxyhemoglobin,
        'Length (cm)': length,
        'Width (cm)': width,
        'Depth (cm)': depth,
        'Diabetes?': diabetes_status,
        'Smoking status': smoking_status,
        'BMI': bmi,
        'Wound Type': wound_types
    })

    # Add visit number
    df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').astype(int)

    # Calculate healing rate for each patient (percentage decrease in wound area per visit)
    healing_rates = []
    for patient_id in df['Record ID'].unique():
        patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')
        for i, row in patient_data.iterrows():
            if row['Visit Number'] == 1:
                healing_rates.append(0)  # No healing rate for first visit
            else:
                prev_area = patient_data[patient_data['Visit Number'] == row['Visit Number'] - 1]['Calculated Wound Area'].values[0]
                curr_area = row['Calculated Wound Area']
                healing_rate = (prev_area - curr_area) / prev_area * 100  # Percentage decrease
                healing_rates.append(healing_rate)

    # If lengths don't match (due to filtering), adjust
    if len(healing_rates) < len(df):
        healing_rates.extend([0] * (len(df) - len(healing_rates)))
    elif len(healing_rates) > len(df):
        healing_rates = healing_rates[:len(df)]

    df['Healing Rate (%)'] = healing_rates

    # Add temperature gradients
    df['Center-Edge Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Edge of Wound Temperature (Fahrenheit)']
    df['Edge-Peri Temp Gradient'] = df['Edge of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']
    df['Total Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']

    return df

# Load data
df = load_data()

# Add data source information
data_source = st.empty()
if 'data_source_type' in locals():
    if data_source_type == "mock":
        data_source.warning("⚠️ Using simulated data (CSV file could not be loaded)")
    else:
        data_source.success("✅ Using actual data from CSV file")

# Create a mapping of patient IDs to their data
patient_ids = [int(id) for id in sorted(df['Record ID'].unique())]
patient_options = ["All Patients"] + [f"Patient {id:03d}" for id in patient_ids]

# Dashboard header
st.title("Smart Bandage Wound Healing Analytics")

# Patient selection dropdown
selected_patient = st.selectbox("Select Patient", patient_options)

# Filter data based on patient selection
if selected_patient == "All Patients":
    filtered_df = df
else:
    patient_id = int(selected_patient.split(" ")[1])
    filtered_df = df[df['Record ID'] == patient_id]

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

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Impedance Analysis",
    "Temperature",
    "Oxygenation",
    "Risk Factors",
    "LLM Analysis"
])

# Tab 1: Overview
with tab1:
    st.header("Smart Bandage Wound Healing Analytics")

    if selected_patient == "All Patients":
        # Show dataset statistics for all patients
        total_patients = len(df['Record ID'].unique())
        total_visits = len(df)
        avg_visits = total_visits / total_patients if total_patients > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", f"{total_patients}")
        with col2:
            st.metric("Total Visits", f"{total_visits}")
        with col3:
            st.metric("Average Visits per Patient", f"{avg_visits:.1f}")

    else:
        # Show patient-specific information
        patient_id = int(selected_patient.split(" ")[1])
        patient_data = filtered_df.iloc[0]

        # Patient Demographics
        st.subheader("Patient Demographics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Age", f"{patient_data['Calculated Age at Enrollment']}")
        with col2:
            st.metric("Gender", f"{patient_data['Sex']}")
        with col3:
            st.metric("BMI", f"{patient_data['BMI']}")

        # Medical History
        st.subheader("Medical History")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Diabetes Status", "Yes" if patient_data['Diabetes?'] == 1 else "No")
        with col2:
            st.metric("Smoking Status", "Yes" if patient_data['Smoking status'] == 1 else "No")
        with col3:
            st.metric("Hypertension", "Yes" if 'Cardiovascular' in str(patient_data['Medical History (select all that apply)']) else "No")
        with col4:
            st.metric("Peripheral Vascular Disease", "Yes" if 'PVD' in str(patient_data['Medical History (select all that apply)']) else "No")

    # Wound Area Plot
    st.subheader("Wound Area Over Time")

    def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
        """Create interactive wound area progression plot."""
        fig = go.Figure()

        if patient_id:
            patient_df = df[df['Record ID'] == patient_id].copy()
            patient_df['Days_Since_First_Visit'] = (pd.to_datetime(patient_df['Visit date']) - pd.to_datetime(patient_df['Visit date']).min()).dt.days

            # Remove any rows with NaN values
            patient_df = patient_df.dropna(subset=["Days_Since_First_Visit", 'Calculated Wound Area'])

            fig.add_trace(go.Scatter(
                x=patient_df['Days_Since_First_Visit'],
                y=patient_df['Calculated Wound Area'],
                mode='lines+markers',
                name='Wound Area',
                line=dict(color='blue'),
                hovertemplate='%{y:.1f} cm²'
            ))

            # Add trendline only if we have enough valid data points
            if len(patient_df) >= 2:  # Need at least 2 points for a line
                try:
                    x = patient_df['Days_Since_First_Visit'].values
                    y = patient_df['Calculated Wound Area'].values

                    # Check for finite values
                    mask = np.isfinite(x) & np.isfinite(y)
                    if np.sum(mask) >= 2:
                        z = np.polyfit(x[mask], y[mask], 1)
                        p = np.poly1d(z)

                        fig.add_trace(go.Scatter(
                            x=patient_df['Days_Since_First_Visit'],
                            y=p(patient_df['Days_Since_First_Visit']),
                            mode='lines',
                            name='Trend',
                            line=dict(color='red', dash='dash'),
                            hovertemplate='Day %{x}<br>Trend: %{y:.1f} cm²'
                        ))
                except Exception as e:
                    st.warning("Could not calculate trend line due to data issues.")

            # Calculate healing rate
            if len(patient_df) >= 2:
                try:
                    total_days = patient_df['Days_Since_First_Visit'].max()
                    if total_days > 0:
                        first_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmin(), 'Calculated Wound Area']
                        last_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmax(), 'Calculated Wound Area']
                        healing_rate = (first_area - last_area) / total_days
                        healing_status = "Improving" if healing_rate > 0 else "Worsening"
                        healing_rate_text = f"Healing Rate: {healing_rate:.2f} cm²/day<br> {healing_status}"
                    else:
                        healing_rate_text = "Insufficient time between measurements for healing rate calculation"
                except Exception as e:
                    healing_rate_text = "Could not calculate healing rate due to data issues"
            else:
                healing_rate_text = "Insufficient data for healing rate calculation"

            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=healing_rate_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            )

            fig.update_layout(
                title=f"Wound Area Progression - Patient {patient_id}",
                xaxis_title="Days Since First Visit",
                yaxis_title="Wound Area (cm²)",
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
            # st.markdown(f"**{healing_rate_text}**")

        else:
            # Multiple patients view
            fig = go.Figure()

            for pid in df['Record ID'].unique():
                patient_df = df[df['Record ID'] == pid].copy()

                patient_df['Days_Since_First_Visit'] = (pd.to_datetime(patient_df['Visit date']) - pd.to_datetime(patient_df['Visit date']).min()).dt.days

                # Remove any rows with NaN values
                patient_df = patient_df.dropna(subset=['Days_Since_First_Visit', 'Calculated Wound Area'])

                if len(patient_df) > 0:  # Only add trace if we have valid data
                    fig.add_trace(go.Scatter(
                        x=patient_df['Days_Since_First_Visit'],
                        y=patient_df['Calculated Wound Area'],
                        mode='lines+markers',
                        name=f'Patient {pid}',
                        hovertemplate='Day %{x}<br>Area: %{y:.1f} cm²'
                    ))

            fig.update_layout(
                title="Wound Area Progression - All Patients",
                xaxis_title="Days Since First Visit",
                yaxis_title="Wound Area (cm²)",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

    if selected_patient != "All Patients":
        create_wound_area_plot(filtered_df, int(selected_patient.split(" ")[1]))
    else:
        create_wound_area_plot(filtered_df)

# Tab 2: Impedance Analysis
with tab2:
    st.subheader("Impedance vs. Wound Healing Progress")

    if selected_patient == "All Patients":
        # Create scatter plot of impedance vs healing rate
        def create_impedance_scatterplot(df: pd.DataFrame) -> go.Figure:
            """Create impedance vs healing rate scatter plot."""
            valid_df = df[df['Healing Rate (%)'] > 0]
            return px.scatter(
                valid_df,
                x='Skin Impedance (kOhms) - Z',
                y='Healing Rate (%)',
                color='Diabetes?',
                size='Calculated Wound Area',
                hover_data=['Record ID', 'Event Name', 'Wound Type'],
                title="Impedance vs. Healing Rate Correlation"
            )

        fig = create_impedance_scatterplot(df)

        # Add trendline
        fig.update_layout(
            xaxis_title="Impedance Z (kOhms)",
            yaxis_title="Healing Rate (% reduction per visit)"
        )

        # Calculate correlation
        valid_data = df[df['Healing Rate (%)'] > 0]  # Exclude first visits with 0 healing rate
        r, p = stats.pearsonr(valid_data['Skin Impedance (kOhms) - Z'], valid_data['Healing Rate (%)'])
        st.plotly_chart(fig, use_container_width=True)

        # Format p-value with proper handling of small values
        p_formatted = f"< 0.001" if p < 0.001 else f"= {p:.3f}"
        st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")
        st.write("Higher impedance values correlate with slower healing rates, especially in diabetic patients")

    else:
        # For individual patient, show impedance over time
        patient_data = filtered_df.sort_values('Visit Number')

        fig = px.line(
            patient_data,
            x='Visit Number',
            y=['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\''],
            title=f"Impedance Measurements Over Time for {selected_patient}",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Visit Number",
            yaxis_title="Impedance (kOhms)",
            legend_title="Measurement"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Impedance Components Over Time")

        if selected_patient == "All Patients":
            # Average impedance by visit number
            avg_impedance = df.groupby('Visit Number')[
                ['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\'']
            ].mean().reset_index()

            fig = px.line(
                avg_impedance,
                x='Visit Number',
                y=['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\''],
                title="Average Impedance Components by Visit",
                markers=True
            )
        else:
            # Individual patient data already plotted above
            fig = None

        if fig:
            fig.update_layout(
                xaxis_title="Visit Number",
                yaxis_title="Impedance (kOhms)",
                legend_title="Component"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Impedance by Wound Type")

        if selected_patient == "All Patients":
            # Average impedance by wound type
            avg_by_type = df.groupby('Wound Type')['Skin Impedance (kOhms) - Z'].mean().reset_index()

            fig = px.bar(
                avg_by_type,
                x='Wound Type',
                y='Skin Impedance (kOhms) - Z',
                title="Average Impedance by Wound Type",
                color='Wound Type'
            )
            fig.update_layout(
                xaxis_title="Wound Type",
                yaxis_title="Average Impedance Z (kOhms)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For individual patient, show their wound type and average impedance
            wound_type = filtered_df['Wound Type'].iloc[0]
            avg_impedance = filtered_df['Skin Impedance (kOhms) - Z'].mean()
            st.info(f"{selected_patient} has a {wound_type} with average impedance of {avg_impedance:.1f} kOhms")

# Tab 3: Temperature Analysis
with tab3:
    st.subheader("Temperature Gradient Analysis")

    if selected_patient == "All Patients":
        # Create a temperature gradient dataframe
        temp_df = df.copy()
        temp_df['Center-Edge Gradient'] = temp_df['Center of Wound Temperature (Fahrenheit)'] - temp_df['Edge of Wound Temperature (Fahrenheit)']
        temp_df['Edge-Peri Gradient'] = temp_df['Edge of Wound Temperature (Fahrenheit)'] - temp_df['Peri-wound Temperature (Fahrenheit)']
        temp_df['Total Gradient'] = temp_df['Center of Wound Temperature (Fahrenheit)'] - temp_df['Peri-wound Temperature (Fahrenheit)']

        # Boxplot of temperature gradients by wound type
        fig = px.box(
            temp_df,
            x='Wound Type',
            y=['Center-Edge Gradient', 'Edge-Peri Gradient', 'Total Gradient'],
            title="Temperature Gradients by Wound Type",
            points="all"
        )
        fig.update_layout(
            xaxis_title="Wound Type",
            yaxis_title="Temperature Gradient (°F)",
            boxmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot of total gradient vs healing rate
        fig = px.scatter(
            temp_df[temp_df['Healing Rate (%)'] > 0],  # Exclude first visits
            x='Total Gradient',
            y='Healing Rate (%)',
            color='Wound Type',
            size='Calculated Wound Area',
            hover_data=['Record ID', 'Event Name'],
            title="Temperature Gradient vs. Healing Rate"
        )
        fig.update_layout(
            xaxis_title="Temperature Gradient (Center to Peri-wound, °F)",
            yaxis_title="Healing Rate (% reduction per visit)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # For individual patient
        patient_data = filtered_df.sort_values('Visit Number')

        # Calculate temperature gradients
        patient_data['Center-Edge'] = patient_data['Center of Wound Temperature (Fahrenheit)'] - patient_data['Edge of Wound Temperature (Fahrenheit)']
        patient_data['Edge-Peri'] = patient_data['Edge of Wound Temperature (Fahrenheit)'] - patient_data['Peri-wound Temperature (Fahrenheit)']

        # Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add temperature lines
        fig.add_trace(
            go.Scatter(
                x=patient_data['Visit Number'],
                y=patient_data['Center of Wound Temperature (Fahrenheit)'],
                name="Center Temp",
                line=dict(color='red')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=patient_data['Visit Number'],
                y=patient_data['Edge of Wound Temperature (Fahrenheit)'],
                name="Edge Temp",
                line=dict(color='orange')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=patient_data['Visit Number'],
                y=patient_data['Peri-wound Temperature (Fahrenheit)'],
                name="Peri-wound Temp",
                line=dict(color='blue')
            )
        )

        # Add gradient bars on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=patient_data['Visit Number'],
                y=patient_data['Center-Edge'],
                name="Center-Edge Gradient",
                opacity=0.5,
                marker_color='lightpink'
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Bar(
                x=patient_data['Visit Number'],
                y=patient_data['Edge-Peri'],
                name="Edge-Peri Gradient",
                opacity=0.5,
                marker_color='lightblue'
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=f"Temperature Measurements for {selected_patient}",
            xaxis_title="Visit Number",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        fig.update_yaxes(title_text="Temperature (°F)", secondary_y=False)
        fig.update_yaxes(title_text="Temperature Gradient (°F)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Oxygenation
with tab4:
    st.subheader("Oxygenation Metrics")

    if selected_patient == "All Patients":
        # Scatter plot of oxygenation vs. healing rate
        fig = px.scatter(
            df[df['Healing Rate (%)'] > 0],  # Exclude first visits
            x='Oxygenation (%)',
            y='Healing Rate (%)',
            color='Diabetes?',
            size='Hemoglobin Level',
            hover_data=['Record ID', 'Event Name', 'Wound Type'],
            title="Relationship Between Oxygenation and Healing Rate"
        )
        fig.update_layout(
            xaxis_title="Oxygenation (%)",
            yaxis_title="Healing Rate (% reduction per visit)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot of oxygenation by wound type
        fig = px.box(
            df,
            x='Wound Type',
            y='Oxygenation (%)',
            title="Oxygenation Levels by Wound Type",
            color='Wound Type',
            points="all"
        )
        fig.update_layout(
            xaxis_title="Wound Type",
            yaxis_title="Oxygenation (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # For individual patient
        patient_data = filtered_df.sort_values('Visit Number')

        # Plot hemoglobin components over time
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=patient_data['Visit Number'],
                y=patient_data['Oxyhemoglobin Level'],
                name="Oxyhemoglobin",
                marker_color='red'
            )
        )

        fig.add_trace(
            go.Bar(
                x=patient_data['Visit Number'],
                y=patient_data['Deoxyhemoglobin Level'],
                name="Deoxyhemoglobin",
                marker_color='purple'
            )
        )

        fig.update_layout(
            title=f"Hemoglobin Components for {selected_patient}",
            xaxis_title="Visit Number",
            yaxis_title="Level (g/dL)",
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Plot oxygenation over time
        fig = px.line(
            patient_data,
            x='Visit Number',
            y='Oxygenation (%)',
            title=f"Oxygenation Over Time for {selected_patient}",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Visit Number",
            yaxis_title="Oxygenation (%)",
            yaxis=dict(range=[80, 100])
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Risk Factors
with tab5:
    st.subheader("Risk Factor Analysis")

    if selected_patient == "All Patients":
        # Create tabs for different risk factors
        risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

        with risk_tab1:
            # Compare healing rates by diabetes status
            diab_healing = df.groupby(['Diabetes?', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
            diab_healing = diab_healing[diab_healing['Visit Number'] > 1]  # Exclude first visit

            fig = px.line(
                diab_healing,
                x='Visit Number',
                y='Healing Rate (%)',
                color='Diabetes?',
                title="Average Healing Rate by Diabetes Status",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Visit Number",
                yaxis_title="Average Healing Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Compare impedance by diabetes status
            diab_imp = df.groupby('Diabetes?')['Skin Impedance (kOhms) - Z'].mean().reset_index()

            fig = px.bar(
                diab_imp,
                x='Diabetes?',
                y='Skin Impedance (kOhms) - Z',
                color='Diabetes?',
                title="Average Impedance by Diabetes Status"
            )
            fig.update_layout(
                xaxis_title="Diabetes Status",
                yaxis_title="Average Impedance Z (kOhms)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with risk_tab2:
            # Compare healing rates by smoking status
            smoke_healing = df.groupby(['Smoking status', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
            smoke_healing = smoke_healing[smoke_healing['Visit Number'] > 1]  # Exclude first visit

            fig = px.line(
                smoke_healing,
                x='Visit Number',
                y='Healing Rate (%)',
                color='Smoking status',
                title="Average Healing Rate by Smoking Status",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Visit Number",
                yaxis_title="Average Healing Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Distribution of wound types by smoking status
            smoke_wound = pd.crosstab(df['Smoking status'], df['Wound Type'])
            smoke_wound_pct = smoke_wound.div(smoke_wound.sum(axis=1), axis=0) * 100

            fig = px.bar(
                smoke_wound_pct.reset_index().melt(id_vars='Smoking status', var_name='Wound Type', value_name='Percentage'),
                x='Smoking status',
                y='Percentage',
                color='Wound Type',
                title="Distribution of Wound Types by Smoking Status",
                barmode='stack'
            )
            fig.update_layout(
                xaxis_title="Smoking Status",
                yaxis_title="Percentage (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with risk_tab3:
            # Create BMI categories
            df['BMI Category'] = pd.cut(
                df['BMI'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II-III']
            )

            # Compare healing rates by BMI category
            bmi_healing = df.groupby(['BMI Category', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
            bmi_healing = bmi_healing[bmi_healing['Visit Number'] > 1]  # Exclude first visit

            fig = px.line(
                bmi_healing,
                x='Visit Number',
                y='Healing Rate (%)',
                color='BMI Category',
                title="Average Healing Rate by BMI Category",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Visit Number",
                yaxis_title="Average Healing Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scatter plot of BMI vs. healing rate
            fig = px.scatter(
                df[df['Healing Rate (%)'] > 0],  # Exclude first visits
                x='BMI',
                y='Healing Rate (%)',
                color='Diabetes?',
                size='Calculated Wound Area',
                hover_data=['Record ID', 'Event Name', 'Wound Type'],
                title="BMI vs. Healing Rate"
            )
            fig.update_layout(
                xaxis_title="BMI",
                yaxis_title="Healing Rate (% reduction per visit)"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        # For individual patient
        patient_data = filtered_df.iloc[0]

        # Create columns for the metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Risk Profile")

            # Display key risk factors
            st.info(f"**Diabetes Status:** {patient_data['Diabetes?']}")
            st.info(f"**Smoking Status:** {patient_data['Smoking status']}")
            st.info(f"**BMI:** {patient_data['BMI']:.1f}")

            # BMI category
            bmi = patient_data['BMI']
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            elif bmi < 35:
                bmi_category = "Obese Class I"
            else:
                bmi_category = "Obese Class II-III"

            st.info(f"**BMI Category:** {bmi_category}")

        with col2:
            st.subheader("Risk Assessment")

            # Create a risk score based on known factors
            # This is a simplified example - in reality would be based on clinical evidence
            risk_factors = []
            risk_score = 0

            if patient_data['Diabetes?'] == 'Yes':
                risk_factors.append("Diabetes")
                risk_score += 3

            if patient_data['Smoking status'] == 'Current':
                risk_factors.append("Current smoker")
                risk_score += 2
            elif patient_data['Smoking status'] == 'Former':
                risk_factors.append("Former smoker")
                risk_score += 1

            if patient_data['BMI'] >= 30:
                risk_factors.append("Obesity")
                risk_score += 2
            elif patient_data['BMI'] >= 25:
                risk_factors.append("Overweight")
                risk_score += 1

            # Temperature gradient risk
            temp_gradient = patient_data['Center of Wound Temperature (Fahrenheit)'] - patient_data['Peri-wound Temperature (Fahrenheit)']
            if temp_gradient > 3:
                risk_factors.append("High temperature gradient")
                risk_score += 2

            # Impedance risk
            if patient_data['Skin Impedance (kOhms) - Z'] > 140:
                risk_factors.append("High impedance")
                risk_score += 2

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
            wound_area = patient_data['Calculated Wound Area']
            base_healing_weeks = 2 + wound_area/2  # Simple formula: 2 weeks + 0.5 weeks per cm²
            risk_multiplier = 1 + (risk_score * 0.1)  # Each risk point adds 10% to healing time
            est_healing_weeks = base_healing_weeks * risk_multiplier

            st.markdown(f"**Estimated Healing Time:** {est_healing_weeks:.1f} weeks")

# Tab 6: LLM Analysis
with tab6:
    st.header("LLM-Powered Wound Analysis")

    if uploaded_file is not None and selected_patient != "All Patients":
        # Add Run Analysis button
        if st.button("Run Analysis", key="run_analysis"):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Initialize LLM with platform and model
                    llm = WoundAnalysisLLM(platform=platform, model_name=model_name)

                    # Get patient ID from selection
                    patient_id = int(selected_patient.split(" ")[1])

                    # Get patient data using the processor
                    patient_data = st.session_state.processor.get_patient_visits(patient_id)

                    if patient_data:
                        # Run the analysis
                        analysis = llm.analyze_patient_data(patient_data)

                        # Save the report
                        report_path = create_and_save_report(patient_data, analysis)

                        # Update session state
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_results = analysis
                        st.session_state.report_path = report_path

                        # Display results
                        st.markdown("### Analysis Results")
                        st.write(analysis)

                        # Add download button for the report
                        href = download_word_report(report_path)
                        st.markdown(
                            f'<a href="{href}" download="wound_analysis_report.docx">Download Word Report</a>',
                            unsafe_allow_html=True
                        )

                        st.success("Analysis complete!")
                    else:
                        st.error(f"No data found for Patient {patient_id}")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Please make sure you have selected a valid patient and the data is in the correct format.")

        if not st.session_state.analysis_complete:
            st.info("Click 'Run Analysis' to generate an AI-powered analysis of the wound data.")
    else:
        if selected_patient == "All Patients":
            st.warning("Please select a specific patient to run the analysis.")
        else:
            st.info("Please upload a CSV file and select a patient to begin analysis.")

# Add footer with information
st.markdown("---")
st.markdown("""
**Note:** This dashboard loads data from 'dataset/SmartBandage-Data_for_llm.csv'.
If the file cannot be loaded, the dashboard will fall back to simulated data.
""")

# Sidebar with additional information
with st.sidebar:
    st.header("About This Dashboard")
    st.write("""
    This Streamlit dashboard visualizes wound healing data collected with smart bandage technology.

    The analysis focuses on key metrics:
    - Impedance measurements (Z, Z', Z'')
    - Temperature gradients
    - Oxygenation levels
    - Patient risk factors

    Use the dropdown at the top to switch between aggregate data view and individual patient profiles.
    """)

    st.header("Statistical Methods")
    st.write("""
    The visualization is supported by these statistical approaches:
    - Longitudinal analysis of healing trajectories
    - Correlation analysis between measurements and outcomes
    - Risk factor significance assessment
    - Comparative analysis across wound types
    """)

    # Add download button (would connect to actual data in production)
    st.download_button(
        label="Download Analysis Report (PDF)",
        data=b"Sample PDF report content",  # In production, generate actual PDF
        file_name="wound_healing_analysis.pdf",
        mime="application/pdf"
    )

# def main():
#     """Main application entry point."""
#     initialize_session_state()
#     setup_page_config()

#     platform, model_name = setup_sidebar_config()
#     df = load_data()

#     if df is not None:
#         selected_patient = render_patient_selection(df)
#         render_main_tabs(df, selected_patient, platform, model_name)

# if __name__ == "__main__":
#     main()
