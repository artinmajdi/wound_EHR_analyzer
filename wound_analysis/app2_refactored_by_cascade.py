"""
Smart Bandage Wound Healing Analytics Dashboard

A Streamlit-based dashboard for analyzing wound healing data from smart bandages.
Provides visualization and analysis of wound progression, healing rates, and various
biometric measurements.

Author: Cascade
Date: 2025-02-21
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import base64
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the application."""
    
    # File paths
    DATA_PATH = Path("/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv")
    REPORT_PATH = Path("wound_analysis_report.docx")
    
    # Plot settings
    PLOT_HEIGHT = 600
    PLOT_WIDTH = 800
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'error': '#d62728',
        'success': '#2ca02c',
        'temperature': {
            'center': 'red',
            'edge': 'orange',
            'peri': 'blue'
        }
    }
    
    # Column names
    NUMERIC_COLUMNS = [
        'Calculated Wound Area',
        'Total Temp Gradient',
        'Skin Impedance (kOhms) - Z',
        'Healing Rate (%)',
        'Oxygenation (%)',
        'Hemoglobin Level',
        'Oxyhemoglobin Level',
        'Deoxyhemoglobin Level',
        'BMI'
    ]
    
    TEMPERATURE_COLUMNS = [
        'Center of Wound Temperature (Fahrenheit)',
        'Edge of Wound Temperature (Fahrenheit)',
        'Peri-wound Temperature (Fahrenheit)'
    ]
    
    # Risk assessment settings
    RISK_SCORES = {
        'diabetes': 3,
        'smoking_current': 2,
        'smoking_former': 1,
        'obesity': 2,
        'overweight': 1,
        'high_temp_gradient': 2,
        'high_impedance': 2
    }
    
    RISK_THRESHOLDS = {
        'high': 6,
        'moderate': 3
    }
    
    # BMI categories
    BMI_CATEGORIES = [
        (0, 18.5, 'Underweight'),
        (18.5, 25, 'Normal'),
        (25, 30, 'Overweight'),
        (30, 35, 'Obese Class I'),
        (35, float('inf'), 'Obese Class II-III')
    ]
    
    # Report settings
    REPORT_TITLE_STYLE = {
        'font_name': 'Calibri',
        'font_size': 16,
        'bold': True
    }
    
    REPORT_HEADING_STYLE = {
        'font_name': 'Calibri',
        'font_size': 14,
        'bold': True
    }
    
    REPORT_BODY_STYLE = {
        'font_name': 'Calibri',
        'font_size': 11
    }
    
class DataLoader:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_data() -> pd.DataFrame:
        """Load and preprocess the wound healing data.
        
        Returns:
            pd.DataFrame: Processed wound healing data
        """
        try:
            logger.info("Loading data from CSV file...")
            if not Config.DATA_PATH.exists():
                raise FileNotFoundError(f"Data file not found: {Config.DATA_PATH}")
            
            # Load the CSV file
            df = pd.read_csv(Config.DATA_PATH)
            df = DataLoader._preprocess_data(df)
            logger.info("Data loaded and preprocessed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        try:
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Extract visit number
            df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)
            
            # Calculate wound area if missing
            if 'Calculated Wound Area' not in df.columns:
                if all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
                    df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']
            
            # Calculate healing rates
            df['Healing Rate (%)'] = DataLoader._calculate_healing_rates(df)
            
            # Convert dates
            if 'Visit date' in df.columns:
                df['Visit date'] = pd.to_datetime(df['Visit date'])
            
            # Sort by patient and visit
            df = df.sort_values(['Record ID', 'Visit Number'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    @staticmethod
    def _calculate_healing_rates(df: pd.DataFrame) -> pd.Series:
        """Calculate healing rates for each patient visit.
        
        Args:
            df: DataFrame with wound measurements
            
        Returns:
            pd.Series: Healing rates for each visit
        """
        healing_rates = []
        
        for patient_id in df['Record ID'].unique():
            patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')
            
            for _, row in patient_data.iterrows():
                if row['Visit Number'] == 1:
                    healing_rates.append(0)  # No healing rate for first visit
                    continue
                    
                prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
                if prev_visits.empty:
                    healing_rates.append(0)
                    continue
                    
                prev_visit = prev_visits.iloc[-1]  # Get most recent previous visit
                
                try:
                    prev_area = prev_visit['Calculated Wound Area']
                    curr_area = row['Calculated Wound Area']
                    
                    if pd.isna(prev_area) or pd.isna(curr_area) or prev_area <= 0:
                        healing_rates.append(0)
                    else:
                        healing_rate = ((prev_area - curr_area) / prev_area) * 100
                        healing_rates.append(healing_rate)
                except Exception as e:
                    logger.warning(f"Error calculating healing rate: {str(e)}")
                    healing_rates.append(0)
        
        return pd.Series(healing_rates, index=df.index)

def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None):
    """Create and display a wound area progression plot.
    If patient_id is provided, filter the data accordingly.
    """
    if patient_id is not None:
        df_plot = df[df['Record ID'] == patient_id]
    else:
        df_plot = df

    if df_plot.empty or 'Calculated Wound Area' not in df_plot.columns:
        st.info("Wound area data not available.")
        return

    fig = px.line(df_plot, x='Visit Number', y='Calculated Wound Area', title="Wound Area Progression")
    st.plotly_chart(fig)


def create_impedance_scatterplot(df: pd.DataFrame):
    """Create a scatter plot of Skin Impedance vs Healing Rate with trendline."""
    if 'Skin Impedance (kOhms) - Z' not in df.columns or 'Healing Rate (%)' not in df.columns:
        st.info("Impedance or Healing Rate data not available.")
        return None
    
    fig = px.scatter(df, x='Skin Impedance (kOhms) - Z', y='Healing Rate (%)', trendline='ols', 
                     title="Impedance vs Healing Rate")
    return fig


def create_temperature_plot(df: pd.DataFrame):
    """Create and display a temperature trend plot using the first temperature column."""
    temp_col = Config.TEMPERATURE_COLUMNS[0] if Config.TEMPERATURE_COLUMNS else None
    if not temp_col or temp_col not in df.columns:
        st.info("Temperature data not available.")
        return
    
    fig = px.line(df, x='Visit Number', y=temp_col, title="Temperature Trend")
    st.plotly_chart(fig)


def create_oxygenation_plot(df: pd.DataFrame):
    """Create and return a scatter plot of Oxygenation Metrics if available."""
    if 'Oxygenation (%)' not in df.columns:
        st.info("Oxygenation data not available.")
        return None
    
    fig = px.scatter(df, x='Visit Number', y='Oxygenation (%)', title="Oxygenation Metrics")
    return fig

class Dashboard:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.setup_page()
        self.initialize_session_state()
        self.load_data()
    
    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Smart Bandage Wound Healing Analytics",
            page_icon="🩹",
            layout="wide"
        )
        st.title("Smart Bandage Wound Healing Analytics")
        st.markdown("""
        This dashboard provides comprehensive analysis of wound healing data collected 
        from smart bandages. Monitor wound progression, healing rates, and various 
        biometric measurements.
        """)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'selected_patient' not in st.session_state:
            st.session_state.selected_patient = "All Patients"
    
    def load_data(self):
        """Load and store the data in session state."""
        self.df = DataLoader.load_data()
        if not self.df.empty:
            st.session_state.data = self.df
    
    def run(self):
        """Run the dashboard application."""
        if self.df.empty:
            st.error("No data available. Please check the data source.")
            return
        
        # Sidebar filters
        self.create_sidebar()
        
        # Create tabs
        tabs = st.tabs([
            "Overview",
            "Wound Progression",
            "Temperature Analysis",
            "Impedance Analysis",
            "Oxygenation Metrics",
            "Risk Factors",
            "LLM Analysis"
        ])
        
        # Populate tabs
        with tabs[0]:
            self.show_overview_tab()
        with tabs[1]:
            self.show_wound_progression_tab()
        with tabs[2]:
            self.show_temperature_tab()
        with tabs[3]:
            self.show_impedance_tab()
        with tabs[4]:
            self.show_oxygenation_tab()
        with tabs[5]:
            self.show_risk_factors_tab()
        with tabs[6]:
            self.show_llm_analysis_tab()
    
    def create_sidebar(self):
        """Create the sidebar with filters and controls."""
        with st.sidebar:
            st.header("Filters")
            # Placeholder for patient selection and other filters
            st.selectbox("Select Patient", ["All Patients"] + 
                         [f"Patient {pid}" for pid in sorted(self.df['Record ID'].unique())] if not self.df.empty else ["All Patients"], key='selected_patient')
    
    def calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics for the dataset."""
        stats = {
            'total_patients': len(self.df['Record ID'].unique()) if 'Record ID' in self.df.columns else 0,
            'total_measurements': len(self.df),
            'avg_wound_area': self.df['Calculated Wound Area'].mean() if 'Calculated Wound Area' in self.df.columns else 0,
            'avg_healing_rate': self.df['Healing Rate (%)'].mean() if 'Healing Rate (%)' in self.df.columns else 0,
            'diabetic_patients': len(self.df[self.df['Diabetes?'] == 'Yes']['Record ID'].unique()) if 'Diabetes?' in self.df.columns and 'Record ID' in self.df.columns else 0,
            'non_diabetic_patients': len(self.df[self.df['Diabetes?'] == 'No']['Record ID'].unique()) if 'Diabetes?' in self.df.columns and 'Record ID' in self.df.columns else 0
        }
        return stats

    def calculate_patient_healing_stats(self, patient_id: int) -> Dict:
        """Calculate healing statistics for a specific patient."""
        patient_data = self.df[self.df['Record ID'] == patient_id].sort_values('Visit Number')
        if patient_data.empty:
            return {}
        initial_area = patient_data.iloc[0]['Calculated Wound Area'] if 'Calculated Wound Area' in patient_data.columns else 0
        current_area = patient_data.iloc[-1]['Calculated Wound Area'] if 'Calculated Wound Area' in patient_data.columns else 0
        healing_progress = ((initial_area - current_area) / initial_area * 100) if initial_area > 0 else 0
        avg_healing_rate = patient_data['Healing Rate (%)'].mean() if 'Healing Rate (%)' in patient_data.columns else 0
        stats = {
            'initial_area': initial_area,
            'current_area': current_area,
            'healing_progress': healing_progress,
            'avg_healing_rate': avg_healing_rate
        }
        return stats

    def show_population_risk_factors(self):
        """Display risk factor analysis for the entire population."""
        risk_factors = ['Diabetes?', 'Smoking status']
        for factor in risk_factors:
            if factor in self.df.columns:
                dist = self.df[factor].value_counts()
                fig = go.Figure(data=[go.Bar(
                    x=dist.index,
                    y=dist.values,
                    text=dist.values,
                    textposition='auto'
                )])
                fig.update_layout(
                    title=f"{factor} Distribution",
                    xaxis_title=factor,
                    yaxis_title="Number of Patients",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    def show_patient_risk_factors(self):
        """Display risk factors for a specific patient."""
        try:
            patient_id = int(st.session_state.selected_patient.split(" ")[1])
        except Exception:
            st.error("Invalid patient selection")
            return
        patient_data = self.df[self.df['Record ID'] == patient_id].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "Yes" if patient_data.get('Diabetes?') == 'Yes' else "No"
            st.metric("Diabetes Status", status)
        with col2:
            smoking = patient_data.get('Smoking status')
            # Assuming 'Current' means yes
            status_smoke = "Yes" if smoking and smoking.lower() in ['yes', 'current'] else "No"
            st.metric("Smoking Status", status_smoke)
        with col3:
            medical_history = patient_data.get('Medical History (select all that apply)', 'N/A')
            st.metric("Medical Conditions", str(medical_history))

    def show_overview_tab(self):
        """Display the Overview tab with data table and summary metrics."""
        st.subheader("Overview")
        if self.df.empty:
            st.info("No data to display.")
            return
        st.dataframe(self.df.head(10))
        st.markdown("### Summary Statistics")
        stats = self.calculate_summary_statistics()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", stats.get('total_patients', 0))
            st.metric("Average Wound Area", f"{stats.get('avg_wound_area', 0):.2f} cm²")
        with col2:
            st.metric("Average Healing Rate", f"{stats.get('avg_healing_rate', 0):.2f}%")
            st.metric("Total Measurements", stats.get('total_measurements', 0))
        with col3:
            st.metric("Diabetic Patients", stats.get('diabetic_patients', 0))
            st.metric("Non-diabetic Patients", stats.get('non_diabetic_patients', 0))
        
        # If an individual patient is selected, show patient-specific healing stats
        if st.session_state.selected_patient != "All Patients":
            try:
                patient_id = int(st.session_state.selected_patient.split(" ")[1])
                patient_stats = self.calculate_patient_healing_stats(patient_id)
                st.markdown("### Patient Healing Statistics")
                st.markdown(f"**Initial Wound Area:** {patient_stats.get('initial_area', 0):.2f} cm²")
                st.markdown(f"**Current Wound Area:** {patient_stats.get('current_area', 0):.2f} cm²")
                st.markdown(f"**Healing Progress:** {patient_stats.get('healing_progress', 0):.1f}%")
                st.markdown(f"**Average Healing Rate:** {patient_stats.get('avg_healing_rate', 0):.2f}% per visit")
            except Exception as e:
                st.error(f"Error calculating patient healing statistics: {e}")

    def show_wound_progression_tab(self):
        """Display the Wound Progression tab with area plots."""
        st.subheader("Wound Progression")
        selected = st.session_state.selected_patient
        if selected != "All Patients":
            try:
                # Extract patient id from string e.g., 'Patient 123'
                patient_id = int(selected.split(" ")[1])
            except Exception:
                patient_id = None
            create_wound_area_plot(self.df, patient_id)
        else:
            create_wound_area_plot(self.df)
    
    def show_temperature_tab(self):
        """Display the Temperature Analysis tab."""
        st.subheader("Temperature Analysis")
        create_temperature_plot(self.df)
    
    def show_impedance_tab(self):
        """Display the Impedance Analysis tab."""
        st.subheader("Impedance Analysis")
        fig = create_impedance_scatterplot(self.df)
        if fig:
            st.plotly_chart(fig)
    
    def show_oxygenation_tab(self):
        """Display the Oxygenation Metrics tab."""
        st.subheader("Oxygenation Metrics")
        fig = create_oxygenation_plot(self.df)
        if fig:
            st.plotly_chart(fig)
    
    def show_risk_factors_tab(self):
        """Display the Risk Factors Analysis tab."""
        st.subheader("Risk Factors")
        if st.session_state.selected_patient == "All Patients":
            self.show_population_risk_factors()
        else:
            self.show_patient_risk_factors()

    def show_llm_analysis_tab(self):
        """Display the LLM-Powered Wound Analysis tab."""
        st.header("LLM-Powered Wound Analysis")
        uploaded_file = st.file_uploader("Upload patient data file for LLM analysis")
        if uploaded_file is not None:
            # Placeholder: Process the uploaded file into a patient_data dict
            patient_data = {}  # Replace with actual file processing logic
            try:
                analysis_manager = AnalysisManager(platform='openai', model_name='gpt-4')
                analysis, report_path = analysis_manager.run_analysis(patient_data)
                st.success("Analysis complete")
                st.write(analysis)
                download_link = DocumentHandler.download_word_report(report_path)
                st.markdown(f"[Download Report]({download_link})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during LLM analysis: {e}")
        else:
            st.info("Please upload a patient data file to begin LLM analysis.")

class DocumentHandler:
    """Handles document creation and report generation."""
    
    @staticmethod
    def create_and_save_report(patient_data: dict, analysis_results: str) -> str:
        """Create and save the analysis report as a Word document.
        
        Args:
            patient_data: Dictionary containing patient data
            analysis_results: String containing analysis results
            
        Returns:
            str: Path to the saved report
        """
        try:
            doc = Document()
            format_word_document(doc, patient_data, analysis_results)
            
            # Save the document
            doc.save(Config.REPORT_PATH)
            return str(Config.REPORT_PATH)
            
        except Exception as e:
            logger.error(f"Error creating report: {str(e)}")
            raise
    
    @staticmethod
    def download_word_report(report_path: str) -> str:
        """Create a download link for the Word report.
        
        Args:
            report_path: Path to the Word document
            
        Returns:
            str: Base64 encoded document data as HTML href
        """
        try:
            with open(report_path, "rb") as file:
                bytes_data = file.read()
            
            b64_data = base64.b64encode(bytes_data).decode()
            href = f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_data}'
            
            return href
            
        except Exception as e:
            logger.error(f"Error creating download link: {str(e)}")
            raise

class RiskAnalyzer:
    """Handles risk analysis and scoring."""
    
    @staticmethod
    def calculate_risk_score(patient_data: pd.Series) -> Tuple[int, List[str], str]:
        """Calculate risk score and identify risk factors for a patient.
        
        Args:
            patient_data: Series containing patient data
            
        Returns:
            Tuple containing:
                - Risk score (int)
                - List of risk factors (List[str])
                - Risk category (str)
        """
        risk_factors = []
        risk_score = 0
        
        # Check diabetes
        if patient_data.get('Diabetes?') == 'Yes':
            risk_factors.append("Diabetes")
            risk_score += Config.RISK_SCORES['diabetes']
        
        # Check smoking status
        if patient_data.get('Smoking status') == 'Current':
            risk_factors.append("Current smoker")
            risk_score += Config.RISK_SCORES['smoking_current']
        elif patient_data.get('Smoking status') == 'Former':
            risk_factors.append("Former smoker")
            risk_score += Config.RISK_SCORES['smoking_former']
        
        # Check BMI
        bmi = patient_data.get('BMI', 0)
        if bmi >= 30:
            risk_factors.append("Obesity")
            risk_score += Config.RISK_SCORES['obesity']
        elif bmi >= 25:
            risk_factors.append("Overweight")
            risk_score += Config.RISK_SCORES['overweight']
        
        # Check temperature gradient
        if all(col in patient_data.index for col in Config.TEMPERATURE_COLUMNS):
            temp_gradient = (
                patient_data['Center of Wound Temperature (Fahrenheit)'] - 
                patient_data['Peri-wound Temperature (Fahrenheit)']
            )
            if temp_gradient > 3:
                risk_factors.append("High temperature gradient")
                risk_score += Config.RISK_SCORES['high_temp_gradient']
        
        # Check impedance
        if 'Skin Impedance (kOhms) - Z' in patient_data.index:
            if patient_data['Skin Impedance (kOhms) - Z'] > 140:
                risk_factors.append("High impedance")
                risk_score += Config.RISK_SCORES['high_impedance']
        
        # Calculate risk category
        if risk_score >= Config.RISK_THRESHOLDS['high']:
            risk_category = "High"
        elif risk_score >= Config.RISK_THRESHOLDS['moderate']:
            risk_category = "Moderate"
        else:
            risk_category = "Low"
        
        return risk_score, risk_factors, risk_category
    
    @staticmethod
    def estimate_healing_time(wound_area: float, risk_score: int) -> float:
        """Estimate healing time based on wound area and risk score.
        
        Args:
            wound_area: Current wound area in cm²
            risk_score: Patient's risk score
            
        Returns:
            float: Estimated healing time in weeks
        """
        base_healing_weeks = 2 + wound_area/2  # Base: 2 weeks + 0.5 weeks per cm²
        risk_multiplier = 1 + (risk_score * 0.1)  # Each risk point adds 10%
        return base_healing_weeks * risk_multiplier
    
    @staticmethod
    def get_bmi_category(bmi: float) -> str:
        """Get BMI category from BMI value.
        
        Args:
            bmi: BMI value
            
        Returns:
            str: BMI category
        """
        for min_bmi, max_bmi, category in Config.BMI_CATEGORIES:
            if min_bmi <= bmi < max_bmi:
                return category
        return "Unknown"

class AnalysisManager:
    """Manages the LLM-based analysis workflow."""
    
    def __init__(self, platform: str, model_name: str):
        """Initialize with platform and model settings.
        
        Args:
            platform: AI platform to use
            model_name: Name of the model to use
        """
        self.platform = platform
        self.model_name = model_name
        self.llm = None
    
    def initialize_llm(self):
        """Initialize the LLM interface."""
        try:
            self.llm = WoundAnalysisLLM(
                platform=self.platform,
                model_name=self.model_name
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def run_analysis(self, patient_data: dict) -> Tuple[str, str]:
        """Run analysis and generate report.
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            Tuple containing:
                - Analysis results (str)
                - Path to report (str)
        """
        try:
            if self.llm is None:
                self.initialize_llm()
            
            # Run the analysis
            analysis = self.llm.analyze_patient_data(patient_data)
            
            # Generate report
            report_path = DocumentHandler.create_and_save_report(
                patient_data,
                analysis
            )
            
            return analysis, report_path
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise

def main():
    """Main application entry point."""
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred while running the application. Please check the logs for details.")

if __name__ == "__main__":
    main()
