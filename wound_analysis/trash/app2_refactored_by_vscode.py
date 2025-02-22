import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List
import pathlib
from docx import Document
import base64
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document

class Config:
    """Application configuration settings."""
    PAGE_TITLE = "Smart Bandage Wound Healing Analytics"
    PAGE_ICON = "🩹"
    LAYOUT = "wide"
    DATA_PATH = pathlib.Path(__file__).parent / "dataset" / "SmartBandage-Data_for_llm.csv"

class DataLoader:
    """Data loading and preprocessing functionality."""
    @staticmethod
    @st.cache_data
    def load_data() -> Optional[pd.DataFrame]:
        """Load and preprocess wound data from CSV."""
        try:
            if not Config.DATA_PATH.exists():
                st.error(f"Data file not found: {Config.DATA_PATH}")
                return None

            # Load the CSV file
            df = pd.read_csv(Config.DATA_PATH)

            # Process the data
            df = DataLoader._preprocess_data(df)

            return df

        except pd.errors.EmptyDataError:
            st.error("The data file is empty")
            return None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    @staticmethod
    def _clean_numeric_value(x) -> float:
        """Clean numeric values that might be percentages."""
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            x = x.strip()
            if x.endswith('%'):
                return float(x.rstrip('%'))
            try:
                return float(x)
            except ValueError:
                return 0.0
        return float(x)

    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the wound data."""
        try:
            # Clean column names
            df.columns = df.columns.str.strip()

            # Handle missing values
            df = df.replace(['', 'NA', 'N/A', 'null', 'NULL'], np.nan)

            # Extract visit number from Event Name
            df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

            # Clean numeric columns first
            numeric_cols = [col for col in df.columns if any(x in col.lower() for x in ['area', 'rate', 'temperature', 'impedance', 'level', 'length', 'width'])]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].apply(DataLoader._clean_numeric_value)

            # Calculate wound area if needed
            if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
                df['Length (cm)'] = pd.to_numeric(df['Length (cm)'], errors='coerce')
                df['Width (cm)'] = pd.to_numeric(df['Width (cm)'], errors='coerce')
                df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

            # Calculate healing rates after cleaning numeric values
            df['Healing Rate (%)'] = DataLoader._calculate_healing_rates(df)

            # Handle missing values in numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())

            # Calculate temperature gradients if available
            temp_cols = [
                'Center of Wound Temperature (Fahrenheit)',
                'Edge of Wound Temperature (Fahrenheit)',
                'Peri-wound Temperature (Fahrenheit)'
            ]

            if all(col in df.columns for col in temp_cols):
                df['Center-Edge Temp Gradient'] = df[temp_cols[0]] - df[temp_cols[1]]
                df['Edge-Peri Temp Gradient'] = df[temp_cols[1]] - df[temp_cols[2]]
                df['Total Temp Gradient'] = df[temp_cols[0]] - df[temp_cols[2]]

            return df

        except Exception as e:
            st.error(f"Error during data preprocessing: {str(e)}")
            return df

    @staticmethod
    def _calculate_healing_rates(df: pd.DataFrame) -> pd.Series:
        """Calculate healing rates for each patient visit."""
        healing_rates = pd.Series(index=df.index, dtype=float)

        for patient_id in df['Record ID'].unique():
            patient_df = df[df['Record ID'] == patient_id].sort_values('Visit Number')

            if len(patient_df) > 1:
                # Get initial wound area (already cleaned to numeric)
                initial_area = patient_df['Calculated Wound Area'].iloc[0]

                # Calculate healing rate only if we have valid initial area
                if pd.notnull(initial_area) and initial_area > 0:
                    current_areas = patient_df['Calculated Wound Area']
                    rates = ((initial_area - current_areas) / initial_area) * 100
                    healing_rates.loc[patient_df.index] = rates

        # Replace any remaining NaN values with 0
        healing_rates = healing_rates.fillna(0)
        return healing_rates

class SessionState:
    """Manage Streamlit session state."""
    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if 'processor' not in st.session_state:
            st.session_state.processor = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'report_path' not in st.session_state:
            st.session_state.report_path = None

class DocumentHandler:
    """Handle document operations."""
    @staticmethod
    def create_report(patient_data: Dict, analysis: str) -> str:
        """Create and save analysis report."""
        try:
            doc = Document()
            format_word_document(doc, analysis, patient_data)
            report_path = "wound_analysis_report.docx"
            doc.save(report_path)
            return report_path
        except Exception as e:
            st.error(f"Error creating report: {str(e)}")
            return None

    @staticmethod
    def get_download_link(report_path: str) -> str:
        """Generate download link for the report."""
        try:
            with open(report_path, "rb") as file:
                bytes_data = file.read()
            b64_data = base64.b64encode(bytes_data).decode()
            return f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_data}'
        except Exception as e:
            st.error(f"Error generating download link: {str(e)}")
            return None

class DataVisualizer:
    """Handle data visualization."""
    @staticmethod
    def plot_wound_measurements(df: pd.DataFrame, patient_id: int) -> None:
        """Create wound measurement visualizations."""
        try:
            patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')

            # Create subplots with additional temperature and oxygenation plots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Wound Dimensions', 'Wound Area', 'Temperature', 'Temperature Gradients', 'Oxygenation', 'Impedance'),
                vertical_spacing=0.12
            )

            # Wound dimensions plot
            fig.add_trace(
                go.Scatter(x=patient_data['Visit Number'], y=patient_data['Length (cm)'],
                          name='Length', line=dict(color='blue')), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=patient_data['Visit Number'], y=patient_data['Width (cm)'],
                          name='Width', line=dict(color='red')), row=1, col=1
            )

            # Wound area plot
            fig.add_trace(
                go.Scatter(x=patient_data['Visit Number'],
                          y=patient_data['Calculated Wound Area'],
                          name='Area', line=dict(color='green')), row=1, col=2
            )

            # Enhanced temperature plot with all gradients
            temp_cols = ['Center of Wound Temperature (Fahrenheit)',
                        'Edge of Wound Temperature (Fahrenheit)',
                        'Peri-wound Temperature (Fahrenheit)']
            for col, color in zip(temp_cols, ['red', 'orange', 'yellow']):
                if col in patient_data.columns:
                    fig.add_trace(
                        go.Scatter(x=patient_data['Visit Number'],
                                 y=patient_data[col],
                                 name=col.replace('Temperature (Fahrenheit)', ''),
                                 line=dict(color=color)), row=2, col=1
                    )

            # Add temperature gradients plot
            fig.add_trace(
                go.Bar(x=patient_data['Visit Number'],
                      y=patient_data['Center-Edge Temp Gradient'],
                      name='Center-Edge Gradient',
                      marker_color='lightpink'), row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=patient_data['Visit Number'],
                      y=patient_data['Edge-Peri Temp Gradient'],
                      name='Edge-Peri Gradient',
                      marker_color='lightblue'), row=2, col=2
            )

            # Add oxygenation plot
            if 'Oxygenation (%)' in patient_data.columns:
                fig.add_trace(
                    go.Scatter(x=patient_data['Visit Number'],
                             y=patient_data['Oxygenation (%)'],
                             name='Oxygenation',
                             line=dict(color='green')), row=3, col=1
                )

            # Enhanced impedance plot with Z' and Z''
            impedance_cols = ['Skin Impedance (kOhms) - Z',
                            'Skin Impedance (kOhms) - Z\'',
                            'Skin Impedance (kOhms) - Z\'\'']
            for col, color in zip(impedance_cols, ['purple', 'magenta', 'violet']):
                if col in patient_data.columns:
                    fig.add_trace(
                        go.Scatter(x=patient_data['Visit Number'],
                                 y=patient_data[col],
                                 name=col.replace('Skin Impedance (kOhms) - ', ''),
                                 line=dict(color=color)), row=3, col=2
                    )

            # Update layout with better spacing and titles
            fig.update_layout(
                height=1200,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")

class RiskAssessment:
    """Handle risk assessment calculations and analysis."""
    @staticmethod
    def calculate_risk_score(patient_data: pd.Series) -> tuple[int, list[str], str]:
        """Calculate risk score and identify risk factors."""
        risk_factors = []
        risk_score = 0

        # Medical history factors
        if patient_data['Diabetes?'] == 'Yes':
            risk_factors.append("Diabetes")
            risk_score += 3
        if patient_data['Smoking status'] == 'Current':
            risk_factors.append("Current smoker")
            risk_score += 2
        elif patient_data['Smoking status'] == 'Former':
            risk_factors.append("Former smoker")
            risk_score += 1

        # BMI risk
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
        elif risk_score >= 3:
            risk_category = "Moderate"
        else:
            risk_category = "Low"

        return risk_score, risk_factors, risk_category

    @staticmethod
    def estimate_healing_time(patient_data: pd.Series, risk_score: int) -> float:
        """Estimate healing time based on wound characteristics and risk factors."""
        wound_area = patient_data['Calculated Wound Area']
        base_healing_weeks = 2 + wound_area/2  # Base: 2 weeks + 0.5 weeks per cm²
        risk_multiplier = 1 + (risk_score * 0.1)  # Each risk point adds 10% to healing time
        return base_healing_weeks * risk_multiplier

    @staticmethod
    def display_risk_assessment(patient_data: pd.Series) -> None:
        """Display comprehensive risk assessment."""
        risk_score, risk_factors, risk_category = RiskAssessment.calculate_risk_score(patient_data)
        est_healing_time = RiskAssessment.estimate_healing_time(patient_data, risk_score)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patient Risk Profile")
            st.info(f"**Diabetes Status:** {patient_data['Diabetes?']}")
            st.info(f"**Smoking Status:** {patient_data['Smoking status']}")
            st.info(f"**BMI:** {patient_data['BMI']:.1f}")
            st.info(f"**Risk Category:** {risk_category} ({risk_score} points)")

        with col2:
            st.subheader("Risk Assessment")
            if risk_factors:
                st.markdown("**Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("**Risk Factors:** None identified")

            st.markdown(f"**Estimated Healing Time:** {est_healing_time:.1f} weeks")

            # Additional metrics
            if 'Oxyhemoglobin Level' in patient_data and 'Deoxyhemoglobin Level' in patient_data:
                st.markdown("**Blood Oxygenation Status:**")
                oxy_ratio = patient_data['Oxyhemoglobin Level'] / (patient_data['Oxyhemoglobin Level'] + patient_data['Deoxyhemoglobin Level'])
                st.progress(oxy_ratio, text=f"Oxygenation Ratio: {oxy_ratio:.2%}")

class StatisticalAnalysis:
    """Handle statistical analysis and correlations."""
    @staticmethod
    def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations between key measurements."""
        columns_of_interest = [
            'Calculated Wound Area',
            'Healing Rate (%)',
            'Skin Impedance (kOhms) - Z',
            'Center of Wound Temperature (Fahrenheit)',
            'Edge of Wound Temperature (Fahrenheit)',
            'Peri-wound Temperature (Fahrenheit)',
            'Oxygenation (%)',
            'BMI'
        ]
        return df[columns_of_interest].corr()

    @staticmethod
    def analyze_healing_trends(df: pd.DataFrame) -> dict:
        """Analyze healing trends across different patient groups."""
        trends = {
            'diabetes': df.groupby(['Diabetes?', 'Visit Number'])['Healing Rate (%)'].mean().unstack(),
            'smoking': df.groupby(['Smoking status', 'Visit Number'])['Healing Rate (%)'].mean().unstack(),
            'wound_type': df.groupby(['Wound Type', 'Visit Number'])['Healing Rate (%)'].mean().unstack()
        }
        return trends

    @staticmethod
    def display_statistical_analysis(df: pd.DataFrame) -> None:
        """Display comprehensive statistical analysis."""
        st.subheader("Statistical Analysis")

        # Correlation analysis
        corr = StatisticalAnalysis.calculate_correlations(df)
        fig = px.imshow(
            corr,
            title="Measurement Correlations",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Healing trends
        trends = StatisticalAnalysis.analyze_healing_trends(df)

        # Display trend analysis
        cols = st.columns(len(trends))
        for i, (group, data) in enumerate(trends.items()):
            with cols[i]:
                st.markdown(f"**{group.title()} Impact on Healing**")
                fig = px.line(
                    data.reset_index(),
                    x='Visit Number',
                    y=data.columns,
                    title=f"Healing Rate by {group.title()}",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

class Dashboard:
    """Main dashboard application."""
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.visualizer = DataVisualizer()
        self.doc_handler = DocumentHandler()
        self.statistical_analysis = StatisticalAnalysis()
        SessionState.initialize()

    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            page_icon=self.config.PAGE_ICON,
            layout=self.config.LAYOUT
        )

    def _run_llm_analysis(self, patient_id: int, platform: str, model_name: str, analysis_options: List[str]) -> None:
        """Enhanced LLM analysis with configurable components."""
        with st.spinner("Analyzing patient data..."):
            try:
                llm = WoundAnalysisLLM(platform=platform, model_name=model_name)
                patient_data = st.session_state.processor.get_patient_visits(patient_id)

                # Include selected analysis components
                analysis_params = {
                    "include_measurements": "Wound Measurements" in analysis_options,
                    "include_temperature": "Temperature Analysis" in analysis_options,
                    "include_impedance": "Impedance Analysis" in analysis_options,
                    "include_risk": "Risk Assessment" in analysis_options
                }

                analysis = llm.analyze_patient_data(patient_data, **analysis_params)
                report_path = self.doc_handler.create_report(patient_data, analysis)

                st.session_state.analysis_complete = True
                st.session_state.analysis_results = analysis
                st.session_state.report_path = report_path

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.report_path = None

    def _display_analysis_results(self, df: pd.DataFrame, patient_id: int, analysis_options: List[str]) -> None:
        """Display comprehensive analysis results."""
        st.markdown("### Analysis Results")
        st.write(st.session_state.analysis_results)

        if st.session_state.report_path:
            href = self.doc_handler.get_download_link(st.session_state.report_path)
            st.markdown(
                f'<a href="{href}" download="wound_analysis_report.docx">Download Word Report</a>',
                unsafe_allow_html=True
            )

        if "Wound Measurements" in analysis_options:
            st.markdown("### Wound Measurements Over Time")
            self.visualizer.plot_wound_measurements(df, patient_id)

        if "Risk Assessment" in analysis_options:
            st.markdown("### Risk Assessment")
            patient_data = df[df['Record ID'] == patient_id].iloc[0]
            RiskAssessment.display_risk_assessment(patient_data)

    def _render_overview(self, df: pd.DataFrame) -> None:
        """Render the overview dashboard tab."""
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

        # Add dataset statistics
        st.subheader("Dataset Overview")
        stats_df = df.describe()
        st.dataframe(stats_df)

    def _render_patient_analysis(self, df: pd.DataFrame) -> None:
        """Render the patient analysis dashboard tab."""
        st.sidebar.title("Configuration")
        platform = st.sidebar.selectbox(
            "Select Platform",
            WoundAnalysisLLM.get_available_platforms(),
            help="Choose the AI platform for wound analysis"
        )
        model_name = st.sidebar.selectbox(
            "Select Model",
            WoundAnalysisLLM.get_available_models(platform),
            help="Select the specific model for analysis"
        )

        # Add statistical methods info in sidebar
        with st.sidebar:
            st.markdown("### Statistical Methods")
            st.markdown("""
            - Longitudinal analysis of healing trajectories
            - Multi-factor correlation analysis
            - Risk stratification modeling
            - Temperature gradient analysis
            - Impedance component analysis
            """)

        # Load and process data
        df = self.data_loader.load_data()
        if df is not None:
            patient_ids = sorted(df['Record ID'].unique())
            selected_patient = st.sidebar.selectbox(
                "Select Patient",
                [f"Patient {id}" for id in patient_ids]
            )

            # Enhanced analysis options
            analysis_options = st.sidebar.multiselect(
                "Analysis Components",
                ["Wound Measurements", "Temperature Analysis", "Impedance Analysis", "Risk Assessment"],
                default=["Wound Measurements", "Temperature Analysis"],
                help="Select which components to include in the analysis"
            )

            if st.sidebar.button("Run Analysis"):
                patient_id = int(selected_patient.split(" ")[1])
                self._run_llm_analysis(patient_id, platform, model_name, analysis_options)

            # Display results with enhanced visualizations
            if st.session_state.analysis_complete:
                self._display_analysis_results(df, patient_id, analysis_options)

    def _render_llm_analysis(self, df: pd.DataFrame) -> None:
        """Render the LLM analysis dashboard tab."""
        st.sidebar.title("Configuration")
        platform = st.sidebar.selectbox(
            "Select Platform",
            WoundAnalysisLLM.get_available_platforms(),
            help="Choose the AI platform for wound analysis"
        )
        model_name = st.sidebar.selectbox(
            "Select Model",
            WoundAnalysisLLM.get_available_models(platform),
            help="Select the specific model for analysis"
        )

        # Add statistical methods info in sidebar
        with st.sidebar:
            st.markdown("### Statistical Methods")
            st.markdown("""
            - Longitudinal analysis of healing trajectories
            - Multi-factor correlation analysis
            - Risk stratification modeling
            - Temperature gradient analysis
            - Impedance component analysis
            """)

        # Load and process data
        df = self.data_loader.load_data()
        if df is not None:
            patient_ids = sorted(df['Record ID'].unique())
            selected_patient = st.sidebar.selectbox(
                "Select Patient",
                [f"Patient {id}" for id in patient_ids]
            )

            # Enhanced analysis options
            analysis_options = st.sidebar.multiselect(
                "Analysis Components",
                ["Wound Measurements", "Temperature Analysis", "Impedance Analysis", "Risk Assessment"],
                default=["Wound Measurements", "Temperature Analysis"],
                help="Select which components to include in the analysis"
            )

            if st.sidebar.button("Run Analysis"):
                patient_id = int(selected_patient.split(" ")[1])
                self._run_llm_analysis(patient_id, platform, model_name, analysis_options)

            # Display results with enhanced visualizations
            if st.session_state.analysis_complete:
                self._display_analysis_results(df, patient_id, analysis_options)

    def run(self):
        """Run the dashboard application."""
        self.setup_page()
        st.title("Wound Care Analysis Dashboard")

        # Enhanced sidebar configuration
        st.sidebar.title("Configuration")

        # Move statistical analysis to its own tab
        tabs = ["Overview", "Patient Analysis", "Statistical Analysis", "LLM Analysis"]
        current_tab = st.tabs(tabs)

        df = self.data_loader.load_data()
        if df is not None:
            with current_tab[0]:  # Overview
                self._render_overview(df)

            with current_tab[1]:  # Patient Analysis
                self._render_patient_analysis(df)

            with current_tab[2]:  # Statistical Analysis
                StatisticalAnalysis.display_statistical_analysis(df)

            with current_tab[3]:  # LLM Analysis
                self._render_llm_analysis(df)

def main():
    """Main application entry point."""
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the data source and configuration.")

if __name__ == "__main__":
    main()
