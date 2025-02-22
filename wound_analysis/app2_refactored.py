"""
Smart Bandage Wound Healing Analytics Dashboard.

This module provides a Streamlit-based dashboard for analyzing wound healing data.
It includes visualizations for wound measurements, temperature, impedance, and
provides AI-powered analysis of wound healing progression.

Author: Artin Majdi
Date: February 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import base64
import os
import re
from docx import Document

from data_processor import WoundDataProcessor
from statistical_analysis import StatisticalAnalysis

# Configuration Constants
class Config:
    """Application configuration constants."""
    PLOT_HEIGHT = 600
    PLOT_CONFIG = {"displayModeBar": True, "displaylogo": False}
    DATA_PATH = Path("/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv")
    PAGE_CONFIG = {
        "page_title": "Smart Bandage Wound Healing Analytics",
        "page_icon": "",
        "layout": "wide"
    }
    DEFAULT_PLATFORM = "ai-verde"
    DEFAULT_MODEL = "llama-3.3-70b-fp8"
    DEFAULT_API_KEY = "sk-h8JtQkCCJUOy-TAdDxCLGw"
    DEFAULT_API_URL = "https://llm-api.cyverse.ai"

class SessionState:
    """Session state management."""
    @staticmethod
    def initialize() -> None:
        """Initialize session state variables."""
        defaults = {
            'processor': None,
            'analysis_complete': False,
            'analysis_results': None,
            'report_path': None,
            'stats_analyzer': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

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
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the wound data."""
        try:
            # Clean column names
            df.columns = df.columns.str.strip()

            # 1. Handle missing values
            df = df.replace(['', 'NA', 'N/A', 'null', 'NULL'], np.nan)

            # 2. Extract visit number from Event Name
            df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

            # 3. Calculate wound area if not present but dimensions are available
            if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
                df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

            # 4. Calculate healing rates
            df['Healing Rate (%)'] = DataLoader._calculate_healing_rates(df)

            # 5. Handle missing values in numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())

            # 6. Calculate temperature gradients if available
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

        return pd.Series(healing_rates, index=df.index)

class Visualization:
    """Visualization components for the dashboard."""

    @staticmethod
    def _ensure_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Ensure columns are numeric."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
        """Create wound area progression plot."""
        # Create a copy and ensure numeric types
        df = df.copy()
        df = Visualization._ensure_numeric(df, ['Visit Number', 'Calculated Wound Area'])

        fig = go.Figure()

        try:
            if patient_id is not None:
                patient_df = df[df['Record ID'] == patient_id].copy()
                if not patient_df.empty:
                    Visualization._add_single_patient_traces(fig, patient_df)
            else:
                Visualization._add_all_patients_traces(fig, df)

            fig.update_layout(
                height=Config.PLOT_HEIGHT,
                title="Wound Area Progression",
                xaxis_title="Visit Number",
                yaxis_title="Wound Area (cm²)",
                hovermode='x unified',
                showlegend=True
            )
        except Exception as e:
            st.error(f"Error creating wound area plot: {str(e)}")

        return fig

    @staticmethod
    def _add_single_patient_traces(fig: go.Figure, patient_df: pd.DataFrame) -> None:
        """Add traces for single patient view."""
        try:
            # Sort by visit number
            patient_df = patient_df.sort_values('Visit Number')

            # Add main trace
            fig.add_trace(go.Scatter(
                x=patient_df['Visit Number'].values.astype(float),
                y=patient_df['Calculated Wound Area'].values.astype(float),
                mode='lines+markers',
                name='Wound Area',
                line=dict(color='blue'),
                hovertemplate='Visit %{x}<br>Area: %{y:.1f} cm²'
            ))

            # Add trendline if enough valid points
            if len(patient_df.dropna()) >= 2:
                Visualization._add_trendline(fig, patient_df)

        except Exception as e:
            st.warning(f"Error adding patient traces: {str(e)}")

    @staticmethod
    def _add_all_patients_traces(fig: go.Figure, df: pd.DataFrame) -> None:
        """Add traces for all patients view."""
        try:
            for pid in df['Record ID'].unique():
                if pd.isna(pid):
                    continue

                patient_df = df[df['Record ID'] == pid].copy()
                patient_df = patient_df.sort_values('Visit Number')

                if len(patient_df.dropna()) > 0:
                    fig.add_trace(go.Scatter(
                        x=patient_df['Visit Number'].values.astype(float),
                        y=patient_df['Calculated Wound Area'].values.astype(float),
                        mode='lines+markers',
                        name=f'Patient {pid}',
                        hovertemplate='Visit %{x}<br>Area: %{y:.1f} cm²'
                    ))
        except Exception as e:
            st.warning(f"Error adding traces for all patients: {str(e)}")

    @staticmethod
    def _add_trendline(fig: go.Figure, df: pd.DataFrame) -> None:
        """Add trendline to the plot."""
        try:
            # Get values and remove NaN
            x = df['Visit Number'].values.astype(float)
            y = df['Calculated Wound Area'].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)

            x = x[mask]
            y = y[mask]

            if len(x) >= 2:
                # Calculate trend
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)

                # Generate points for trend line
                x_trend = np.linspace(x.min(), x.max(), 100)
                y_trend = p(x_trend)

                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash'),
                    hovertemplate='Visit %{x}<br>Trend: %{y:.1f} cm²'
                ))
        except Exception as e:
            st.warning(f"Could not calculate trend line: {str(e)}")

class Dashboard:
    """Main dashboard functionality."""
    def __init__(self):
        """Initialize dashboard components."""
        SessionState.initialize()
        self.setup_page_config()
        self.df = DataLoader.load_data()

        if self.df is not None:
            if 'processor' not in st.session_state or st.session_state.processor is None:
                st.session_state.processor = WoundDataProcessor(self.df)
            if 'stats_analyzer' not in st.session_state or st.session_state.stats_analyzer is None:
                st.session_state.stats_analyzer = StatisticalAnalysis(self.df)

    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(**Config.PAGE_CONFIG)

    def setup_sidebar(self) -> Tuple[str, str]:
        """Configure sidebar and model settings."""
        st.sidebar.title("Model Configuration")

        platform = st.sidebar.selectbox(
            "Select Platform",
            ["ai-verde"],
            index=0,
            help="Currently only AI Verde models are supported"
        )

        model_name = st.sidebar.selectbox(
            "Select Model",
            [Config.DEFAULT_MODEL],
            index=0
        )

        with st.sidebar.expander("Advanced Settings"):
            api_key = st.text_input(
                "API Key",
                value=Config.DEFAULT_API_KEY,
                type="password"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        return platform, model_name

    def run(self) -> None:
        """Run the dashboard application."""
        if self.df is None:
            st.error("Failed to load data. Please check the data source.")
            return

        try:
            platform, model_name = self.setup_sidebar()

            # Patient selection
            patient_ids = sorted(self.df['Record ID'].unique())
            patient_options = ["All Patients"] + [f"Patient {id:03d}" for id in patient_ids]
            selected_patient = st.selectbox("Select Patient", patient_options)

            # Create tabs
            tabs = st.tabs([
                "Overview",
                "Impedance Analysis",
                "Temperature",
                "Oxygenation",
                "Risk Factors",
                "LLM Analysis"
            ])

            # Render tabs
            self.render_overview_tab(tabs[0], selected_patient)
            self.render_impedance_tab(tabs[1], selected_patient)
            self.render_temperature_tab(tabs[2], selected_patient)
            self.render_oxygenation_tab(tabs[3], selected_patient)
            self.render_risk_factors_tab(tabs[4], selected_patient)
            self.render_llm_analysis_tab(tabs[5], selected_patient, platform, model_name)

        except Exception as e:
            st.error(f"Error running dashboard: {str(e)}")

    def render_overview_tab(self, tab: Any, selected_patient: str) -> None:
        """Render overview tab content."""
        with tab:
            st.header("Smart Bandage Wound Healing Analytics")

            if selected_patient == "All Patients":
                self.render_dataset_statistics()
            else:
                self.render_patient_statistics(selected_patient)

            st.subheader("Wound Area Over Time")
            patient_id = None if selected_patient == "All Patients" else int(selected_patient.split()[1])
            fig = Visualization.create_wound_area_plot(self.df, patient_id)
            st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)

    def render_dataset_statistics(self) -> None:
        """Render statistics for the entire dataset."""
        try:
            if self.df is None or self.df.empty:
                st.warning("No data available for analysis")
                return

            total_patients = len(self.df['Record ID'].unique())
            total_visits = len(self.df)
            avg_visits = total_visits / total_patients if total_patients > 0 else 0.0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Patients", f"{total_patients}")
            with col2:
                st.metric("Total Visits", f"{total_visits}")
            with col3:
                st.metric("Average Visits per Patient", f"{avg_visits:.1f}")

            if st.session_state.stats_analyzer:
                overall_stats = st.session_state.stats_analyzer.get_overall_statistics()
                st.write("## Overall Statistics")
                st.write(overall_stats)

        except Exception as e:
            st.error(f"Error calculating dataset statistics: {str(e)}")

    def render_patient_statistics(self, selected_patient: str) -> None:
        """Render statistics for a single patient."""
        try:
            if self.df is None or self.df.empty:
                st.warning("No data available for analysis")
                return

            patient_id = int(selected_patient.split()[1])
            patient_data = self.df[self.df['Record ID'] == patient_id]

            if patient_data.empty:
                st.error(f"No data found for {selected_patient}")
                return

            patient_data = patient_data.iloc[0]

            st.subheader("Patient Demographics")
            col1, col2, col3 = st.columns(3)

            # Safely get demographic data with fallbacks
            age = patient_data.get('Calculated Age at Enrollment', 'N/A')
            gender = patient_data.get('Gender', 'N/A')
            bmi = patient_data.get('BMI', 'N/A')

            with col1:
                st.metric("Age", str(age))
            with col2:
                st.metric("Gender", str(gender))
            with col3:
                st.metric("BMI", f"{bmi:.1f}" if isinstance(bmi, (int, float)) else str(bmi))

            if st.session_state.stats_analyzer:
                patient_stats = st.session_state.stats_analyzer.get_patient_statistics(patient_id)
                st.write("## Patient Statistics")
                st.write(patient_stats)

        except Exception as e:
            st.error(f"Error displaying patient statistics: {str(e)}")

    def render_impedance_tab(self, tab: Any, selected_patient: str) -> None:
        """Render impedance analysis tab content."""
        with tab:
            try:
                if self.df is None or self.df.empty:
                    st.warning("No impedance data available")
                    return

                st.subheader("Impedance vs. Wound Healing Progress")

                if selected_patient == "All Patients":
                    # Scatter plot of impedance vs healing rate
                    valid_data = self.df[
                        (self.df['Healing Rate (%)'].notna()) &
                        (self.df['Skin Impedance (kOhms) - Z'].notna())
                    ]

                    if not valid_data.empty:
                        fig = px.scatter(
                            valid_data,
                            x='Skin Impedance (kOhms) - Z',
                            y='Healing Rate (%)',
                            color='Diabetes?',
                            size='Calculated Wound Area',
                            hover_data=['Record ID', 'Event Name', 'Wound Type'],
                            title="Relationship Between Skin Impedance and Healing Rate"
                        )
                        fig.update_layout(
                            height=Config.PLOT_HEIGHT,
                            xaxis_title="Impedance Z (kOhms)",
                            yaxis_title="Healing Rate (% reduction per visit)"
                        )
                        st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)

                        # Calculate correlation only if we have valid data
                        valid_healing = valid_data[valid_data['Healing Rate (%)'] > 0]
                        if len(valid_healing) >= 2:  # Need at least 2 points for correlation
                            r, p = stats.pearsonr(
                                valid_healing['Skin Impedance (kOhms) - Z'],
                                valid_healing['Healing Rate (%)']
                            )
                            p_formatted = f"< 0.001" if p < 0.001 else f"= {p:.3f}"
                            st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")
                    else:
                        st.warning("Insufficient data for impedance analysis")
                else:
                    # Individual patient impedance over time
                    patient_id = int(selected_patient.split()[1])
                    patient_data = self.df[self.df['Record ID'] == patient_id].sort_values('Visit Number')

                    if not patient_data.empty:
                        impedance_cols = [
                            'Skin Impedance (kOhms) - Z',
                            'Skin Impedance (kOhms) - Z\'',
                            'Skin Impedance (kOhms) - Z\'\''
                        ]

                        # Check if we have any impedance data
                        if any(patient_data[col].notna().any() for col in impedance_cols):
                            fig = px.line(
                                patient_data,
                                x='Visit Number',
                                y=impedance_cols,
                                title=f"Impedance Measurements Over Time - {selected_patient}",
                                markers=True
                            )
                            fig.update_layout(
                                height=Config.PLOT_HEIGHT,
                                xaxis_title="Visit Number",
                                yaxis_title="Impedance (kOhms)",
                                legend_title="Measurement"
                            )
                            st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)
                        else:
                            st.warning("No impedance measurements available for this patient")
                    else:
                        st.warning("No data available for this patient")

            except Exception as e:
                st.error(f"Error in impedance analysis: {str(e)}")

    def render_temperature_tab(self, tab: Any, selected_patient: str) -> None:
        """Render temperature analysis tab content."""
        with tab:
            st.subheader("Temperature Gradient Analysis")

            if selected_patient == "All Patients":
                # Temperature gradients by wound type
                fig = px.box(
                    self.df,
                    x='Wound Type',
                    y=['Center-Edge Temp Gradient', 'Edge-Peri Temp Gradient', 'Total Temp Gradient'],
                    title="Temperature Gradients by Wound Type",
                    points="all"
                )
                fig.update_layout(
                    height=Config.PLOT_HEIGHT,
                    xaxis_title="Wound Type",
                    yaxis_title="Temperature Gradient (°F)",
                    boxmode='group'
                )
                st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)
            else:
                # Individual patient temperature progression
                patient_id = int(selected_patient.split()[1])
                patient_data = self.df[self.df['Record ID'] == patient_id].sort_values('Visit Number')

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Temperature measurements
                for temp_type, color in [
                    ('Center of Wound Temperature (Fahrenheit)', 'red'),
                    ('Edge of Wound Temperature (Fahrenheit)', 'orange'),
                    ('Peri-wound Temperature (Fahrenheit)', 'blue')
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=patient_data['Visit Number'],
                            y=patient_data[temp_type],
                            name=temp_type.split('(')[0].strip(),
                            line=dict(color=color)
                        )
                    )

                # Temperature gradients
                for gradient, color in [
                    ('Center-Edge Temp Gradient', 'rgba(255,0,0,0.3)'),
                    ('Edge-Peri Temp Gradient', 'rgba(0,0,255,0.3)')
                ]:
                    fig.add_trace(
                        go.Bar(
                            x=patient_data['Visit Number'],
                            y=patient_data[gradient],
                            name=gradient,
                            marker_color=color
                        ),
                        secondary_y=True
                    )

                fig.update_layout(
                    height=Config.PLOT_HEIGHT,
                    title=f"Temperature Analysis - {selected_patient}",
                    xaxis_title="Visit Number",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig.update_yaxes(title_text="Temperature (°F)", secondary_y=False)
                fig.update_yaxes(title_text="Temperature Gradient (°F)", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)

    def render_oxygenation_tab(self, tab: Any, selected_patient: str) -> None:
        """Render oxygenation analysis tab content."""
        with tab:
            st.subheader("Oxygenation Metrics")

            if selected_patient == "All Patients":
                # Oxygenation vs healing rate
                fig = px.scatter(
                    self.df[self.df['Healing Rate (%)'] > 0],
                    x='Oxygenation (%)',
                    y='Healing Rate (%)',
                    color='Diabetes?',
                    size='Hemoglobin Level',
                    hover_data=['Record ID', 'Event Name', 'Wound Type'],
                    title="Relationship Between Oxygenation and Healing Rate"
                )
                fig.update_layout(
                    height=Config.PLOT_HEIGHT,
                    xaxis_title="Oxygenation (%)",
                    yaxis_title="Healing Rate (% reduction per visit)"
                )
                st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)
            else:
                # Individual patient oxygenation trends
                patient_id = int(selected_patient.split()[1])
                patient_data = self.df[self.df['Record ID'] == patient_id].sort_values('Visit Number')

                # Hemoglobin components
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=patient_data['Visit Number'],
                    y=patient_data['Oxyhemoglobin Level'],
                    name="Oxyhemoglobin",
                    marker_color='red'
                ))
                fig.add_trace(go.Bar(
                    x=patient_data['Visit Number'],
                    y=patient_data['Deoxyhemoglobin Level'],
                    name="Deoxyhemoglobin",
                    marker_color='purple'
                ))

                fig.update_layout(
                    height=Config.PLOT_HEIGHT,
                    title=f"Hemoglobin Components - {selected_patient}",
                    xaxis_title="Visit Number",
                    yaxis_title="Level (g/dL)",
                    barmode='stack'
                )
                st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)

    def render_risk_factors_tab(self, tab: Any, selected_patient: str) -> None:
        """Render risk factors analysis tab content."""
        with tab:
            st.subheader("Risk Factor Analysis")

            if selected_patient == "All Patients":
                # Create subtabs for different risk factors
                subtabs = st.tabs(["Diabetes", "Smoking", "BMI"])

                # Diabetes analysis
                with subtabs[0]:
                    # Healing rates by diabetes status
                    diab_healing = self.df.groupby(['Diabetes?', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
                    diab_healing = diab_healing[diab_healing['Visit Number'] > 1]

                    fig = px.line(
                        diab_healing,
                        x='Visit Number',
                        y='Healing Rate (%)',
                        color='Diabetes?',
                        title="Average Healing Rate by Diabetes Status",
                        markers=True
                    )
                    fig.update_layout(height=Config.PLOT_HEIGHT)
                    st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)

                # Smoking analysis
                with subtabs[1]:
                    smoke_healing = self.df.groupby(['Smoking status', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
                    smoke_healing = smoke_healing[smoke_healing['Visit Number'] > 1]

                    fig = px.line(
                        smoke_healing,
                        x='Visit Number',
                        y='Healing Rate (%)',
                        color='Smoking status',
                        title="Average Healing Rate by Smoking Status",
                        markers=True
                    )
                    fig.update_layout(height=Config.PLOT_HEIGHT)
                    st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)

                # BMI analysis
                with subtabs[2]:
                    fig = px.scatter(
                        self.df,
                        x='BMI',
                        y='Healing Rate (%)',
                        color='Diabetes?',
                        title="BMI vs Healing Rate",
                        trendline="ols"
                    )
                    fig.update_layout(height=Config.PLOT_HEIGHT)
                    st.plotly_chart(fig, use_container_width=True, config=Config.PLOT_CONFIG)
            else:
                patient_id = int(selected_patient.split()[1])
                patient_data = self.df[self.df['Record ID'] == patient_id].iloc[0]

                # Display risk factor summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Diabetes Status", patient_data['Diabetes?'])
                with col2:
                    st.metric("Smoking Status", patient_data['Smoking status'])
                with col3:
                    st.metric("BMI", f"{patient_data['BMI']:.1f}")

    def render_llm_analysis_tab(self, tab: Any, selected_patient: str, platform: str, model_name: str) -> None:
        """Render LLM analysis tab content."""
        with tab:
            st.header("LLM-Powered Wound Analysis")

            if selected_patient != "All Patients":
                if st.button("Run Analysis"):
                    patient_id = int(selected_patient.split()[1])
                    self._run_llm_analysis(patient_id, platform, model_name)

                if st.session_state.analysis_complete and st.session_state.analysis_results:
                    st.markdown("### Analysis Results")
                    st.write(st.session_state.analysis_results)

                    if st.session_state.report_path:
                        with open(st.session_state.report_path, "rb") as file:
                            bytes_data = file.read()
                            b64_data = base64.b64encode(bytes_data).decode()
                            href = f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_data}'
                            st.markdown(
                                f'<a href="{href}" download="wound_analysis_report.docx">Download Word Report</a>',
                                unsafe_allow_html=True
                            )
            else:
                st.warning("Please select a specific patient to run the analysis.")

    def _run_llm_analysis(self, patient_id: int, platform: str, model_name: str) -> None:
        """Run LLM analysis for a specific patient."""
        with st.spinner("Analyzing patient data..."):
            try:
                # Validate inputs
                if not platform or not model_name:
                    st.error("Invalid platform or model configuration")
                    return

                if not isinstance(patient_id, int) or patient_id <= 0:
                    st.error("Invalid patient ID")
                    return

                # Initialize LLM and get patient data
                llm = WoundAnalysisLLM(platform=platform, model_name=model_name)
                if not st.session_state.processor:
                    st.error("Data processor not initialized")
                    return

                patient_data = st.session_state.processor.get_patient_visits(patient_id)

                if not patient_data:
                    st.error(f"No data found for Patient {patient_id}")
                    return

                # Run analysis and generate report
                analysis = llm.analyze_patient_data(patient_data)
                if not analysis:
                    st.error("Failed to generate analysis")
                    return

                report_path = self._create_and_save_report(patient_data, analysis)

                # Update session state
                st.session_state.analysis_complete = True
                st.session_state.analysis_results = analysis
                st.session_state.report_path = report_path

                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.error("Please check your API configuration and try again.")
                # Reset analysis state on error
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.report_path = None

    def _create_and_save_report(self, patient_data: Dict, analysis: str) -> Optional[str]:
        """Create and save the analysis report.

        Args:
            patient_data: Dictionary containing patient data
            analysis: String containing the analysis results

        Returns:
            Optional[str]: Path to the saved report file, or None if creation fails
        """
        try:
            if not patient_data or not analysis:
                st.error("Missing data for report generation")
                return None

            doc = Document()
            format_word_document(doc, patient_data, analysis)

            report_path = "wound_analysis_report.docx"
            try:
                doc.save(report_path)
            except Exception as e:
                st.error(f"Failed to save report: {str(e)}")
                return None

            return report_path

        except Exception as e:
            st.error(f"Error creating report: {str(e)}")
            return None

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
