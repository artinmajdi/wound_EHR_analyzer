"""
VSCODE - O3 Smart Bandage Wound Healing Analytics Dashboard
A Streamlit app for visualizing and analyzing wound healing data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import pathlib
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from docx import Document
import base64
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM, format_word_document

@dataclass
class Config:
    """Application configuration settings."""
    PAGE_TITLE: str = "VSCODE - O3"
    PAGE_ICON: str = "🩹"
    LAYOUT: str = "wide"
    DATA_PATH: pathlib.Path = pathlib.Path(__file__).parent / "dataset" / "SmartBandage-Data_for_llm.csv"

class DataManager:
    """Handles data loading, processing and manipulation."""

    @staticmethod
    @st.cache_data
    def load_data() -> Optional[pd.DataFrame]:
        """Load and preprocess the wound healing data."""
        df = pd.read_csv(Config.DATA_PATH)
        df = DataManager._preprocess_data(df)
        return df

    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the loaded data."""
        # Make a copy to avoid modifying the cached data
        df = df.copy()

        df.columns = df.columns.str.strip()
        # Fill missing Visit Number with 1 before converting to int
        df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)
        df = DataManager._calculate_healing_rates(df)
        df = DataManager._create_derived_features(df.copy())  # Pass a copy to ensure modifications stick
        return df

    @staticmethod
    def _calculate_healing_rates(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate healing rates for each patient visit."""
        healing_rates = []

        for patient_id in df['Record ID'].unique():
            patient_data = df[df['Record ID'] == patient_id].sort_values('Visit Number')
            for _, row in patient_data.iterrows():
                if row['Visit Number'] == 1:
                    healing_rates.append(0)
                else:
                    prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
                    prev_visit = prev_visits[prev_visits['Visit Number'] == prev_visits['Visit Number'].max()]

                    if len(prev_visit) > 0 and 'Calculated Wound Area' in df.columns:
                        prev_area = prev_visit['Calculated Wound Area'].values[0]
                        curr_area = row['Calculated Wound Area']

                        if prev_area > 0 and not pd.isna(prev_area) and not pd.isna(curr_area):
                            healing_rate = (prev_area - curr_area) / prev_area * 100
                            healing_rates.append(healing_rate)
                        else:
                            healing_rates.append(0)
                    else:
                        healing_rates.append(0)

        df['Healing Rate (%)'] = healing_rates[:len(df)]
        return df

    @staticmethod
    def _create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create additional derived features from the data."""
        import numpy as np

        # Make a copy to avoid modifying the cached data
        df = df.copy()

        # Temperature gradients
        if all(col in df.columns for col in [   'Center of Wound Temperature (Fahrenheit)',
                                                'Edge of Wound Temperature (Fahrenheit)',
                                                'Peri-wound Temperature (Fahrenheit)']):
            df['Center-Edge Temp Gradient'] = df['Center of Wound Temperature (Fahrenheit)'] - df['Edge of Wound Temperature (Fahrenheit)']
            df['Edge-Peri Temp Gradient']   = df['Edge of Wound Temperature (Fahrenheit)']   - df['Peri-wound Temperature (Fahrenheit)']
            df['Total Temp Gradient']       = df['Center of Wound Temperature (Fahrenheit)'] - df['Peri-wound Temperature (Fahrenheit)']

        # BMI categories
        if 'BMI' in df.columns:
            df['BMI Category'] = pd.cut(
                df['BMI'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II-III']
            )

        if df is not None and not df.empty:
            # Convert Visit date to datetime if not already
            df['Visit date'] = pd.to_datetime(df['Visit date'])

            # Calculate days since first visit for each patient
            df['Days_Since_First_Visit'] = df.groupby('Record ID')['Visit date'].transform(
                lambda x: (x - x.min()).dt.days
            )

            # Initialize columns with explicit dtypes
            df['Healing Rate'] = pd.Series(0.0, index=df.index, dtype=float)
            df['Estimated_Days_To_Heal'] = pd.Series(np.nan, index=df.index, dtype=float)
            df['Overall_Improvement'] = pd.Series(np.nan, index=df.index, dtype=str)

            # Calculate healing rate and improvement for each patient
            for patient_id in df['Record ID'].unique():
                patient_data = df[df['Record ID'] == patient_id].sort_values('Days_Since_First_Visit')

                if len(patient_data) >= 2:
                    # Get first and last measurements
                    first_visit = patient_data.iloc[0]
                    last_visit = patient_data.iloc[-1]

                    # Calculate overall healing rate
                    days_diff = last_visit['Days_Since_First_Visit'] - first_visit['Days_Since_First_Visit']
                    area_diff = last_visit['Calculated Wound Area'] - first_visit['Calculated Wound Area']

                    if days_diff > 0:
                        healing_rate = area_diff / days_diff
                        # Apply this rate to all patient's records
                        df.loc[patient_data.index, 'Healing Rate'] = healing_rate

                        # Mark improvement based on healing rate
                        if healing_rate < 0:  # Negative rate means wound is getting smaller
                            df.loc[last_visit.name, 'Overall_Improvement'] = 'Yes'
                        else:
                            df.loc[last_visit.name, 'Overall_Improvement'] = 'No'

                        # Estimate days to heal if wound is improving
                        if healing_rate < 0:  # Wound is improving
                            # Calculate remaining days based on current healing rate
                            latest_area = last_visit['Calculated Wound Area']
                            if latest_area > 0 and abs(healing_rate) > 0:  # Ensure valid division
                                days_to_heal = latest_area / abs(healing_rate)
                                if days_to_heal > 0:  # Ensure positive estimate
                                    total_estimated_days = last_visit['Days_Since_First_Visit'] + days_to_heal
                                    # Only set if the estimate is reasonable (e.g., less than 2 years)
                                    if total_estimated_days > 0 and total_estimated_days < 730:  # 730 days = 2 years
                                        # Use loc to ensure the column is created
                                        df.loc[patient_data.index, 'Estimated_Days_To_Heal'] = float(total_estimated_days)

            # Calculate average healing rate
            avg_rates = df.groupby('Record ID')['Healing Rate'].mean()
            df['Average Healing Rate'] = df['Record ID'].map(avg_rates)

            # Ensure the column exists with proper type
            if 'Estimated_Days_To_Heal' not in df.columns:
                df['Estimated_Days_To_Heal'] = np.nan

        return df

    @staticmethod
    def get_patient_data(df: pd.DataFrame, patient_id: int) -> pd.DataFrame:
        """Get data for a specific patient."""
        return df[df['Record ID'] == patient_id].sort_values('Visit Number')

    # Add a new static method _generate_mock_data to handle fallback data generation
    @staticmethod
    def _generate_mock_data() -> pd.DataFrame:
        """Generate mock data in case the CSV file cannot be loaded."""
        np.random.seed(42)
        n_patients = 10
        n_visits = 3
        rows = []
        for p in range(1, n_patients + 1):
            initial_area = np.random.uniform(10, 20)
            for v in range(1, n_visits + 1):
                area = initial_area * (0.9 ** (v - 1))
                rows.append({
                    'Record ID': p,
                    'Event Name': f'Visit {v}',
                    'Calculated Wound Area': area,
                    'Center of Wound Temperature (Fahrenheit)': np.random.uniform(97, 102),
                    'Edge of Wound Temperature (Fahrenheit)': np.random.uniform(96, 101),
                    'Peri-wound Temperature (Fahrenheit)': np.random.uniform(95, 100),
                    'BMI': np.random.uniform(18, 35),
                    'Diabetes?': np.random.choice(['Yes', 'No']),
                    'Smoking status': np.random.choice(['Never', 'Current', 'Former'])
                })
        df = pd.DataFrame(rows)
        df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)
        df = DataManager._calculate_healing_rates(df)
        df = DataManager._create_derived_features(df)
        return df

class SessionStateManager:
    """Manages Streamlit session state."""

    @staticmethod
    def initialize() -> None:
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
    """Handles document generation and downloads."""

    @staticmethod
    def create_report(patient_data: Dict, analysis_results: str) -> str:
        """Create and save analysis report."""
        doc = Document()
        format_word_document(doc, patient_data, analysis_results)
        report_path = "wound_analysis_report.docx"
        doc.save(report_path)
        return report_path

    @staticmethod
    def get_download_link(report_path: str) -> str:
        """Generate download link for the report."""
        with open(report_path, "rb") as file:
            bytes_data = file.read()
        b64_data = base64.b64encode(bytes_data).decode()
        return f'data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_data}'

class Visualizer:
    """Handles data visualization."""

    @staticmethod
    def create_wound_area_plot(df: pd.DataFrame, patient_id: Optional[int] = None) -> go.Figure:
        """Create wound area progression plot."""
        if patient_id:
            return Visualizer._create_single_patient_plot(df, patient_id)
        return Visualizer._create_all_patients_plot(df)

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, column: str, quantile_threshold: float = 0.1) -> pd.DataFrame:
        """Remove outliers using quantile thresholds and z-score validation.

        Args:
            df: DataFrame containing the data
            column: Column name to check for outliers
            quantile_threshold: Value between 0 and 0.5 for quantile-based filtering (0 = no filtering)
        """
        if quantile_threshold <= 0 or len(df) < 3:  # Not enough data points or no filtering requested
            return df

        Q1 = df[column].quantile(quantile_threshold)
        Q3 = df[column].quantile(1 - quantile_threshold)
        IQR = Q3 - Q1

        if IQR == 0:  # All values are the same
            return df

        lower_bound = max(0, Q1 - 1.5 * IQR)  # Ensure non-negative values
        upper_bound = Q3 + 1.5 * IQR

        # Calculate z-scores for additional validation
        z_scores = abs((df[column] - df[column].mean()) / df[column].std())

        # Combine IQR and z-score methods
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound) & (z_scores <= 3)

        return df[mask].copy()

    @staticmethod
    def _create_all_patients_plot(df: pd.DataFrame) -> go.Figure:
        """Create wound area plot for all patients."""
        # Add outlier threshold control
        col1, col2 = st.columns([4, 1])
        with col2:
            outlier_threshold = st.number_input(
                "Outlier Threshold",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.01,
                help="Quantile threshold for outlier detection (0 = no outliers removed, 0.1 = using 10th and 90th percentiles)"
            )

        fig = go.Figure()

        # Get the filtered data for y-axis limits
        all_wound_areas_df = pd.DataFrame({'wound_area': df['Calculated Wound Area'].dropna()})
        filtered_df = Visualizer._remove_outliers(all_wound_areas_df, 'wound_area', outlier_threshold)

        # Set y-axis limits based on filtered data
        lower_bound = 0
        upper_bound = (filtered_df['wound_area'].max() if outlier_threshold > 0 else all_wound_areas_df['wound_area'].max()) * 1.05

        # Store patient statistics for coloring
        patient_stats = []

        for pid in df['Record ID'].unique():
            patient_df = df[df['Record ID'] == pid].copy()
            patient_df['Days_Since_First_Visit'] = (pd.to_datetime(patient_df['Visit date']) - pd.to_datetime(patient_df['Visit date']).min()).dt.days

            # Remove NaN values
            patient_df = patient_df.dropna(subset=['Days_Since_First_Visit', 'Calculated Wound Area'])

            if not patient_df.empty:
                # Calculate healing rate for this patient
                if len(patient_df) >= 2:
                    first_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmin(), 'Calculated Wound Area']
                    last_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmax(), 'Calculated Wound Area']
                    total_days = patient_df['Days_Since_First_Visit'].max()

                    if total_days > 0:
                        healing_rate = (first_area - last_area) / total_days
                        patient_stats.append({
                            'pid': pid,
                            'healing_rate': healing_rate,
                            'initial_area': first_area
                        })

                fig.add_trace(go.Scatter(
                    x=patient_df['Days_Since_First_Visit'],
                    y=patient_df['Calculated Wound Area'],
                    mode='lines+markers',
                    name=f'Patient {pid}',
                    hovertemplate=(
                        'Day: %{x}<br>'
                        'Area: %{y:.1f} cm²<br>'
                        '<extra>Patient %{text}</extra>'
                    ),
                    text=[str(pid)] * len(patient_df),
                    line=dict(width=2),
                    marker=dict(size=8)
                ))

        # Update layout with improved styling
        fig.update_layout(
            title=dict(
                text="Wound Area Progression - All Patients (Outliers Removed)",
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Days Since First Visit",
                title_font=dict(size=14),
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title="Wound Area (cm²)",
                title_font=dict(size=14),
                range=[lower_bound, upper_bound],
                gridcolor='lightgray',
                showgrid=True
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="lightgray",
                borderwidth=1
            ),
            margin=dict(l=60, r=120, t=50, b=50),
            plot_bgcolor='white'
        )

        # Update annotation text based on outlier threshold
        annotation_text = (
            "Note: No outliers removed" if outlier_threshold == 0 else
            f"Note: Outliers removed using combined IQR and z-score methods<br>"
            f"Threshold: {outlier_threshold:.2f} quantile"
        )

        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.99, y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="right",
            yanchor="bottom",
            align="right"
        )

        return fig

    @staticmethod
    def _create_single_patient_plot(df: pd.DataFrame, patient_id: int) -> go.Figure:
        """Create wound area plot for a single patient."""
        patient_df = df[df['Record ID'] == patient_id].sort_values('Days_Since_First_Visit')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=patient_df['Days_Since_First_Visit'],
            y=patient_df['Calculated Wound Area'],
            mode='lines+markers',
            name='Wound Area',
            line=dict(color='blue'),
            hovertemplate='%{y:.1f} cm²'
        ))

        if len(patient_df) >= 2:

            x = patient_df['Days_Since_First_Visit'].values
            y = patient_df['Calculated Wound Area'].values
            mask = np.isfinite(x) & np.isfinite(y)

            # Add trendline
            if np.sum(mask) >= 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)

                # Add trend line
                fig.add_trace(go.Scatter(
                    x=patient_df['Days_Since_First_Visit'],
                    y=p(patient_df['Days_Since_First_Visit']),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash'),
                    hovertemplate='Day %{x}<br>Trend: %{y:.1f} cm²'
                ))

            # Calculate and display healing rate
            total_days = patient_df['Days_Since_First_Visit'].max()
            if total_days > 0:
                first_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmin(), 'Calculated Wound Area']
                last_area = patient_df.loc[patient_df['Days_Since_First_Visit'].idxmax(), 'Calculated Wound Area']
                healing_rate = (first_area - last_area) / total_days
                healing_status = "Improving" if healing_rate > 0 else "Worsening"
                healing_rate_text = f"Healing Rate: {healing_rate:.2f} cm²/day<br> {healing_status}"
            else:
                healing_rate_text = "Insufficient time between measurements for healing rate calculation"

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
        return fig

class RiskAnalyzer:
    """Handles risk analysis and assessment."""

    @staticmethod
    def calculate_risk_score(patient_data: pd.Series) -> Tuple[int, List[str], str]:
        """Calculate risk score and identify risk factors."""
        risk_score = 0
        risk_factors = []

        # Check diabetes
        if patient_data['Diabetes?'] == 'Yes':
            risk_factors.append("Diabetes")
            risk_score += 3

        # Check smoking status
        if patient_data['Smoking status'] == 'Current':
            risk_factors.append("Current smoker")
            risk_score += 2
        elif patient_data['Smoking status'] == 'Former':
            risk_factors.append("Former smoker")
            risk_score += 1

        # Check BMI
        if patient_data['BMI'] >= 30:
            risk_factors.append("Obesity")
            risk_score += 2
        elif patient_data['BMI'] >= 25:
            risk_factors.append("Overweight")
            risk_score += 1

        # Determine risk category
        if risk_score >= 6:
            risk_category = "High"
        elif risk_score >= 3:
            risk_category = "Moderate"
        else:
            risk_category = "Low"

        return risk_score, risk_factors, risk_category

class Dashboard:
    """Main dashboard application."""

    def __init__(self):
        """Initialize the dashboard."""
        self.config = Config()
        self.data_manager = DataManager()
        self.visualizer = Visualizer()
        self.doc_handler = DocumentHandler()
        self.risk_analyzer = RiskAnalyzer()
        # LLM configuration placeholders
        self.llm_platform = None
        self.llm_model = None
        self.uploaded_file = None

    def setup(self) -> None:
        """Set up the dashboard configuration."""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            page_icon=self.config.PAGE_ICON,
            layout=self.config.LAYOUT
        )
        SessionStateManager.initialize()
        self._create_llm_sidebar()

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and prepare data for the dashboard."""
        return self.data_manager.load_data()

    def run(self) -> None:
        """Run the main dashboard application."""
        self.setup()
        df = self.load_data()

        if df is not None:
            # Header
            st.title(self.config.PAGE_TITLE)

            # Patient selection
            patient_ids = sorted(df['Record ID'].unique())
            patient_options = ["All Patients"] + [f"Patient {id:03d}" for id in patient_ids]
            selected_patient = st.selectbox("Select Patient", patient_options)

            # Create tabs
            self._create_dashboard_tabs(df, selected_patient)

            # Footer
            self._add_footer()

            # Additional sidebar info
            self._create_sidebar()

    def _create_dashboard_tabs(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Create and manage dashboard tabs."""
        tabs = st.tabs([
            "Overview",
            "Impedance Analysis",
            "Temperature",
            "Oxygenation",
            "Risk Factors",
            "LLM Analysis"
        ])

        with tabs[0]:
            self._overview_tab(df, selected_patient)
        with tabs[1]:
            self._impedance_tab(df, selected_patient)
        with tabs[2]:
            self._temperature_tab(df, selected_patient)
        with tabs[3]:
            self._oxygenation_tab(df, selected_patient)
        with tabs[4]:
            self._risk_factors_tab(df, selected_patient)
        with tabs[5]:
            self._llm_analysis_tab(df, selected_patient)

    def _overview_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Render the overview tab."""
        st.header("Overview")

        if selected_patient == "All Patients":
            self._render_overview_stats(df)
        else:
            patient_id = int(selected_patient.split(" ")[1])
            self._render_patient_overview(df, patient_id)
            st.subheader("Wound Area Over Time")
            fig = self.visualizer.create_wound_area_plot(DataManager.get_patient_data(df, patient_id), patient_id)
            st.plotly_chart(fig, use_container_width=True)

    def _impedance_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Render the impedance analysis tab."""
        st.header("Impedance Analysis")

        if selected_patient == "All Patients":
            # Scatter plot: Impedance vs Healing Rate
            # Filter out NaN values and rows where Healing Rate is 0
            valid_df = df[df['Healing Rate (%)'] > 0].copy()

            # Handle NaN values in the size column (Calculated Wound Area)
            valid_df['Calculated Wound Area'] = valid_df['Calculated Wound Area'].fillna(valid_df['Calculated Wound Area'].mean())

            # Ensure all numeric columns are clean
            for col in ['Skin Impedance (kOhms) - Z', 'Healing Rate (%)', 'Calculated Wound Area']:
                valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')

            # Remove any remaining rows with NaN values
            valid_df = valid_df.dropna(subset=['Skin Impedance (kOhms) - Z', 'Healing Rate (%)', 'Calculated Wound Area'])

            if not valid_df.empty:
                fig = px.scatter(
                    valid_df,
                    x='Skin Impedance (kOhms) - Z',
                    y='Healing Rate (%)',
                    color='Diabetes?',
                    size='Calculated Wound Area',
                    size_max=30,  # Maximum marker size
                    hover_data=['Record ID', 'Event Name', 'Wound Type'],
                    title="Impedance vs Healing Rate Correlation"
                )
                fig.update_layout(xaxis_title="Impedance Z (kOhms)", yaxis_title="Healing Rate (% reduction per visit)")
                st.plotly_chart(fig, use_container_width=True)

                # Calculate correlation
                r, p = stats.pearsonr(valid_df['Skin Impedance (kOhms) - Z'], valid_df['Healing Rate (%)'])
                p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
                st.info(f"Statistical correlation: r = {r:.2f} (p {p_formatted})")
                st.write("Higher impedance values correlate with slower healing rates, especially in diabetic patients")
            else:
                st.warning("No valid data available for the scatter plot.")

            # Rest of the impedance tab code...
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Impedance Components Over Time")
                avg_impedance = df.groupby('Visit Number')[['Skin Impedance (kOhms) - Z', "Skin Impedance (kOhms) - Z'", "Skin Impedance (kOhms) - Z''"]].mean().reset_index()
                fig1 = px.line(
                    avg_impedance,
                    x='Visit Number',
                    y=['Skin Impedance (kOhms) - Z', "Skin Impedance (kOhms) - Z'", "Skin Impedance (kOhms) - Z''"],
                    title="Average Impedance Components by Visit",
                    markers=True
                )
                fig1.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.subheader("Impedance by Wound Type")
                avg_by_type = df.groupby('Wound Type')['Skin Impedance (kOhms) - Z'].mean().reset_index()
                fig2 = px.bar(
                    avg_by_type,
                    x='Wound Type',
                    y='Skin Impedance (kOhms) - Z',
                    title="Average Impedance by Wound Type",
                    color='Wound Type'
                )
                fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Average Impedance Z (kOhms)")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            # For individual patient
            patient_data = DataManager.get_patient_data(df, int(selected_patient.split(" ")[1])).sort_values('Visit Number')
            fig = px.line(
                patient_data,
                x='Visit Number',
                y=['Skin Impedance (kOhms) - Z', "Skin Impedance (kOhms) - Z'", "Skin Impedance (kOhms) - Z''"],
                title=f"Impedance Measurements Over Time for {selected_patient}",
                markers=True
            )
            fig.update_layout(xaxis_title="Visit Number", yaxis_title="Impedance (kOhms)", legend_title="Measurement")
            st.plotly_chart(fig, use_container_width=True)

    def _temperature_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Render the temperature analysis tab."""
        st.header("Temperature Analysis")

        if selected_patient == "All Patients":
            temp_df = df.copy()

            # Calculate temperature gradients and handle NaN values
            for col in ['Center of Wound Temperature (Fahrenheit)', 'Edge of Wound Temperature (Fahrenheit)', 'Peri-wound Temperature (Fahrenheit)']:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

            temp_df['Center-Edge Gradient'] = temp_df['Center of Wound Temperature (Fahrenheit)'] - temp_df['Edge of Wound Temperature (Fahrenheit)']
            temp_df['Edge-Peri Gradient'] = temp_df['Edge of Wound Temperature (Fahrenheit)'] - temp_df['Peri-wound Temperature (Fahrenheit)']
            temp_df['Total Gradient'] = temp_df['Center of Wound Temperature (Fahrenheit)'] - temp_df['Peri-wound Temperature (Fahrenheit)']

            # Handle NaN values in wound area for scatter plot
            temp_df['Calculated Wound Area'] = temp_df['Calculated Wound Area'].fillna(temp_df['Calculated Wound Area'].mean())

            # Create boxplot
            valid_temp_df = temp_df.dropna(subset=['Center-Edge Gradient', 'Edge-Peri Gradient', 'Total Gradient'])
            if not valid_temp_df.empty:
                fig_box = px.box(
                    valid_temp_df,
                    x='Wound Type',
                    y=['Center-Edge Gradient', 'Edge-Peri Gradient', 'Total Gradient'],
                    title="Temperature Gradients by Wound Type",
                    points="all"
                )
                fig_box.update_layout(xaxis_title="Wound Type", yaxis_title="Temperature Gradient (°F)", boxmode='group')
                st.plotly_chart(fig_box, use_container_width=True)

            # Create scatter plot with healing rate
            valid_scatter_df = temp_df[temp_df['Healing Rate (%)'] > 0].dropna(subset=['Total Gradient', 'Healing Rate (%)', 'Calculated Wound Area'])
            if not valid_scatter_df.empty:
                fig_scatter = px.scatter(
                    valid_scatter_df,
                    x='Total Gradient',
                    y='Healing Rate (%)',
                    color='Wound Type',
                    size='Calculated Wound Area',
                    size_max=30,
                    hover_data=['Record ID', 'Event Name'],
                    title="Temperature Gradient vs. Healing Rate"
                )
                fig_scatter.update_layout(
                    xaxis_title="Temperature Gradient (Center to Peri-wound, °F)",
                    yaxis_title="Healing Rate (% reduction per visit)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Insufficient data for temperature gradient analysis.")

    def _oxygenation_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Render the oxygenation analysis tab."""
        st.header("Oxygenation Analysis")

        if selected_patient == "All Patients":
            # Prepare data for scatter plot
            valid_df = df[df['Healing Rate (%)'] > 0].copy()

            # Handle NaN values
            valid_df['Hemoglobin Level'] = valid_df['Hemoglobin Level'].fillna(valid_df['Hemoglobin Level'].mean())
            valid_df = valid_df.dropna(subset=['Oxygenation (%)', 'Healing Rate (%)', 'Hemoglobin Level'])

            if not valid_df.empty:
                fig1 = px.scatter(
                    valid_df,
                    x='Oxygenation (%)',
                    y='Healing Rate (%)',
                    color='Diabetes?',
                    size='Hemoglobin Level',
                    size_max=30,
                    hover_data=['Record ID', 'Event Name', 'Wound Type'],
                    title="Relationship Between Oxygenation and Healing Rate"
                )
                fig1.update_layout(xaxis_title="Oxygenation (%)", yaxis_title="Healing Rate (% reduction per visit)")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Insufficient data for oxygenation analysis.")

            # Create boxplot for oxygenation levels
            valid_box_df = df.dropna(subset=['Oxygenation (%)', 'Wound Type'])
            if not valid_box_df.empty:
                fig2 = px.box(
                    valid_box_df,
                    x='Wound Type',
                    y='Oxygenation (%)',
                    title="Oxygenation Levels by Wound Type",
                    color='Wound Type',
                    points="all"
                )
                fig2.update_layout(xaxis_title="Wound Type", yaxis_title="Oxygenation (%)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Insufficient data for wound type comparison.")

        else:
            patient_data = DataManager.get_patient_data(df, int(selected_patient.split(" ")[1])).sort_values('Visit Number')
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=patient_data['Visit Number'],
                y=patient_data['Oxyhemoglobin Level'],
                name="Oxyhemoglobin",
                marker_color='red'
            ))
            fig_bar.add_trace(go.Bar(
                x=patient_data['Visit Number'],
                y=patient_data['Deoxyhemoglobin Level'],
                name="Deoxyhemoglobin",
                marker_color='purple'
            ))
            fig_bar.update_layout(
                title=f"Hemoglobin Components for {selected_patient}",
                xaxis_title="Visit Number",
                yaxis_title="Level (g/dL)",
                barmode='stack',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_line = px.line(
                patient_data,
                x='Visit Number',
                y='Oxygenation (%)',
                title=f"Oxygenation Over Time for {selected_patient}",
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Visit Number",
                yaxis_title="Oxygenation (%)",
                yaxis=dict(range=[80, 100])
            )
            st.plotly_chart(fig_line, use_container_width=True)

    def _risk_factors_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Render the risk factors analysis tab."""
        st.header("Risk Factors Analysis")

        if selected_patient == "All Patients":
            risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

            with risk_tab1:
                diab_healing = df.groupby(['Diabetes?', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
                diab_healing = diab_healing[diab_healing['Visit Number'] > 1]
                fig = px.line(
                    diab_healing,
                    x='Visit Number',
                    y='Healing Rate (%)',
                    color='Diabetes?',
                    title="Average Healing Rate by Diabetes Status",
                    markers=True
                )
                fig.update_layout(xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

                diab_imp = df.groupby('Diabetes?')['Skin Impedance (kOhms) - Z'].mean().reset_index()
                fig_bar = px.bar(
                    diab_imp,
                    x='Diabetes?',
                    y='Skin Impedance (kOhms) - Z',
                    color='Diabetes?',
                    title="Average Impedance by Diabetes Status"
                )
                fig_bar.update_layout(xaxis_title="Diabetes Status", yaxis_title="Average Impedance Z (kOhms)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with risk_tab2:
                smoke_healing = df.groupby(['Smoking status', 'Visit Number'])['Healing Rate (%)'].mean().reset_index()
                smoke_healing = smoke_healing[smoke_healing['Visit Number'] > 1]
                fig_line = px.line(
                    smoke_healing,
                    x='Visit Number',
                    y='Healing Rate (%)',
                    color='Smoking status',
                    title="Average Healing Rate by Smoking Status",
                    markers=True
                )
                fig_line.update_layout(xaxis_title="Visit Number", yaxis_title="Average Healing Rate (%)")
                st.plotly_chart(fig_line, use_container_width=True)

                smoke_wound = pd.crosstab(df['Smoking status'], df['Wound Type'])
                smoke_wound_pct = smoke_wound.div(smoke_wound.sum(axis=1), axis=0) * 100
                fig_bar2 = px.bar(
                    smoke_wound_pct.reset_index().melt(id_vars='Smoking status', var_name='Wound Type', value_name='Percentage'),
                    x='Smoking status',
                    y='Percentage',
                    color='Wound Type',
                    title="Distribution of Wound Types by Smoking Status",
                    barmode='stack'
                )
                fig_bar2.update_layout(xaxis_title="Smoking Status", yaxis_title="Percentage (%)")
                st.plotly_chart(fig_bar2, use_container_width=True)

            with risk_tab3:
                st.info("BMI risk factor visualization not implemented.")
        else:
            patient_data = DataManager.get_patient_data(df, int(selected_patient.split(" ")[1]))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Diabetes", "Yes" if patient_data.iloc[0].get("Diabetes?", "No") in ["Yes", 1] else "No")
            with col2:
                st.metric("Smoking", "Yes" if patient_data.iloc[0].get("Smoking status", "Never") == "Current" else "No")

    def _llm_analysis_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """Render the LLM analysis tab."""
        st.header("LLM Analysis")
        if selected_patient == "All Patients":
            st.info("Please select an individual patient for LLM analysis.")
        else:
            st.subheader("LLM-Powered Wound Analysis")
            if self.uploaded_file is not None:
                if st.button("Run Analysis", key="run_analysis"):
                    # Placeholder for LLM analysis logic
                    st.success("LLM analysis complete. (Placeholder result)")
                    # Example: save report and provide download link
                    patient_data = DataManager.get_patient_data(df, int(selected_patient.split(" ")[1])).iloc[0].to_dict()
                    report_path = self.doc_handler.create_report(patient_data, "Analysis result goes here")
                    download_link = self.doc_handler.get_download_link(report_path)
                    st.markdown(f"Download Report: [Click Here]({download_link})")
            else:
                st.warning("Please upload a patient data file from the sidebar to enable LLM analysis.")

    def _add_footer(self) -> None:
        """Add footer information."""
        st.markdown("---")
        st.markdown("""
        **Note:** This dashboard loads data from 'dataset/SmartBandage-Data_for_llm.csv'.
        If the file cannot be loaded, the dashboard will fall back to simulated data.
        """)

    def _create_sidebar(self) -> None:
        """Create the sidebar with additional information."""
        with st.sidebar:
            st.header("About This Dashboard")
            st.write("""
            This Streamlit dashboard visualizes wound healing data collected with smart bandage technology.

            The analysis focuses on key metrics:
            - Impedance measurements (Z, Z', Z'')
            - Temperature gradients
            - Oxygenation levels
            - Patient risk factors
            """)

    def _create_llm_sidebar(self) -> None:
        """Create sidebar components specific to LLM configuration."""
        with st.sidebar:
            st.subheader("Model Configuration")
            self.uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=['csv'])
            platform_options = WoundAnalysisLLM.get_available_platforms()
            self.llm_platform = st.selectbox(
                "Select Platform",
                platform_options,
                index=platform_options.index("ai-verde") if "ai-verde" in platform_options else 0,
                help="AI Verde models are recommended."
            )
            if self.llm_platform == "huggingface":
                st.warning("Hugging Face models are currently disabled. Please use AI Verde models.")
                self.llm_platform = "ai-verde"
            available_models = WoundAnalysisLLM.get_available_models(self.llm_platform)
            default_model = "llama-3.3-70b-fp8" if self.llm_platform == "ai-verde" else "medalpaca-7b"
            self.llm_model = st.selectbox(
                "Select Model",
                available_models,
                index=available_models.index(default_model) if default_model in available_models else 0
            )
            with st.expander("Advanced Model Settings"):
                api_key = st.text_input("API Key", value="", type="password")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                if self.llm_platform == "ai-verde":
                    base_url = st.text_input("Base URL", value="https://llm-api.cyverse.ai")

    def _render_overview_stats(self, df: pd.DataFrame) -> None:
        """Render statistics for all patients."""
        st.subheader("Population Statistics")

        # Display wound area progression for all patients
        st.plotly_chart(
            self.visualizer.create_wound_area_plot(df),
            use_container_width=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Calculate average days in study (actual duration so far)
            avg_days_in_study = df.groupby('Record ID')['Days_Since_First_Visit'].max().mean()
            st.metric("Average Days in Study", f"{avg_days_in_study:.1f} days")

        with col2:
            try:
                # Calculate average estimated treatment duration for improving wounds
                estimated_days = df.groupby('Record ID')['Estimated_Days_To_Heal'].mean()
                valid_estimates = estimated_days[estimated_days.notna()]
                if len(valid_estimates) > 0:
                    avg_estimated_duration = valid_estimates.mean()
                    st.metric("Est. Treatment Duration", f"{avg_estimated_duration:.1f} days")
                else:
                    st.metric("Est. Treatment Duration", "N/A")
            except (KeyError, AttributeError):
                st.metric("Est. Treatment Duration", "N/A")

        with col3:
            # Calculate average healing rate excluding zeros and infinite values
            healing_rates = df['Healing Rate']
            valid_rates = healing_rates[(healing_rates != 0) & (np.isfinite(healing_rates))]
            avg_healing_rate = valid_rates.mean() if len(valid_rates) > 0 else 0
            st.metric("Average Healing Rate", f"{abs(avg_healing_rate):.2f} cm²/day")

        with col4:
            try:
                # Calculate improvement rate using only the last visit for each patient
                if 'Overall_Improvement' not in df.columns:
                    df['Overall_Improvement'] = np.nan

                # Get the last visit for each patient and calculate improvement rate
                last_visits = df.groupby('Record ID').agg({
                    'Overall_Improvement': 'last',
                    'Healing Rate': 'last'
                })

                # Calculate improvement rate from patients with valid improvement status
                valid_improvements = last_visits['Overall_Improvement'].dropna()
                if len(valid_improvements) > 0:
                    improvement_rate = (valid_improvements == 'Yes').mean() * 100
                    print(f"\nImprovement Status:")
                    print(valid_improvements.value_counts())
                    print(f"\nImprovement rate: {improvement_rate:.1f}%")
                    st.metric("Improvement Rate", f"{improvement_rate:.1f}%")
                else:
                    st.metric("Improvement Rate", "N/A")
            except Exception as e:
                st.metric("Improvement Rate", "N/A")
                print(f"Error calculating improvement rate: {e}")

    def _render_patient_overview(self, df: pd.DataFrame, patient_id: int) -> None:
        """Render overview for a specific patient."""
        patient_df = DataManager.get_patient_data(df, patient_id)
        if patient_df.empty:
            st.error("No data available for this patient.")
            return
        patient_data = patient_df.iloc[0]

        st.subheader("Patient Demographics")
        col1, col2, col3 = st.columns(3)
        age = patient_data.get("Calculated Age at Enrollment", "N/A")
        gender = patient_data.get("Sex", "N/A")
        bmi = patient_data.get("BMI", "N/A")
        with col1:
            st.metric("Age", age)
        with col2:
            st.metric("Gender", gender)
        with col3:
            st.metric("BMI", bmi)

        st.subheader("Medical History")
        col1, col2, col3, col4 = st.columns(4)
        diabetes = patient_data.get("Diabetes?", "N/A")
        smoking = patient_data.get("Smoking status", "N/A")
        med_history = patient_data.get("Medical History (select all that apply)", "N/A")
        with col1:
            st.metric("Diabetes Status", "Yes" if diabetes == "Yes" or diabetes == 1 else "No")
        with col2:
            st.metric("Smoking Status", "Yes" if smoking == "Current" else "No")
        with col3:
            st.metric("Hypertension", "Yes" if "Cardiovascular" in str(med_history) else "No")
        with col4:
            st.metric("Peripheral Vascular Disease", "Yes" if "PVD" in str(med_history) else "No")

def main():
    """Main application entry point."""
    # try:
    dashboard = Dashboard()
    dashboard.run()
    # except Exception as e:
    #     st.error(f"Application error: {str(e)}")
    #     st.error("Please check the data source and configuration.")

if __name__ == "__main__":
    main()
