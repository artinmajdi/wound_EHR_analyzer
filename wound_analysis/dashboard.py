# Standard library imports
import os
import pathlib
from typing import Optional

# Third-party imports
import pandas as pd
import plotly.express as px
import streamlit as st

# Load environment variables from .env file
from dotenv import load_dotenv

# Local application imports
try:
    # First try direct imports (for Streamlit Cloud deployment)
    from wound_analysis.dashboard_components.exudate_tab import ExudateTab
    from wound_analysis.dashboard_components.impedance_tab import ImpedanceTab
    from wound_analysis.dashboard_components.llm_analysis_tab import LLMAnalysisTab
    from wound_analysis.dashboard_components.overview_tab import OverviewTab
    from wound_analysis.dashboard_components.oxygenation_tab import OxygenationTab
    from wound_analysis.dashboard_components.risk_factors_tab import RiskFactorsTab
    from wound_analysis.dashboard_components.temperature_tab import TemperatureTab
    from wound_analysis.dashboard_components.settings import DashboardSettings
    from wound_analysis.dashboard_components.visualizer import Visualizer
    
    from wound_analysis.utils.correlation_analysis import CorrelationAnalysis
    from wound_analysis.utils.data_manager import DataManager
    from wound_analysis.utils.impedance_analyzer import ImpedanceAnalyzer
    from wound_analysis.utils.llm_analyzer import WoundAnalysisLLM
    from wound_analysis.utils.wound_data_processor import WoundDataProcessor
except ImportError as e:
    # If direct imports fail, try package imports
    try:
        from wound_analysis.dashboard_components import (
            ExudateTab,
            ImpedanceTab,
            LLMAnalysisTab,
            OverviewTab,
            OxygenationTab,
            RiskFactorsTab,
            TemperatureTab,
        )
        from wound_analysis.utils import (
            CorrelationAnalysis,
            DataManager,
            ImpedanceAnalyzer,
            WoundAnalysisLLM,
            WoundDataProcessor,
        )
        from wound_analysis.dashboard_components.settings import DashboardSettings
        from wound_analysis.dashboard_components.visualizer import Visualizer
    except ImportError:
        # If absolute imports fail, try relative imports (for development environment)
        from .dashboard_components import (
            ExudateTab,
            ImpedanceTab,
            LLMAnalysisTab,
            OverviewTab,
            OxygenationTab,
            RiskFactorsTab,
            TemperatureTab,
        )
        from .utils import (
            CorrelationAnalysis,
            DataManager,
            ImpedanceAnalyzer,
            WoundAnalysisLLM,
            WoundDataProcessor,
        )
        from .dashboard_components.settings import DashboardSettings
        from .dashboard_components.visualizer import Visualizer

# Try to load environment variables from different possible locations
env_paths = [
    pathlib.Path(__file__).parent.parent / '.env',  # Project root .env
    pathlib.Path.cwd() / '.env',                    # Current working directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break

# Debug mode disabled
st.set_option('client.showErrorDetails', True)

def debug_log(message):
    """Write debug messages to a file and display in Streamlit"""
    try:
        # Only attempt to write to file in environments where it might work
        if os.environ.get('WRITE_DEBUG_LOG', 'false').lower() == 'true':
            with open('/app/debug.log', 'a') as f:
                f.write(f"{message}\n")
    except Exception:
        pass
    # Always show in sidebar for debugging
    st.sidebar.text(message)


class Dashboard:
    """Main dashboard class for the Wound Analysis application.

    This class serves as the core controller for the wound analysis dashboard, integrating
    data processing, visualization, and analysis components. It handles the initialization,
    setup, and rendering of the Streamlit application interface.

    The dashboard provides comprehensive wound analysis features including:
    - Overview of patient data and population statistics
    - Impedance analysis with clinical interpretation
    - Temperature gradient analysis
    - Tissue oxygenation assessment
    - Exudate characterization
    - Risk factor evaluation
    - LLM-powered wound analysis

    Attributes:
        DashboardSettings (DashboardSettings): Configuration settings for the application
        data_manager (DataManager): Handles data loading and processing operations
        visualizer (Visualizer): Creates data visualizations for the dashboard
        impedance_analyzer (ImpedanceAnalyzer): Processes and interprets impedance measurements
        llm_platform (str): Selected platform for LLM analysis (e.g., "ai-verde")
        llm_model (str): Selected LLM model for analysis
        csv_dataset_path (str): Path to the uploaded CSV dataset
        data_processor (WoundDataProcessor): Processes wound data for analysis
        impedance_freq_sweep_path (pathlib.Path): Path to impedance frequency sweep data files

    Methods:
        setup(): Initializes the Streamlit page configuration and sidebar
        load_data(uploaded_file): Loads data from the uploaded CSV file
        run(): Main execution method that runs the dashboard application
        _create_dashboard_tabs(): Creates and manages the main dashboard tabs
        _overview_tab(): Renders the overview tab for patient data
        _impedance_tab(): Renders impedance analysis visualization and interpretation
        _temperature_tab(): Renders temperature analysis visualization
        _oxygenation_tab(): Renders oxygenation analysis visualization
        _exudate_tab(): Renders exudate analysis visualization
        _risk_factors_tab(): Renders risk factors analysis visualization
        _llm_analysis_tab(): Renders the LLM-powered analysis interface
        _create_left_sidebar(): Creates the sidebar with configuration options
        """
    def __init__(self):
        """Initialize the dashboard."""
        self.DashboardSettings             = DashboardSettings()
        self.data_manager       = DataManager()
        self.visualizer         = Visualizer()
        self.impedance_analyzer = ImpedanceAnalyzer()

        # LLM configuration placeholders
        self.llm_platform   = None
        self.llm_model      = None
        self.csv_dataset_path  = None
        self.data_processor = None
        self.impedance_freq_sweep_path: pathlib.Path = None

    def setup(self) -> None:
        """Set up the dashboard configuration."""
        st.set_page_config(
            page_title = self.DashboardSettings.PAGE_TITLE,
            page_icon  = self.DashboardSettings.PAGE_ICON,
            layout     = self.DashboardSettings.LAYOUT
        )
        DashboardSettings.initialize()
        self._create_left_sidebar()

    @staticmethod
    # @st.cache_data
    def load_data(uploaded_file) -> Optional[pd.DataFrame]:
        """
        Loads data from an uploaded file into a pandas DataFrame.

        This function serves as a wrapper around the DataManager's load_data method,
        providing consistent data loading functionality for the dashboard.

        Args:
            uploaded_file: The file uploaded by the user through the application interface (typically a csv, excel, or other supported format)

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data, or None if the file couldn't be loaded

        Note:
            The actual loading logic is handled by the DataManager class.
        """
        df = DataManager.load_data(uploaded_file)
        return df

    def run(self) -> None:
        """
        Run the main dashboard application.

        This method initializes the dashboard, loads the dataset, processes wound data,
        sets up the page layout including title and patient selection dropdown,
        and creates the dashboard tabs.

        If no CSV file is uploaded, displays an information message.
        If data loading fails, displays an error message.

        Returns:
            None
        """

        self.setup()
        if not self.csv_dataset_path:
            st.info("Please upload a CSV file to proceed.")
            return

        df = self.load_data(self.csv_dataset_path)

        if df is None:
            st.error("Failed to load data. Please check the CSV file.")
            return

        self.data_processor = WoundDataProcessor(df=df, impedance_freq_sweep_path=self.impedance_freq_sweep_path)

        # Header
        st.title(self.DashboardSettings.PAGE_TITLE)

        # Patient selection
        patient_ids = sorted(df['Record ID'].unique())
        patient_options = ["All Patients"] + [f"Patient {id:d}" for id in patient_ids]
        selected_patient = st.selectbox("Select Patient", patient_options)

        # Create tabs
        self._create_dashboard_tabs(df, selected_patient)

    def _create_dashboard_tabs(self, df: pd.DataFrame, selected_patient: str) -> None:
        """
        Create and manage dashboard tabs for displaying patient wound data.

        This method sets up the main dashboard interface with multiple tabs for different
        wound analysis categories. Each tab is populated with specific visualizations and
        data analyses related to the selected patient.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe containing wound data for analysis
        selected_patient : str
            The identifier of the currently selected patient

        Returns:
        --------
        None
            This method updates the Streamlit UI directly without returning values

        Notes:
        ------
        The following tabs are created:
        - Overview          : General patient information and wound summary
        - Impedance Analysis: Electrical measurements of wound tissue
        - Temperature       : Thermal measurements and analysis
        - Oxygenation       : Oxygen saturation and related metrics
        - Exudate           : Analysis of wound drainage
        - Risk Factors      : Patient-specific risk factors for wound healing
        - LLM Analysis      : Natural language processing analysis of wound data
        """

        tabs = st.tabs([
            "Overview",
            "Impedance Analysis",
            "Temperature",
            "Oxygenation",
            "Exudate",
            "Risk Factors",
            "LLM Analysis"
        ])

        argsv = dict(df=df, selected_patient=selected_patient, data_processor=self.data_processor)
        with tabs[0]:
            OverviewTab(**argsv).render()
        with tabs[1]:
            ImpedanceTab(**argsv).render()
        with tabs[2]:
            TemperatureTab(**argsv).render()
        with tabs[3]:
            OxygenationTab(**argsv).render()
        with tabs[4]:
            ExudateTab(**argsv).render()
        with tabs[5]:
            RiskFactorsTab(**argsv).render()
        with tabs[6]:
            LLMAnalysisTab(selected_patient=selected_patient, data_processor=self.data_processor, llm_platform=self.llm_platform, llm_model=self.llm_model, csv_dataset_path=self.csv_dataset_path).render()

    def _exudate_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """
            Create and display the exudate analysis tab in the wound management dashboard.

            This tab provides detailed analysis of wound exudate characteristics including volume,
            viscosity, and type. For aggregate patient data, it shows statistical correlations and
            visualizations comparing exudate properties across different wound types. For individual
            patients, it displays a timeline of exudate changes and provides clinical interpretations
            for each visit.

            Parameters
            ----------
            df : pd.DataFrame
                The dataframe containing wound assessment data for all patients
            selected_patient : str
                The currently selected patient ID or "All Patients"

            Returns:
            -------
            None
                The method updates the Streamlit UI directly

            Notes:
            -----
            For aggregate analysis, this method:
            - Calculates correlations between exudate characteristics and healing rates
            - Creates boxplots comparing exudate properties across wound types
            - Generates scatter plots to visualize relationships between variables
            - Shows distributions of exudate types by wound category

            For individual patient analysis, this method:
            - Displays a timeline chart of exudate changes
            - Provides clinical interpretations for each visit
            - Offers treatment recommendations based on exudate characteristics
        """

        st.header("Exudate Analysis")

        if selected_patient == "All Patients":
            # Create a copy of the dataframe for exudate analysis
            exudate_df = df.copy()

            # Define exudate columns
            exudate_cols = ['Exudate Volume', 'Exudate Viscosity', 'Exudate Type']

            # Drop rows with missing exudate data
            exudate_df = exudate_df.dropna(subset=exudate_cols)

            # Create numerical mappings for volume and viscosity
            level_mapping = {
                'Low': 1,
                'Medium': 2,
                'High': 3
            }

            # Convert volume and viscosity to numeric values
            exudate_df['Volume_Numeric']    = exudate_df['Exudate Volume'].map(level_mapping)
            exudate_df['Viscosity_Numeric'] = exudate_df['Exudate Viscosity'].map(level_mapping)

            if not exudate_df.empty:
                # Create two columns for volume and viscosity analysis
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Volume Analysis")
                    # Calculate correlation between volume and healing rate
                    valid_df = exudate_df.dropna(subset=['Volume_Numeric', 'Healing Rate (%)'])

                    if len(valid_df) > 1:
                        stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Volume_Numeric', y_col='Healing Rate (%)', REMOVE_OUTLIERS=False)
                        _, _, _ = stats_analyzer.calculate_correlation()
                        st.info(stats_analyzer.get_correlation_text(text="Volume correlation vs Healing Rate"))

                    # Boxplot of exudate volume by wound type
                    fig_vol = px.box(
                        exudate_df,
                        x='Wound Type',
                        y='Exudate Volume',
                        title="Exudate Volume by Wound Type",
                        points="all"
                    )

                    fig_vol.update_layout(
                        xaxis_title="Wound Type",
                        yaxis_title="Exudate Volume",
                        boxmode='group'
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                with col2:
                    st.subheader("Viscosity Analysis")
                    # Calculate correlation between viscosity and healing rate
                    valid_df = exudate_df.dropna(subset=['Viscosity_Numeric', 'Healing Rate (%)'])

                    if len(valid_df) > 1:
                        stats_analyzer = CorrelationAnalysis(data=valid_df, x_col='Viscosity_Numeric', y_col='Healing Rate (%)', REMOVE_OUTLIERS=False)
                        _, _, _ = stats_analyzer.calculate_correlation()
                        st.info(stats_analyzer.get_correlation_text(text="Viscosity correlation vs Healing Rate"))


                    # Boxplot of exudate viscosity by wound type
                    fig_visc = px.box(
                        exudate_df,
                        x='Wound Type',
                        y='Exudate Viscosity',
                        title="Exudate Viscosity by Wound Type",
                        points="all"
                    )
                    fig_visc.update_layout(
                        xaxis_title="Wound Type",
                        yaxis_title="Exudate Viscosity",
                        boxmode='group'
                    )
                    st.plotly_chart(fig_visc, use_container_width=True)

                # Create scatter plot matrix for volume, viscosity, and healing rate
                st.subheader("Relationship Analysis")

                exudate_df['Healing Rate (%)']      = exudate_df['Healing Rate (%)'].clip(-100, 100)
                exudate_df['Calculated Wound Area'] = exudate_df['Calculated Wound Area'].fillna(exudate_df['Calculated Wound Area'].mean())

                fig_scatter = px.scatter(
                    exudate_df,
                    x='Volume_Numeric',
                    y='Healing Rate (%)',
                    color='Wound Type',
                    size='Calculated Wound Area',
                    hover_data=['Record ID', 'Event Name', 'Exudate Volume', 'Exudate Viscosity', 'Exudate Type'],
                    title="Exudate Characteristics vs. Healing Rate"
                )
                fig_scatter.update_layout(
                    xaxis_title="Exudate Volume (1=Low, 2=Medium, 3=High)",
                    yaxis_title="Healing Rate (% reduction per visit)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Display distribution of exudate types
                st.subheader("Exudate Type Distribution")
                col1, col2 = st.columns(2)

                with col1:
                    # Distribution by wound type
                    type_by_wound = pd.crosstab(exudate_df['Wound Type'], exudate_df['Exudate Type'])
                    fig_type = px.bar(
                        type_by_wound.reset_index().melt(id_vars='Wound Type', var_name='Exudate Type', value_name='Percentage'),
                        x='Wound Type',
                        y='Percentage',
                        color='Exudate Type',
                    )

                    st.plotly_chart(fig_type, use_container_width=True)

                with col2:
                    # Overall distribution
                    type_counts = exudate_df['Exudate Type'].value_counts()
                    fig_pie = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title="Overall Distribution of Exudate Types"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            else:
                st.warning("No valid exudate data available for analysis.")
        else:
            visits = self.data_processor.get_patient_visits(record_id=int(selected_patient.split(" ")[1]))['visits']

            # Display the exudate chart
            fig = Visualizer.create_exudate_chart(visits)
            st.plotly_chart(fig, use_container_width=True)

            # Clinical interpretation section
            st.subheader("Clinical Interpretation of Exudate Characteristics")

            # Create tabs for each visit
            visit_tabs = st.tabs([visit.get('visit_date', 'N/A') for visit in visits])

            # Process each visit in its corresponding tab
            for tab, visit in zip(visit_tabs, visits):
                with tab:
                    col1, col2 = st.columns(2)
                    volume           = visit['wound_info']['exudate'].get('volume', 'N/A')
                    viscosity        = visit['wound_info']['exudate'].get('viscosity', 'N/A')
                    exudate_type_str = str(visit['wound_info']['exudate'].get('type', 'N/A'))
                    exudate_analysis = DashboardSettings.get_exudate_analysis(volume=volume, viscosity=viscosity, exudate_types=exudate_type_str)

                    with col1:
                        st.markdown("### Volume Analysis")
                        st.write(f"**Current Level:** {volume}")
                        st.info(exudate_analysis['volume_analysis'])

                    with col2:
                        st.markdown("### Viscosity Analysis")
                        st.write(f"**Current Level:** {viscosity}")
                        if viscosity == 'High':
                            st.warning(exudate_analysis['viscosity_analysis'])
                        elif viscosity == 'Low':
                            st.info(exudate_analysis['viscosity_analysis'])

                    # Exudate Type Analysis
                    st.markdown('----')
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Type Analysis")
                        if exudate_type_str != 'N/A':
                            exudate_types = [t.strip() for t in exudate_type_str.split(',')]
                            st.write(f"**Current Types:** {exudate_types}")
                        else:
                            exudate_types = ['N/A']
                            st.write("**Current Type:** N/A")

                        # Process each exudate type
                        highest_severity = 'info'  # Track highest severity for overall implications
                        for exudate_type in exudate_types:
                            if exudate_type in DashboardSettings.EXUDATE_TYPE_INFO:
                                info = DashboardSettings.EXUDATE_TYPE_INFO[exudate_type]
                                message = f"""
                                **Description:** {info['description']} \n
                                **Clinical Indication:** {info['indication']}
                                """
                                if info['severity'] == 'error':
                                    st.error(message)
                                    highest_severity = 'error'
                                elif info['severity'] == 'warning' and highest_severity != 'error':
                                    st.warning(message)
                                    highest_severity = 'warning'
                                else:
                                    st.info(message)

                        # Overall Clinical Assessment based on multiple types
                        if len(exudate_types) > 1 and 'N/A' not in exudate_types:
                            st.markdown("#### Overall Clinical Assessment")
                            if highest_severity == 'error':
                                st.error("⚠️ Multiple exudate types present with signs of infection. Immediate clinical attention recommended.")
                            elif highest_severity == 'warning':
                                st.warning("⚠️ Mixed exudate characteristics suggest active wound processes. Close monitoring required.")
                            else:
                                st.info("Multiple exudate types present. Continue regular monitoring of wound progression.")

                    with col2:
                        st.markdown("### Treatment Implications")
                        if exudate_analysis['treatment_implications']:
                            st.write("**Recommended Actions:**")
                            st.success("\n".join(exudate_analysis['treatment_implications']))

    def _risk_factors_tab(self, df: pd.DataFrame, selected_patient: str) -> None:
        """
        Renders the Risk Factors Analysis tab in the Streamlit application.
        This tab provides analysis of how different risk factors (diabetes, smoking, BMI)
        affect wound healing rates. For the aggregate view ('All Patients'), it shows
        statistical distributions and comparisons across different patient groups.
        For individual patients, it displays a personalized risk profile and assessment.
        Features:
        - For all patients: Interactive tabs showing wound healing statistics across diabetes status, smoking status, and BMI categories, with visualizations and statistical summaries
        - For individual patients: Risk profile with factors like diabetes, smoking status and BMI, plus a computed risk score and estimated healing time
        Args:
            df (pd.DataFrame): The dataframe containing all patient wound data
            selected_patient (str): The currently selected patient ID as a string, or "All Patients" for aggregate view
        Returns:
            None: This method renders UI components directly in Streamlit
        """

        st.header("Risk Factors Analysis")

        if selected_patient == "All Patients":
            risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Diabetes", "Smoking", "BMI"])

            valid_df = df.dropna(subset=['Healing Rate (%)', 'Visit Number']).copy()

            # Add detailed warning for outliers with statistics
            # outliers = valid_df[abs(valid_df['Healing Rate (%)']) > 100]
            # if not outliers.empty:
            #     st.warning(
            #         f"⚠️ Data Quality Alert:\n\n"
            #         f"Found {len(outliers)} measurements ({(len(outliers)/len(valid_df)*100):.1f}% of data) "
            #         f"with healing rates outside the expected range (-100% to 100%).\n\n"
            #         f"Statistics:\n"
            #         f"- Minimum value: {outliers['Healing Rate (%)'].min():.1f}%\n"
            #         f"- Maximum value: {outliers['Healing Rate (%)'].max():.1f}%\n"
            #         f"- Mean value: {outliers['Healing Rate (%)'].mean():.1f}%\n"
            #         f"- Number of unique patients affected: {len(outliers['Record ID'].unique())}\n\n"
            #         "These values will be clipped to [-100%, 100%] range for visualization purposes."
            #     )

            # Clip healing rates to reasonable range
            # valid_df['Healing Rate (%)'] = valid_df['Healing Rate (%)'].clip(-100, 100)

            for col in ['Diabetes?', 'Smoking status', 'BMI']:
                # Add consistent diabetes status for each patient
                first_diabetes_status = valid_df.groupby('Record ID')[col].first()
                valid_df[col] = valid_df['Record ID'].map(first_diabetes_status)

            valid_df['Healing_Color'] = valid_df['Healing Rate (%)'].apply(
                lambda x: 'green' if x < 0 else 'red'
            )

            with risk_tab1:

                st.subheader("Impact of Diabetes on Wound Healing")

                # Ensure diabetes status is consistent for each patient
                valid_df['Diabetes?'] = valid_df['Diabetes?'].fillna('No')

                # Compare average healing rates by diabetes status
                diab_stats = valid_df.groupby('Diabetes?').agg({ 'Healing Rate (%)': ['mean', 'count', 'std'] }).round(2)

                # Create a box plot for healing rates with color coding
                fig1 = px.box(
                    valid_df,
                    x='Diabetes?',
                    y='Healing Rate (%)',
                    title="Healing Rate Distribution by Diabetes Status",
                    color='Healing_Color',
                    color_discrete_map={'green': 'green', 'red': 'red'},
                    points='all'
                )
                fig1.update_layout(
                    xaxis_title="Diabetes Status",
                    yaxis_title="Healing Rate (%)",
                    showlegend=True,
                    legend_title="Wound Status",
                    legend={'traceorder': 'reversed'},
                    yaxis=dict(
                        range=[-100, 100],
                        tickmode='linear',
                        tick0=-100,
                        dtick=25
                    )
                )

                # Update legend labels
                fig1.for_each_trace(lambda t: t.update(name='Improving' if t.name == 'green' else 'Worsening'))
                st.plotly_chart(fig1, use_container_width=True)

                # Display statistics
                st.write("**Statistical Summary:**")
                for status in diab_stats.index:
                    stats_data = diab_stats.loc[status]
                    improvement_rate = (valid_df[valid_df['Diabetes?'] == status]['Healing Rate (%)'] < 0).mean() * 100

                    st.write(f"- {status}: Average Healing Rate = {stats_data[('Healing Rate (%)', 'mean')]}% "
                            "(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
                            "SD={stats_data[('Healing Rate (%)', 'std')]}, "
                            "Improvement Rate={improvement_rate:.1f}%)")

                # Compare wound types distribution
                wound_diab = pd.crosstab(valid_df['Diabetes?'], valid_df['Wound Type'], normalize='index') * 100
                fig2 = px.bar(
                    wound_diab.reset_index().melt(id_vars='Diabetes?', var_name='Wound Type', value_name='Percentage'),
                    x='Diabetes?',
                    y='Percentage',
                    color='Wound Type',
                    title="Wound Type Distribution by Diabetes Status",
                    labels={'Percentage': 'Percentage of Wounds (%)'}
                )
                st.plotly_chart(fig2, use_container_width=True)

            with risk_tab2:
                st.subheader("Impact of Smoking on Wound Healing")

                # Clean smoking status
                valid_df['Smoking status'] = valid_df['Smoking status'].fillna('Never')

                # Create healing rate distribution by smoking status with color coding
                fig1 = px.box(
                    valid_df,
                    x='Smoking status',
                    y='Healing Rate (%)',
                    title="Healing Rate Distribution by Smoking Status",
                    color='Healing_Color',
                    color_discrete_map={'green': 'green', 'red': 'red'},
                    points='all'
                )
                fig1.update_layout(
                    showlegend=True,
                    legend_title="Wound Status",
                    legend={'traceorder': 'reversed'},
                    yaxis=dict(
                        range=[-100, 100],
                        tickmode='linear',
                        tick0=-100,
                        dtick=25
                    )
                )
                # Update legend labels
                fig1.for_each_trace(lambda t: t.update(name='Improving' if t.name == 'green' else 'Worsening'))
                st.plotly_chart(fig1, use_container_width=True)

                # Calculate and display statistics
                smoke_stats = valid_df.groupby('Smoking status').agg({
                    'Healing Rate (%)': ['mean', 'count', 'std']
                }).round(2)

                st.write("**Statistical Summary:**")
                for status in smoke_stats.index:
                    stats_data = smoke_stats.loc[status]
                    improvement_rate = (valid_df[valid_df['Smoking status'] == status]['Healing Rate (%)'] < 0).mean() * 100
                    st.write(f"- {status}: Average Healing Rate = {stats_data[('Healing Rate (%)', 'mean')]}% "
                            "(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
                            "SD={stats_data[('Healing Rate (%)', 'std')]}, "
                            "Improvement Rate={improvement_rate:.1f}%)")

                # Wound type distribution by smoking status
                wound_smoke = pd.crosstab(valid_df['Smoking status'], valid_df['Wound Type'], normalize='index') * 100
                fig2 = px.bar(
                    wound_smoke.reset_index().melt(id_vars='Smoking status', var_name='Wound Type', value_name='Percentage'),
                    x='Smoking status',
                    y='Percentage',
                    color='Wound Type',
                    title="Wound Type Distribution by Smoking Status",
                    labels={'Percentage': 'Percentage of Wounds (%)'}
                )
                st.plotly_chart(fig2, use_container_width=True)

            with risk_tab3:
                st.subheader("Impact of BMI on Wound Healing")

                # Create BMI categories
                bins = [0, 18.5, 24.9, 29.9, float('inf')]
                labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
                valid_df['BMI Category'] = pd.cut(valid_df['BMI'], bins=bins, labels=labels)

                # Create healing rate distribution by BMI category with color coding
                fig1 = px.box(
                    valid_df,
                    x='BMI Category',
                    y='Healing Rate (%)',
                    title="Healing Rate Distribution by BMI Category",
                    color='Healing_Color',
                    color_discrete_map={'green': 'green', 'red': 'red'},
                    points='all'
                )
                fig1.update_layout(
                    showlegend=True,
                    legend_title="Wound Status",
                    legend={'traceorder': 'reversed'},
                    yaxis=dict(
                        range=[-100, 100],
                        tickmode='linear',
                        tick0=-100,
                        dtick=25
                    )
                )
                # Update legend labels
                fig1.for_each_trace(lambda t: t.update(name='Improving' if t.name == 'green' else 'Worsening'))
                st.plotly_chart(fig1, use_container_width=True)

                # Calculate and display statistics
                bmi_stats = valid_df.groupby('BMI Category', observed=False).agg({
                    'Healing Rate (%)': ['mean', 'count', 'std']
                }).round(2)

                st.write("**Statistical Summary:**")
                for category in bmi_stats.index:
                    stats_data = bmi_stats.loc[category]
                    improvement_rate = (valid_df[valid_df['BMI Category'] == category]['Healing Rate (%)'] < 0).mean() * 100
                    st.write(f"- {category}: Average Healing Rate = {stats_data[('Healing Rate (%)', 'mean')]}% "
                            "(n={int(stats_data[('Healing Rate (%)', 'count')])}, "
                            "SD={stats_data[('Healing Rate (%)', 'std')]}, "
                            "Improvement Rate={improvement_rate:.1f}%)")

                # Wound type distribution by BMI category
                wound_bmi = pd.crosstab(valid_df['BMI Category'], valid_df['Wound Type'], normalize='index') * 100
                fig2 = px.bar(
                    wound_bmi.reset_index().melt(id_vars='BMI Category', var_name='Wound Type', value_name='Percentage'),
                    x='BMI Category',
                    y='Percentage',
                    color='Wound Type',
                    title="Wound Type Distribution by BMI Category",
                    labels={'Percentage': 'Percentage of Wounds (%)'}
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            # For individual patient
            df_temp = df[df['Record ID'] == int(selected_patient.split(" ")[1])].copy()
            patient_data = df_temp.iloc[0]

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

    def _get_input_user_data(self) -> None:
        """
        Get user inputs from Streamlit interface for data paths and validate them.

        This method provides UI components for users to:
        1. Upload a CSV file containing patient data
        2. Specify a path to the folder containing impedance frequency sweep XLSX files
        3. Validate the path and check for the existence of XLSX files

        The method populates:
        - self.csv_dataset_path: The uploaded CSV file
        - self.impedance_freq_sweep_path: Path to the folder containing impedance XLSX files

        Returns:
            None
        """

        self.csv_dataset_path = st.file_uploader("Upload Patient Data (CSV)", type=['csv'])

        default_path = str(pathlib.Path(__file__).parent.parent / "dataset/impedance_frequency_sweep")

        if self.csv_dataset_path is not None:
            # Text input for dataset path
            dataset_path_input = st.text_input(
                "Path to impedance_frequency_sweep folder",
                value=default_path,
                help="Enter the absolute path to the folder containing impedance frequency sweep XLSX files",
                key="dataset_path_input_1"
            )

            # Convert to Path object
            self.impedance_freq_sweep_path = pathlib.Path(dataset_path_input)

            # Button to check if files exist
            # if st.button("Check XLSX Files"):
            try:
                # Check if path exists
                if not self.impedance_freq_sweep_path.exists():
                    st.error(f"Path does not exist: {self.impedance_freq_sweep_path}")
                else:
                    # Count xlsx files
                    xlsx_files = list(self.impedance_freq_sweep_path.glob("**/*.xlsx"))

                    if xlsx_files:
                        st.success(f"Found {len(xlsx_files)} XLSX files in the directory")
                        # Show files in an expander
                        with st.expander("View Found Files"):
                            for file in xlsx_files:
                                st.text(f"- {file.name}")
                    else:
                        st.warning(f"No XLSX files found in {self.dataset_path}")
            except Exception as e:
                st.error(f"Error checking path: {e}")

    def _create_left_sidebar(self) -> None:
        """
        Creates the left sidebar of the Streamlit application.

        This method sets up the sidebar with model configuration options, file upload functionality,
        and informational sections about the dashboard. The sidebar includes:

        1. Model Configuration section:
            - File uploader for patient data (CSV files)
            - Platform selector (defaulting to ai-verde)
            - Model selector with appropriate defaults based on the platform
            - Advanced settings expandable section for API keys and base URLs

        2. Information sections:
            - About This Dashboard: Describes the purpose and data visualization focus
            - Statistical Methods: Outlines analytical approaches used in the dashboard

        Returns:
            None
        """

        with st.sidebar:
            st.markdown("### Dataset Configuration")
            self._get_input_user_data()

            st.markdown("---")
            st.subheader("Model Configuration")

            platform_options = WoundAnalysisLLM.get_available_platforms()

            self.llm_platform = st.selectbox( "Select Platform", platform_options,
                index=platform_options.index("ai-verde") if "ai-verde" in platform_options else 0,
                help="Hugging Face models are currently disabled. Please use AI Verde models."
            )

            if self.llm_platform == "huggingface":
                st.warning("Hugging Face models are currently disabled. Please use AI Verde models.")
                self.llm_platform = "ai-verde"

            available_models = WoundAnalysisLLM.get_available_models(self.llm_platform)
            default_model = "llama-3.3-70b-fp8" if self.llm_platform == "ai-verde" else "medalpaca-7b"
            self.llm_model = st.selectbox( "Select Model", available_models,
                index=available_models.index(default_model) if default_model in available_models else 0
            )

            # Add warning for deepseek-r1 model
            if self.llm_model == "deepseek-r1":
                st.warning("**Warning:** The DeepSeek R1 model is currently experiencing connection issues. Please select a different model for reliable results.", icon="⚠️")
                self.llm_model = "llama-3.3-70b-fp8"

            with st.expander("Advanced Model Settings"):

                api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

                if self.llm_platform == "ai-verde":
                    base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", ""))
                    if base_url:
                        os.environ["OPENAI_BASE_URL"] = base_url


def main():
    """Main application entry point."""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
