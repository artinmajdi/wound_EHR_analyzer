from dataclasses import dataclass, field
import streamlit as st

from wound_analysis.utils.column_schema import DataColumns

@dataclass
class DashboardSettings:
    """Configuration class for the Wound Care Management & Interpreter Dashboard.

    This class defines constants, settings, and utility methods used throughout the
    dashboard application, including page layout settings, data schema definitions,
    and analysis parameters.

    Attributes:
        PAGE_TITLE (str): The title displayed on the dashboard.
        PAGE_ICON (str): The emoji icon used for the page favicon.
        LAYOUT (str): The Streamlit layout type, set to "wide".
        COLUMN_SCHEMA (DataColumns): Schema definition for data columns.
        EXUDATE_TYPE_INFO (dict): Dictionary containing clinical information about different
            types of wound exudate, including descriptions, clinical indications, and severity levels.

    Methods:
        get_exudate_analysis: Analyzes exudate characteristics and provides clinical interpretation.
        initialize: Sets up the Streamlit session state with necessary variables.
    """

    PAGE_TITLE: str = "Wound Care Management & Interpreter Dashboard"
    PAGE_ICON : str = "🩹"
    LAYOUT    : str = "wide"
    COLUMN_SCHEMA: DataColumns            = field(default_factory=DataColumns)

	# Constants for bioimpedance analysis
	# ANOMALY_THRESHOLD                       : float = 2.5  # Z-score threshold for anomaly detection
	# MIN_VISITS_FOR_ANALYSIS                 : int   = 3  # Minimum visits needed for trend analysis
	# SIGNIFICANT_CHANGE_THRESHOLD            : float = 15.0  # Percentage change threshold
	# INFECTION_RISK_RATIO_THRESHOLD          : float = 15.0  # Low/high frequency ratio threshold
	# SIGNIFICANT_CHANGE_THRESHOLD_IMPEDANCE  : float = 15.0  # Percentage change threshold for resistance/absolute impedance
	# SIGNIFICANT_CHANGE_THRESHOLD_CAPACITANCE: float = 20.0  # Percentage change threshold for capacitance
	# INFLAMMATORY_INCREASE_THRESHOLD         : float = 30.0  # Percentage increase threshold for low-frequency resistance


    # Exudate analysis information
    EXUDATE_TYPE_INFO = {
        'Serous': {
            'description': 'Straw-colored, clear, thin',
            'indication': 'Normal healing process',
            'severity': 'info'
        },
        'Serosanguineous': {
            'description': 'Pink or light red, thin',
            'indication': 'Presence of blood cells in early healing',
            'severity': 'info'
        },
        'Sanguineous': {
            'description': 'Red, thin',
            'indication': 'Active bleeding or trauma',
            'severity': 'warning'
        },
        'Seropurulent': {
            'description': 'Cloudy, milky, or creamy',
            'indication': 'Possible early infection or inflammation',
            'severity': 'warning'
        },
        'Purulent': {
            'description': 'Yellow, tan, or green, thick',
            'indication': 'Active infection present',
            'severity': 'error'
        }
    }

    @staticmethod
    def get_exudate_analysis(volume: str, viscosity: str, exudate_types: str = None) -> dict:
        """
            Provides clinical interpretation and treatment implications for exudate characteristics.

            Args:
                volume: The exudate volume level ('High', 'Medium', or 'Low')
                viscosity: The exudate viscosity level ('High', 'Medium', or 'Low')
                exudate_type: The type of exudate (e.g., 'Serous', 'Purulent')

            Returns:
                A dictionary containing:
                - volume_analysis: Interpretation of volume level
                - viscosity_analysis: Interpretation of viscosity level
                - type_info: Information about the exudate type
                - treatment_implications: List of treatment recommendations
        """

        result = {
            "volume_analysis": "",
            "viscosity_analysis": "",
            "type_info": {},
            "treatment_implications": []
        }

        # Volume interpretation
        if volume == 'High':
            result["volume_analysis"] = """
            **High volume exudate** is common in:
            - Chronic venous leg ulcers
            - Dehisced surgical wounds
            - Inflammatory ulcers
            - Burns

            This may indicate active inflammation or healing processes.
            """
        elif volume == 'Low':
            result["volume_analysis"] = """
            **Low volume exudate** is typical in:
            - Necrotic wounds
            - Ischaemic/arterial wounds
            - Neuropathic diabetic foot ulcers

            Monitor for signs of insufficient moisture.
            """

        # Viscosity interpretation
        if viscosity == 'High':
            result["viscosity_analysis"] = """
            **High viscosity** (thick) exudate may indicate:
            - High protein content
            - Possible infection
            - Inflammatory processes
            - Presence of necrotic material

            Consider reassessing treatment approach.
            """
        elif viscosity == 'Low':
            result["viscosity_analysis"] = """
            **Low viscosity** (thin) exudate may suggest:
            - Low protein content
            - Possible venous condition
            - Potential malnutrition
            - Presence of fistulas

            Monitor fluid balance and nutrition.
            """
        for exudate_type in [t.strip() for t in exudate_types.split(',')]:
            if exudate_type and exudate_type in DashboardSettings.EXUDATE_TYPE_INFO:
                result["type_info"][exudate_type] = DashboardSettings.EXUDATE_TYPE_INFO[exudate_type]

        # Treatment implications based on volume and viscosity
        if volume == 'High' and viscosity == 'High':
            result["treatment_implications"] = [
                "- Consider highly absorbent dressings",
                "- More frequent dressing changes may be needed",
                "- Monitor for maceration of surrounding tissue"
            ]
        elif volume == 'Low' and viscosity == 'Low':
            result["treatment_implications"] = [
                "- Use moisture-retentive dressings",
                "- Protect wound bed from desiccation",
                "- Consider hydrating dressings"
            ]

        return result

    @staticmethod
    def initialize() -> None:
        """
        Initializes the Streamlit session state with necessary variables.

        This function sets up the following session state variables if they don't exist:
        - processor: Stores the wound image processor instance
        - analysis_complete: Boolean flag indicating if analysis has been completed
        - analysis_results: Stores the results of wound analysis
        - report_path: Stores the file path to the generated report

        Returns:
            None
        """

        if 'processor' not in st.session_state:
            st.session_state.processor = None

        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False

        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None

        if 'report_path' not in st.session_state:
            st.session_state.report_path = None


