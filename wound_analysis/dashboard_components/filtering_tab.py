# Standard library imports
import logging
from typing import Dict, List, Literal, Union, Optional, Any

# Third-party imports
import pandas as pd
import numpy as np
import streamlit as st

# Local application imports
from wound_analysis.utils.column_schema import DColumns
from wound_analysis.utils.data_processor import WoundDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilteringTab:
    """
    A class for managing and rendering the Filtering tab in the wound analysis dashboard.

    This class contains methods to filter wound data based on various medical criteria
    such as demographics, medical history, wound characteristics, and treatment details.

    Attributes:
        wound_data_processor (WoundDataProcessor): The data processor containing the DataFrame
        df (pd.DataFrame): A reference to the DataFrame in the data processor
        CN (DColumns): Column name schema for the DataFrame
    """

    def __init__(self, wound_data_processor: WoundDataProcessor, selected_patient: Optional[str]=None):
        """
        Initialize the FilteringTab.

        Args:
            wound_data_processor (WoundDataProcessor): The data processor containing the DataFrame
        """
        self.wound_data_processor = wound_data_processor
        self.df = wound_data_processor.df
        self.CN = DColumns(df=self.df)

        # Initialize session state for filters if not already present
        if 'filters_applied' not in st.session_state:
            st.session_state.filters_applied = False
            st.session_state.filtered_df = None
            st.session_state.filter_stats = {
                'original_count': 0,
                'filtered_count': 0,
                'filter_summary': {}
            }

    def _filter_demographics(self) -> Dict[str, Any]:
        """
        Create UI controls and collect settings for demographic filters.

        Returns:
            Dict[str, Any]: Dictionary containing the demographic filter settings
        """
        st.markdown("### Demographics")

        col1, col2 = st.columns(2)

        filters = {}

        with col1:
            # Age range filter
            if self.CN.AGE in self.df.columns:
                min_age = int(self.df[self.CN.AGE].min())
                max_age = int(self.df[self.CN.AGE].max())
                age_range = st.slider("Age Range", min_value=min_age, max_value=max_age,
                                      value=(min_age, max_age))
                filters['age'] = age_range

            # Sex filter (not gender)
            if self.CN.SEX in self.df.columns:
                sex_options = ["All"] + list(self.df[self.CN.SEX].unique())
                selected_sex = st.selectbox("Sex", options=sex_options)
                if selected_sex != "All":
                    filters['sex'] = selected_sex

        with col2:
            # BMI range filter
            if self.CN.BMI in self.df.columns:
                min_bmi = float(self.df[self.CN.BMI].min())
                max_bmi = float(self.df[self.CN.BMI].max())
                bmi_range = st.slider("BMI Range", min_value=min_bmi, max_value=max_bmi,
                                     value=(min_bmi, max_bmi), step=0.1)
                filters['bmi'] = bmi_range

            # Race/Ethnicity filter
            if self.CN.RACE in self.df.columns:
                race_options = ["All"] + list(self.df[self.CN.RACE].unique())
                selected_race = st.selectbox("Race", options=race_options)
                if selected_race != "All":
                    filters['race'] = selected_race

        return filters

    def _filter_medical_history(self) -> Dict[str, Any]:
        """
        Create UI controls and collect settings for medical history filters.

        Returns:
            Dict[str, Any]: Dictionary containing the medical history filter settings
        """
        st.markdown("### Medical History")

        filters = {}

        # Debug section for showing unique values - using a container instead of an expander
        debug_container = st.container()
        with debug_container:
            show_debug = st.checkbox("Debug: Show unique values for medical history fields", value=False)
            if show_debug:
                if self.CN.DIABETES in self.df.columns:
                    diabetes_values = self.df[self.CN.DIABETES].unique()
                    st.write(f"Diabetes unique values: {diabetes_values}")

                if self.CN.CARDIOVASCULAR in self.df.columns:
                    cardio_values = self.df[self.CN.CARDIOVASCULAR].unique()
                    st.write(f"Cardiovascular unique values: {cardio_values}")

                if self.CN.SMOKING_STATUS in self.df.columns:
                    smoking_values = self.df[self.CN.SMOKING_STATUS].unique()
                    st.write(f"Smoking status unique values: {smoking_values}")

        col1, col2 = st.columns(2)

        with col1:
            # Diabetes filter
            if self.CN.DIABETES in self.df.columns:
                diabetes_values = sorted(self.df[self.CN.DIABETES].dropna().unique())
                if len(diabetes_values) > 0:
                    diabetes_options = ["All"] + [str(val) for val in diabetes_values]
                    selected_diabetes = st.selectbox("Diabetes", options=diabetes_options)
                    if selected_diabetes != "All":
                        # Store the actual value, not the string "Yes"/"No"
                        filters['diabetes'] = selected_diabetes

            # Cardiovascular filter
            if self.CN.CARDIOVASCULAR in self.df.columns:
                cardio_values = sorted(self.df[self.CN.CARDIOVASCULAR].dropna().unique())
                if len(cardio_values) > 0:
                    cardio_options = ["All"] + [str(val) for val in cardio_values]
                    selected_cardio = st.selectbox("Cardiovascular Issues", options=cardio_options)
                    if selected_cardio != "All":
                        # Store the actual value, not the string "Yes"/"No"
                        filters['cardiovascular'] = selected_cardio

        with col2:
            # Smoking status filter
            if self.CN.SMOKING_STATUS in self.df.columns:
                smoking_values = sorted(self.df[self.CN.SMOKING_STATUS].dropna().unique())
                if len(smoking_values) > 0:
                    smoking_options = ["All"] + [str(val) for val in smoking_values]
                    selected_smoking = st.selectbox("Smoking Status", options=smoking_options)
                    if selected_smoking != "All":
                        filters['smoking_status'] = selected_smoking

        return filters

    def _filter_wound_characteristics(self) -> Dict[str, Any]:
        """
        Create UI controls and collect settings for wound characteristic filters.

        Returns:
            Dict[str, Any]: Dictionary containing the wound characteristic filter settings
        """
        st.markdown("### Wound Characteristics")

        filters = {}

        col1, col2 = st.columns(2)

        with col1:
            # Wound area range filter
            if self.CN.WOUND_AREA in self.df.columns:
                min_area = float(self.df[self.CN.WOUND_AREA].min())
                max_area = float(self.df[self.CN.WOUND_AREA].max())
                area_range = st.slider("Wound Area (sq cm)", min_value=min_area, max_value=max_area,
                                      value=(min_area, max_area), step=0.1)
                filters['wound_area'] = area_range

            # Wound depth range filter
            if self.CN.DEPTH in self.df.columns:
                min_depth = float(self.df[self.CN.DEPTH].min())
                max_depth = float(self.df[self.CN.DEPTH].max())
                depth_range = st.slider("Wound Depth (cm)", min_value=min_depth, max_value=max_depth,
                                       value=(min_depth, max_depth), step=0.1)
                filters['depth'] = depth_range

        with col2:
            # Wound location filter
            if self.CN.WOUND_LOCATION in self.df.columns:
                location_options = ["All"] + list(self.df[self.CN.WOUND_LOCATION].unique())
                selected_location = st.selectbox("Wound Location", options=location_options)
                if selected_location != "All":
                    filters['wound_location'] = selected_location

            # Wound type filter
            if self.CN.WOUND_TYPE in self.df.columns:
                type_options = ["All"] + list(self.df[self.CN.WOUND_TYPE].unique())
                selected_type = st.selectbox("Wound Type", options=type_options)
                if selected_type != "All":
                    filters['wound_type'] = selected_type

        return filters

    def _filter_exudate(self) -> Dict[str, Any]:
        """
        Create UI controls and collect settings for exudate filters.

        Returns:
            Dict[str, Any]: Dictionary containing the exudate filter settings
        """
        st.markdown("### Exudate")

        filters = {}

        col1, col2 = st.columns(2)

        with col1:
            # Exudate volume filter
            if self.CN.EXUDATE_VOLUME in self.df.columns:
                volume_options = ["All"] + list(self.df[self.CN.EXUDATE_VOLUME].unique())
                selected_volume = st.selectbox("Exudate Volume", options=volume_options)
                if selected_volume != "All":
                    filters['exudate_volume'] = selected_volume

        with col2:
            # Exudate type filter
            if self.CN.EXUDATE_TYPE in self.df.columns:
                type_options = ["All"] + list(self.df[self.CN.EXUDATE_TYPE].unique())
                selected_type = st.selectbox("Exudate Type", options=type_options)
                if selected_type != "All":
                    filters['exudate_type'] = selected_type

        return filters

    def _filter_healing_metrics(self) -> Dict[str, Any]:
        """
        Create UI controls and collect settings for healing metric filters.

        Returns:
            Dict[str, Any]: Dictionary containing the healing metric filter settings
        """
        st.markdown("### Healing Metrics")

        filters = {}

        col1, col2 = st.columns(2)

        with col1:
            # Healing rate range filter
            if self.CN.HEALING_RATE in self.df.columns:
                min_rate = float(self.df[self.CN.HEALING_RATE].min())
                max_rate = float(self.df[self.CN.HEALING_RATE].max())
                rate_range = st.slider("Healing Rate (%)", min_value=min_rate, max_value=max_rate,
                                      value=(min_rate, max_rate), step=0.1)
                filters['healing_rate'] = rate_range

        with col2:
            # Estimated days to heal range filter
            if self.CN.ESTIMATED_DAYS_TO_HEAL in self.df.columns:
                min_days = int(self.df[self.CN.ESTIMATED_DAYS_TO_HEAL].min())
                max_days = int(self.df[self.CN.ESTIMATED_DAYS_TO_HEAL].max())
                days_range = st.slider("Estimated Days to Heal", min_value=min_days, max_value=max_days,
                                      value=(min_days, max_days))
                filters['estimated_days_to_heal'] = days_range

        return filters

    def _apply_filters(self, filter_settings: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply the selected filters to the DataFrame.

        Args:
            filter_settings (Dict[str, Dict[str, Any]]): Dictionary containing all filter settings

        Returns:
            pd.DataFrame: The filtered DataFrame
        """
        filtered_df = self.df.copy()
        original_count = len(filtered_df)
        filter_summary = {}

        # Show debug information before the expander
        show_debug = st.checkbox("Show Debug Filter Information", value=False)
        if show_debug:
            st.write("Filter settings:", filter_settings)
            st.write("Original dataframe shape:", filtered_df.shape)

            # Show a sample of the first visit data
            st.write("Sample of first visits (before filtering):")
            first_visits_sample = filtered_df.sort_values([self.CN.RECORD_ID, self.CN.VISIT_DATE]).groupby(self.CN.RECORD_ID).first().head(3)
            st.dataframe(first_visits_sample)

        # For demographic and medical history filters, we need to filter by record_id
        # These attributes are typically only recorded in the first visit

        # First, create a DataFrame with just the first visit for each patient
        first_visits_df = filtered_df.sort_values([self.CN.RECORD_ID, self.CN.VISIT_DATE]).groupby(self.CN.RECORD_ID).first().reset_index()

        # Track record_ids to keep - start with all record_ids
        record_ids_to_keep = set(first_visits_df[self.CN.RECORD_ID].unique())

        # Apply demographic filters to first visits only
        if 'demographics' in filter_settings:
            if 'age' in filter_settings['demographics']:
                age_range = filter_settings['demographics']['age']
                age_filtered_ids = set(first_visits_df[
                    (first_visits_df[self.CN.AGE] >= age_range[0]) &
                    (first_visits_df[self.CN.AGE] <= age_range[1])
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= age_filtered_ids
                filter_summary['Age Range'] = f"{age_range[0]} to {age_range[1]} years"

            if 'sex' in filter_settings['demographics']:
                sex = filter_settings['demographics']['sex']
                sex_filtered_ids = set(first_visits_df[
                    first_visits_df[self.CN.SEX] == sex
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= sex_filtered_ids
                filter_summary['Sex'] = sex

            if 'bmi' in filter_settings['demographics']:
                bmi_range = filter_settings['demographics']['bmi']
                bmi_filtered_ids = set(first_visits_df[
                    (first_visits_df[self.CN.BMI] >= bmi_range[0]) &
                    (first_visits_df[self.CN.BMI] <= bmi_range[1])
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= bmi_filtered_ids
                filter_summary['BMI Range'] = f"{bmi_range[0]:.1f} to {bmi_range[1]:.1f}"

            if 'race' in filter_settings['demographics']:
                race = filter_settings['demographics']['race']
                race_filtered_ids = set(first_visits_df[
                    first_visits_df[self.CN.RACE] == race
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= race_filtered_ids
                filter_summary['Race'] = race

        # Apply medical history filters to first visits only
        if 'medical_history' in filter_settings:
            if 'diabetes' in filter_settings['medical_history']:
                diabetes_value = filter_settings['medical_history']['diabetes']
                # Convert to proper type if needed (string to int/float/bool)
                if diabetes_value.lower() in ['true', 'yes', '1']:
                    diabetes_value = 1
                elif diabetes_value.lower() in ['false', 'no', '0']:
                    diabetes_value = 0
                else:
                    try:
                        diabetes_value = int(diabetes_value)
                    except (ValueError, TypeError):
                        pass

                diabetes_filtered_ids = set(first_visits_df[
                    first_visits_df[self.CN.DIABETES].astype(str) == str(diabetes_value)
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= diabetes_filtered_ids
                filter_summary['Diabetes'] = str(diabetes_value)

            if 'cardiovascular' in filter_settings['medical_history']:
                cardio_value = filter_settings['medical_history']['cardiovascular']
                # Convert to proper type if needed (string to int/float/bool)
                if cardio_value.lower() in ['true', 'yes', '1']:
                    cardio_value = 1
                elif cardio_value.lower() in ['false', 'no', '0']:
                    cardio_value = 0
                else:
                    try:
                        cardio_value = int(cardio_value)
                    except (ValueError, TypeError):
                        pass

                cardio_filtered_ids = set(first_visits_df[
                    first_visits_df[self.CN.CARDIOVASCULAR].astype(str) == str(cardio_value)
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= cardio_filtered_ids
                filter_summary['Cardiovascular Issues'] = str(cardio_value)

            if 'smoking_status' in filter_settings['medical_history']:
                smoking_status = filter_settings['medical_history']['smoking_status']
                smoking_filtered_ids = set(first_visits_df[
                    first_visits_df[self.CN.SMOKING_STATUS].astype(str) == str(smoking_status)
                ][self.CN.RECORD_ID])
                record_ids_to_keep &= smoking_filtered_ids
                filter_summary['Smoking Status'] = smoking_status

        # Filter the DataFrame to only include records for the filtered record_ids
        patient_filtered_df = filtered_df[filtered_df[self.CN.RECORD_ID].isin(record_ids_to_keep)]

        # Show patient filtering results if debug is on
        if show_debug:
            st.write(f"Patient filtering stage - Records remaining: {len(patient_filtered_df)}")
            st.write(f"Patients remaining: {len(record_ids_to_keep)} out of {len(first_visits_df[self.CN.RECORD_ID].unique())}")

            # Show remaining patient IDs
            st.write("Filtered Patient IDs:", sorted(list(record_ids_to_keep)))

        # Apply visit-specific filters to the already filtered DataFrame
        visit_filtered_df = patient_filtered_df.copy()

        # Apply wound characteristics filters (these apply to each visit)
        if 'wound_characteristics' in filter_settings:
            if 'wound_area' in filter_settings['wound_characteristics']:
                area_range = filter_settings['wound_characteristics']['wound_area']
                visit_filtered_df = visit_filtered_df[
                    (visit_filtered_df[self.CN.WOUND_AREA] >= area_range[0]) &
                    (visit_filtered_df[self.CN.WOUND_AREA] <= area_range[1])
                ]
                filter_summary['Wound Area'] = f"{area_range[0]:.1f} to {area_range[1]:.1f} sq cm"

            if 'depth' in filter_settings['wound_characteristics']:
                depth_range = filter_settings['wound_characteristics']['depth']
                visit_filtered_df = visit_filtered_df[
                    (visit_filtered_df[self.CN.DEPTH] >= depth_range[0]) &
                    (visit_filtered_df[self.CN.DEPTH] <= depth_range[1])
                ]
                filter_summary['Wound Depth'] = f"{depth_range[0]:.1f} to {depth_range[1]:.1f} cm"

            if 'wound_location' in filter_settings['wound_characteristics']:
                location = filter_settings['wound_characteristics']['wound_location']
                visit_filtered_df = visit_filtered_df[visit_filtered_df[self.CN.WOUND_LOCATION] == location]
                filter_summary['Wound Location'] = location

            if 'wound_type' in filter_settings['wound_characteristics']:
                wound_type = filter_settings['wound_characteristics']['wound_type']
                visit_filtered_df = visit_filtered_df[visit_filtered_df[self.CN.WOUND_TYPE] == wound_type]
                filter_summary['Wound Type'] = wound_type

        # Apply exudate filters (these apply to each visit)
        if 'exudate' in filter_settings:
            if 'exudate_volume' in filter_settings['exudate']:
                volume = filter_settings['exudate']['exudate_volume']
                visit_filtered_df = visit_filtered_df[visit_filtered_df[self.CN.EXUDATE_VOLUME] == volume]
                filter_summary['Exudate Volume'] = volume

            if 'exudate_type' in filter_settings['exudate']:
                exudate_type = filter_settings['exudate']['exudate_type']
                visit_filtered_df = visit_filtered_df[visit_filtered_df[self.CN.EXUDATE_TYPE] == exudate_type]
                filter_summary['Exudate Type'] = exudate_type

        # Apply healing metrics filters (these apply to each visit)
        if 'healing_metrics' in filter_settings:
            if 'healing_rate' in filter_settings['healing_metrics']:
                rate_range = filter_settings['healing_metrics']['healing_rate']
                visit_filtered_df = visit_filtered_df[
                    ((visit_filtered_df[self.CN.HEALING_RATE] >= rate_range[0]) &
                     (visit_filtered_df[self.CN.HEALING_RATE] <= rate_range[1])) |
                    (visit_filtered_df[self.CN.HEALING_RATE].isna())
                ]
                filter_summary['Healing Rate'] = f"{rate_range[0]:.1f} to {rate_range[1]:.1f}% (NaN values preserved)"

            if 'estimated_days_to_heal' in filter_settings['healing_metrics']:
                days_range = filter_settings['healing_metrics']['estimated_days_to_heal']
                visit_filtered_df = visit_filtered_df[
                    ((visit_filtered_df[self.CN.ESTIMATED_DAYS_TO_HEAL] >= days_range[0]) &
                     (visit_filtered_df[self.CN.ESTIMATED_DAYS_TO_HEAL] <= days_range[1])) |
                    (visit_filtered_df[self.CN.ESTIMATED_DAYS_TO_HEAL].isna())
                ]
                filter_summary['Estimated Days to Heal'] = f"{days_range[0]} to {days_range[1]} days (NaN values preserved)"

        # Show final debug information if debug is on
        if show_debug:
            st.write(f"Final dataframe shape: {visit_filtered_df.shape}")
            st.write(f"Records: {len(visit_filtered_df)} of {original_count}")
            st.write(f"Patients: {len(visit_filtered_df[self.CN.RECORD_ID].unique())} of {len(filtered_df[self.CN.RECORD_ID].unique())}")

        # Store filter statistics in session state
        st.session_state.filter_stats = {
            'original_count'        : original_count,
            'filtered_count'        : len(visit_filtered_df),
            'filter_summary'        : filter_summary,
            'patient_count'         : len(visit_filtered_df[self.CN.RECORD_ID].unique()),
            'original_patient_count': len(filtered_df[self.CN.RECORD_ID].unique())
        }

        return visit_filtered_df

    def _display_filter_summary(self) -> None:
        """
        Display a summary of the applied filters and their effects.
        """
        stats = st.session_state.filter_stats
        original_count = stats['original_count']
        filtered_count = stats['filtered_count']
        patient_count = stats['patient_count']
        original_patient_count = stats['original_patient_count']
        filter_summary = stats['filter_summary']

        st.markdown(f"### Filter Summary")

        # Display counts for both records and patients
        st.markdown(f"**Records:** {filtered_count} of {original_count} ({(filtered_count/original_count*100):.1f}%)")
        st.markdown(f"**Patients:** {patient_count} of {original_patient_count} ({(patient_count/original_patient_count*100):.1f}%)")

        # If no filters applied
        if not filter_summary:
            st.info("No filters applied. Showing all records.")
            return

        # Display applied filters
        st.markdown("**Applied Filters:**")

        col1, col2 = st.columns(2)

        # Split the filters between the two columns
        filters_list = list(filter_summary.items())
        half_index = len(filters_list) // 2 + len(filters_list) % 2

        with col1:
            for key, value in filters_list[:half_index]:
                st.markdown(f"- **{key}:** {value}")

        with col2:
            for key, value in filters_list[half_index:]:
                st.markdown(f"- **{key}:** {value}")

    def _reset_filters(self) -> None:
        """
        Reset all filters and return to the original dataset.
        """
        st.session_state.filters_applied = False
        st.session_state.filtered_df = None
        st.session_state.filter_stats = {
            'original_count': 0,
            'filtered_count': 0,
            'filter_summary': {}
        }

    def render(self) -> pd.DataFrame:
        """
        Render the filtering tab and return the filtered DataFrame.

        Returns:
            pd.DataFrame: The filtered DataFrame
        """
        st.write("Filter patient data based on medical criteria")

        with st.expander("Filter Settings", expanded=True):
            # Create filter sections
            filter_settings = {
                'demographics'         : self._filter_demographics(),
                'medical_history'      : self._filter_medical_history(),
                'wound_characteristics': self._filter_wound_characteristics(),
                'exudate'              : self._filter_exudate(),
                'healing_metrics'      : self._filter_healing_metrics()
            }

            # Add filter information
            st.info("Note: Demographic and medical history filters apply to all visits of a patient based on first visit data")

        # Add filter buttons
        col1, col2 = st.columns(2)

        with col1:
            apply_filter = st.button("Apply Filters")

        with col2:
            reset_filter = st.button("Reset Filters")

        if apply_filter:
            st.session_state.filtered_df = self._apply_filters(filter_settings)
            st.session_state.filters_applied = True

        if reset_filter:
            self._reset_filters()

        # Display filter summary if filters have been applied
        if st.session_state.filters_applied and st.session_state.filtered_df is not None:
            self._display_filter_summary()

            # Return the filtered dataframe
            return st.session_state.filtered_df

        # Return the original dataframe if no filters applied or reset
        return self.df
