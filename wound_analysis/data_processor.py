import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoundDataProcessor:
    def __init__(self):
        # Load CSVs and clean column names
        self.measurements_df = pd.read_csv('SmartBandage_measurements.csv')
        self.wound_df = pd.read_csv('SmartBandage_wound.csv')
        self.other_df = pd.read_csv('SmartBandage_other.csv')

        # Clean column names in all dataframes
        self.measurements_df.columns = self.measurements_df.columns.str.strip()
        self.wound_df.columns = self.wound_df.columns.str.strip()
        self.other_df.columns = self.other_df.columns.str.strip()

    def get_patient_visits(self, record_id: int) -> Dict:
        """Get all visit data for a specific patient."""
        try:
            patient_measurements = self.measurements_df[self.measurements_df['Record ID'] == record_id]
            if patient_measurements.empty:
                raise ValueError(f"No measurements found for patient {record_id}")

            patient_other = self.other_df[self.other_df['Record ID'] == record_id]
            first_visit = patient_measurements.iloc[0]

            # Get metadata from both measurements and other data
            metadata = self._extract_patient_metadata(
                patient_other.iloc[0] if not patient_other.empty else None,
                first_visit
            )

            patient_wound = self.wound_df[self.wound_df['Record ID'] == record_id]
            visits_data = []

            for _, visit in patient_measurements.iterrows():
                try:
                    if pd.isna(visit.get('Skipped Visit?')) or visit['Skipped Visit?'] != 'Yes':
                        visit_data = self._process_visit_data(visit)
                        if visit_data:
                            wound_info = self._get_wound_info(patient_wound, visit['Event Name'])
                            visit_data['wound_info'] = wound_info
                            visits_data.append(visit_data)
                except Exception as e:
                    logger.warning(f"Error processing visit data: {str(e)}")
                    continue

            return {
                'patient_metadata': metadata,
                'visits': visits_data
            }

        except Exception as e:
            logger.error(f"Error processing patient {record_id}: {str(e)}")
            raise

    def _extract_patient_metadata(self, patient_data, first_visit) -> Dict:
        """Extract relevant patient metadata from both other data and measurements."""
        metadata = {}

        # Basic demographics from first visit
        if first_visit is not None:
            metadata.update({
                'age': first_visit['Age'] if not pd.isna(first_visit.get('Age')) else None,
                'wound_onset_date': first_visit['Target wound onset date'] if not pd.isna(first_visit.get('Target wound onset date')) else None
            })

        # Demographics and physical characteristics from other data
        if patient_data is not None:
            metadata.update({
                'sex': patient_data['Sex'] if not pd.isna(patient_data.get('Sex')) else None,
                'race': patient_data['Race'] if not pd.isna(patient_data.get('Race')) else None,
                'weight': patient_data['Weight'] if not pd.isna(patient_data.get('Weight')) else None,
                'height': patient_data['Height'] if not pd.isna(patient_data.get('Height')) else None,
                'bmi': patient_data['BMI'] if not pd.isna(patient_data.get('BMI')) else None,
                'study_cohort': patient_data['Study Cohort'] if not pd.isna(patient_data.get('Study Cohort')) else None
            })

            # Lifestyle factors
            metadata.update({
                'smoking_status': patient_data['Smoking status'] if not pd.isna(patient_data.get('Smoking status')) else None,
                'packs_per_day': patient_data['Number of Packs per Day'] if not pd.isna(patient_data.get('Number of Packs per Day')) else None,
                'years_smoking': patient_data['Number of Years smoked'] if not pd.isna(patient_data.get('Number of Years smoked')) else None,
                'alcohol_use': patient_data['Alcohol Use Status'] if not pd.isna(patient_data.get('Alcohol Use Status')) else None,
                'alcohol_frequency': patient_data['Number of alcohol drinks consumed'] if not pd.isna(patient_data.get('Number of alcohol drinks consumed')) else None
            })

            # Medical history
            metadata['medical_history'] = {
                condition.replace('Medical History (select all that apply) (choice=', '').replace(')', ''): 'Yes'
                for condition in patient_data.index
                if 'Medical History (select all that apply) (choice=' in condition
                and patient_data[condition] == 'Checked'
            }

            # Diabetes status
            metadata['diabetes'] = {
                'status': 'T1DM' if patient_data.get('Medical History (select all that apply) (choice=T1DM)') == 'Checked'
                          else 'T2DM' if patient_data.get('Medical History (select all that apply) (choice=T2DM)') == 'Checked'
                          else None,
                'hemoglobin_a1c': patient_data['Hemoglobin A1c (%)'] if not pd.isna(patient_data.get('Hemoglobin A1c (%)')) else None,
                'a1c_date': patient_data['Result Date'] if not pd.isna(patient_data.get('Result Date')) else None
            }

        return metadata

    def _get_wound_info(self, patient_wound: pd.DataFrame, event_name: str) -> Dict:
        """Get detailed wound information for a specific visit."""
        try:
            visit_wound = patient_wound[patient_wound['Event Name'] == event_name]
            if visit_wound.empty:
                return {}

            wound_data = visit_wound.iloc[0]

            # Base wound characteristics
            wound_info = {
                'location': wound_data.get('Describe the wound location') if not pd.isna(wound_data.get('Describe the wound location')) else None,
                'type': wound_data.get('Wound Type') if not pd.isna(wound_data.get('Wound Type')) else None,
                'current_care': wound_data.get('Current wound care') if not pd.isna(wound_data.get('Current wound care')) else None,
                'clinical_events': wound_data.get('Clinical events') if not pd.isna(wound_data.get('Clinical events')) else None
            }

            # Undermining/tunneling info
            wound_info['undermining'] = {
                'present': wound_data.get('Is there undermining/ tunneling?') == 'Yes',
                'location': wound_data.get('Undermining Location Description') if not pd.isna(wound_data.get('Undermining Location Description')) else None,
                'tunneling': wound_data.get('Tunneling Location Description') if not pd.isna(wound_data.get('Tunneling Location Description')) else None
            }

            # Infection status
            wound_info['infection'] = {
                'status': next((col.replace('Infection (choice=', '').replace(')', '')
                              for col in visit_wound.columns
                              if col.startswith('Infection (choice=') and wound_data.get(col) == 'Checked'), None),
                'classification': wound_data.get('Diabetic Foot Wound - WIfI Classification: foot Infection (fI)') if not pd.isna(wound_data.get('Diabetic Foot Wound - WIfI Classification: foot Infection (fI)')) else None
            }

            # Tissue characteristics
            wound_info['tissue'] = {
                'granulation': {
                    'coverage': wound_data.get('Granulation') if not pd.isna(wound_data.get('Granulation')) else None,
                    'quality': next((col.replace('Granulation Quality (choice=', '').replace(')', '')
                                   for col in visit_wound.columns
                                   if col.startswith('Granulation Quality') and wound_data.get(col) == 'Checked'), None)
                },
                'necrosis': wound_data.get('Necrosis') if not pd.isna(wound_data.get('Necrosis')) else None
            }

            # Exudate characteristics
            wound_info['exudate'] = {
                'volume': wound_data.get('Exudate Volume') if not pd.isna(wound_data.get('Exudate Volume')) else None,
                'viscosity': wound_data.get('Exudate Viscosity') if not pd.isna(wound_data.get('Exudate Viscosity')) else None,
                'type': next((col.replace('Exudate Type (choice=', '').replace(')', '')
                            for col in visit_wound.columns
                            if col.startswith('Exudate Type') and wound_data.get(col) == 'Checked'), None)
            }

            return wound_info

        except Exception as e:
            logger.warning(f"Error getting wound info: {str(e)}")
            return {}

    def _process_visit_data(self, visit) -> Optional[Dict]:
        """Process visit measurement data with validation."""
        try:
            visit_date = pd.to_datetime(visit['Visit date']).strftime('%Y-%m-%d') if not pd.isna(visit.get('Visit date')) else None
            if not visit_date:
                return None

            return {
                'visit_date': visit_date,
                'wound_measurements': {
                    'length': float(visit['Length (cm)']) if not pd.isna(visit.get('Length (cm)')) else None,
                    'width': float(visit['Width (cm)']) if not pd.isna(visit.get('Width (cm)')) else None,
                    'depth': float(visit['Depth (cm)']) if not pd.isna(visit.get('Depth (cm)')) else None,
                    'area': float(visit['Wound Area']) if not pd.isna(visit.get('Wound Area')) else None,
                    'surface_area': float(visit['Total Surface Area']) if not pd.isna(visit.get('Total Surface Area')) else None
                },
                'sensor_data': {
                    'oxygenation': float(visit['Oxygenation (%)']) if not pd.isna(visit.get('Oxygenation (%)')) else None,
                    'temperature': {
                        'center': float(visit['Center of Wound Temperature (Fahrenheit)']) if not pd.isna(visit.get('Center of Wound Temperature (Fahrenheit)')) else None,
                        'edge': float(visit['Edge of Wound Temperature (Fahrenheit)']) if not pd.isna(visit.get('Edge of Wound Temperature (Fahrenheit)')) else None,
                        'peri': float(visit['Peri-wound Temperature (Fahrenheit)']) if not pd.isna(visit.get('Peri-wound Temperature (Fahrenheit)')) else None
                    },
                    'impedance': float(visit['Skin Impedance (kOhms) - Z']) if not pd.isna(visit.get('Skin Impedance (kOhms) - Z')) else None,
                    'hemoglobin': float(visit['Hemoglobin Level']) if not pd.isna(visit.get('Hemoglobin Level')) else None,
                    'oxyhemoglobin': float(visit['Oxyhemoglobin Level']) if not pd.isna(visit.get('Oxyhemoglobin Level')) else None,
                    'deoxyhemoglobin': float(visit['Deoxyhemoglobin Level']) if not pd.isna(visit.get('Deoxyhemoglobin Level')) else None
                }
            }

        except Exception as e:
            logger.warning(f"Error processing visit data: {str(e)}")
            return None
