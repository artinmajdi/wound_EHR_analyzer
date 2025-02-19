import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoundDataProcessor:
    def __init__(self, dataset_path: pathlib.Path):
        self.dataset_path = dataset_path
        # Load single consolidated CSV file
        csv_path = dataset_path / 'SmartBandage-Data_for_llm.csv'
        self.df = pd.read_csv(csv_path)
        # Clean column names
        self.df.columns = self.df.columns.str.strip()

    def get_patient_visits(self, record_id: int) -> Dict:
        """Get all visit data for a specific patient."""
        try:
            patient_data = self.df[self.df['Record ID'] == record_id]
            if patient_data.empty:
                raise ValueError(f"No measurements found for patient {record_id}")

            # Get metadata from first visit
            first_visit = patient_data.iloc[0]
            metadata = self._extract_patient_metadata(first_visit)

            visits_data = []
            for _, visit in patient_data.iterrows():
                try:
                    if pd.isna(visit.get('Skipped Visit?')) or visit['Skipped Visit?'] != 'Yes':
                        visit_data = self._process_visit_data(visit)
                        if visit_data:
                            wound_info = self._get_wound_info(visit)
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

    def _extract_patient_metadata(self, patient_data) -> Dict:
        """Extract relevant patient metadata from a single row."""
        metadata = {
            'age': patient_data['Calculated Age at Enrollment'] if not pd.isna(patient_data.get('Calculated Age at Enrollment')) else None,
            'sex': patient_data['Sex'] if not pd.isna(patient_data.get('Sex')) else None,
            'race': patient_data['Race'] if not pd.isna(patient_data.get('Race')) else None,
            'ethnicity': patient_data['Ethnicity'] if not pd.isna(patient_data.get('Ethnicity')) else None,
            'weight': patient_data['Weight'] if not pd.isna(patient_data.get('Weight')) else None,
            'height': patient_data['Height'] if not pd.isna(patient_data.get('Height')) else None,
            'bmi': patient_data['BMI'] if not pd.isna(patient_data.get('BMI')) else None,
            'study_cohort': patient_data['Study Cohort'] if not pd.isna(patient_data.get('Study Cohort')) else None,
            'smoking_status': patient_data['Smoking status'] if not pd.isna(patient_data.get('Smoking status')) else None,
            'packs_per_day': patient_data['Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes'] if not pd.isna(patient_data.get('Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes')) else None,
            'years_smoking': patient_data['Number of Years smoked/has been smoking cigarettes'] if not pd.isna(patient_data.get('Number of Years smoked/has been smoking cigarettes')) else None,
            'alcohol_use': patient_data['Alcohol Use Status'] if not pd.isna(patient_data.get('Alcohol Use Status')) else None,
            'alcohol_frequency': patient_data['Number of alcohol drinks consumed/has been consuming'] if not pd.isna(patient_data.get('Number of alcohol drinks consumed/has been consuming')) else None
        }

        # Medical history from individual columns
        medical_conditions = [
            'Respiratory', 'Cardiovascular', 'Gastrointestinal', 'Musculoskeletal',
            'Endocrine/ Metabolic', 'Hematopoietic', 'Hepatic/Renal', 'Neurologic', 'Immune'
        ]
        # Get medical history from standard columns
        metadata['medical_history'] = {
            condition: patient_data[condition]
            for condition in medical_conditions if not pd.isna(patient_data.get(condition))
        }

        # Check additional medical history from free text field
        other_history = patient_data.get('Medical History (select all that apply)')
        if not pd.isna(other_history):
            existing_conditions = set(medical_conditions)
            other_conditions = [cond.strip() for cond in str(other_history).split(',')]
            other_conditions = [cond for cond in other_conditions if cond and cond not in existing_conditions]
            if other_conditions:
                metadata['medical_history']['other'] = ', '.join(other_conditions)


        # Diabetes information
        metadata['diabetes'] = {
            'status': patient_data.get('Diabetes?'),
            'hemoglobin_a1c': patient_data.get('Hemoglobin A1c (%)'),
            'a1c_available': patient_data.get('A1c  available within the last 3 months?')
        }

        return metadata

    def _get_wound_info(self, visit_data) -> Dict:
        """Get detailed wound information from a single visit row."""
        try:

            def clean_field(data, field):
                return data.get(field) if not pd.isna(data.get(field)) else None

            present = clean_field(visit_data, 'Is there undermining/ tunneling?')

            wound_info = {
                'location'       : clean_field(visit_data, 'Describe the wound location'),
                'type'           : clean_field(visit_data, 'Wound Type'),
                'current_care'   : clean_field(visit_data, 'Current wound care'),
                'clinical_events': clean_field(visit_data, 'Clinical events'),
                'undermining': {
                    'present'  : None if present is None else present == 'Yes',
                    'location' : visit_data.get('Undermining Location Description'),
                    'tunneling': visit_data.get('Tunneling Location Description')
                },
                'infection': {
                    'status'             : clean_field(visit_data, 'Infection'),
                    'wifi_classification': visit_data.get('Diabetic Foot Wound - WIfI Classification: foot Infection (fI)')
                },
                'granulation': {
                    'coverage': clean_field(visit_data, 'Granulation'),
                    'quality' : clean_field(visit_data, 'Granulation Quality')
                },
                'necrosis': visit_data.get('Necrosis'),
                'exudate': {
                    'volume'   : visit_data.get('Exudate Volume'),
                    'viscosity': visit_data.get('Exudate Viscosity'),
                    'type'     : visit_data.get('Exudate Type')
                }
            }

            return wound_info

        except Exception as e:
            logger.warning(f"Error getting wound info: {str(e)}")
            return {}

    def _process_visit_data(self, visit) -> Optional[Dict]:
        """Process visit measurement data from a single row."""
        try:

            visit_date = pd.to_datetime(visit['Visit date']).strftime('%Y-%m-%d') if not pd.isna(visit.get('Visit date')) else None

            if not visit_date:
                return None

            def get_float(data, key):
                return float(data[key]) if not pd.isna(data.get(key)) else None

            # Create impedance data structure for both frequencies
            impedance_imaginary = get_float(visit, "Skin Impedance (kOhms) - Z''")

            impedance_data = {
                'high_frequency': {  # 80kHz measurements
                    'Z'          : get_float(visit, 'Skin Impedance (kOhms) - Z'),
                    'resistance' : get_float(visit, "Skin Impedance (kOhms) - Z'"),
                    'capacitance': None if impedance_imaginary is None else 1 / (2 * 3.14 * 80000 * impedance_imaginary)
                },
                'low_frequency': {  # 100Hz measurements (placeholder for new data)
                    'Z'          : None, # Will be populated from new CSV columns when available
                    'resistance' : None,
                    'capacitance': None
                }
            }

            wound_measurements = {
                'length': get_float(visit, 'Length (cm)'),
                'width' : get_float(visit, 'Width (cm)'),
                'depth' : get_float(visit, 'Depth (cm)'),
                'area'  : get_float(visit, 'Calculated Wound Area')
            }

            temperature_readings = {
                'center': get_float(visit, "Center of Wound Temperature (Fahrenheit)"),
                'edge'  : get_float(visit, "Edge of Wound Temperature (Fahrenheit)"),
                'peri'  : get_float(visit, "Peri-wound Temperature (Fahrenheit)")
                }

            hemoglobin_types = {
                'hemoglobin'     : 'Hemoglobin Level',
                'oxyhemoglobin'  : 'Oxyhemoglobin Level',
                'deoxyhemoglobin': 'Deoxyhemoglobin Level'
            }

            return {
                'visit_date': visit_date,
                'wound_measurements': wound_measurements,
                'sensor_data': {
                    'oxygenation': get_float(visit, 'Oxygenation (%)'),
                    'temperature': temperature_readings,
                    'impedance'  : impedance_data,
                    **{key: get_float(visit, value) for key, value in hemoglobin_types.items()}
                }
            }


        except Exception as e:
            logger.warning(f"Error processing visit data: {str(e)}")
            return None
