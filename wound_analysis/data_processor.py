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
        metadata['medical_history'] = {
            condition: patient_data[condition]
            for condition in medical_conditions if not pd.isna(patient_data.get(condition))
        }


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
            wound_info = {
                'location': visit_data.get('Describe the wound location') if not pd.isna(visit_data.get('Describe the wound location')) else None,
                'type': visit_data.get('Wound Type') if not pd.isna(visit_data.get('Wound Type')) else None,
                'current_care': visit_data.get('Current wound care') if not pd.isna(visit_data.get('Current wound care')) else None,
                'clinical_events': visit_data.get('Clinical events') if not pd.isna(visit_data.get('Clinical events')) else None,
                'undermining': {
                    'present': visit_data.get('Is there undermining/ tunneling?') == 'Yes',
                    'location': visit_data.get('Undermining Location Description'),
                    'tunneling': visit_data.get('Tunneling Location Description')
                },
                'infection': {
                    'Infection': visit_data.get('Infection'),
                    'Diabetic Foot Wound - WIfI Classification: foot Infection (fI)': visit_data.get('Diabetic Foot Wound - WIfI Classification: foot Infection (fI)')
                },
                'granulation': {
                    'coverage': visit_data.get('Granulation'),
                    'quality': visit_data.get('Granulation Quality')
                },
                'necrosis': visit_data.get('Necrosis'),
                'exudate': {
                    'volume': visit_data.get('Exudate Volume'),
                    'viscosity': visit_data.get('Exudate Viscosity'),
                    'type': visit_data.get('Exudate Type')
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

            # Create impedance data structure for both frequencies
            impedance_data = {
                'high_frequency': {  # 80kHz measurements (existing data)
                    'Z': float(visit['Skin Impedance (kOhms) - Z']) if not pd.isna(visit.get('Skin Impedance (kOhms) - Z')) else None,
                    'resistance': float(visit["Skin Impedance (kOhms) - Z'"]) if not pd.isna(visit.get("Skin Impedance (kOhms) - Z'")) else None,
                    'capacitance': 1 / (2 * 3.14 * 80000 * float(visit["Skin Impedance (kOhms) - Z''"])) if not pd.isna(visit.get("Skin Impedance (kOhms) - Z''")) else None
                },
                'low_frequency': {  # 100Hz measurements (placeholder for new data)
                    'Z': None,  # Will be populated from new CSV columns when available
                    'resistance': None,
                    'capacitance': None
                }
            }

            return {
                'visit_date': visit_date,
                'wound_measurements': {
                    'length': float(visit['Length (cm)']) if not pd.isna(visit.get('Length (cm)')) else None,
                    'width': float(visit['Width (cm)']) if not pd.isna(visit.get('Width (cm)')) else None,
                    'depth': float(visit['Depth (cm)']) if not pd.isna(visit.get('Depth (cm)')) else None,
                    'area': float(visit['Calculated Wound Area']) if not pd.isna(visit.get('Calculated Wound Area')) else None
                },
                'sensor_data': {
                    'oxygenation': float(visit['Oxygenation (%)']) if not pd.isna(visit.get('Oxygenation (%)')) else None,
                    'temperature': {
                        'center': float(visit['Center of Wound Temperature (Fahrenheit)']) if not pd.isna(visit.get('Center of Wound Temperature (Fahrenheit)')) else None,
                        'edge': float(visit['Edge of Wound Temperature (Fahrenheit)']) if not pd.isna(visit.get('Edge of Wound Temperature (Fahrenheit)')) else None,
                        'peri': float(visit['Peri-wound Temperature (Fahrenheit)']) if not pd.isna(visit.get('Peri-wound Temperature (Fahrenheit)')) else None
                    },
                    'impedance': impedance_data,
                    'hemoglobin': float(visit['Hemoglobin Level']) if not pd.isna(visit.get('Hemoglobin Level')) else None,
                    'oxyhemoglobin': float(visit['Oxyhemoglobin Level']) if not pd.isna(visit.get('Oxyhemoglobin Level')) else None,
                    'deoxyhemoglobin': float(visit['Deoxyhemoglobin Level']) if not pd.isna(visit.get('Deoxyhemoglobin Level')) else None
                }
            }
        except Exception as e:
            logger.warning(f"Error processing visit data: {str(e)}")
            return None
