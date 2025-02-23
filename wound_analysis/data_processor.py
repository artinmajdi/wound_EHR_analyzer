import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging
import pathlib
import re
from docx import Document
from llm_interface import format_word_document
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoundDataProcessor:
	def __init__(self, dataset_path: pathlib.Path=None, df: pd.DataFrame= None):

		self.dataset_path = dataset_path
		if df is None:
			csv_path = dataset_path / 'SmartBandage-Data_for_llm.csv'
			self.df = pd.read_csv(csv_path)
		else:
			self.df = df

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

				if pd.isna(visit.get('Skipped Visit?')) or visit['Skipped Visit?'] != 'Yes':
					visit_data = self._process_visit_data(visit=visit, record_id=record_id)
					if visit_data:
						wound_info = self._get_wound_info(visit)
						visit_data['wound_info'] = wound_info
						visits_data.append(visit_data)

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
		""" Get detailed wound information from a single visit row."""
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

	def _process_visit_data(self, visit, record_id: int) -> Optional[Dict]:
		"""Process visit measurement data from a single row."""
		visit_date = pd.to_datetime(visit['Visit date']).strftime('%m-%d-%Y') if not pd.isna(visit.get('Visit date')) else None

		if not visit_date:
			logger.warning("Missing visit date")
			return None

		def get_float(data, key):
			return float(data[key]) if not pd.isna(data.get(key)) else None

		def get_impedance_data():

			impedance_data = {
				'high_frequency': {'Z': None, 'resistance': None, 'capacitance': None, 'frequency': None},
				'center_frequency': {'Z': None, 'resistance': None, 'capacitance': None, 'frequency': None},
				'low_frequency': {'Z': None, 'resistance': None, 'capacitance': None, 'frequency': None}
			}

			def transform_impedance_data(freq_data, frequency):
				if freq_data is not None:
					return {
						'Z': freq_data['Z'],
						'resistance': freq_data['Z_prime'],
						'capacitance': None if freq_data['Z_double_prime'] is None else 1 / (2 * 3.14 * frequency * freq_data['Z_double_prime']),
						'frequency': frequency
					}
				return {'Z': None, 'resistance': None, 'capacitance': None, 'frequency': None}

			df = self.process_impedance_sweep_xlsx(record_id=record_id, visit_date_being_processed=visit_date)

			if df is not None:
				# Get data for this visit date
				visit_df = df[df['Visit date'] == visit_date]

				if not visit_df.empty:
					# Get data for all three frequencies using the index
					high_freq   = visit_df[visit_df.index == 'highest_freq'].iloc[0] if not visit_df[visit_df.index == 'highest_freq'].empty else None
					center_freq = visit_df[visit_df.index == 'center_freq'].iloc[0] if not visit_df[visit_df.index == 'center_freq'].empty else None
					low_freq    = visit_df[visit_df.index == 'lowest_freq'].iloc[0] if not visit_df[visit_df.index == 'lowest_freq'].empty else None

					impedance_data['high_frequency']   = transform_impedance_data(high_freq, float(high_freq['frequency']) if high_freq is not None else None)
					impedance_data['center_frequency'] = transform_impedance_data(center_freq, float(center_freq['frequency']) if center_freq is not None else None)
					impedance_data['low_frequency']    = transform_impedance_data(low_freq, float(low_freq['frequency']) if low_freq is not None else None)

			else:
				# Get impedance data from visit parameters if no sweep data
				high_freq = {
					'Z': get_float(visit, 'Skin Impedance (kOhms) - Z'),
					'Z_prime': get_float(visit, "Skin Impedance (kOhms) - Z'"),
					'Z_double_prime': get_float(visit, 'Skin Impedance (kOhms) - Z"')
				}

				impedance_data['high_frequency'] = transform_impedance_data(high_freq, 80000)

			return impedance_data

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
				'impedance'  : get_impedance_data(),
				**{key: get_float(visit, value) for key, value in hemoglobin_types.items()}
			}
		}

	def process_impedance_sweep_xlsx(self, record_id: int, visit_date_being_processed) -> Optional[pd.DataFrame]:
		"""
		Process impedance data from an Excel file for a specific record ID.

		Args:
			record_id (int): The patient record ID

		Returns:
			Optional[pd.DataFrame]: DataFrame containing processed impedance data, or None if processing fails
		"""
		# try:
		# Find the Excel file for this record
		excel_file = self.dataset_path / f'palmsense files Jan 2025/{record_id}.xlsx'
		if not excel_file.exists():
			return None

		# Read all sheets from the Excel file
		xl = pd.ExcelFile(excel_file)

		# Process each sheet (visit date)
		dfs = []
		for sheet_name in xl.sheet_names:

			# Extract visit date from filename
			match = re.search(rf"{record_id}_visit_(\d+)_(\d+)-(\d+)-(\d+)", sheet_name)
			if not match:
				logger.warning(f"Could not extract visit date from filename: {sheet_name}")
				return None
			visit_number = match.group(1)
			visit_date = datetime.strptime(f"{match.group(2)}/{match.group(3)}/{match.group(4)}", "%m/%d/%Y").strftime('%m-%d-%Y')

			if visit_date != visit_date_being_processed:
				continue

			# Read the sheet
			df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=1)

			# Clean column names and select relevant columns
			df.columns = df.columns.str.strip()

			# Get only the bottom half of the dataframe
			half_point = len(df) // 2
			df_bottom = df.iloc[half_point:]

			# Find frequencies of interest in the bottom half
			lowest_freq = df_bottom['freq / Hz'].min()
			highest_freq = df_bottom['freq / Hz'].max()

			# Calculate the difference between Z' and -Z" and find where it's minimum
			# df_bottom['z_diff'] = abs(df_bottom["Z' / Ohm"] - df_bottom["-Z'' / Ohm"])
			center_freq = df_bottom.loc[df_bottom['neg. Phase / °'].idxmax(), 'freq / Hz']

			# Get data for these three frequencies
			freq_data = []
			for freq_type, freq in [('lowest_freq', lowest_freq), ('center_freq', center_freq), ('highest_freq', highest_freq)]:
				row = df_bottom[df_bottom['freq / Hz'] == freq].iloc[0]
				freq_data.append({
					'Visit date': visit_date,
					'Visit number': visit_number,
					'frequency': str(freq),
					'Z': row["Z / Ohm"],
					'Z_prime': row["Z' / Ohm"],
					'Z_double_prime': row["-Z'' / Ohm"],
					'neg. Phase / °': row["neg. Phase / °"],
					'index': freq_type
				})

			dfs.extend(freq_data)

		if not dfs:
			return None

		# Create DataFrame from processed data
		df = pd.DataFrame(dfs).set_index('index')

		# Convert Visit date to datetime and format
		df['Visit date'] = pd.to_datetime(df['Visit date']).dt.strftime('%m-%d-%Y')

		return df

		# except Exception as e:
		# 	logger.error(f"Error processing Excel file for record {record_id}: {str(e)}")
		# 	return None

	def save_report(self, report_path: str, analysis_results: str, patient_data: dict) -> str:
		"""
		Save analysis results to a Word document.
		Args:
			report_path: Path where to save the Word document
			analysis_results: The analysis text from the LLM
			patient_data: Dictionary containing patient data and visit history
		Returns:
			str: Path to the saved document
		"""
		try:
			doc = Document()
			return format_word_document(doc, analysis_results, patient_data, report_path)
		except Exception as e:
			logger.error(f"Error saving report: {str(e)}")
			raise
