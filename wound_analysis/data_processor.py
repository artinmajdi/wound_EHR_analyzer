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
import numpy as np

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

	def get_population_statistics(self) -> Dict:
		"""
		Gather comprehensive population-level statistics for LLM analysis.
		Includes data from all dashboard tabs (Overview, Impedance, Temperature,
		Exudate, Risk Factors) for the entire patient population.

		Returns:
			Dict: A comprehensive dictionary containing population statistics including:
				- Demographics (age, gender, race, BMI, etc.)
				- Wound characteristics (type, location, size)
				- Healing progression metrics
				- Sensor data analysis (impedance, temperature, oxygenation)
				- Risk factor analysis
				- Treatment effectiveness
				- Temporal trends
		"""
		# Get processed data
		df = self.get_processed_data()
		if df.empty:
			raise ValueError("No data available for population statistics")

		# Calculate derived metrics
		df['BMI Category'] = pd.cut(
			df['BMI'],
			bins=[0, 18.5, 24.9, 29.9, float('inf')],
			labels=['Underweight', 'Normal', 'Overweight', 'Obese']
		)

		# Calculate healing metrics
		df['Healing_Color'] = df['Healing Rate (%)'].apply(lambda x: 'green' if x < 0 else 'red')
		df['Healing_Status'] = df['Healing Rate (%)'].apply(
			lambda x: 'Improving' if x < 0 else ('Stable' if -5 <= x <= 5 else 'Worsening')
		)

		stats = {
			'summary': {
				'total_patients': len(df['Record ID'].unique()),
				'total_visits': len(df),
				'avg_visits_per_patient': len(df) / len(df['Record ID'].unique()),
				'overall_improvement_rate': (df['Healing Rate (%)'] < 0).mean() * 100,
				'avg_treatment_duration_days': (df.groupby('Record ID')['Days from First Visit'].max().mean()),
				'completion_rate': (df['Visit Status'] == 'Completed').mean() * 100 if 'Visit Status' in df.columns else None
			},
			'demographics': {
				'age_stats': {
					'summary': f"Mean: {df['Calculated Age at Enrollment'].mean():.1f}, Median: {df['Calculated Age at Enrollment'].median():.1f}",
					'distribution': df['Calculated Age at Enrollment'].value_counts().to_dict(),
					'age_groups': pd.cut(df['Calculated Age at Enrollment'],
						bins=[0, 30, 50, 70, float('inf')],
						labels=['<30', '30-50', '50-70', '>70']).value_counts().to_dict()
				},
				'gender_distribution': df['Sex'].value_counts().to_dict(),
				'race_distribution': df['Race'].value_counts().to_dict(),
				'ethnicity_distribution': df['Ethnicity'].value_counts().to_dict(),
				'bmi_stats': {
					'summary': f"Mean: {df['BMI'].mean():.1f}, Range: {df['BMI'].min():.1f}-{df['BMI'].max():.1f}",
					'distribution': df['BMI Category'].value_counts().to_dict(),
					'by_healing_status': df.groupby('BMI Category')['Healing Rate (%)'].agg(['mean', 'count']).to_dict()
				}
			},
			'risk_factors': {
				'primary_conditions': {
					'diabetes': {
						'distribution': df['Diabetes?'].value_counts().to_dict(),
						'healing_impact': df.groupby('Diabetes?')['Healing Rate (%)'].agg(['mean', 'std', 'count']).to_dict()
					},
					'smoking': {
						'distribution': df['Smoking status'].value_counts().to_dict(),
						'healing_impact': df.groupby('Smoking status')['Healing Rate (%)'].agg(['mean', 'std', 'count']).to_dict()
					}
				},
				'comorbidity_analysis': {
					'diabetes_smoking': df.groupby(['Diabetes?', 'Smoking status'])['Healing Rate (%)'].agg(['mean', 'count']).to_dict(),
					'diabetes_bmi': df.groupby(['Diabetes?', 'BMI Category'])['Healing Rate (%)'].agg(['mean', 'count']).to_dict()
				}
			},
			'wound_characteristics': {
				'type_distribution': {
					'overall': df['Wound Type'].value_counts().to_dict(),
					'by_healing_status': df.groupby(['Wound Type', 'Healing_Status']).size().to_dict()
				},
				'location_analysis': {
					'distribution': df['Describe the wound location'].value_counts().to_dict(),
					'healing_by_location': df.groupby('Describe the wound location')['Healing Rate (%)'].mean().to_dict()
				},
				'size_progression': {
					'initial_vs_final': {
						'area': {
							'initial': df.groupby('Record ID')['Calculated Wound Area'].first().agg(['mean', 'median', 'std']).to_dict(),
							'final': df.groupby('Record ID')['Calculated Wound Area'].last().agg(['mean', 'median', 'std']).to_dict(),
							'percent_change': ((df.groupby('Record ID')['Calculated Wound Area'].last() -
								df.groupby('Record ID')['Calculated Wound Area'].first()) /
								df.groupby('Record ID')['Calculated Wound Area'].first() * 100).mean()
						}
					},
					'healing_by_initial_size': {
						'small': df[df['Calculated Wound Area'] < df['Calculated Wound Area'].quantile(0.33)]['Healing Rate (%)'].mean(),
						'medium': df[(df['Calculated Wound Area'] >= df['Calculated Wound Area'].quantile(0.33)) &
							(df['Calculated Wound Area'] < df['Calculated Wound Area'].quantile(0.67))]['Healing Rate (%)'].mean(),
						'large': df[df['Calculated Wound Area'] >= df['Calculated Wound Area'].quantile(0.67)]['Healing Rate (%)'].mean()
					}
				}
			},
			'healing_progression': {
				'overall_stats': {
					'summary': f"Mean: {df['Healing Rate (%)'].mean():.1f}%, Median: {df['Healing Rate (%)'].median():.1f}%",
					'distribution': df['Healing_Status'].value_counts().to_dict(),
					'percentiles': df['Healing Rate (%)'].quantile([0.25, 0.5, 0.75]).to_dict()
				},
				'temporal_analysis': {
					'by_visit_number': df.groupby('Visit Number')['Healing Rate (%)'].agg(['mean', 'std', 'count']).to_dict(),
					'by_treatment_duration': pd.cut(df['Days from First Visit'],
						bins=[0, 30, 90, 180, float('inf')],
						labels=['<30 days', '30-90 days', '90-180 days', '>180 days']
					).value_counts().to_dict()
				}
			},
			'exudate_analysis': {
				'characteristics': {
					'volume': {
						'distribution': df['Exudate Volume'].value_counts().to_dict(),
						'healing_correlation': df.groupby('Exudate Volume')['Healing Rate (%)'].mean().to_dict()
					},
					'type': {
						'distribution': df['Exudate Type'].value_counts().to_dict(),
						'healing_correlation': df.groupby('Exudate Type')['Healing Rate (%)'].mean().to_dict()
					},
					'viscosity': {
						'distribution': df['Exudate Viscosity'].value_counts().to_dict(),
						'healing_correlation': df.groupby('Exudate Viscosity')['Healing Rate (%)'].mean().to_dict()
					}
				},
				'temporal_patterns': {
					'volume_progression': df.groupby('Visit Number')['Exudate Volume'].value_counts().to_dict(),
					'type_progression': df.groupby('Visit Number')['Exudate Type'].value_counts().to_dict()
				}
			}
		}

		# Add sensor data analysis if available
		if all(col in df.columns for col in ['Skin Impedance (kOhms) - Z', 'Skin Impedance (kOhms) - Phase']):
			stats['sensor_data'] = {
				'impedance': {
					'magnitude': {
						'overall': df['Skin Impedance (kOhms) - Z'].agg(['mean', 'std', 'min', 'max']).to_dict(),
						'by_healing_status': df.groupby('Healing_Status')['Skin Impedance (kOhms) - Z'].mean().to_dict(),
						'temporal_trend': df.groupby('Visit Number')['Skin Impedance (kOhms) - Z'].mean().to_dict()
					},
					'phase': {
						'overall': df['Skin Impedance (kOhms) - Phase'].agg(['mean', 'std', 'min', 'max']).to_dict(),
						'by_healing_status': df.groupby('Healing_Status')['Skin Impedance (kOhms) - Phase'].mean().to_dict(),
						'temporal_trend': df.groupby('Visit Number')['Skin Impedance (kOhms) - Phase'].mean().to_dict()
					}
				}
			}

		if all(col in df.columns for col in ['Temperature (°F)', 'Peri-wound Temperature (°F)']):
			if 'sensor_data' not in stats:
				stats['sensor_data'] = {}

			stats['sensor_data']['temperature'] = {
				'wound_temp': {
					'overall': df['Temperature (°F)'].agg(['mean', 'std', 'min', 'max']).to_dict(),
					'by_healing_status': df.groupby('Healing_Status')['Temperature (°F)'].mean().to_dict()
				},
				'gradient': {
					'overall': (df['Temperature (°F)'] - df['Peri-wound Temperature (°F)']).agg(['mean', 'std']).to_dict(),
					'by_healing_status': df.groupby('Healing_Status')
						.apply(lambda x: (x['Temperature (°F)'] - x['Peri-wound Temperature (°F)']).mean()).to_dict()
				},
				'temporal_patterns': {
					'wound_temp': df.groupby('Visit Number')['Temperature (°F)'].mean().to_dict(),
					'gradient': df.groupby('Visit Number')
						.apply(lambda x: (x['Temperature (°F)'] - x['Peri-wound Temperature (°F)']).mean()).to_dict()
				}
			}

		if 'Oxygenation (%)' in df.columns:
			if 'sensor_data' not in stats:
				stats['sensor_data'] = {}

			stats['sensor_data']['oxygenation'] = {
				'overall': df['Oxygenation (%)'].agg(['mean', 'std', 'min', 'max']).to_dict(),
				'by_healing_status': df.groupby('Healing_Status')['Oxygenation (%)'].mean().to_dict(),
				'temporal_trend': df.groupby('Visit Number')['Oxygenation (%)'].mean().to_dict(),
				'correlation_with_healing': df['Healing Rate (%)'].corr(df['Oxygenation (%)']),
				'distribution_quartiles': pd.qcut(df['Oxygenation (%)'], q=4).value_counts().to_dict()
			}

		return stats

	def get_processed_data(self) -> pd.DataFrame:
		"""
		Process the raw data for population-level analysis.
		Handles data cleaning, type conversion, and derived metrics calculation.

		Returns:
			pd.DataFrame: Processed dataframe ready for population analysis
		"""
		if self.df is None:
			raise ValueError("No data available. Please load data first.")

		# Create a copy to avoid modifying original data
		df = self.df.copy()

		# Clean column names
		df.columns = df.columns.str.strip()

		# Filter out skipped visits
		df = df[df['Skipped Visit?'] != 'Yes']

		# Extract visit number from Event Name
		df['Visit Number'] = df['Event Name'].str.extract(r'Visit (\d+)').fillna(1).astype(int)

		# Convert and format dates
		df['Visit date'] = pd.to_datetime(df['Visit date'])

		# Calculate days since first visit for each patient
		df['Days_Since_First_Visit'] = df.groupby('Record ID')['Visit date'].transform(
			lambda x: (x - x.min()).dt.days
		)

		# Handle Wound Type categorization
		if 'Wound Type' in df.columns:
			# First replace NaN with 'Unknown'
			df['Wound Type'] = df['Wound Type'].fillna('Unknown')
			# Get unique categories including 'Unknown'
			categories = sorted(df['Wound Type'].unique())
			# Now create categorical with all possible categories
			df['Wound Type'] = pd.Categorical(df['Wound Type'], categories=categories)

		# Calculate wound area if not present
		if 'Calculated Wound Area' not in df.columns and all(col in df.columns for col in ['Length (cm)', 'Width (cm)']):
			df['Calculated Wound Area'] = df['Length (cm)'] * df['Width (cm)']

		# Create derived features
		# Temperature gradients
		center = 'Center of Wound Temperature (Fahrenheit)'
		edge = 'Edge of Wound Temperature (Fahrenheit)'
		peri = 'Peri-wound Temperature (Fahrenheit)'
		if all(col in df.columns for col in [center, edge, peri]):
			df['Center-Edge Temp Gradient'] = df[center] - df[edge]
			df['Edge-Peri Temp Gradient'] = df[edge] - df[peri]
			df['Total Temp Gradient'] = df[center] - df[peri]

		# BMI categories
		if 'BMI' in df.columns:
			df['BMI Category'] = pd.cut(
				df['BMI'],
				bins=[0, 18.5, 25, 30, 35, 100],
				labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II-III']
			)

		# Calculate healing rates
		MAX_TREATMENT_DAYS = 730  # 2 years in days
		MIN_WOUND_AREA = 0

		def calculate_patient_healing_metrics(patient_data: pd.DataFrame) -> tuple:
			"""Calculate healing rate and estimated days for a patient."""
			if len(patient_data) < 2:
				return [0.0], False, np.nan

			healing_rates = []
			for i, row in patient_data.iterrows():
				if row['Visit Number'] == 1 or len(patient_data[patient_data['Visit Number'] < row['Visit Number']]) == 0:
					healing_rates.append(0)
				else:
					prev_visits = patient_data[patient_data['Visit Number'] < row['Visit Number']]
					prev_visit = prev_visits[prev_visits['Visit Number'] == prev_visits['Visit Number'].max()]

					if len(prev_visit) > 0 and 'Calculated Wound Area' in patient_data.columns:
						prev_area = prev_visit['Calculated Wound Area'].values[0]
						curr_area = row['Calculated Wound Area']

						if prev_area > 0 and not pd.isna(prev_area) and not pd.isna(curr_area):
							healing_rate = (prev_area - curr_area) / prev_area * 100
							healing_rates.append(healing_rate)
						else:
							healing_rates.append(0)
					else:
						healing_rates.append(0)

			valid_rates = [rate for rate in healing_rates if rate > 0]
			avg_healing_rate = np.mean(valid_rates) if valid_rates else 0
			is_improving = avg_healing_rate > 0

			estimated_days = np.nan
			if is_improving and len(patient_data) > 0:
				last_visit = patient_data.iloc[-1]
				current_area = last_visit['Calculated Wound Area']

				if current_area > MIN_WOUND_AREA and avg_healing_rate > 0:
					daily_healing_rate = (avg_healing_rate / 100) * current_area
					if daily_healing_rate > 0:
						days_to_heal = current_area / daily_healing_rate
						total_days = last_visit['Days_Since_First_Visit'] + days_to_heal
						if 0 < total_days < MAX_TREATMENT_DAYS:
							estimated_days = float(total_days)

			return healing_rates, is_improving, estimated_days

		# Process each patient's data
		for patient_id in df['Record ID'].unique():
			patient_data = df[df['Record ID'] == patient_id].sort_values('Days_Since_First_Visit')
			healing_rates, is_improving, estimated_days = calculate_patient_healing_metrics(patient_data)

			# Update patient records with healing rates
			for i, (idx, row) in enumerate(patient_data.iterrows()):
				if i < len(healing_rates):
					df.loc[idx, 'Healing Rate (%)'] = healing_rates[i]

			# Update the last visit with overall improvement status
			df.loc[patient_data.iloc[-1].name, 'Overall_Improvement'] = 'Yes' if is_improving else 'No'

			if not np.isnan(estimated_days):
				df.loc[patient_data.index, 'Estimated_Days_To_Heal'] = estimated_days

		# Calculate and store average healing rates
		df['Average Healing Rate (%)'] = df.groupby('Record ID')['Healing Rate (%)'].transform('mean')

		# Ensure estimated days column exists
		if 'Estimated_Days_To_Heal' not in df.columns:
			df['Estimated_Days_To_Heal'] = pd.Series(np.nan, index=df.index, dtype=float)

		return df

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
