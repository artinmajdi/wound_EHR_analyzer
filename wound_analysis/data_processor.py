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
from column_schema import DataColumns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoundDataProcessor:
	def __init__(self, dataset_path: pathlib.Path=None, df: pd.DataFrame= None):
		# Initialize the column schema
		self.columns = DataColumns()
		
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
			patient_id_col = self.columns.patient_identifiers.record_id
			skipped_visit_col = self.columns.visit_info.skipped_visit
			
			patient_data = self.df[self.df[patient_id_col] == record_id]
			if patient_data.empty:
				raise ValueError(f"No measurements found for patient {record_id}")

			# Get metadata from first visit
			first_visit = patient_data.iloc[0]
			metadata = self._extract_patient_metadata(first_visit)

			visits_data = []
			for _, visit in patient_data.iterrows():
				if pd.isna(visit.get(skipped_visit_col)) or visit[skipped_visit_col] != 'Yes':
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
		# Get schema columns
		pi = self.columns.patient_identifiers
		vis = self.columns.visit_info
		wc = self.columns.wound_characteristics
		dem = self.columns.demographics
		ls = self.columns.lifestyle
		mh = self.columns.medical_history
		temp = self.columns.temperature
		oxy = self.columns.oxygenation
		imp = self.columns.impedance
		ca = self.columns.clinical_assessment
		hm = self.columns.healing_metrics
		
		# Get processed data
		df = self.get_processed_data()
		if df.empty:
			raise ValueError("No data available for population statistics")

		# Calculate derived metrics
		df['BMI Category'] = pd.cut(
			df[dem.bmi],
			bins=[0, 18.5, 24.9, 29.9, float('inf')],
			labels=['Underweight', 'Normal', 'Overweight', 'Obese']
		)

		# Calculate healing metrics
		df['Healing_Color'] = df[hm.healing_rate].apply(lambda x: 'green' if x < 0 else 'red')
		df['Healing_Status'] = df[hm.healing_rate].apply(
			lambda x: 'Improving' if x < 0 else ('Stable' if -5 <= x <= 5 else 'Worsening')
		)

		stats = {
			'summary': {
				'total_patients': len(df[pi.record_id].unique()),
				'total_visits': len(df),
				'avg_visits_per_patient': len(df) / len(df[pi.record_id].unique()),
				'overall_improvement_rate': (df[hm.healing_rate] < 0).mean() * 100,
				'avg_treatment_duration_days': (df.groupby(pi.record_id)[vis.days_since_first_visit].max().mean()),
				'completion_rate': (df['Visit Status'] == 'Completed').mean() * 100 if 'Visit Status' in df.columns else None
			},
			'demographics': {
				'age_stats': {
					'summary': f"Mean: {df[dem.age_at_enrollment].mean():.1f}, Median: {df[dem.age_at_enrollment].median():.1f}",
					'distribution': df[dem.age_at_enrollment].value_counts().to_dict(),
					'age_groups': pd.cut(df[dem.age_at_enrollment],
						bins=[0, 30, 50, 70, float('inf')],
						labels=['<30', '30-50', '50-70', '>70']).value_counts().to_dict()
				},
				'gender_distribution': df[dem.sex].value_counts().to_dict(),
				'race_distribution': df[dem.race].value_counts().to_dict(),
				'ethnicity_distribution': df[dem.ethnicity].value_counts().to_dict(),
				'bmi_stats': {
					'summary': f"Mean: {df[dem.bmi].mean():.1f}, Range: {df[dem.bmi].min():.1f}-{df[dem.bmi].max():.1f}",
					'distribution': df['BMI Category'].value_counts().to_dict(),
					'by_healing_status': df.groupby('BMI Category')[hm.healing_rate].agg(['mean', 'count']).to_dict()
				}
			},
			'risk_factors': {
				'primary_conditions': {
					'diabetes': {
						'distribution': df[mh.diabetes].value_counts().to_dict(),
						'healing_impact': df.groupby(mh.diabetes)[hm.healing_rate].agg(['mean', 'std', 'count']).to_dict()
					},
					'smoking': {
						'distribution': df[ls.smoking_status].value_counts().to_dict(),
						'healing_impact': df.groupby(ls.smoking_status)[hm.healing_rate].agg(['mean', 'std', 'count']).to_dict()
					}
				},
				'comorbidity_analysis': {
					'diabetes_smoking': df.groupby([mh.diabetes, ls.smoking_status])[hm.healing_rate].agg(['mean', 'count']).to_dict(),
					'diabetes_bmi': df.groupby([mh.diabetes, 'BMI Category'])[hm.healing_rate].agg(['mean', 'count']).to_dict()
				}
			},
			'wound_characteristics': {
				'type_distribution': {
					'overall': df[wc.wound_type].value_counts().to_dict(),
					'by_healing_status': df.groupby([wc.wound_type, 'Healing_Status']).size().to_dict()
				},
				'location_analysis': {
					'distribution': df[wc.wound_location].value_counts().to_dict(),
					'healing_by_location': df.groupby(wc.wound_location)[hm.healing_rate].mean().to_dict()
				},
				'size_progression': {
					'initial_vs_final': {
						'area': {
							'initial': df.groupby(pi.record_id)[wc.wound_area].first().agg(['mean', 'median', 'std']).to_dict(),
							'final': df.groupby(pi.record_id)[wc.wound_area].last().agg(['mean', 'median', 'std']).to_dict(),
							'percent_change': ((df.groupby(pi.record_id)[wc.wound_area].last() -
								df.groupby(pi.record_id)[wc.wound_area].first()) /
								df.groupby(pi.record_id)[wc.wound_area].first() * 100).mean()
						}
					},
					'healing_by_initial_size': {
						'small': df[df[wc.wound_area] < df[wc.wound_area].quantile(0.33)][hm.healing_rate].mean(),
						'medium': df[(df[wc.wound_area] >= df[wc.wound_area].quantile(0.33)) &
							(df[wc.wound_area] < df[wc.wound_area].quantile(0.67))][hm.healing_rate].mean(),
						'large': df[df[wc.wound_area] >= df[wc.wound_area].quantile(0.67)][hm.healing_rate].mean()
					}
				}
			},
			'healing_progression': {
				'overall_stats': {
					'summary': f"Mean: {df[hm.healing_rate].mean():.1f}%, Median: {df[hm.healing_rate].median():.1f}%",
					'distribution': df['Healing_Status'].value_counts().to_dict(),
					'percentiles': df[hm.healing_rate].quantile([0.25, 0.5, 0.75]).to_dict()
				},
				'temporal_analysis': {
					'by_visit_number': df.groupby('Visit Number')[hm.healing_rate].agg(['mean', 'std', 'count']).to_dict(),
					'by_treatment_duration': pd.cut(df[vis.days_since_first_visit],
						bins=[0, 30, 90, 180, float('inf')],
						labels=['<30 days', '30-90 days', '90-180 days', '>180 days']
					).value_counts().to_dict()
				}
			},
			'exudate_analysis': {
				'characteristics': {
					'volume': {
						'distribution': df[ca.exudate_volume].value_counts().to_dict(),
						'healing_correlation': df.groupby(ca.exudate_volume)[hm.healing_rate].mean().to_dict()
					},
					'type': {
						'distribution': df[ca.exudate_type].value_counts().to_dict(),
						'healing_correlation': df.groupby(ca.exudate_type)[hm.healing_rate].mean().to_dict()
					},
					'viscosity': {
						'distribution': df[ca.exudate_viscosity].value_counts().to_dict(),
						'healing_correlation': df.groupby(ca.exudate_viscosity)[hm.healing_rate].mean().to_dict()
					}
				},
				'temporal_patterns': {
					'volume_progression': df.groupby('Visit Number')[ca.exudate_volume].value_counts().to_dict(),
					'type_progression': df.groupby('Visit Number')[ca.exudate_type].value_counts().to_dict()
				}
			}
		}

		# Add sensor data analysis if available
		stats['sensor_data'] = {}

		# Temperature Analysis
		if temp.center_temp in df.columns:
			stats['sensor_data']['temperature'] = {
				'center_temp': {
					'overall': df[temp.center_temp].agg(['mean', 'std', 'min', 'max']).to_dict(),
					'by_healing_status': df.groupby('Healing_Status')[temp.center_temp].mean().to_dict(),
					'temporal_trend': df.groupby('Visit Number')[temp.center_temp].mean().to_dict()
				}
			}

			# Add edge and peri-wound temperatures if available
			if all(col in df.columns for col in [temp.edge_temp, temp.peri_temp]):
				stats['sensor_data']['temperature'].update({
					'edge_temp': {
						'overall': df[temp.edge_temp].agg(['mean', 'std', 'min', 'max']).to_dict(),
						'by_healing_status': df.groupby('Healing_Status')[temp.edge_temp].mean().to_dict()
					},
					'peri_temp': {
						'overall': df[temp.peri_temp].agg(['mean', 'std', 'min', 'max']).to_dict(),
						'by_healing_status': df.groupby('Healing_Status')[temp.peri_temp].mean().to_dict()
					},
					'gradients': {
						'center_to_edge': (df[temp.center_temp] - df[temp.edge_temp]).agg(['mean', 'std']).to_dict(),
						'center_to_peri': (df[temp.center_temp] - df[temp.peri_temp]).agg(['mean', 'std']).to_dict(),
						'by_healing_status': df.groupby('Healing_Status').apply(
							lambda x: {
								'center_to_edge': (x[temp.center_temp] - x[temp.edge_temp]).mean(),
								'center_to_peri': (x[temp.center_temp] - x[temp.peri_temp]).mean()
							}
						).to_dict()
					}
				})

		# Impedance Analysis
		if imp.highest_freq_z in df.columns:
			stats['sensor_data']['impedance'] = {
				'magnitude': {
					'overall': df[imp.highest_freq_z].agg(['mean', 'std', 'min', 'max']).to_dict(),
					'by_healing_status': df.groupby('Healing_Status')[imp.highest_freq_z].mean().to_dict(),
					'temporal_trend': df.groupby('Visit Number')[imp.highest_freq_z].mean().to_dict()
				}
			}

			# Add complex impedance components if available
			if all(col in df.columns for col in [imp.highest_freq_z_prime, imp.highest_freq_z_double_prime]):
				stats['sensor_data']['impedance'].update({
					'complex_components': {
						'real': {
							'overall': df[imp.highest_freq_z_prime].agg(['mean', 'std', 'min', 'max']).to_dict(),
							'by_healing_status': df.groupby('Healing_Status')[imp.highest_freq_z_prime].mean().to_dict()
						},
						'imaginary': {
							'overall': df[imp.highest_freq_z_double_prime].agg(['mean', 'std', 'min', 'max']).to_dict(),
							'by_healing_status': df.groupby('Healing_Status')[imp.highest_freq_z_double_prime].mean().to_dict()
						}
					}
				})

		# Oxygenation Analysis
		if oxy.oxygenation in df.columns:
			stats['sensor_data']['oxygenation'] = {
				'oxygenation': {
					'overall': df[oxy.oxygenation].agg(['mean', 'std', 'min', 'max']).to_dict(),
					'by_healing_status': df.groupby('Healing_Status')[oxy.oxygenation].mean().to_dict(),
					'temporal_trend': df.groupby('Visit Number')[oxy.oxygenation].mean().to_dict(),
					'correlation_with_healing': df[oxy.oxygenation].corr(df[hm.healing_rate]),
					'distribution_quartiles': pd.qcut(df[oxy.oxygenation], q=4).value_counts().to_dict()
				}
			}

			# Add hemoglobin measurements if available
			for hb_type, col in {'hemoglobin': oxy.hemoglobin,
								'oxyhemoglobin': oxy.oxyhemoglobin,
								'deoxyhemoglobin': oxy.deoxyhemoglobin}.items():
				if col in df.columns:
					stats['sensor_data']['oxygenation'][hb_type] = {
						'overall': df[col].agg(['mean', 'std', 'min', 'max']).to_dict(),
						'by_healing_status': df.groupby('Healing_Status')[col].mean().to_dict(),
						'temporal_trend': df.groupby('Visit Number')[col].mean().to_dict(),
						'correlation_with_healing': df[col].corr(df[hm.healing_rate])
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

		# Get schema columns
		pi = self.columns.patient_identifiers
		vis = self.columns.visit_info
		wc = self.columns.wound_characteristics
		temp = self.columns.temperature
		oxy = self.columns.oxygenation
		dem = self.columns.demographics
		hm = self.columns.healing_metrics
		
		# Create a copy to avoid modifying original data
		df = self.df.copy()

		# Clean column names
		df.columns = df.columns.str.strip()

		# Filter out skipped visits
		df = df[df[vis.skipped_visit] != 'Yes']

		# Extract visit number from Event Name
		df['Visit Number'] = df[pi.event_name].str.extract(r'Visit (\d+)').fillna(1).astype(int)

		# Convert and format dates
		df[vis.visit_date] = pd.to_datetime(df[vis.visit_date])

		# Calculate days since first visit for each patient
		df[vis.days_since_first_visit] = df.groupby(pi.record_id)[vis.visit_date].transform(
			lambda x: (x - x.min()).dt.days
		)

		# Handle Wound Type categorization
		if wc.wound_type in df.columns:
			# Convert to string type first to handle any existing categorical
			df[wc.wound_type] = df[wc.wound_type].astype(str)
			# Replace NaN with 'Unknown'
			df[wc.wound_type] = df[wc.wound_type].replace('nan', 'Unknown')
			# Get unique categories including 'Unknown'
			categories = sorted(df[wc.wound_type].unique())
			# Now create categorical with all possible categories
			df[wc.wound_type] = pd.Categorical(df[wc.wound_type], categories=categories)

		# Calculate wound area if not present
		if wc.wound_area not in df.columns and all(col in df.columns for col in [wc.length, wc.width]):
			df[wc.wound_area] = df[wc.length] * df[wc.width]

		# Convert numeric columns
		numeric_columns = [wc.length, wc.width, hm.healing_rate, oxy.oxygenation]
		for col in numeric_columns:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')

		# Create derived features
		# Temperature gradients
		if all(col in df.columns for col in [temp.center_temp, temp.edge_temp, temp.peri_temp]):
			df[temp.center_edge_gradient] = df[temp.center_temp] - df[temp.edge_temp]
			df[temp.edge_peri_gradient] = df[temp.edge_temp] - df[temp.peri_temp]
			df[temp.total_temp_gradient] = df[temp.center_temp] - df[temp.peri_temp]

		# BMI categories
		if dem.bmi in df.columns:
			df['BMI Category'] = pd.cut(
				df[dem.bmi],
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

					if len(prev_visit) > 0 and wc.wound_area in patient_data.columns:
						prev_area = prev_visit[wc.wound_area].values[0]
						curr_area = row[wc.wound_area]

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
				current_area = last_visit[wc.wound_area]

				if current_area > MIN_WOUND_AREA and avg_healing_rate > 0:
					daily_healing_rate = (avg_healing_rate / 100) * current_area
					if daily_healing_rate > 0:
						days_to_heal = current_area / daily_healing_rate
						total_days = last_visit[vis.days_since_first_visit] + days_to_heal
						if 0 < total_days < MAX_TREATMENT_DAYS:
							estimated_days = float(total_days)

			return healing_rates, is_improving, estimated_days

		# Process each patient's data
		for patient_id in df[pi.record_id].unique():
			patient_data = df[df[pi.record_id] == patient_id].sort_values(vis.days_since_first_visit)
			healing_rates, is_improving, estimated_days = calculate_patient_healing_metrics(patient_data)

			# Update patient records with healing rates
			for i, (idx, row) in enumerate(patient_data.iterrows()):
				if i < len(healing_rates):
					df.loc[idx, hm.healing_rate] = healing_rates[i]

			# Update the last visit with overall improvement status
			df.loc[patient_data.iloc[-1].name, hm.overall_improvement] = 'Yes' if is_improving else 'No'

			if not np.isnan(estimated_days):
				df.loc[patient_data.index, hm.estimated_days_to_heal] = estimated_days

		# Calculate and store average healing rates
		df[hm.average_healing_rate] = df.groupby(pi.record_id)[hm.healing_rate].transform('mean')

		# Ensure estimated days column exists
		if hm.estimated_days_to_heal not in df.columns:
			df[hm.estimated_days_to_heal] = pd.Series(np.nan, index=df.index, dtype=float)

		return df

	def _extract_patient_metadata(self, patient_data) -> Dict:
		"""Extract relevant patient metadata from a single row."""
		
		# Get column names from the schema
		dem = self.columns.demographics
		pi = self.columns.patient_identifiers
		ls = self.columns.lifestyle
		mh = self.columns.medical_history
		
		metadata = {
			'age': patient_data[dem.age_at_enrollment] if not pd.isna(patient_data.get(dem.age_at_enrollment)) else None,
			'sex': patient_data[dem.sex] if not pd.isna(patient_data.get(dem.sex)) else None,
			'race': patient_data[dem.race] if not pd.isna(patient_data.get(dem.race)) else None,
			'ethnicity': patient_data[dem.ethnicity] if not pd.isna(patient_data.get(dem.ethnicity)) else None,
			'weight': patient_data[dem.weight] if not pd.isna(patient_data.get(dem.weight)) else None,
			'height': patient_data[dem.height] if not pd.isna(patient_data.get(dem.height)) else None,
			'bmi': patient_data[dem.bmi] if not pd.isna(patient_data.get(dem.bmi)) else None,
			'study_cohort': patient_data[dem.study_cohort] if not pd.isna(patient_data.get(dem.study_cohort)) else None,
			'smoking_status': patient_data[ls.smoking_status] if not pd.isna(patient_data.get(ls.smoking_status)) else None,
			'packs_per_day': patient_data[ls.packs_per_day] if not pd.isna(patient_data.get(ls.packs_per_day)) else None,
			'years_smoking': patient_data[ls.years_smoked] if not pd.isna(patient_data.get(ls.years_smoked)) else None,
			'alcohol_use': patient_data[ls.alcohol_status] if not pd.isna(patient_data.get(ls.alcohol_status)) else None,
			'alcohol_frequency': patient_data[ls.alcohol_drinks] if not pd.isna(patient_data.get(ls.alcohol_drinks)) else None
		}

		# Medical history from individual columns
		medical_conditions = [
			mh.respiratory, mh.cardiovascular, mh.gastrointestinal, mh.musculoskeletal,
			mh.endocrine_metabolic, mh.hematopoietic, mh.hepatic_renal, mh.neurologic, mh.immune
		]
		
		# Get medical history from standard columns
		metadata['medical_history'] = {
			condition: patient_data[condition]
			for condition in medical_conditions if not pd.isna(patient_data.get(condition))
		}

		# Check additional medical history from free text field
		other_history = patient_data.get(mh.medical_history)
		if not pd.isna(other_history):
			existing_conditions = set(medical_conditions)
			other_conditions = [cond.strip() for cond in str(other_history).split(',')]
			other_conditions = [cond for cond in other_conditions if cond and cond not in existing_conditions]
			if other_conditions:
				metadata['medical_history']['other'] = ', '.join(other_conditions)

		# Diabetes information
		metadata['diabetes'] = {
			'status': patient_data.get(mh.diabetes),
			'hemoglobin_a1c': patient_data.get(mh.a1c),
			'a1c_available': patient_data.get(mh.a1c_available)
		}

		return metadata

	def _get_wound_info(self, visit_data) -> Dict:
		""" Get detailed wound information from a single visit row."""
		try:
			# Get column schema for wound characteristics and clinical assessment
			wc = self.columns.wound_characteristics
			ca = self.columns.clinical_assessment

			def clean_field(data, field):
				return data.get(field) if not pd.isna(data.get(field)) else None

			present = clean_field(visit_data, wc.undermining)

			wound_info = {
				'location'       : clean_field(visit_data, wc.wound_location),
				'type'           : clean_field(visit_data, wc.wound_type),
				'current_care'   : clean_field(visit_data, wc.current_wound_care),
				'clinical_events': clean_field(visit_data, ca.clinical_events),
				'undermining': {
					'present'  : None if present is None else present == 'Yes',
					'location' : visit_data.get(wc.undermining_location),
					'tunneling': visit_data.get(wc.tunneling_location)
				},
				'infection': {
					'status'             : clean_field(visit_data, ca.infection),
					'wifi_classification': visit_data.get(ca.wifi_classification)
				},
				'granulation': {
					'coverage': clean_field(visit_data, ca.granulation),
					'quality' : clean_field(visit_data, ca.granulation_quality)
				},
				'necrosis': visit_data.get(ca.necrosis),
				'exudate': {
					'volume'   : visit_data.get(ca.exudate_volume),
					'viscosity': visit_data.get(ca.exudate_viscosity),
					'type'     : visit_data.get(ca.exudate_type)
				}
			}

			return wound_info

		except Exception as e:
			logger.warning(f"Error getting wound info: {str(e)}")
			return {}

	def _process_visit_data(self, visit, record_id: int) -> Optional[Dict]:
		"""Process visit measurement data from a single row."""
		# Get column schema
		vis = self.columns.visit_info
		wc = self.columns.wound_characteristics
		temp = self.columns.temperature
		oxy = self.columns.oxygenation
		imp = self.columns.impedance
		
		visit_date = pd.to_datetime(visit[vis.visit_date]).strftime('%m-%d-%Y') if not pd.isna(visit.get(vis.visit_date)) else None

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
					'Z': get_float(visit, imp.highest_freq_z),
					'Z_prime': get_float(visit, imp.highest_freq_z_prime),
					'Z_double_prime': get_float(visit, imp.highest_freq_z_double_prime)
				}

				impedance_data['high_frequency'] = transform_impedance_data(high_freq, 80000)

			return impedance_data

		wound_measurements = {
			'length': get_float(visit, wc.length),
			'width' : get_float(visit, wc.width),
			'depth' : get_float(visit, wc.depth),
			'area'  : get_float(visit, wc.wound_area)
		}

		temperature_readings = {
			'center': get_float(visit, temp.center_temp),
			'edge'  : get_float(visit, temp.edge_temp),
			'peri'  : get_float(visit, temp.peri_temp)
		}

		hemoglobin_types = {
			'hemoglobin'     : oxy.hemoglobin,
			'oxyhemoglobin'  : oxy.oxyhemoglobin,
			'deoxyhemoglobin': oxy.deoxyhemoglobin
		}

		return {
			'visit_date': visit_date,
			'wound_measurements': wound_measurements,
			'sensor_data': {
				'oxygenation': get_float(visit, oxy.oxygenation),
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
			visit_date_being_processed: The visit date to process

		Returns:
			Optional[pd.DataFrame]: DataFrame containing processed impedance data, or None if processing fails
		"""
		# Get the visit date column name
		vis = self.columns.visit_info
		
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
