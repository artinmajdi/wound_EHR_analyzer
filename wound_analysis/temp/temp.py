
import pandas as pd
import numpy as np
import pandas as pd
from ..llm_interface import WoundAnalysisLLM
from ..data_processor import WoundDataProcessor

dataset_path = '/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv'

df = pd.read_csv(dataset_path)
df.head()


def test_getting_llm_prompt():

	print(df['Wound Type '].value_counts())


	llm = WoundAnalysisLLM(platform="ai-verde", model_name="llama-3.3-70b-fp8")
	data_processor = WoundDataProcessor(df=df, dataset_path=dataset_path)

	mode = 'all'
	if mode=='all':
		patient_data = data_processor.get_population_statistics()
		prompt = llm._format_population_prompt(patient_data)
		# analysis = llm.analyze_population_data(patient_data)
	else:
		patient_id = 3
		patient_data = data_processor.get_patient_visits(record_id=int(patient_id))
		prompt = llm._format_per_patient_prompt(patient_data)
		# analysis = llm.analyze_population_data(patient_data)

	print(prompt)
	print(patient_data.keys())
	print(patient_data['sensor_data'])




def _calculate_healing_rates(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate healing rates for each patient visit."""
	# Constants
	MAX_TREATMENT_DAYS = 730  # 2 years in days
	MIN_WOUND_AREA = 0

	def calculate_patient_healing_metrics(patient_data: pd.DataFrame) -> tuple[float, bool, float]:
		"""Calculate healing rate and estimated days for a patient.

		Returns:
			tuple: (healing_rate, is_improving, estimated_days_to_heal)
		"""
		if len(patient_data) < 2:
			return 0.0, False, np.nan

		first_visit = patient_data.iloc[0]
		last_visit = patient_data.iloc[-1]

		days_elapsed = last_visit['Days_Since_First_Visit'] - first_visit['Days_Since_First_Visit']
		if days_elapsed <= 0:
			return 0.0, False, np.nan

		area_change = last_visit['Calculated Wound Area'] - first_visit['Calculated Wound Area']
		healing_rate = area_change / days_elapsed
		is_improving = healing_rate < 0

		estimated_days = np.nan
		if is_improving:
			current_area = last_visit['Calculated Wound Area']
			if current_area > MIN_WOUND_AREA:
				healing_speed = abs(healing_rate)
				if healing_speed > 0:
					days_to_heal = current_area / healing_speed
					total_days = last_visit['Days_Since_First_Visit'] + days_to_heal
					if 0 < total_days < MAX_TREATMENT_DAYS:
						estimated_days = float(total_days)

		return healing_rate, is_improving, estimated_days

	# Process each patient's data
	for patient_id in df['Record ID'].unique():
		patient_data = df[df['Record ID'] == patient_id].sort_values('Days_Since_First_Visit')

		healing_rate, is_improving, estimated_days = calculate_patient_healing_metrics(patient_data)

		# Update patient records
		df.loc[patient_data.index, 'Healing Rate (%)'] = healing_rate
		df.loc[patient_data.iloc[-1].name, 'Overall_Improvement'] = 'Yes' if is_improving else 'No'
		if not np.isnan(estimated_days):
			df.loc[patient_data.index, 'Estimated_Days_To_Heal'] = estimated_days

	# Calculate and store average healing rates
	df['Average Healing Rate (%)'] = df.groupby('Record ID')['Healing Rate (%)'].transform('mean')

	# Ensure estimated days column exists
	if 'Estimated_Days_To_Heal' not in df.columns:
		df['Estimated_Days_To_Heal'] = pd.Series(np.nan, index=df.index, dtype=float)

	return df


if __name__ == "__main__":


    dataset_path = '/Users/artinmajdi/Documents/GitHubs/postdoc/Wound_management_interpreter_LLM/dataset/SmartBandage-Data_for_llm.csv'
    df = pd.read_csv(dataset_path)

    # Process data
    df = _calculate_healing_rates(df)

    print(df)
