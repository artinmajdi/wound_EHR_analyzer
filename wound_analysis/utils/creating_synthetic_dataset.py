#!/usr/bin/env python3
"""
Synthetic SmartBandage Data Generator

This script generates synthetic data for the SmartBandage dataset and saves it to the dataset directory.
It creates completely fictional patient data that follows the same structure as the original CSV.
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

def generate_synthetic_smartbandage_data(num_rows=436):
    """
    Generate synthetic data for the SmartBandage dataset.
    This creates completely fictional patient data that follows
    the same structure as the original CSV.

    Args:
        num_rows (int): Number of rows to generate in the dataset

    Returns:
        pandas.DataFrame: Generated synthetic dataset
    """
    # Define helper functions
    def random_date(start, end):
        """Generate a random date between start and end."""
        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = random.randrange(int_delta)
        return start + timedelta(seconds=random_second)

    def format_date(date):
        """Format date to MM/DD/YYYY."""
        return date.strftime("%m/%d/%Y")

    # Define possible values for categorical variables
    sex_options               = ['Male', 'Female']
    race_options              = ['White', 'Black or African American', 'Asian', 'American Indian/Alaska Native', 'Native Hawaiian or Other Pacific Islander', 'Other']
    ethnicity_options         = ['Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown']
    cohort_options            = ['Cohort A', 'Cohort B', 'Cohort C']
    smoking_options           = ['Current smoker', 'Former smoker', 'Never smoker']
    alcohol_options           = ['Current', 'Former', 'Never']
    yes_no_options            = ['Yes', 'No']
    drug_use_types            = ['None', 'Marijuana', 'Cocaine', 'Heroin', 'Methamphetamine', 'Multiple']
    frequency_options         = ['Daily', 'Weekly', 'Monthly', 'Less than monthly', 'Never']
    wound_location_options    = ['Lower extremity', 'Upper extremity', 'Sacrum', 'Abdomen', 'Foot', 'Back']
    wound_type_options        = ['Pressure Ulcer', 'Diabetic Foot Ulcer', 'Venous Leg Ulcer', 'Arterial Ulcer', 'Surgical Wound', 'Traumatic Wound']
    wound_care_options        = ['Foam dressing', 'Hydrocolloid', 'Alginate', 'Silver dressing', 'Negative pressure therapy']
    infection_options         = ['None', 'Mild', 'Moderate', 'Severe']
    granulation_options       = ['Poor', 'Moderate', 'Good', 'Excellent']
    necrosis_options          = ['None', 'Minimal', 'Moderate', 'Extensive']
    exudate_volume_options    = ['None', 'Minimal', 'Moderate', 'Heavy']
    exudate_viscosity_options = ['Watery', 'Thick', 'Sticky']
    exudate_type_options      = ['Serous', 'Serosanguineous', 'Sanguineous', 'Purulent']

    # Prepare data container
    data = []

    # Generate unique patients (75% of records will be unique patients)
    patient_count = int(num_rows * 0.75)

    # Define today's date for reference
    today = datetime.now()

    # Generate patients
    for i in range(patient_count):
        # Generate a random birth date for an adult (18-90 years old)
        min_birth_date = today - timedelta(days=90*365)
        max_birth_date = today - timedelta(days=18*365)
        birth_date = random_date(min_birth_date, max_birth_date)
        age = (today - birth_date).days // 365

        # Random chance of diabetes - higher for older patients
        diabetes_chance = 0.2 + (age / 200)  # Increases with age
        is_diabetic = 'Yes' if random.random() < diabetes_chance else 'No'

        # MRN - medical record number (6-digit number)
        mrn = random.randint(100000, 999999)

        # Physical measurements
        weight = round(random.uniform(50, 150), 1)  # 50-150 kg
        height = round(random.uniform(150, 200), 1)  # 150-200 cm
        bmi = round(weight / ((height/100) ** 2), 1)

        # Generate wound onset date (between 1 week and 1 year before today)
        min_wound_date = today - timedelta(days=365)
        max_wound_date = today - timedelta(days=7)
        wound_onset_date = random_date(min_wound_date, max_wound_date)

        # Generate first visit date (after wound onset but before today)
        visit_date = random_date(wound_onset_date, today)

        # Smoking data
        smoking_status = random.choice(smoking_options)
        packs_per_day = 0 if smoking_status == 'Never smoker' else round(random.uniform(0.1, 2), 1)
        years_smoked = 0 if smoking_status == 'Never smoker' else random.randint(1, 40)

        # A1c data - higher for diabetic patients
        a1c_available = random.choice(['Yes', 'No'])
        a1c = None
        if a1c_available == 'Yes':
            a1c = round(random.uniform(6.5, 9.5), 1) if is_diabetic == 'Yes' else round(random.uniform(4.5, 6.0), 1)

        # Medical history - more likely for older patients
        medical_history = []
        if random.random() < 0.3 + (age / 200):
            medical_history.append('Hypertension')
        if random.random() < 0.2 + (age / 250):
            medical_history.append('Hyperlipidemia')
        if is_diabetic == 'Yes':
            medical_history.append('Diabetes')
        if random.random() < 0.15 + (age / 300):
            medical_history.append('Coronary Artery Disease')
        if random.random() < 0.1 + (age / 400):
            medical_history.append('Stroke')
        if random.random() < 0.2:
            medical_history.append('Obesity')

        medical_history_str = ', '.join(medical_history) if medical_history else 'None'

        # Wound measurements
        length = round(random.uniform(0.5, 10.5), 1)
        width = round(random.uniform(0.5, 8.5), 1)
        depth = round(random.uniform(0.1, 3.1), 1)
        wound_area = round(length * width, 2)

        # Generate a patient record
        patient = {
            'Record ID': i + 1,
            'Event Name': 'Visit 1',
            'MRN': float(mrn),
            'Date of Birth': format_date(birth_date),
            'Sex': random.choice(sex_options),
            'Race ': random.choice(race_options),
            'Race: Other - Specify ': '',
            'Ethnicity ': random.choice(ethnicity_options),
            'Weight': weight,
            'Height ': height,
            'BMI ': bmi,
            'Study Cohort ': random.choice(cohort_options),
            'Smoking status': smoking_status,
            'Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes': packs_per_day,
            'Number of Years smoked/has been smoking cigarettes': years_smoked,
            'Alcohol Use Status ': random.choice(alcohol_options),
            'Number of alcohol drinks consumed/has been consuming': str(random.randint(0, 15)),
            'Illicit drug use?': random.choice(yes_no_options),
            'Current type of drug use:': random.choice(drug_use_types),
            'Former type of drug use:': random.choice(drug_use_types),
            'How often has the patient used these substances in the past 3 months?': random.choice(frequency_options),
            'IV drug use?': random.choice(yes_no_options),
            'Medical History (select all that apply)': medical_history_str,
            'Diabetes?': is_diabetic,
            'Respiratory': random.choice(['COPD', 'Asthma', 'None']),
            'Cardiovascular ': random.choice(['Hypertension', 'CHF', 'None']),
            'Gastrointestinal': random.choice(['GERD', 'IBD', 'None']),
            'Musculoskeletal': random.choice(['Osteoarthritis', 'Rheumatoid Arthritis', 'None']),
            'Endocrine/ Metabolic': 'Diabetes' if is_diabetic == 'Yes' else random.choice(['Hypothyroidism', 'None']),
            'Hematopoietic': float(1 if random.random() < 0.1 else 0),
            'Hepatic/Renal': random.choice(['CKD', 'Hepatitis', 'None']),
            'Neurologic': random.choice(['Neuropathy', 'Stroke', 'None']),
            'Immune': float(1 if random.random() < 0.1 else 0),
            'Other Medical History': '',
            'A1c  available within the last 3 months?': a1c_available,
            'Hemoglobin A1c (%)': a1c,
            'Skipped Visit?': 'No',
            'Visit date ': format_date(visit_date),
            'Calculated Age at Enrollment': float(age),
            'Oxygenation (%)': f"{round(random.uniform(85, 100), 1)}%",
            'Hemoglobin Level': round(random.uniform(10, 16), 1),
            'Oxyhemoglobin Level': round(random.uniform(10, 14), 1),
            'Deoxyhemoglobin Level': round(random.uniform(1, 3), 1),
            'Center of Wound Temperature (Fahrenheit)': round(random.uniform(96, 102), 1),
            'Edge of Wound Temperature (Fahrenheit)': round(random.uniform(94, 100), 1),
            'Peri-wound Temperature (Fahrenheit)': round(random.uniform(92, 98), 1),
            'Skin Impedance (kOhms) - Z': round(random.uniform(10, 60), 2),
            'Skin Impedance (kOhms) - Z\'': round(random.uniform(5, 35), 2),
            'Skin Impedance (kOhms) - Z\'\'': round(random.uniform(2, 22), 2),
            'Target wound onset date ': format_date(wound_onset_date),
            'Length (cm)': length,
            'Width (cm)': width,
            'Depth (cm)': depth,
            'Calculated Wound Area': wound_area,
            'Describe the wound location': random.choice(wound_location_options),
            'Is there undermining/ tunneling? ': random.choice(yes_no_options),
            'Undermining Location Description': random.choice(['Lateral edge', 'Medial edge', 'None']),
            'Tunneling Location Description': random.choice(['Lateral edge', 'Medial edge', 'None']),
            'Wound Type ': random.choice(wound_type_options),
            'Current wound care ': random.choice(wound_care_options),
            'Clinical events ': random.choice(['Infection', 'Debridement', 'None']),
            'Diabetic Foot Wound - WIfI Classification: foot Infection (fI)': '',
            'Infection': random.choice(infection_options),
            'Infection/ Biomarker Measurement ': '',
            'Granulation': random.choice(yes_no_options),
            'Granulation Quality ': random.choice(granulation_options),
            'Necrosis ': random.choice(necrosis_options),
            'Exudate Volume': random.choice(exudate_volume_options),
            'Exudate Viscosity': random.choice(exudate_viscosity_options),
            'Exudate Type ': random.choice(exudate_type_options)
        }

        data.append(patient)

    # Generate follow-up visits for some patients
    follow_ups = []

    # Create 1-3 follow-up visits for randomly selected patients
    for i in range(patient_count):
        patient = data[i]
        # 50% chance of no follow-up, 30% one follow-up, 15% two follow-ups, 5% three follow-ups
        follow_up_count = 0 if random.random() < 0.5 else (
                          1 if random.random() < 0.6 else (
                          2 if random.random() < 0.7 else 3))

        for visit in range(1, follow_up_count + 1):
            # Create a follow-up visit by copying the patient data
            follow_up = patient.copy()

            # Update event name for follow-up
            follow_up['Event Name'] = f'Visit {visit + 1}'

            # Update visit date (2-4 weeks after previous visit)
            original_visit = datetime.strptime(patient['Visit date '], '%m/%d/%Y')
            follow_up_date = original_visit + timedelta(days=(visit * (random.randint(14, 28))))
            follow_up['Visit date '] = format_date(follow_up_date)

            # Update wound measurements - show healing over time
            healing_factor = (random.uniform(0.1, 0.3) * visit)  # 10-30% improvement per visit
            follow_up['Length (cm)'] = max(0.1, round(patient['Length (cm)'] * (1 - healing_factor), 1))
            follow_up['Width (cm)'] = max(0.1, round(patient['Width (cm)'] * (1 - healing_factor), 1))
            follow_up['Depth (cm)'] = max(0.1, round(patient['Depth (cm)'] * (1 - healing_factor), 1))
            follow_up['Calculated Wound Area'] = round(follow_up['Length (cm)'] * follow_up['Width (cm)'], 2)

            # Improve wound status over time
            granulation_map = {'Poor': 'Moderate', 'Moderate': 'Good', 'Good': 'Excellent', 'Excellent': 'Excellent'}
            necrosis_map = {'Extensive': 'Moderate', 'Moderate': 'Minimal', 'Minimal': 'None', 'None': 'None'}
            exudate_map = {'Heavy': 'Moderate', 'Moderate': 'Minimal', 'Minimal': 'None', 'None': 'None'}

            follow_up['Granulation Quality '] = granulation_map.get(patient['Granulation Quality '], 'Good')
            follow_up['Necrosis '] = necrosis_map.get(patient['Necrosis '], 'Minimal')
            follow_up['Exudate Volume'] = exudate_map.get(patient['Exudate Volume'], 'Minimal')

            # Add some variation to biomarkers
            follow_up['Hemoglobin Level'] = round(patient['Hemoglobin Level'] + random.uniform(-0.3, 0.6), 1)
            follow_up['Oxyhemoglobin Level'] = round(patient['Oxyhemoglobin Level'] + random.uniform(-0.3, 0.6), 1)
            follow_up['Deoxyhemoglobin Level'] = round(patient['Deoxyhemoglobin Level'] + random.uniform(-0.2, 0.4), 1)

            follow_ups.append(follow_up)

    # Add follow-up visits to data (up to the total number of rows)
    remaining_slots = num_rows - patient_count
    follow_ups_to_add = follow_ups[:remaining_slots]
    data.extend(follow_ups_to_add)

    # Ensure we have exactly the requested number of rows
    if len(data) < num_rows:
        # Add more unique patients if needed
        for i in range(len(data), num_rows):
            # Copy a random patient and modify
            template = random.choice(data[:patient_count])
            new_patient = template.copy()
            new_patient['Record ID'] = i + 1
            new_patient['MRN'] = float(random.randint(100000, 999999))

            # Adjust other values slightly
            new_patient['Weight'] = round(template['Weight'] * random.uniform(0.9, 1.1), 1)
            new_patient['Height '] = round(template['Height '] * random.uniform(0.95, 1.05), 1)
            new_patient['BMI '] = round(new_patient['Weight'] / ((new_patient['Height ']/100) ** 2), 1)

            data.append(new_patient)

    # Update all Record IDs to be sequential
    for i, record in enumerate(data):
        record['Record ID'] = i + 1

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df

def save_synthetic_data(df, output_dir, filename='synthetic_smartbandage_data.csv'):
    """
    Save the synthetic data to a CSV file in the specified directory.

    Args:
        df (pandas.DataFrame): The synthetic data to save
        output_dir (str): Directory path where to save the CSV file
        filename (str): Name of the output CSV file

    Returns:
        str: Full path to the saved file
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the full path
    output_path = os.path.join(output_dir, filename)

    # Save to CSV
    df.to_csv(output_path, index=False)

    return output_path

def print_dataset_stats(df):
    """
    Print statistics about the generated dataset.

    Args:
        df (pandas.DataFrame): The synthetic data
    """
    print(f"Synthetic data generated with {len(df)} records.")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Diabetes percentage: {(df['Diabetes?'] == 'Yes').mean() * 100:.1f}%")
    print(f"Male percentage: {(df['Sex'] == 'Male').mean() * 100:.1f}%")

    # Display wound type distribution
    wound_counts = df['Wound Type '].value_counts()
    print("\nWound Type Distribution:")
    for wound_type, count in wound_counts.items():
        print(f"  {wound_type}: {count} ({count/len(df)*100:.1f}%)")

def main():
    """
    Main function to generate and save synthetic data.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate synthetic SmartBandage data')
        parser.add_argument('--rows', type=int, default=436,
                            help='Number of rows to generate (default: 436)')
        parser.add_argument('--output', type=str, default=None,
                            help='Output directory (default: dataset directory in project root)')
        parser.add_argument('--filename', type=str, default='synthetic_smartbandage_data.csv',
                            help='Output filename (default: synthetic_smartbandage_data.csv)')
        args = parser.parse_args()

        # Determine the output directory
        if args.output:
            output_dir = args.output
        else:
            # Get the project root directory (assuming this script is in wound_analysis/utils/)
            project_root = Path(__file__).resolve().parents[2]
            output_dir = os.path.join(project_root, 'dataset')

        # Generate synthetic data
        print(f"Generating {args.rows} rows of synthetic data...")
        synthetic_data = generate_synthetic_smartbandage_data(args.rows)

        # Save the data
        output_path = save_synthetic_data(synthetic_data, output_dir, args.filename)
        print(f"Data saved to: {output_path}")

        # Print statistics
        print_dataset_stats(synthetic_data)

        return 0  # Success exit code
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1  # Error exit code

if __name__ == "__main__":
    sys.exit(main())
