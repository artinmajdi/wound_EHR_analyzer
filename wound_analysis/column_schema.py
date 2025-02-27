from pydantic import BaseModel, Field
from typing import Optional

# # Group columns into logical categories
# patient_info_columns = [
#     'Record ID', 'MRN', 'Date of Birth', 'Sex', 'Race', 'Race: Other - Specify',
#     'Ethnicity', 'Weight', 'Height', 'BMI', 'Calculated Age at Enrollment'
# ]

# visit_info_columns = [
#     'Event Name', 'Visit date', 'Skipped Visit?', 'Study Cohort'
# ]

# medical_history_columns = [
#     'Medical History (select all that apply)', 'Diabetes?', 'Respiratory', 'Cardiovascular',
#     'Gastrointestinal', 'Musculoskeletal', 'Endocrine/ Metabolic', 'Hematopoietic',
#     'Hepatic/Renal', 'Neurologic', 'Immune', 'Other Medical History',
#     'A1c available within the last 3 months?', 'Hemoglobin A1c (%)'
# ]

# lifestyle_habits_columns = [
#     'Smoking status', 'Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes',
#     'Number of Years smoked/has been smoking cigarettes', 'Alcohol Use Status',
#     'Number of alcohol drinks consumed/has been consuming', 'Illicit drug use?',
#     'Current type of drug use:', 'Former type of drug use:',
#     'How often has the patient used these substances in the past 3 months?', 'IV drug use?'
# ]

# wound_characteristics_columns = [
#     'Target wound onset date', 'Length (cm)', 'Width (cm)', 'Depth (cm)',
#     'Calculated Wound Area', 'Describe the wound location', 'Is there undermining/ tunneling?',
#     'Undermining Location Description', 'Tunneling Location Description', 'Wound Type'
# ]

# wound_assessment_columns = [
#     'Granulation', 'Granulation Quality', 'Necrosis', 'Exudate Volume',
#     'Exudate Viscosity', 'Exudate Type', 'Infection', 'Infection/ Biomarker Measurement',
#     'Diabetic Foot Wound - WIfI Classification: foot Infection (fI)'
# ]

# treatment_columns = [
#     'Current wound care', 'Clinical events'
# ]

# sensor_data_columns = [
#     'Oxygenation (%)', 'Hemoglobin Level', 'Oxyhemoglobin Level', 'Deoxyhemoglobin Level',
#     'Center of Wound Temperature (Fahrenheit)', 'Edge of Wound Temperature (Fahrenheit)',
#     'Peri-wound Temperature (Fahrenheit)', 'Skin Impedance (kOhms) - Z',
#     'Skin Impedance (kOhms) - Z\'', 'Skin Impedance (kOhms) - Z\'\''
# ]

# # Create a dictionary of column categories
# column_categories = {
#     "Patient Information": patient_info_columns,
#     "Visit Information": visit_info_columns,
#     "Medical History": medical_history_columns,
#     "Lifestyle Habits": lifestyle_habits_columns,
#     "Wound Characteristics": wound_characteristics_columns,
#     "Wound Assessment": wound_assessment_columns,
#     "Treatment": treatment_columns,
#     "Sensor Data": sensor_data_columns
# }



class PatientIdentifiers(BaseModel):
    record_id : str = Field('Record ID', description="Patient's unique record identifier")
    event_name: str = Field('Event Name', description="Name of the clinical event")
    mrn       : str = Field('MRN', description="Medical Record Number")

class PatientDemographics(BaseModel):
    dob              : str = Field('Date of Birth', description="Patient's date of birth")
    sex              : str = Field('Sex', description="Patient's biological sex")
    race             : str = Field('Race', description="Patient's race")
    race_other       : str = Field('Race: Other - Specify', description="Other race specification")
    ethnicity        : str = Field('Ethnicity', description="Patient's ethnicity")
    weight           : str = Field('Weight', description="Patient's weight")
    height           : str = Field('Height', description="Patient's height")
    bmi              : str = Field('BMI', description="Body Mass Index")
    bmi_category     : str = Field('BMI Category', description="BMI classification category")
    study_cohort     : str = Field('Study Cohort', description="Study group assignment")
    age_at_enrollment: str = Field('Calculated Age at Enrollment', description="Age when enrolled in study")

class LifestyleFactors(BaseModel):
    smoking_status         : str = Field('Smoking status', description="Current smoking status")
    packs_per_day          : str = Field('Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes')
    years_smoked           : str = Field('Number of Years smoked/has been smoking cigarettes')
    alcohol_status         : str = Field('Alcohol Use Status')
    alcohol_drinks         : str = Field('Number of alcohol drinks consumed/has been consuming')
    illicit_drug_use       : str = Field('Illicit drug use?')
    current_drug_type      : str = Field('Current type of drug use:')
    former_drug_type       : str = Field('Former type of drug use:')
    substance_use_frequency: str = Field('How often has the patient used these substances in the past 3 months?')
    iv_drug_use            : str = Field('IV drug use?')

class MedicalHistory(BaseModel):
    medical_history      : str = Field('Medical History (select all that apply)')
    diabetes             : str = Field('Diabetes?')
    respiratory          : str = Field('Respiratory')
    cardiovascular       : str = Field('Cardiovascular')
    gastrointestinal     : str = Field('Gastrointestinal')
    musculoskeletal      : str = Field('Musculoskeletal')
    endocrine_metabolic  : str = Field('Endocrine/ Metabolic')
    hematopoietic        : str = Field('Hematopoietic')
    hepatic_renal        : str = Field('Hepatic/Renal')
    neurologic           : str = Field('Neurologic')
    immune               : str = Field('Immune')
    other_medical_history: str = Field('Other Medical History')
    a1c_available        : str = Field('A1c  available within the last 3 months?')
    a1c                  : str = Field('Hemoglobin A1c (%)')

class VisitInformation(BaseModel):
    skipped_visit         : str = Field('Skipped Visit?')
    visit_date            : str = Field('Visit date')
    visit_number          : str = Field('Visit Number')
    days_since_first_visit: str = Field('Days_Since_First_Visit')

class OxygenationMeasurements(BaseModel):
    oxygenation    : str = Field('Oxygenation (%)')
    hemoglobin     : str = Field('Hemoglobin Level')
    oxyhemoglobin  : str = Field('Oxyhemoglobin Level')
    deoxyhemoglobin: str = Field('Deoxyhemoglobin Level')

class TemperatureMeasurements(BaseModel):
    center_temp         : str = Field('Center of Wound Temperature (Fahrenheit)')
    edge_temp           : str = Field('Edge of Wound Temperature (Fahrenheit)')
    peri_temp           : str = Field('Peri-wound Temperature (Fahrenheit)')
    center_edge_gradient: str = Field('Center-Edge Temp Gradient')
    edge_peri_gradient  : str = Field('Edge-Peri Temp Gradient')
    total_temp_gradient : str = Field('Total Temp Gradient')

class ImpedanceMeasurements(BaseModel):
    # High frequency impedance measurements (corresponding to CSV data)
    highest_freq_z             : str = Field('Skin Impedance (kOhms) - Z')
    highest_freq_z_prime       : str = Field("Skin Impedance (kOhms) - Z'")
    highest_freq_z_double_prime: str = Field("Skin Impedance (kOhms) - Z''")

    # Center frequency impedance measurements
    center_freq_z             : str = Field('Center Frequency Impedance (kOhms) - Z')
    center_freq_z_prime       : str = Field("Center Frequency Impedance (kOhms) - Z'")
    center_freq_z_double_prime: str = Field("Center Frequency Impedance (kOhms) - Z''")

    # Low frequency impedance measurements
    lowest_freq_z       : str = Field('Low Frequency Impedance (kOhms) - Z')
    lowest_freq_z_prime : str = Field("Low Frequency Impedance (kOhms) - Z'")
    lowest_freq_z_double_prime: str = Field("Low Frequency Impedance (kOhms) - Z''")

class WoundCharacteristics(BaseModel):
    wound_onset_date    : str = Field('Target wound onset date')
    length              : str = Field('Length (cm)')
    width               : str = Field('Width (cm)')
    depth               : str = Field('Depth (cm)')
    wound_area          : str = Field('Calculated Wound Area')
    wound_location      : str = Field('Describe the wound location')
    undermining         : str = Field('Is there undermining/ tunneling?')
    undermining_location: str = Field('Undermining Location Description')
    tunneling_location  : str = Field('Tunneling Location Description')
    wound_type          : str = Field('Wound Type')
    current_wound_care  : str = Field('Current wound care')

class ClinicalAssessment(BaseModel):
    clinical_events    : str = Field('Clinical events')
    wifi_classification: str = Field('Diabetic Foot Wound - WIfI Classification: foot Infection (fI)')
    infection          : str = Field('Infection')
    infection_biomarker: str = Field('Infection/ Biomarker Measurement')
    granulation        : str = Field('Granulation')
    granulation_quality: str = Field('Granulation Quality')
    necrosis           : str = Field('Necrosis')
    exudate_volume     : str = Field('Exudate Volume')
    exudate_viscosity  : str = Field('Exudate Viscosity')
    exudate_type       : str = Field('Exudate Type')

class HealingMetrics(BaseModel):
    healing_rate          : str = Field('Healing Rate (%)')
    estimated_days_to_heal: str = Field('Estimated_Days_To_Heal')
    overall_improvement   : str = Field('Overall_Improvement')
    average_healing_rate  : str = Field('Average Healing Rate (%)')

class DataColumns(BaseModel):
    """Main model containing all column categories"""
    patient_identifiers  : PatientIdentifiers      = Field(default_factory=PatientIdentifiers)
    demographics         : PatientDemographics     = Field(default_factory=PatientDemographics)
    lifestyle            : LifestyleFactors        = Field(default_factory=LifestyleFactors)
    medical_history      : MedicalHistory          = Field(default_factory=MedicalHistory)
    visit_info           : VisitInformation        = Field(default_factory=VisitInformation)
    oxygenation          : OxygenationMeasurements = Field(default_factory=OxygenationMeasurements)
    temperature          : TemperatureMeasurements = Field(default_factory=TemperatureMeasurements)
    impedance            : ImpedanceMeasurements   = Field(default_factory=ImpedanceMeasurements)
    wound_characteristics: WoundCharacteristics    = Field(default_factory=WoundCharacteristics)
    clinical_assessment  : ClinicalAssessment      = Field(default_factory=ClinicalAssessment)
    healing_metrics      : HealingMetrics          = Field(default_factory=HealingMetrics)

    def get_all_columns(self) -> dict:
        """Get all column names as a flat dictionary"""
        all_columns = {}
        for category in self.__dict__.values():
            if isinstance(category, BaseModel):
                all_columns.update({k: v for k, v in category.__dict__.items()})
        return all_columns

    @classmethod
    def get_column_name(cls, field_name: str) -> Optional[str]:
        """Get the actual column name for a given field name"""
        instance = cls()
        all_columns = instance.get_all_columns()
        return all_columns.get(field_name)
