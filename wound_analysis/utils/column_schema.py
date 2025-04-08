from typing import Optional, List, Dict
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum

class PatientIdentifiers(BaseModel):
    record_id : str = Field('Record ID')
    event_name: str = Field('Event Name')
    mrn       : str = Field('MRN')

class PatientDemographics(BaseModel):
    dob              : str = Field('Date of Birth')
    sex              : str = Field('Sex')
    race             : str = Field('Race')
    race_other       : str = Field('Race: Other - Specify')
    ethnicity        : str = Field('Ethnicity')
    weight           : str = Field('Weight')
    height           : str = Field('Height')
    bmi              : str = Field('BMI')
    bmi_category     : str = Field('BMI Category')
    study_cohort     : str = Field('Study Cohort')
    age: str = Field('Calculated Age at Enrollment')

class LifestyleFactors(BaseModel):
    smoking_status         : str = Field('Smoking status')
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
    total_gradient      : str = Field('Total Temp Gradient')

class ImpedanceMeasurements(BaseModel):
    # High frequency impedance measurements (corresponding to CSV data)
    highest_freq_absolute  : str = Field('Skin Impedance (kOhms) - Z')
    highest_freq_real      : str = Field("Skin Impedance (kOhms) - Z'")
    highest_freq_imaginary : str = Field("Skin Impedance (kOhms) - Z''")

    # Center frequency impedance measurements
    center_freq_absolute  : str = Field('Center Frequency Impedance (kOhms) - Z')
    center_freq_real      : str = Field("Center Frequency Impedance (kOhms) - Z'")
    center_freq_imaginary : str = Field("Center Frequency Impedance (kOhms) - Z''")

    # Low frequency impedance measurements
    lowest_freq_absolute  : str = Field('Low Frequency Impedance (kOhms) - Z')
    lowest_freq_real      : str = Field("Low Frequency Impedance (kOhms) - Z'")
    lowest_freq_imaginary : str = Field("Low Frequency Impedance (kOhms) - Z''")



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
    patient_identifiers      : PatientIdentifiers      = Field(default_factory=PatientIdentifiers)
    demographics             : PatientDemographics     = Field(default_factory=PatientDemographics)
    lifestyle                : LifestyleFactors        = Field(default_factory=LifestyleFactors)
    medical_history          : MedicalHistory          = Field(default_factory=MedicalHistory)
    visit_info               : VisitInformation        = Field(default_factory=VisitInformation)
    oxygenation_measurements : OxygenationMeasurements = Field(default_factory=OxygenationMeasurements)
    temperature_measurements : TemperatureMeasurements = Field(default_factory=TemperatureMeasurements)
    impedance_measurements   : ImpedanceMeasurements   = Field(default_factory=ImpedanceMeasurements)
    wound_characteristics    : WoundCharacteristics    = Field(default_factory=WoundCharacteristics)
    clinical_assessment      : ClinicalAssessment      = Field(default_factory=ClinicalAssessment)
    healing_metrics          : HealingMetrics          = Field(default_factory=HealingMetrics)

    # Internal column mapping
    column_map: Dict[str, str] = Field(default_factory=dict)
    df_columns: List[str]      = Field(default_factory=list)

    def get_all_columns(self) -> dict:
        """Get all column names as a flat dictionary"""
        all_columns = {}
        for category in self.__dict__.values():
            if isinstance(category, BaseModel):
                all_columns.update({k: v for k, v in category.__dict__.items() if not k.startswith('_')})
        return all_columns

    def update(self, df=None):
        """Update the column mapping based on the dataframe columns

        This method dynamically updates the field values in each category based on
        whether the dataframe uses 'Label' or 'Raw' column naming conventions.

        Args:
            df: pandas DataFrame with columns to map

        Returns:
            self: Returns the instance for method chaining
        """
        if df is None:
            return self

        # Store dataframe columns for lookup
        self.df_columns = df.columns.tolist()

        # Map all field names to actual column names in the dataframe
        for category in self.__dict__.values():
            if isinstance(category, BaseModel):
                for field_name, field_value in category.__dict__.items():

                    # Try to find the column in the dataframe
                    column_name = self._find_column_in_dataframe(field_value)
                    if column_name is not None:
                        # Store the mapping and update the field
                        self.column_map[field_name] = column_name
                        setattr(category, field_name, column_name)

        return self

    def _find_column_in_dataframe(self, field_value: str) -> Optional[str]:
        """
        Find a column in the dataframe based on its field value.

        This method attempts to find a column in two ways:
        1. Direct match: Checks if the field_value exists as a column name.
        2. Mapping match: Checks if field_value has a mapping in COLUMN_MAPPING and if that mapped value exists as a column.

        Args:
            field_value (str): The field value to look for in the dataframe columns.

        Returns:
            Optional[str]: The name of the found column if it exists, None otherwise.
        """

        # First check if the Label column exists in the dataframe
        if field_value in self.df_columns:
            return field_value

        # Then check if the Raw column exists in the dataframe
        raw_column = self.COLUMN_MAPPING.get(field_value)
        if raw_column and raw_column in self.df_columns:
            return raw_column

        return None

    # Mapping between Label and Raw column names based on raw_label_mapping.csv
    COLUMN_MAPPING: Dict[str, str] = {
        # Patient Identifiers
        'Record ID' : 'record_id',
        'Event Name': 'redcap_event_name',
        'MRN'       : 'mrn',

        # Demographics
        'Date of Birth'               : 'dob',
        'Sex'                         : 'sex',
        'Race'                        : 'race',
        'Race: Other - Specify'       : 'race_oth',
        'Ethnicity'                   : 'ethnicity',
        'Weight'                      : 'weight',
        'Height'                      : 'height',
        'BMI'                         : 'bmi',
        'BMI Category'                : 'bmi_category',
        'Study Cohort'                : 'cohort',
        'Calculated Age at Enrollment': 'calc_age',

        # Lifestyle
        'Smoking status'                                                                                : 'smoking_status',
        'Number of Packs per Day(average number of cigarette packs smoked per day)1 Pack= 20 Cigarettes': 'smoking_ppd',
        'Number of Years smoked/has been smoking cigarettes'                                            : 'smoking_years',
        'Alcohol Use Status'                                                                            : 'alcohol_status',
        'Number of alcohol drinks consumed/has been consuming'                                          : 'alcohol_frequency',
        'Illicit drug use?'                                                                             : 'drug_use',
        'Current type of drug use:'                                                                     : 'drug_type',
        'Former type of drug use:'                                                                      : 'drug_type_former',
        'How often has the patient used these substances in the past 3 months?'                         : 'drug_use_frequency',
        'IV drug use?'                                                                                  : 'iv_drug',

        # Medical History
        'Medical History (select all that apply)': 'pmhx_731312',
        'Diabetes?'                              : 'diabetes',
        'Respiratory'                            : 'pmhx_resp',
        'Cardiovascular'                         : 'pmhx_cardio',
        'Gastrointestinal'                       : 'pmhx_gi',
        'Musculoskeletal'                        : 'pmhx_msk',
        'Endocrine/ Metabolic'                   : 'pmhx_endo',
        'Hematopoietic'                          : 'pmhx_hem',
        'Hepatic/Renal'                          : 'pmhx_hepren',
        'Neurologic'                             : 'pmhx_neuro',
        'Immune'                                 : 'pmhx_immune',
        'Other Medical History'                  : 'pmhx_oth',
        'A1c available within the last 3 months?': 'labs',
        'Hemoglobin A1c (%)'                     : 'a1c',

        # Visit Info
        'Skipped Visit?'        : 'skipped_visit',
        'Visit date'            : 'visit_date',
        'Visit Number'          : 'visit_number',
        'Days_Since_First_Visit': 'days_since_first_visit',

        # Oxygenation
        'Oxygenation (%)'      : 'oxygenation',
        'Hemoglobin Level'     : 'hgb_level',
        'Oxyhemoglobin Level'  : 'oxhgb_level',
        'Deoxyhemoglobin Level': 'deoxhgb_level',

        # Temperature
        'Center of Wound Temperature (Fahrenheit)': 'temp',
        'Edge of Wound Temperature (Fahrenheit)'  : 'temp_edge',
        'Peri-wound Temperature (Fahrenheit)'     : 'temp_periw',
        'Center-Edge Temp Gradient'               : 'center_edge_gradient',
        'Edge-Peri Temp Gradient'                 : 'edge_peri_gradient',
        'Total Temp Gradient'                     : 'total_gradient',

        # Impedance
        'Skin Impedance (kOhms) - Z'              : 'impedance',
        "Skin Impedance (kOhms) - Z'"             : 'impedance_2',
        "Skin Impedance (kOhms) - Z''"            : 'impedance_3',
        'Center Frequency Impedance (kOhms) - Z'  : 'center_freq_impedance',
        "Center Frequency Impedance (kOhms) - Z'" : 'center_freq_impedance_2',
        "Center Frequency Impedance (kOhms) - Z''": 'center_freq_impedance_3',
        'Low Frequency Impedance (kOhms) - Z'     : 'low_freq_impedance',
        "Low Frequency Impedance (kOhms) - Z'"    : 'low_freq_impedance_2',
        "Low Frequency Impedance (kOhms) - Z''"   : 'low_freq_impedance_3',

        # Wound Characteristics
        'Target wound onset date'         : 'onset_date',
        'Length (cm)'                     : 'length',
        'Width (cm)'                      : 'width',
        'Depth (cm)'                      : 'depth',
        'Calculated Wound Area'           : 'wound_area',
        'Describe the wound location'     : 'wound_location',
        'Is there undermining/ tunneling?': 'tunneling',
        'Undermining Location Description': 'undermining_desc',
        'Tunneling Location Description'  : 'tunneling_desc',
        'Wound Type'                      : 'etiology',
        'Current wound care'              : 'wound_care',

        # Clinical Assessment
        'Clinical events'                                               : 'clincal_events',
        'Diabetic Foot Wound - WIfI Classification: foot Infection (fI)': 'wifi_fi',
        'Infection'                                                     : 'infection',
        'Infection/ Biomarker Measurement'                              : 'biomarker',
        'Granulation'                                                   : 'granulation',
        'Granulation Quality'                                           : 'gran_quality',
        'Necrosis'                                                      : 'necrosis_type',
        'Exudate Volume'                                                : 'exudate_volume',
        'Exudate Viscosity'                                             : 'exudate_viscosity',
        'Exudate Type'                                                  : 'exudate_type',

        # Healing Metrics
        'Healing Rate (%)'        : 'healing_rate',
        'Estimated_Days_To_Heal'  : 'estimated_days_to_heal',
        'Overall_Improvement'     : 'overall_improvement',
        'Average Healing Rate (%)': 'average_healing_rate'
    }

    # Reverse mapping (Raw to Label)
    REVERSE_MAPPING: Dict[str, str] = {v: k for k, v in COLUMN_MAPPING.items()}

    def get_column_name(self, field_name: str) -> Optional[str]:

        # If we have a mapped column, return it
        if field_name in self.column_map:
            return self.column_map[field_name]

        # Otherwise get it from the schema
        for category in self.__dict__.values():
            if isinstance(category, BaseModel) and hasattr(category, field_name):
                return getattr(category, field_name)

        return None


class ExcelSheetColumns(Enum):
    FREQ      = 'freq / Hz'
    ABSOLUTE  = "Z / Ohm"
    REAL      = "Z' / Ohm"
    IMAGINARY = "-Z'' / Ohm"
    NEG_PHASE = "neg. Phase / °"
    VISIT_DATE_FREQ_SWEEP = 'visit_date_freq_sweep'


class DColumns:
    def __init__(self, df: pd.DataFrame = None):
        """Creates a class-level dictionary of all column names from a dataframe"""

        # Initialize DataColumns and update with dataframe if provided
        dc = DataColumns()
        if df is not None:
            dc.update(df=df)

        # Add class annotations for type checking
        if not hasattr(self.__class__, '__annotations__'):
            self.__class__.__annotations__ = {}

        # Map all fields from each category to uppercase class attributes
        for category in vars(dc).values():
            if isinstance(category, BaseModel):
                for field_name, field_value in vars(category).items():
                    if not field_name.startswith('_'):
                        upper_name = field_name.upper()
                        setattr(self, upper_name, field_value)
                        self.__class__.__annotations__[upper_name] = str
