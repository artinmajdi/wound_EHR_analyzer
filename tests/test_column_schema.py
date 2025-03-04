"""
Tests for the column_schema module.
"""
import unittest
from wound_analysis.utils.column_schema import (
    DataColumns, 
    PatientIdentifiers, 
    PatientDemographics,
    LifestyleFactors,
    MedicalHistory,
    VisitInformation,
    OxygenationMeasurements,
    TemperatureMeasurements,
    ImpedanceMeasurements,
    WoundCharacteristics,
    ClinicalAssessment,
    HealingMetrics
)

class TestDataColumns(unittest.TestCase):
    """Tests for the DataColumns class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.columns = DataColumns()
    
    def test_init(self):
        """Test initialization of DataColumns."""
        self.assertIsInstance(self.columns, DataColumns)
        self.assertIsInstance(self.columns.patient_identifiers, PatientIdentifiers)
        self.assertIsInstance(self.columns.demographics, PatientDemographics)
        self.assertIsInstance(self.columns.lifestyle, LifestyleFactors)
        self.assertIsInstance(self.columns.medical_history, MedicalHistory)
        self.assertIsInstance(self.columns.visit_info, VisitInformation)
        self.assertIsInstance(self.columns.oxygenation, OxygenationMeasurements)
        self.assertIsInstance(self.columns.temperature, TemperatureMeasurements)
        self.assertIsInstance(self.columns.impedance, ImpedanceMeasurements)
        self.assertIsInstance(self.columns.wound_characteristics, WoundCharacteristics)
        self.assertIsInstance(self.columns.clinical_assessment, ClinicalAssessment)
        self.assertIsInstance(self.columns.healing_metrics, HealingMetrics)
    
    def test_get_all_columns(self):
        """Test getting all columns as a flat dictionary."""
        all_columns = self.columns.get_all_columns()
        
        # Check that it's a dictionary
        self.assertIsInstance(all_columns, dict)
        
        # Check that it contains expected columns
        self.assertIn('record_id', all_columns)
        self.assertIn('wound_area', all_columns)
        self.assertIn('exudate_type', all_columns)
        
        # Check that values are the expected column names
        self.assertEqual(all_columns['record_id'], 'Record ID')
        self.assertEqual(all_columns['wound_area'], 'Calculated Wound Area')
        self.assertEqual(all_columns['exudate_type'], 'Exudate Type')
    
    def test_get_column_name(self):
        """Test getting a column name from field name."""
        # Test with valid field names
        self.assertEqual(self.columns.get_column_name('record_id'), 'Record ID')
        self.assertEqual(self.columns.get_column_name('wound_area'), 'Calculated Wound Area')
        self.assertEqual(self.columns.get_column_name('exudate_type'), 'Exudate Type')
        
        # Test with invalid field name
        self.assertIsNone(self.columns.get_column_name('non_existent_field'))

class TestPatientIdentifiers(unittest.TestCase):
    """Tests for the PatientIdentifiers class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.identifiers = PatientIdentifiers()
    
    def test_field_values(self):
        """Test field values of PatientIdentifiers."""
        self.assertEqual(self.identifiers.record_id, 'Record ID')
        self.assertEqual(self.identifiers.event_name, 'Event Name')
        self.assertEqual(self.identifiers.mrn, 'MRN')

class TestWoundCharacteristics(unittest.TestCase):
    """Tests for the WoundCharacteristics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.characteristics = WoundCharacteristics()
    
    def test_field_values(self):
        """Test field values of WoundCharacteristics."""
        self.assertEqual(self.characteristics.wound_area, 'Calculated Wound Area')
        self.assertEqual(self.characteristics.length, 'Length (cm)')
        self.assertEqual(self.characteristics.width, 'Width (cm)')
        self.assertEqual(self.characteristics.depth, 'Depth (cm)')
        self.assertEqual(self.characteristics.wound_type, 'Wound Type')

class TestTemperatureMeasurements(unittest.TestCase):
    """Tests for the TemperatureMeasurements class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.measurements = TemperatureMeasurements()
    
    def test_field_values(self):
        """Test field values of TemperatureMeasurements."""
        self.assertEqual(self.measurements.center_temp, 'Center of Wound Temperature (Fahrenheit)')
        self.assertEqual(self.measurements.edge_temp, 'Edge of Wound Temperature (Fahrenheit)')
        self.assertEqual(self.measurements.peri_temp, 'Peri-wound Temperature (Fahrenheit)')
        self.assertEqual(self.measurements.center_edge_gradient, 'Center-Edge Temp Gradient')
        self.assertEqual(self.measurements.edge_peri_gradient, 'Edge-Peri Temp Gradient')
        self.assertEqual(self.measurements.total_temp_gradient, 'Total Temp Gradient')

if __name__ == '__main__':
    unittest.main()
