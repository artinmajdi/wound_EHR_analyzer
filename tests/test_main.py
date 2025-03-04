"""
Tests for the main module.
"""
import unittest
from unittest.mock import patch, MagicMock
import argparse
import pathlib
import sys
import os

# Update the import paths to use the direct import for main instead of through the package
@patch('wound_analysis.utils.data_processor.WoundDataProcessor')
@patch('wound_analysis.utils.data_processor.DataManager')
@patch('wound_analysis.utils.llm_interface.WoundAnalysisLLM')
class TestMainModule(unittest.TestCase):
    """Tests for the main module."""
    
    def test_parse_arguments(self, *mocks):
        """Test parsing of command line arguments."""
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=argparse.Namespace(
                      record_id=42,
                      csv_dataset_path=pathlib.Path('test/path.csv'),
                      impedance_freq_sweep_path=pathlib.Path('test/impedance'),
                      output_dir=pathlib.Path('test/output'),
                      platform='ai-verde',
                      api_key='test-key',
                      model_name='test-model'
                  )):
            # Import here to avoid import errors with mocked dependencies
            try:
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                from wound_analysis.main import parse_arguments
                args = parse_arguments()
                
                # Verify arguments were parsed correctly
                self.assertEqual(args.record_id, 42)
                self.assertEqual(args.csv_dataset_path, pathlib.Path('test/path.csv'))
                self.assertEqual(args.platform, 'ai-verde')
                self.assertEqual(args.api_key, 'test-key')
                self.assertEqual(args.model_name, 'test-model')
            finally:
                # Clean up the path modification
                if sys.path[0] == os.path.abspath(os.path.join(os.path.dirname(__file__), '..')):
                    sys.path.pop(0)
    
    def test_setup_logging(self, *mocks):
        """Test setting up of logging."""
        with patch('logging.basicConfig') as mock_logging:
            # Import here to avoid import errors with mocked dependencies
            try:
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                from wound_analysis.main import setup_logging
                
                # Create a temp dir for logs with parents=True to ensure it exists
                temp_log_dir = pathlib.Path('test_temp/log')
                temp_log_dir.parent.mkdir(exist_ok=True)
                if not temp_log_dir.exists():
                    temp_log_dir.mkdir(exist_ok=True)
                
                setup_logging(temp_log_dir)
                
                # Verify that logging was set up correctly
                mock_logging.assert_called_once()
                
                # Clean up - remove the test directory
                import shutil
                if temp_log_dir.exists():
                    shutil.rmtree(temp_log_dir.parent)
            finally:
                # Clean up the path modification
                if sys.path[0] == os.path.abspath(os.path.join(os.path.dirname(__file__), '..')):
                    sys.path.pop(0)


if __name__ == '__main__':
    unittest.main()
