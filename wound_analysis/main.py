import argparse
import logging
import os
import pathlib
from datetime import datetime
from typing import Tuple

from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.llm_interface import WoundAnalysisLLM
from wound_analysis.dashboard import Dashboard

def setup_logging(log_dir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """Set up logging configuration and return log file paths."""
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename  = log_dir / f'wound_analysis_{timestamp}.log'
    word_filename = log_dir / f'wound_analysis_{timestamp}.docx'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    return log_filename, word_filename

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze wound care data using LLMs')
    parser.add_argument('--record-id', type=int, default=41, help='Patient record ID to analyze')
    parser.add_argument('--csv-dataset-path', type=pathlib.Path, default='/Users/artinmajdi/Documents/GitHubs/postdoc/wound_EHR_analyzer_private/dataset/csv_files/SmartBandage-2025-03-26_labels.csv', help='Path to the CSV dataset file')
    parser.add_argument('--impedance-freq-sweep-path', type=pathlib.Path, default='/Users/artinmajdi/Documents/GitHubs/postdoc/wound_EHR_analyzer_private/dataset/impedance_frequency_sweep', help='Path to the impedance frequency sweep directory')
    parser.add_argument('--output-dir', type=pathlib.Path, default='/Users/artinmajdi/Documents/GitHubs/postdoc/wound_EHR_analyzer_private/wound_analysis/utils/logs', help='Directory to save output files')
    parser.add_argument('--platform', type=str, default='ai-verde', choices=WoundAnalysisLLM.get_available_platforms(), help='LLM platform to use')
    parser.add_argument('--api-key', type=str, help='API key for the LLM platform')
    parser.add_argument('--model-name', type=str, default='llama-3.3-70b-fp8', help='Name of the LLM model to use')
    return parser.parse_args()



logger = logging.getLogger(__name__)

def main():
    args = parse_arguments()
    _, word_filename = setup_logging(args.output_dir)

    try:
        logger.debug(f"Starting Wound Analysis Dashboard")

        # Set up environment
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key

        # Initialize the dashboard
        dashboard = Dashboard()

        # Set the necessary paths and configurations
        dashboard.csv_dataset_path = args.csv_dataset_path
        dashboard.impedance_freq_sweep_path = args.impedance_freq_sweep_path
        dashboard.llm_platform = args.platform
        dashboard.llm_model = args.model_name

        # Run the dashboard application
        dashboard.run()

    except Exception as e:
        logger.error(f"Dashboard initialization failed: {e}")
        raise


if __name__ == "__main__":
    main()
