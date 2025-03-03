import argparse
import logging
import os
import pathlib
from datetime import datetime
from typing import Tuple

from utils.data_processor import WoundDataProcessor, DataManager
from utils.llm_interface import WoundAnalysisLLM

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
    parser.add_argument('--csv-dataset-path', type=pathlib.Path, default=pathlib.Path(__file__).parent.parent / 'dataset' / 'SmartBandage-Data_for_llm.csv', help='Path to the CSV dataset file')
    parser.add_argument('--impedance-freq-sweep-path', type=pathlib.Path, default=pathlib.Path(__file__).parent.parent / 'dataset' / 'impedance_frequency_sweep', help='Path to the impedance frequency sweep directory')
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path(__file__).parent.parent / 'wound_analysis/utils/logs', help='Directory to save output files')
    parser.add_argument('--platform', type=str, default='ai-verde', choices=WoundAnalysisLLM.get_available_platforms(), help='LLM platform to use')
    parser.add_argument('--api-key', type=str, help='API key for the LLM platform')
    parser.add_argument('--model-name', type=str, default='llama-3.3-70b-fp8', help='Name of the LLM model to use')
    return parser.parse_args()



logger = logging.getLogger(__name__)

def main():
    args = parse_arguments()
    _, word_filename = setup_logging(args.output_dir)

    try:
        logger.info(f"Starting analysis for patient {args.record_id}")

        # Set up environment
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key

        # Process patient data
        processor = WoundDataProcessor(csv_dataset_path=args.csv_dataset_path, impedance_freq_sweep_path=args.impedance_freq_sweep_path)
        patient_data = processor.get_patient_visits(record_id=args.record_id)

        # Analyze data using LLM
        llm = WoundAnalysisLLM(platform=args.platform, model_name=args.model_name)
        analysis_results = llm.analyze_patient_data(patient_data=patient_data)

        # Save report
        DataManager.create_and_save_report(patient_metadata=patient_data, analysis_results=analysis_results, report_path=word_filename)

        # Output results
        logger.info(f"Report saved: {word_filename}")
        print("\nWound Care Analysis Results:")
        print("=" * 30)
        print(analysis_results)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
