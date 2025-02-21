import logging
import pathlib
import argparse
from datetime import datetime
from typing import Tuple
from llm_interface import WoundAnalysisLLM

def setup_logging(log_dir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """Set up logging configuration and return log file paths."""
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f'wound_analysis_{timestamp}.log'
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
    parser.add_argument('--dataset-path', type=pathlib.Path, default=pathlib.Path(__file__).parent.parent / 'dataset', help='Path to the dataset directory containing SmartBandage-Data_for_llm.csv')
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path(__file__).parent.parent / 'wound_analysis/logs', help='Directory to save output files')
    parser.add_argument('--platform', type=str, default='ai-verde', choices=WoundAnalysisLLM.get_available_platforms(), help='LLM platform to use')
    parser.add_argument('--api-key', type=str, default='sk-h8JtQkCCJUOy-TAdDxCLGw', help='API key for the LLM platform')
    parser.add_argument('--model-name', type=str, default='llama-3.3-70b-fp8', help='Name of the LLM model to use')
    return parser.parse_args()
