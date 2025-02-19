import os
import argparse
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM
import logging
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze wound care data using LLMs')
    parser.add_argument('--record-id', type=int, default=3, help='Patient record ID to analyze')
    parser.add_argument('--model-name', type=str, default='ai-verde',
                        choices=['falcon-7b-medical', 'llama2-medical', 'biogpt', 'clinical-bert', 'ai-verde'],
                        help='Specific model name to use')
    parser.add_argument('--dataset-path', type=pathlib.Path,
                        help='path to three csv files',
                        default=pathlib.Path('/Users/artinmajdi/Documents/GitHubs/postdoc/TDT_copilot_2'))
    parser.add_argument('--api-key', type=str, default="sk-h8JtQkCCJUOy-TAdDxCLGw", help='API key for AI Verde model')

    args = parser.parse_args()

    try:
        # Set API key for AI Verde if provided
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key

        # Initialize data processor
        processor = WoundDataProcessor(args.dataset_path)

        # Get patient data
        logger.info(f"Processing data for patient {args.record_id}")
        patient_data = processor.get_patient_visits(args.record_id)

        # Initialize LLM interface
        llm = WoundAnalysisLLM(model_name=args.model_name)

        # Get analysis
        logger.info("Analyzing patient data with LLM")
        analysis = llm.analyze_patient_data(patient_data)

        # Print results
        print("\nWound Care Analysis Results:")
        print("-" * 50)
        print(analysis)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
