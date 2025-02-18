import os
import argparse
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze wound care data using LLMs')
    parser.add_argument('record_id', type=int, help='Patient record ID to analyze')
    parser.add_argument('--model-name', type=str, default='biogpt',
                        choices=['falcon-7b-medical', 'llama2-medical', 'med42', 'biogpt', 'clinical-bert'],
                        help='Specific model name to use')

    args = parser.parse_args()

    try:
        # Initialize data processor
        processor = WoundDataProcessor()

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
