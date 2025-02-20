import os
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM
from utils import setup_logging, parse_arguments
import logging

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
        processor = WoundDataProcessor(args.dataset_path)
        patient_data = processor.get_patient_visits(args.record_id)

        # Analyze data using LLM
        llm = WoundAnalysisLLM(platform=args.platform, model_name=args.model_name)
        analysis_results = llm.analyze_patient_data(patient_data)

        # Save report
        processor.save_report(word_filename, analysis_results, patient_data)

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

# TODO: select the 3rd point with impedance real == imaginary
