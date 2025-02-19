import os
import argparse
from data_processor import WoundDataProcessor
from llm_interface import WoundAnalysisLLM
import logging
import pathlib
from datetime import datetime
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

print('------')

# Create logs directory if it doesn't exist
log_dir = pathlib.Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

# Configure logging to both file and console
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = log_dir / f'wound_analysis_{timestamp}.log'
word_filename = log_dir / f'wound_analysis_{timestamp}.docx'

# Create and configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def format_word_document(doc: Document, analysis_text: str, patient_data: dict) -> None:
    """Format the analysis results in a professional Word document layout."""
    # Add title
    title = doc.add_heading('Wound Care Analysis Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add patient information section
    doc.add_heading('Patient Information', level=1)
    metadata = patient_data['patient_metadata']
    patient_info = doc.add_paragraph()
    patient_info.add_run('Patient Demographics:\n').bold = True
    patient_info.add_run(f"Age: {metadata.get('age', 'Unknown')} years\n")
    patient_info.add_run(f"Sex: {metadata.get('sex', 'Unknown')}\n")
    patient_info.add_run(f"BMI: {metadata.get('bmi', 'Unknown')}\n")

    # Add diabetes information
    diabetes_info = doc.add_paragraph()
    diabetes_info.add_run('Diabetes Status:\n').bold = True
    if 'diabetes' in metadata:
        diabetes_info.add_run(f"Type: {metadata['diabetes'].get('status', 'Unknown')}\n")
        diabetes_info.add_run(f"HbA1c: {metadata['diabetes'].get('hemoglobin_a1c', 'Unknown')}%\n")

    # Add analysis section
    doc.add_heading('Analysis Results', level=1)

    # Split analysis into sections and format them
    sections = analysis_text.split('\n\n')
    for section in sections:
        if section.strip():
            if '**' in section:  # Handle markdown-style headers
                # Convert markdown headers to proper formatting
                section = section.replace('**', '')
                p = doc.add_paragraph()
                p.add_run(section.strip()).bold = True
            else:
                # Handle bullet points
                if section.strip().startswith('- ') or section.strip().startswith('* '):
                    p = doc.add_paragraph(section.strip()[2:], style='List Bullet')
                else:
                    p = doc.add_paragraph(section.strip())

    # Add footer with timestamp
    doc.add_paragraph(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    parser = argparse.ArgumentParser(description='Analyze wound care data using LLMs')
    parser.add_argument('--record-id', type=int, default=7, help='Patient record ID to analyze')
    parser.add_argument('--dataset-path', type=pathlib.Path, default='/Users/artinmajdi/Documents/GitHubs/postdoc/TDT_copilot_2/dataset', help='Path to the dataset directory containing SmartBandage-Data_for_llm.csv')
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path(__file__).parent / 'logs', help='Directory to save output files')
    parser.add_argument('--platform', type=str, default='ai-verde', choices=WoundAnalysisLLM.get_available_platforms(), help='LLM platform to use')
    parser.add_argument('--api-key', type=str, default='sk-h8JtQkCCJUOy-TAdDxCLGw', help='API key for the LLM platform')
    parser.add_argument('--model-name', type=str, default='llama-3.3-70b-fp8', help='Name of the LLM model to use')

    args = parser.parse_args()


    try:
        logger.info(f"Starting analysis for patient {args.record_id}")

        # Environment setup
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key

        # Data processing
        processor = WoundDataProcessor(args.dataset_path)
        patient_data = processor.get_patient_visits(args.record_id)

        # Analysis
        llm = WoundAnalysisLLM(
            platform=args.platform,
            model_name=args.model_name
        )
        analysis_results = llm.analyze_patient_data(patient_data)

        # Report generation
        doc = Document()
        format_word_document(doc, analysis_results, patient_data)
        doc.save(word_filename)

        # Output
        logger.info(f"Report saved: {word_filename}")
        print("\nWound Care Analysis Results:")
        print("="* 30)
        print(analysis_results)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
