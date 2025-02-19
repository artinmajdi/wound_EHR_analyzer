from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForMaskedLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
import logging
import torch
import os
import pathlib
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

logger = logging.getLogger(__name__)

class WoundAnalysisLLM:
    MODEL_CATEGORIES = {
        "huggingface": {
            "medalpaca-7b": "medalpaca/medalpaca-7b",
            "BioMistral-7B": "BioMistral/BioMistral-7B",
            "ClinicalCamel-70B": "wanglab/ClinicalCamel-70B",
        },
        "ai-verde": {
            # "llama-3-90b-vision": "llama-3-2-90b-vision-instruct-quantized",
            "llama-3.3-70b-fp8": "js2/Llama-3.3-70B-Instruct-FP8-Dynamic",
            "meta-llama-3.1-70b": "Meta-Llama-3.1-70B-Instruct-quantized",  # default
            "deepseek-r1": "js2/DeepSeek-R1",
            "llama-3.2-11b-vision": "Llama-3.2-11B-Vision-Instruct"
        }
    }

    def __init__(self, platform: str = "ai-verde", model_name: str = "llama-3.3-70b-fp8"):
        """
        Initialize the LLM interface with HuggingFace models or AI Verde.
        Args:
            platform: The platform to use ("huggingface" or "ai-verde")
            model_name: Name of the model to use within the selected platform
        """
        if platform not in self.MODEL_CATEGORIES:
            raise ValueError(f"Platform {platform} not supported. Choose from: {list(self.MODEL_CATEGORIES.keys())}")

        if model_name not in self.MODEL_CATEGORIES[platform]:
            raise ValueError(f"Model {model_name} not supported for platform {platform}. Choose from: {list(self.MODEL_CATEGORIES[platform].keys())}")

        self.platform = platform
        self.model_name = model_name
        self.model_path = self.MODEL_CATEGORIES[platform][model_name]
        self.model = None  # Initialize as None, will be loaded on first use

    def _load_model(self):
        """Lazy loading of the model to avoid Streamlit file watcher issues"""
        if self.model is not None:
            return

        try:
            if self.platform == "ai-verde":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY environment variable must be set for AI Verde")

                self.model = ChatOpenAI(
                    model=self.model_path,
                    base_url="https://llm-api.cyverse.ai"
                )
                logger.info(f"Successfully loaded AI Verde model {self.model_name}")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

                if self.model_name == "clinical-bert":
                    self.model = pipeline(
                        "fill-mask",
                        model=self.model_path,
                        device=device
                    )
                else:
                    self.model = pipeline(
                        "text-generation",
                        model=self.model_path,
                        device=device,
                        max_new_tokens=512
                    )
                logger.info(f"Successfully loaded model {self.model_name} on {device}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise

    @classmethod
    def get_available_platforms(cls) -> List[str]:
        """Get list of supported platforms."""
        return list(cls.MODEL_CATEGORIES.keys())

    @classmethod
    def get_available_models(cls, platform: str) -> List[str]:
        """Get list of supported models for a specific platform."""
        if platform not in cls.MODEL_CATEGORIES:
            raise ValueError(f"Platform {platform} not supported")
        return list(cls.MODEL_CATEGORIES[platform].keys())

    def _format_prompt(self, patient_data: Dict) -> str:
        """Format patient data into a prompt for the LLM."""
        metadata = patient_data['patient_metadata']
        visits = patient_data['visits']

        # Create a more detailed patient profile
        prompt = (
            "Task: Analyze wound healing progression and provide clinical recommendations.\n\n"
            "Patient Profile:\n"
            f"- {metadata.get('age', 'Unknown')}y/o {metadata.get('sex', 'Unknown')}\n"
            f"- Race/Ethnicity: {metadata.get('race', 'Unknown')}, {metadata.get('ethnicity', 'Unknown')}\n"
            f"- BMI: {metadata.get('bmi', 'Unknown')}\n"
            f"- Study Cohort: {metadata.get('study_cohort', 'Unknown')}\n"
            f"- Smoking: {metadata.get('smoking_status', 'None')}"
        )

        # Add diabetes information
        diabetes_info = metadata.get('diabetes', {})
        if diabetes_info:
            prompt += (
                f"\n- Diabetes: {diabetes_info.get('status', 'None')}\n"
                f"- HbA1c: {diabetes_info.get('hemoglobin_a1c', 'Unknown')}%"
            )

        # Add medical history if available
        medical_history = metadata.get('medical_history', {})
        if medical_history:
            conditions = [cond for cond, val in medical_history.items() if val and val != 'None']
            if conditions:
                prompt += f"\n- Medical History: {', '.join(conditions)}"

        prompt += "\n\nVisit History:\n"

        # Format visit information
        for visit in visits:
            visit_date = visit.get('visit_date', 'Unknown')
            measurements = visit.get('wound_measurements', {})
            wound_info = visit.get('wound_info', {})
            sensor = visit.get('sensor_data', {})

            prompt += (
                f"\nDate: {visit_date}\n"
                f"Location: {wound_info.get('location', 'Unknown')}\n"
                f"Type: {wound_info.get('type', 'Unknown')}\n"
                f"Size: {measurements.get('length')}cm x {measurements.get('width')}cm x {measurements.get('depth')}cm\n"
                f"Area: {measurements.get('area')}cm²\n"
            )

            # Add granulation information
            granulation = wound_info.get('granulation', {})
            if granulation:
                prompt += f"Tissue: {granulation.get('quality', 'Unknown')}, coverage {granulation.get('coverage', 'Unknown')}\n"

            # Add exudate information
            exudate = wound_info.get('exudate', {})
            if exudate:
                prompt += (
                    f"Exudate: {exudate.get('volume', 'Unknown')} volume, "
                    f"viscosity {exudate.get('viscosity', 'Unknown')}, "
                    f"type {exudate.get('type', 'Unknown')}\n"
                )

            # Add sensor data
            if sensor:
                temp = sensor.get('temperature', {})
                impedance = sensor.get('impedance', {})
                high_freq_imp = impedance.get('high_frequency', {})
                low_freq_imp = impedance.get('low_frequency', {})

                prompt += (
                    f"Measurements:\n"
                    f"- O₂: {sensor.get('oxygenation')}%\n"
                    f"- Temperature: center {temp.get('center')}°F, edge {temp.get('edge')}°F, peri-wound {temp.get('peri')}°F\n"
                    f"- Hemoglobin: {sensor.get('hemoglobin')}, Oxy: {sensor.get('oxyhemoglobin')}, Deoxy: {sensor.get('deoxyhemoglobin')}\n"
                    f"- Impedance (80kHz): |Z|={high_freq_imp.get('Z')}, resistance={high_freq_imp.get('resistance')}, capacitance={high_freq_imp.get('capacitance')}\n"
                )

                # Add low frequency impedance if available
                if any(v is not None for v in low_freq_imp.values()):
                    prompt += f"- Impedance (100Hz): |Z|={low_freq_imp.get('Z')}, resistance={low_freq_imp.get('resistance')}, capacitance={low_freq_imp.get('capacitance')}\n"

            # Add infection and treatment information
            infection = wound_info.get('infection', {})
            if infection.get('status') or infection.get('classification'):
                prompt += f"Infection Status: {infection.get('status', 'None')}, Classification: {infection.get('classification', 'None')}\n"

            if wound_info.get('current_care'):
                prompt += f"Current Care: {wound_info.get('current_care')}\n"

        prompt += (
            "\nPlease provide a comprehensive analysis including:\n"
            "1. Wound healing trajectory (analyze changes in size, exudate, and tissue characteristics)\n"
            "2. Concerning patterns (identify any worrying trends in measurements or characteristics)\n"
            "3. Care recommendations (based on wound type, characteristics, and healing progress)\n"
            "4. Complication risks (assess based on patient profile and wound characteristics)\n"
            "5. Significance of sensor measurements (interpret oxygenation, temperature, and impedance trends)\n"
        )

        return prompt

    def analyze_patient_data(self, patient_data: Dict) -> str:
        """
        Analyze patient data using the configured model.
        Args:
            patient_data: Processed patient data dictionary
        Returns:
            Analysis results as string
        """
        # Load model if not already loaded
        self._load_model()
        prompt = self._format_prompt(patient_data)

        try:
            if self.platform == "ai-verde":
                # Use AI Verde with system and human messages
                messages = [
                    SystemMessage(content="You are a medical expert specializing in wound care analysis. Analyze the provided wound data and provide clinical recommendations."),
                    HumanMessage(content=prompt)
                ]
                response = self.model.invoke(messages)
                return response.content

            elif self.model_name == "clinical-bert":
                # For BERT, we'll use a simpler approach focused on key aspects
                latest_visit = patient_data['visits'][-1]
                measurements = latest_visit.get('wound_measurements', {})
                wound_info = latest_visit.get('wound_info', {})

                # Create shorter, focused prompts
                healing_prompt = f"The wound healing progress is [MASK]. Size is {measurements.get('length')}x{measurements.get('width')}cm."
                risk_prompt = f"The wound infection risk is [MASK]. Exudate is {wound_info.get('exudate', {}).get('volume', 'Unknown')}."

                # Get predictions for each aspect
                healing_result = self.model(healing_prompt)[0]['token_str']
                risk_result = self.model(risk_prompt)[0]['token_str']

                # Combine results
                analysis = f"""
                Wound Analysis Summary:
                - Healing Progress: {healing_result}
                - Risk Assessment: {risk_result}
                - Latest Measurements: {measurements.get('length')}cm x {measurements.get('width')}cm
                - Area: {measurements.get('area')}cm²
                """
                return analysis
            else:
                # Standard text generation for other models
                response = self.model(
                    prompt,
                    do_sample=True,
                    temperature=0.3,
                    max_new_tokens=512,
                    num_return_sequences=1
                )

                # Extract and clean the generated text
                generated_text = response[0]['generated_text']
                # Remove the prompt from the response if it's included
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                return generated_text

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

def format_word_document(doc: Document, analysis_text: str, patient_data: dict, report_path: str = None) -> str:
    """
    Format the analysis results in a professional Word document layout.
    Returns the path to the saved document.
    """
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

    # Save the document
    if report_path is None:
        # Create logs directory if it doesn't exist
        log_dir = pathlib.Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = log_dir / f'wound_analysis_{timestamp}.docx'

    doc.save(report_path)
    return str(report_path)
