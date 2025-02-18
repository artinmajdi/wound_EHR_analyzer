from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForMaskedLM
import logging
import torch

logger = logging.getLogger(__name__)

class WoundAnalysisLLM:
    SUPPORTED_MODELS = {
        "falcon-7b-medical": "medicalai/FalconMedicalCoder-7B",
        "llama2-medical": "TheBloke/Llama-2-7B-Medical",
        "med42": "medicalai/med42-70b",
        "biogpt": "microsoft/BioGPT",
        "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT"
    }

    def __init__(self, model_name: str = "biogpt"):
        """
        Initialize the LLM interface with HuggingFace models.
        Args:
            model_name: Name of the model to use (must be one of SUPPORTED_MODELS)
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")

        self.model_name = model_name
        self.model_path = self.SUPPORTED_MODELS[model_name]

        try:
            # Different configuration based on model type
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if model_name == "clinical-bert":
                # Special handling for BERT models
                self.model = pipeline(
                    "fill-mask",
                    model=self.model_path,
                    device=device
                )
            elif model_name == "biogpt":
                self.model = pipeline(
                    "text-generation",
                    model=self.model_path,
                    device=device,
                    max_new_tokens=512
                )
            else:
                self.model = pipeline(
                    "text-generation",
                    model=self.model_path,
                    device=device,
                    max_new_tokens=512
                )
            logger.info(f"Successfully loaded model {model_name} on {device}")

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def _format_prompt(self, patient_data: Dict) -> str:
        """Format patient data into a prompt for the LLM."""
        metadata = patient_data['patient_metadata']
        visits = patient_data['visits']

        # Create a more structured, concise prompt
        prompt = (
            "Task: Analyze wound healing progression and provide clinical recommendations.\n\n"
            "Patient Profile:\n"
            f"- {metadata.get('age', 'Unknown')}y/o {metadata.get('sex', 'Unknown')}\n"
            f"- BMI: {metadata.get('bmi', 'Unknown')}\n"
            f"- Diabetes: {metadata.get('diabetes_type', 'None')}\n"
            f"- HbA1c: {metadata.get('hemoglobin_a1c', 'Unknown')}\n\n"
            "Visit History:\n"
        )

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
                f"Tissue: {wound_info.get('granulation', {}).get('quality', 'Unknown')} granulation, "
                f"coverage {wound_info.get('granulation', {}).get('coverage', 'Unknown')}\n"
                f"Exudate: {wound_info.get('exudate', {}).get('volume', 'Unknown')} volume, "
                f"type {wound_info.get('exudate', {}).get('type', 'Unknown')}\n"
                f"O₂: {sensor.get('oxygenation')}%, Temp: {sensor.get('temperature', {}).get('center')}°F\n"
            )

        prompt += (
            "\nPlease analyze:\n"
            "1. Wound healing trajectory\n"
            "2. Concerning patterns\n"
            "3. Care recommendations\n"
            "4. Complication risks\n"
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
        prompt = self._format_prompt(patient_data)

        try:
            if self.model_name == "clinical-bert":
                # Special handling for BERT masked language modeling
                # Use a simpler prompt for BERT
                masked_text = f"The wound is [MASK]. {prompt}"
                results = self.model(masked_text)
                return results[0]['sequence']
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

    def get_available_models(self) -> List[str]:
        """Get list of supported models."""
        return list(self.SUPPORTED_MODELS.keys())
