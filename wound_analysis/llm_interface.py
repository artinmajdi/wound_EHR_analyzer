from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForMaskedLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging
import torch
import os

logger = logging.getLogger(__name__)

class WoundAnalysisLLM:
    SUPPORTED_MODELS = {
        "falcon-7b-medical": "medicalai/FalconMedicalCoder-7B",
        "biogpt": "microsoft/BioGPT",
        "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
        "ai-verde": "Meta-Llama-3.1-70B-Instruct-quantized"  # AI Verde model
    }

    def __init__(self, model_name: str = "biogpt"):
        """
        Initialize the LLM interface with HuggingFace models or AI Verde.
        Args:
            model_name: Name of the model to use (must be one of SUPPORTED_MODELS)
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")

        self.model_name = model_name
        self.model_path = self.SUPPORTED_MODELS[model_name]

        try:
            if model_name == "ai-verde":
                # Initialize AI Verde model through LangChain
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY environment variable must be set for AI Verde")

                self.model = ChatOpenAI(
                    model=self.model_path,
                    base_url="https://llm-api.cyverse.ai"
                )
                logger.info(f"Successfully loaded AI Verde model {model_name}")
            else:
                # Existing HuggingFace model initialization
                device = "cuda" if torch.cuda.is_available() else "cpu"

                if model_name == "clinical-bert":
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
            if self.model_name == "ai-verde":
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

    def get_available_models(self) -> List[str]:
        """Get list of supported models."""
        return list(self.SUPPORTED_MODELS.keys())
