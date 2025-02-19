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
                    f"- Impedance (80kHz): Z={high_freq_imp.get('z')}, Z'={high_freq_imp.get('z_prime')}, Z''={high_freq_imp.get('z_double_prime')}\n"
                )

                # Add low frequency impedance if available
                if any(v is not None for v in low_freq_imp.values()):
                    prompt += f"- Impedance (100Hz): Z={low_freq_imp.get('z')}, Z'={low_freq_imp.get('z_prime')}, Z''={low_freq_imp.get('z_double_prime')}\n"

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
