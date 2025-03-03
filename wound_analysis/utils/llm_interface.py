import logging
import os
from typing import Dict, List

import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

def dict_to_bullets(d: Dict) -> str:
    """Convert a dictionary to a bullet-point string format."""
    return "\n".join([f"- {k}: {v}" for k, v in d.items()])

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
        self.thinking_process = None  # Store thinking process for models that support it

    def _load_model(self):
        """Lazy loading of the model to avoid Streamlit file watcher issues"""
        if self.model is not None:
            return

        try:

            if self.platform == "ai-verde":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY environment variable must be set for AI Verde")

                model_settings = {
                    'model': self.model_path,
                    'base_url': os.getenv("OPENAI_BASE_URL"),
                }

                if self.model_name == "deepseek-r1":
                    model_settings['model_kwargs'] = {
                        'response_format': {
                            'type': 'json_object'
                        }}
                    model_settings['request_timeout'] = 180.0  # Increase timeout to 3 minutes
                    model_settings['streaming'] = True  # Enable streaming for DeepSeek R1


                self.model = ChatOpenAI(**model_settings)
                logger.info(f"Successfully loaded AI Verde model {self.model_name}")

            elif self.platform == "huggingface":

                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
            else:
                raise ValueError(f"Platform {self.platform} not supported. Choose from: {list(self.MODEL_CATEGORIES.keys())}")

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

    def _format_per_patient_prompt(self, patient_data: Dict) -> str:
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
                center_freq_imp = impedance.get('center_frequency', {})
                low_freq_imp = impedance.get('low_frequency', {})

                prompt += (
                    f"Measurements:\n"
                    f"- O₂: {sensor.get('oxygenation')}%\n"
                    f"- Temperature: center {temp.get('center')}°F, edge {temp.get('edge')}°F, peri-wound {temp.get('peri')}°F\n"
                    f"- Hemoglobin: {sensor.get('hemoglobin')}, Oxy: {sensor.get('oxyhemoglobin')}, Deoxy: {sensor.get('deoxyhemoglobin')}\n"
                )

                # Add all three frequency impedance measurements
                if high_freq_imp and any(v is not None for v in high_freq_imp.values()):
                    prompt += f"- High Frequency Impedance ({high_freq_imp.get('frequency', '80kHz')}): |Z|={high_freq_imp.get('Z')}, resistance={high_freq_imp.get('resistance')}, capacitance={high_freq_imp.get('capacitance')}\n"

                if center_freq_imp and any(v is not None for v in center_freq_imp.values()):
                    prompt += f"- Center Frequency Impedance ({center_freq_imp.get('frequency', '40kHz')}): |Z|={center_freq_imp.get('Z')}, resistance={center_freq_imp.get('resistance')}, capacitance={center_freq_imp.get('capacitance')}\n"

                if low_freq_imp and any(v is not None for v in low_freq_imp.values()):
                    prompt += f"- Low Frequency Impedance ({low_freq_imp.get('frequency', '100Hz')}): |Z|={low_freq_imp.get('Z')}, resistance={low_freq_imp.get('resistance')}, capacitance={low_freq_imp.get('capacitance')}\n"

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

    def _format_population_prompt(self, population_data: Dict) -> str:
        """Format population-level data into a clinical analysis prompt for the LLM."""

        # Helper function to safely format float values
        def safe_float_format(value, format_spec='.1f'):
            return f"{value:{format_spec}}" if value is not None else "N/A"

        prompt = (
            "## Clinical Population Analysis Task\n\n"
            "**Context**: Analyze this wound care dataset from a smart bandage clinical trial. "
            "Provide insights to improve wound care protocols and patient outcomes.\n\n"
            "**Dataset Overview**:\n"
            f"- Total Patients: {population_data['summary']['total_patients']}\n"
            f"- Total Visits: {population_data['summary']['total_visits']}\n"
            f"- Average Visits per Patient: {safe_float_format(population_data['summary']['avg_visits_per_patient'])}\n"
            f"- Overall Improvement Rate: {safe_float_format(population_data['summary']['overall_improvement_rate'])}%\n"
            f"- Average Treatment Duration: {safe_float_format(population_data['summary']['avg_treatment_duration_days'])} days\n"
        )

        # Add completion rate only if available
        if population_data['summary'].get('completion_rate') is not None:
            prompt += f"- Completion Rate: {safe_float_format(population_data['summary']['completion_rate'])}%\n"

        prompt += (
            "\n**Demographics Profile**:\n"
            f"- Age Statistics: {population_data['demographics']['age_stats']['summary']}\n"
            "Age Distribution:\n"
            f"{dict_to_bullets(population_data['demographics']['age_stats']['age_groups'])}\n"
            "Gender Distribution:\n"
            f"{dict_to_bullets(population_data['demographics']['gender_distribution'])}\n"
            "Race Distribution:\n"
            f"{dict_to_bullets(population_data['demographics']['race_distribution'])}\n"
            "Ethnicity Distribution:\n"
            f"{dict_to_bullets(population_data['demographics']['ethnicity_distribution'])}\n"
            "BMI Profile:\n"
            f"- Summary: {population_data['demographics']['bmi_stats']['summary']}\n"
            "BMI Categories:\n"
            f"{dict_to_bullets(population_data['demographics']['bmi_stats']['distribution'])}\n\n"

            "**Risk Factor Analysis**:\n"
            "Diabetes Status:\n"
            f"{dict_to_bullets(population_data['risk_factors']['primary_conditions']['diabetes']['distribution'])}\n"
            "Diabetes Impact on Healing:\n"
            f"{dict_to_bullets(population_data['risk_factors']['primary_conditions']['diabetes']['healing_impact'])}\n"
            "Smoking Status:\n"
            f"{dict_to_bullets(population_data['risk_factors']['primary_conditions']['smoking']['distribution'])}\n"
            "Smoking Impact on Healing:\n"
            f"{dict_to_bullets(population_data['risk_factors']['primary_conditions']['smoking']['healing_impact'])}\n\n"
            "Comorbidity Analysis:\n"
            "- Diabetes & Smoking Interaction:\n"
            f"{dict_to_bullets(population_data['risk_factors']['comorbidity_analysis']['diabetes_smoking'])}\n"
            "- Diabetes & BMI Interaction:\n"
            f"{dict_to_bullets(population_data['risk_factors']['comorbidity_analysis']['diabetes_bmi'])}\n\n"

            "**Wound Characteristics**:\n"
            "Type Distribution:\n"
            f"{dict_to_bullets(population_data['wound_characteristics']['type_distribution']['overall'])}\n"
            "Healing Status by Wound Type:\n"
            f"{dict_to_bullets(population_data['wound_characteristics']['type_distribution']['by_healing_status'])}\n"
            "Location Analysis:\n"
            f"{dict_to_bullets(population_data['wound_characteristics']['location_analysis']['distribution'])}\n"
            "Healing by Location:\n"
            f"{dict_to_bullets(population_data['wound_characteristics']['location_analysis']['healing_by_location'])}\n\n"
            "Size Progression:\n"
            "Initial Area Statistics:\n"
            f"{dict_to_bullets(population_data['wound_characteristics']['size_progression']['initial_vs_final']['area']['initial'])}\n"
            "Final Area Statistics:\n"
            f"{dict_to_bullets(population_data['wound_characteristics']['size_progression']['initial_vs_final']['area']['final'])}\n"
            f"Percent Change in Area: {safe_float_format(population_data['wound_characteristics']['size_progression']['initial_vs_final']['area']['percent_change'])}%\n"
            "Healing by Initial Size:\n"
            f"- Small Wounds: {safe_float_format(population_data['wound_characteristics']['size_progression']['healing_by_initial_size']['small'])}%\n"
            f"- Medium Wounds: {safe_float_format(population_data['wound_characteristics']['size_progression']['healing_by_initial_size']['medium'])}%\n"
            f"- Large Wounds: {safe_float_format(population_data['wound_characteristics']['size_progression']['healing_by_initial_size']['large'])}%\n\n"

            "**Healing Progression**:\n"
            f"Overall: {population_data['healing_progression']['overall_stats']['summary']}\n"
            "Status Distribution:\n"
            f"{dict_to_bullets(population_data['healing_progression']['overall_stats']['distribution'])}\n"
            "Healing Rate Percentiles:\n"
            f"{dict_to_bullets(population_data['healing_progression']['overall_stats']['percentiles'])}\n"
            "Temporal Analysis:\n"
            "By Visit Number:\n"
            f"{dict_to_bullets(population_data['healing_progression']['temporal_analysis']['by_visit_number'])}\n"
            "By Treatment Duration:\n"
            f"{dict_to_bullets(population_data['healing_progression']['temporal_analysis']['by_treatment_duration'])}\n\n"

            "**Exudate Analysis**:\n"
            "Volume Characteristics:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['characteristics']['volume']['distribution'])}\n"
            "Volume vs Healing:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['characteristics']['volume']['healing_correlation'])}\n"
            "Type Distribution:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['characteristics']['type']['distribution'])}\n"
            "Type vs Healing:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['characteristics']['type']['healing_correlation'])}\n"
            "Viscosity Distribution:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['characteristics']['viscosity']['distribution'])}\n"
            "Viscosity vs Healing:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['characteristics']['viscosity']['healing_correlation'])}\n"
            "Temporal Patterns:\n"
            "Volume Progression:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['temporal_patterns']['volume_progression'])}\n"
            "Type Progression:\n"
            f"{dict_to_bullets(population_data['exudate_analysis']['temporal_patterns']['type_progression'])}\n\n"
        )

        # Add sensor data section if available
        if 'sensor_data' in population_data:
            prompt += "**Sensor Data Analysis**:\n"

            if 'temperature' in population_data['sensor_data']:
                temp_data = population_data['sensor_data']['temperature']
                prompt += "Temperature Measurements:\n"

                # Center temperature
                center_stats = temp_data['center_temp']['overall']
                prompt += (f"Center of Wound:\n"
                          f"- Mean: {safe_float_format(center_stats['mean'])}°F\n"
                          f"- Range: {safe_float_format(center_stats['min'])} - {safe_float_format(center_stats['max'])}°F\n")

                # Edge and peri-wound temperatures if available
                if 'edge_temp' in temp_data and 'peri_temp' in temp_data:
                    edge_stats = temp_data['edge_temp']['overall']
                    peri_stats = temp_data['peri_temp']['overall']
                    prompt += (
                        f"Edge of Wound:\n"
                        f"- Mean: {safe_float_format(edge_stats['mean'])}°F\n"
                        f"Peri-wound:\n"
                        f"- Mean: {safe_float_format(peri_stats['mean'])}°F\n"
                    )

                    # Temperature gradients
                    if 'gradients' in temp_data:
                        grad_stats = temp_data['gradients']
                        prompt += (
                            "Temperature Gradients:\n"
                            f"- Center to Edge: {safe_float_format(grad_stats['center_to_edge']['mean'])}°F\n"
                            f"- Center to Peri-wound: {safe_float_format(grad_stats['center_to_peri']['mean'])}°F\n"
                        )
                prompt += "\n"

            if 'impedance' in population_data['sensor_data']:
                imp_data = population_data['sensor_data']['impedance']
                prompt += "Impedance Measurements:\n"

                # Magnitude
                mag_stats = imp_data['magnitude']['overall']
                prompt += (f"Magnitude (|Z|):\n"
                          f"- Mean: {safe_float_format(mag_stats['mean'])} kOhms\n"
                          f"- Range: {safe_float_format(mag_stats['min'])} - {safe_float_format(mag_stats['max'])} kOhms\n")

                # Complex components if available
                if 'complex_components' in imp_data:
                    real_stats = imp_data['complex_components']['real']['overall']
                    imag_stats = imp_data['complex_components']['imaginary']['overall']
                    prompt += (
                        f"Real Component (Z'):\n"
                        f"- Mean: {safe_float_format(real_stats['mean'])} kOhms\n"
                        f"Imaginary Component (Z\"):\n"
                        f"- Mean: {safe_float_format(imag_stats['mean'])} kOhms\n"
                    )
                prompt += "\n"

            if 'oxygenation' in population_data['sensor_data']:
                oxy_data = population_data['sensor_data']['oxygenation']
                prompt += "Oxygenation Measurements:\n"

                # Basic oxygenation
                if 'oxygenation' in oxy_data:
                    oxy_stats = oxy_data['oxygenation']['overall']
                    prompt += (
                        f"Oxygenation Percentage:\n"
                        f"- Mean: {safe_float_format(oxy_stats['mean'])}%\n"
                        f"- Range: {safe_float_format(oxy_stats['min'])} - {safe_float_format(oxy_stats['max'])}%\n"
                        f"- Correlation with Healing: {safe_float_format(oxy_data['oxygenation']['correlation_with_healing'])}\n"
                    )

                # Hemoglobin measurements
                for hb_type in ['hemoglobin', 'oxyhemoglobin', 'deoxyhemoglobin']:
                    if hb_type in oxy_data:
                        hb_stats = oxy_data[hb_type]['overall']
                        prompt += (
                            f"{hb_type.title()}:\n"
                            f"- Mean: {safe_float_format(hb_stats['mean'])}\n"
                            f"- Range: {safe_float_format(hb_stats['min'])} - {safe_float_format(hb_stats['max'])}\n"
                            f"- Correlation with Healing: {safe_float_format(oxy_data[hb_type]['correlation_with_healing'])}\n"
                        )
                prompt += "\n"

        prompt += (
            "\n**Analysis Requirements**:\n"
            "1. Identify key patterns and correlations in:\n"
            "   - Demographics vs healing outcomes\n"
            "   - Risk factors' impact on healing\n"
            "   - Wound characteristics vs healing time\n"
            "   - Sensor data trends (if available)\n\n"
            "2. Provide evidence-based recommendations for:\n"
            "   - Risk stratification\n"
            "   - Treatment optimization\n"
            "   - Monitoring protocols\n\n"
            "3. Format your analysis as a structured clinical report with:\n"
            "   - Key findings with statistical significance\n"
            "   - Clinical implications\n"
            "   - Actionable recommendations\n"
        )

        return prompt

    def _stream_analysis(self, messages, callback):
        """
        Stream analysis results from the model.

        Args:
            messages: List of messages to send to the model
            callback: Function to call with each chunk of streamed content

        Returns:
            Final analysis results as string
        """
        import json
        import re
        import httpx
        import time

        # Initialize buffers for streaming content
        stream_buffer = ""
        thinking_buffer = ""
        conclusion = ""
        
        try:
            # Process the streaming response
            for chunk in self.model.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    # Add the new content to the buffer
                    stream_buffer += chunk.content

                    # Try to parse JSON as it comes in
                    try:
                        # Try to find complete JSON objects in the stream
                        # Look for matching braces to identify potential JSON objects
                        json_pattern = r'\{.*\}'
                        match = re.search(json_pattern, stream_buffer)

                        if match:
                            potential_json = match.group(0)
                            try:
                                partial_data = json.loads(potential_json)

                                # Extract thinking and conclusion if available
                                if "thinking" in partial_data:
                                    new_thinking = partial_data["thinking"]
                                    if new_thinking != thinking_buffer:
                                        thinking_buffer = new_thinking
                                        # Call the callback with updated thinking
                                        if callback:
                                            callback({"type": "thinking", "content": thinking_buffer})

                                if "conclusion" in partial_data:
                                    conclusion = partial_data["conclusion"]
                            except json.JSONDecodeError:
                                # Not a valid JSON object yet
                                pass
                    except Exception as e:
                        # Handle any errors in regex or parsing
                        logger.debug(f"Error parsing streaming JSON: {str(e)}")
                        pass

                    # Call the callback with the raw chunk for any custom handling
                    if callback:
                        callback({"type": "raw", "content": chunk.content})
        except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout, 
                httpx.ReadError, ConnectionError, httpx.HTTPError) as e:
            # Handle connection-related errors
            logger.warning(f"Streaming connection error: {str(e)}. Continuing with partial results.")
            if callback:
                callback({"type": "error", "content": f"Connection interrupted: {str(e)}. Using partial results."})
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error during streaming: {str(e)}")
            if callback:
                callback({"type": "error", "content": f"Error: {str(e)}. Using partial results."})

        # Final processing of the response (complete or partial)
        try:
            # Try to find a complete JSON object in the final buffer
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, stream_buffer, re.DOTALL)

            if match:
                potential_json = match.group(0)
                final_data = json.loads(potential_json)
                self.thinking_process = final_data.get("thinking", "No thinking process provided")
                conclusion = final_data.get("conclusion", stream_buffer)
            else:
                # If no JSON found, use the raw buffer
                self.thinking_process = thinking_buffer or "No structured thinking process available"
                conclusion = conclusion or stream_buffer
        except Exception as e:
            logger.warning(f"Failed to parse JSON from streamed response: {str(e)}")
            self.thinking_process = thinking_buffer or "Error parsing thinking process"
            conclusion = conclusion or stream_buffer

        # If we didn't get any conclusion but have thinking, create a simple conclusion
        if not conclusion and thinking_buffer:
            conclusion = "Analysis was interrupted. Please see the thinking process for partial analysis."
            
        return conclusion

    def analyze_patient_data(self, patient_data: Dict, callback=None) -> str:
        """
        Analyze patient data using the configured model.
        Args:
            patient_data: Processed patient data dictionary
            callback: Optional callback function to receive streaming updates
        Returns:
            Analysis results as string
        """
        # Load model if not already loaded
        self._load_model()
        prompt = self._format_per_patient_prompt(patient_data)
        self.thinking_process = None  # Reset thinking process

        try:
            if self.platform == "ai-verde":
                # Use AI Verde with system and human messages
                if self.model_name == "deepseek-r1":
                    # For DeepSeek R1, request structured output with thinking
                    system_message = (
                        "You are a medical expert specializing in wound care analysis. "
                        "Analyze the provided wound data and provide clinical recommendations. "
                        "Structure your response as a JSON object with two keys: "
                        "'thinking' (your step-by-step analysis process) and "
                        "'conclusion' (your final analysis and recommendations)."
                    )
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=prompt)
                    ]

                    # Check if streaming is enabled and callback is provided
                    if getattr(self.model, 'streaming', False) and callback:
                        return self._stream_analysis(messages, callback)
                    else:
                        response = self.model.invoke(messages)

                        # Parse the JSON response
                        try:
                            import json
                            response_data = json.loads(response.content)
                            self.thinking_process = response_data.get("thinking", "No thinking process provided")
                            return response_data.get("conclusion", response.content)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON response from DeepSeek R1")
                            return response.content
                else:
                    # Standard approach for other models
                    messages = [
                        SystemMessage(content="You are a medical expert specializing in wound care analysis. Analyze the provided wound data and provide clinical recommendations."),
                        HumanMessage(content=prompt)
                    ]
                    response = self.model.invoke(messages)
                    return response.content
            else:
                # For HuggingFace models, we'll use a simplified version of the prompt
                combined_prompt = (
                    "You are a wound care expert analyzing patient data. "
                    "Provide a detailed analysis focusing on patterns, risk factors, "
                    "and evidence-based recommendations.\n\n" + prompt
                )

                response = self.model(
                    combined_prompt,
                    do_sample=True,
                    temperature=0.3,
                    max_new_tokens=512,
                    num_return_sequences=1
                )

                if isinstance(response, list):
                    generated_text = response[0]['generated_text']
                else:
                    generated_text = response['generated_text']

                return generated_text
                # # Clean up the response
                # analysis = generated_text.replace(combined_prompt, '').strip()

                # # Format the analysis with clear sections
                # sections = [
                #     "Key Patterns and Trends",
                #     "Risk Factor Analysis",
                #     "Treatment Effectiveness",
                #     "Sensor Data Insights",
                #     "Clinical Recommendations",
                #     "Future Directions"
                # ]

                # formatted_analysis = ""
                # current_section = ""
                # for line in analysis.split('\n'):
                #     # Check if line starts a new section
                #     is_section = False
                #     for section in sections:
                #         if section.lower() in line.lower():
                #             current_section = section
                #             formatted_analysis += f"\n## {section}\n"
                #             is_section = True
                #             break

                #     if not is_section and line.strip():
                #         if not current_section:
                #             current_section = "Analysis"
                #             formatted_analysis += "\n## General Analysis\n"
                #         formatted_analysis += line + "\n"

                # return formatted_analysis.strip()

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

    def analyze_population_data(self, population_data: Dict, callback=None) -> str:
        """
        Analyze population-level data using the configured model.
        Args:
            population_data: Processed population data dictionary
            callback: Optional callback function to receive streaming updates
        Returns:
            Analysis results as string
        """
        # Load model if not already loaded
        self._load_model()
        prompt = self._format_population_prompt(population_data)
        self.thinking_process = None  # Reset thinking process

        try:
            if self.platform == "ai-verde":
                if self.model_name == "deepseek-r1":
                    # For DeepSeek R1, request structured output with thinking
                    system_message = (
                        "You are a medical expert specializing in wound care analysis. "
                        "Analyze the provided population-level wound data and provide clinical insights. "
                        "Structure your response as a JSON object with two keys: "
                        "'thinking' (your step-by-step analysis process) and "
                        "'conclusion' (your final analysis and recommendations)."
                    )
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=prompt)
                    ]

                    # Check if streaming is enabled and callback is provided
                    if getattr(self.model, 'streaming', False) and callback:
                        return self._stream_analysis(messages, callback)
                    else:
                        response = self.model.invoke(messages)

                        # Parse the JSON response
                        try:
                            import json
                            response_data = json.loads(response.content)
                            self.thinking_process = response_data.get("thinking", "No thinking process provided")
                            return response_data.get("conclusion", response.content)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON response from DeepSeek R1")
                            return response.content
                else:
                    # Standard approach for other models
                    system_prompt = (
                        "You are a senior wound care specialist and data scientist analyzing population-level wound healing data. "
                        "Your expertise spans clinical wound care, biostatistics, and medical data analysis. Your task is to:\n\n"
                        "1. Analyze complex wound healing patterns across diverse patient populations\n"
                        "2. Identify clinically significant trends and correlations\n"
                        "3. Evaluate the effectiveness of different treatment approaches\n"
                        "4. Provide evidence-based recommendations for improving wound care outcomes\n\n"
                        "Focus on actionable insights that have practical clinical applications. Support your analysis with "
                        "specific data points and statistics when available. Consider both statistical significance and "
                        "clinical relevance in your interpretations. When making recommendations, consider implementation "
                        "feasibility and potential resource constraints."
                    )

                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt)
                    ]
                    response = self.model.invoke(messages)
                    return response.content
            else:
                # For HuggingFace models, use a simplified approach
                combined_prompt = (
                    "You are a wound care expert analyzing population data. "
                    "Provide a detailed analysis focusing on patterns, risk factors, "
                    "and evidence-based recommendations.\n\n" + prompt
                )

                response = self.model(
                    combined_prompt,
                    do_sample=True,
                    temperature=0.3,
                    max_new_tokens=512,
                    top_p=0.95
                )

                if isinstance(response, list):
                    generated_text = response[0]['generated_text']
                else:
                    generated_text = response['generated_text']

                return generated_text

        except Exception as e:
            logger.error(f"Error analyzing population data: {str(e)}")
            raise

    def get_thinking_process(self) -> str:
        """
        Get the thinking process from the last analysis (if available).

        Returns:
            The thinking process as a string, or None if not available
        """
        return self.thinking_process
