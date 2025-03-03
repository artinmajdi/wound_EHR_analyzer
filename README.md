# Wound Management Interpreter LLM

A comprehensive wound care analysis system that leverages large language models (LLMs) and sensor data to provide clinical insights and recommendations for wound management.

## Overview

This project provides an intelligent system for analyzing wound care data using bioimpedance measurements, medical observations, and LLM-based interpretation. It enables healthcare providers to track wound healing progress, detect potential complications, and receive personalized treatment recommendations.

The system includes both a command-line interface for individual patient analysis and a Streamlit dashboard for interactive exploration of patient data and population-level insights.

## Features

- **Patient Data Analysis**: Process and analyze individual patient wound care records
- **Bioimpedance Analysis**: Advanced electrical impedance spectroscopy interpretation
- **LLM-Powered Clinical Insights**: Leverages state-of-the-art language models to generate clinical interpretations
- **Interactive Dashboard**: Visualize wound healing trajectories, impedance data, and clinical recommendations
- **Population-Level Analytics**: Analyze trends across patient demographics and wound types
- **Report Generation**: Export comprehensive wound analysis reports in Word format

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (listed in `requirements.txt`)
- Access to LLM API (optional, models can be run locally)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/artinmajdi/Wound_management_interpreter_LLM.git
   cd Wound_management_interpreter_LLM
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys if using external LLM services
   ```

### Dataset Structure

The system expects a specific dataset structure:

1. Main CSV data file: `SmartBandage-Data_for_llm.csv` in the `/dataset` directory
2. Impedance data: The `palmsense files Jan 2025` folder must be placed inside the same directory as the CSV file

Your dataset directory should look like this:
```
dataset/
├── SmartBandage-Data_for_llm.csv
└── palmsense files Jan 2025/
    ├── 1.xlsx
    ├── 2.xlsx
    └── ... (impedance data files by patient ID)
```

## Usage

### Command Line Interface

To analyze a specific patient record:

```bash
python wound_analysis/main.py --record-id 41 --platform ai-verde --model-name llama-3.3-70b-fp8
```

Options:
- `--record-id`: Patient record ID to analyze
- `--dataset-path`: Path to the dataset directory
- `--output-dir`: Directory to save output files
- `--platform`: LLM platform to use (ai-verde or huggingface)
- `--model-name`: Name of the LLM model to use
- `--api-key`: API key for the LLM platform (if required)

### Interactive Dashboard

To launch the Streamlit dashboard:

```bash
streamlit run wound_analysis/dashboard.py
```

The dashboard provides:
1. Patient selection and data visualization
2. Wound healing trajectory analysis
3. Bioimpedance measurement interpretation
4. Clinical recommendations based on LLM analysis
5. Population-level statistical insights
6. Downloadable reports

## System Architecture

The project is organized into these main components:

- **Data Processing**: Handles loading, cleaning, and transforming wound care data
- **Bioimpedance Analysis**: Interprets electrical impedance measurements to assess tissue health
- **LLM Interface**: Connects to various language models to generate clinical insights
- **Dashboard**: Interactive Streamlit interface for data exploration and analysis
- **Reporting**: Generates formatted reports with analysis results

## Models

The system supports multiple language model platforms:

### AI Verde Models
- llama-3.3-70b-fp8 (default)
- meta-llama-3.1-70b
- deepseek-r1
- llama-3.2-11b-vision

### HuggingFace Models
- medalpaca-7b
- BioMistral-7B
- ClinicalCamel-70B

## License

[License information]

## Acknowledgments

[Any acknowledgments for data sources, collaborators, or inspiration]
