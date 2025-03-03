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

#### Option 1: Using Python venv

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

#### Option 2: Using Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/artinmajdi/Wound_management_interpreter_LLM.git
   cd Wound_management_interpreter_LLM
   ```

2. Create and activate a Conda environment:

   Method A: Using the provided environment.yml file (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate wound_analysis
   ```

   Method B: Creating a new environment manually:
   ```bash
   conda create -n wound_analysis python=3.12
   conda activate wound_analysis
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys if using external LLM services
   ```

#### Option 3: Using Docker

The project includes Docker support for easy deployment and consistent environments.

#### Prerequisites

- Docker and Docker Compose installed on your system
- Dataset files properly organized in the `dataset` directory

#### Running with Docker

1. Build and start the Streamlit dashboard:

```bash
docker compose up
```

2. Or run the CLI version for a specific patient:

```bash
docker compose run --rm cli
```

3. To analyze a different patient record:

```bash
docker compose run --rm cli python -m wound_analysis.main --record-id <PATIENT_ID>
```

4. To use a specific LLM platform:

```bash
docker compose run --rm cli python -m wound_analysis.main --platform <PLATFORM_NAME> --api-key <YOUR_API_KEY>
```

5. If you make changes to the code, rebuild the Docker image:

```bash
docker compose build
```

6. To verify your dataset is properly mounted and accessible to the Docker container:

```bash
docker compose --profile verify up verify
```

This runs the `wound_analysis/utils/verify_dataset.py` script which checks:
- If the dataset directory exists and is accessible
- If the impedance_frequency_sweep directory exists
- If the main CSV file exists and is readable
- File and directory permissions

Use this tool to troubleshoot any dataset access issues when running the application in Docker.

#### Docker Environment Variables

You can set environment variables in a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your-base-url-here
```

You can also specify a custom dataset path when running Docker Compose:

```bash
DATASET_PATH=/path/to/your/dataset docker compose up
```

This allows you to use a dataset located anywhere on your system without modifying the docker-compose.yml file. By default, the system will use the `./dataset` directory if no custom path is specified.

#### Using the setup_env.sh Script

For convenience, you can use the provided `setup_env.sh` script to set up your environment variables:

1. Make the script executable:
   ```bash
   chmod +x setup_env.sh
   ```

2. Run the script:
   ```bash
   ./setup_env.sh
   ```

3. Follow the prompts to enter your OpenAI API key and base URL.

The script will:
- Create a `.env` file from `.env.example` if it doesn't exist
- Update the OpenAI API key in the `.env` file
- Update the OpenAI base URL in the `.env` file (optional)
- Provide instructions for running Docker Compose

### Dataset Structure

The system expects a specific dataset structure:

1. Main CSV data file: `SmartBandage-Data_for_llm.csv` in the `/dataset` directory
2. Impedance data: The `impedance_frequency_sweep` folder must be placed inside the same directory as the CSV file

Your dataset directory should look like this:
```
dataset/
├── SmartBandage-Data_for_llm.csv
└── impedance_frequency_sweep/
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

### Streamlit Dashboard

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
