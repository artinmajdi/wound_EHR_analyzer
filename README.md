# Wound EHR Analyzer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wound-analysis7.streamlit.app/)

An AI-powered tool for analyzing and interpreting wound care management data, providing healthcare professionals with advanced insights and recommendations.

<!-- [📚 View Full Documentation](docs/index.md) -->

## Overview

This application leverages large language models (LLMs) to analyze wound care data, generating comprehensive insights and evidence-based recommendations for healthcare providers. The system includes both an interactive web dashboard and command-line tools for efficient data processing and analysis.

## Key Features

- **Interactive Analysis Dashboard**: Streamlit-based interface for real-time data visualization and AI-powered insights
- **Multi-Model LLM Support**: Compatible with various LLM platforms including OpenAI and custom endpoints
- **Advanced Statistical Analysis**: Comprehensive wound healing trend analysis and progression tracking
- **Flexible Data Handling**: Support for diverse data types including images, time-series data, and clinical notes
- **Robust Error Handling**: Graceful recovery from API interruptions and connection issues
- **Containerized Deployment**: Docker support for consistent deployment across environments

### Dashboard Components

The interactive dashboard provides comprehensive wound analysis through specialized tabs:

- **Overview**: Patient demographics, wound summary statistics, and population-level trends
- **Impedance Analysis**: Electrical measurements visualization with clinical interpretations
- **Temperature**: Thermal gradient analysis for wound assessment
- **Oxygenation**: Tissue oxygen saturation monitoring and analysis
- **Exudate**: Characterization and trending of wound drainage
- **Risk Factors**: Patient-specific risk factor evaluation and impact analysis
- **LLM Analysis**: AI-powered natural language processing for comprehensive wound assessment

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for containerized deployment)
- OpenAI API key or compatible service

### Installation & Setup

We provide convenient scripts for all setup operations. Choose the deployment method that best fits your needs:

#### Option 0: Pip Installation (Simplest Method)

```bash
# Install directly from PyPI (once published)
pip install wound-analysis

# Or install the latest version from GitHub
pip install git+https://github.com/artinmajdi/Wound_management_interpreter_LLM.git

# Run the dashboard
wound-dashboard

# Or run analysis from command line
wound-analysis --record-id 41
```

See [documentation/INSTALL.md](documentation/INSTALL.md) for detailed pip installation instructions.

#### Option 1: Docker Deployment (Recommended for Production)

```bash
# 1. Set up environment variables (API keys, etc.)
./scripts/setup_env_variables.sh

# 2. Start the application in Docker
./scripts/run_docker.sh start

# 3. Access the dashboard at http://localhost:8501

# 4. Run CLI analysis for a specific patient record
./scripts/run_docker.sh cli 41

# 5. Verify dataset structure and integrity
./scripts/run_docker.sh verify
```

#### Option 2: Conda Environment (Recommended for Development)

```bash
# 1. Create and configure the conda environment
./scripts/install.sh

# 2. Activate the environment
conda activate wound_analysis

# 3. Run the dashboard
streamlit run wound_analysis/dashboard.py
```

#### Option 3: Python Virtual Environment

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies and set up environment
pip install -r setup_config/requirements.txt
pip install -e .
./scripts/setup_env_variables.sh

# 3. Run the dashboard
streamlit run wound_analysis/dashboard.py
```

## Documentation

- [**Configuration Guide**](documentation/configuration.md): Environment variables and configuration options
- [**Docker Usage Guide**](documentation/docker_usage.md): Detailed containerization instructions
- [**API Documentation**](documentation/index.md): API reference and component documentation
- [**Installation Guide**](documentation/INSTALL.md): Pip installation instructions
- [**Data Processing**](documentation/data_processing.md): Information on supported data formats and processing pipelines
- [**LLM Integration**](documentation/llm_integration.md): Guide to configuring and using different LLM models

## Project Structure

```
wound_management_interpreter_LLM/
├── setup.py                  # Package configuration
├── setup_config/                   # Configuration files
│   ├── .env.example          # Template for environment variables
│   ├── environment.yml       # Conda environment specification
│   ├── MANIFEST.in           # Package manifest file
│   ├── pyproject.toml        # Modern Python project metadata
│   ├── pytest.ini            # PyTest configuration
│   └── requirements.txt      # Python dependencies
├── documentation/            # Documentation files
│   ├── INSTALL.md            # Installation instructions
│   ├── LICENSE               # License file
│   ├── configuration.md      # Configuration guide
│   ├── docker_usage.md       # Docker deployment instructions
│   ├── data_processing.md    # Data processing guide
│   ├── llm_integration.md    # LLM integration guide
│   └── index.md              # Documentation index
├── docker/                   # Docker configuration
│   ├── Dockerfile            # Container definition
│   ├── docker-compose.yml    # Service orchestration
│   └── .dockerignore         # Build exclusions
├── scripts/                  # Utility scripts
│   ├── run_docker.sh         # Docker management script
│   ├── install.sh        # Conda environment setup
│   └── setup_env_variables.sh          # Environment configuration
├── tests/                    # Test suite
├── wound_analysis/           # Core application code
│   ├── dashboard.py          # Streamlit interface
│   ├── main.py               # CLI entry point
│   ├── cli.py                # Command line interface
│   ├── utils/                # Utility modules
│   └── dashboard_components/ # Dashboard components
│       ├── overview.py       # Overview tab component
│       ├── impedance.py      # Impedance analysis component
│       ├── temperature.py    # Temperature analysis component
│       ├── oxygenation.py    # Oxygenation analysis component
│       ├── exudate.py        # Exudate analysis component
│       └── risk_factors.py   # Risk factors analysis component
├── dataset/                  # Data directory (mounted at runtime)
├── .env                      # Environment variables
└── ide_config/               # IDE configuration
    └── Wound_management_interpreter_LLM.code-workspace  # VSCode workspace file
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC 4.0), which permits non-commercial use with attribution. See the [documentation/LICENSE](documentation/LICENSE) file for details.

## Acknowledgments

- This project was developed as part of advanced research in AI-assisted healthcare
- Special thanks to the healthcare professionals who provided domain expertise
