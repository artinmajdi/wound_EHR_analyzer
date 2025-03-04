# Wound Management Interpreter LLM

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A professional AI-powered tool for analyzing and interpreting wound care management data, providing healthcare professionals with advanced insights and recommendations.

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

See [INSTALL.md](INSTALL.md) for detailed pip installation instructions.

#### Option 1: Docker Deployment (Recommended for Production)

```bash
# 1. Set up environment variables (API keys, etc.)
./scripts/setup_env.sh

# 2. Start the application in Docker
./scripts/run_docker.sh start

# 3. Access the dashboard at http://localhost:8501
```

#### Option 2: Conda Environment (Recommended for Development)

```bash
# 1. Create and configure the conda environment
./scripts/setup_conda.sh

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
pip install -r requirements.txt
pip install -e .
./scripts/setup_env.sh

# 3. Run the dashboard
streamlit run wound_analysis/dashboard.py
```

## Documentation

- [**Configuration Guide**](docs/configuration.md): Environment variables and configuration options
- [**Docker Usage Guide**](docs/docker_usage.md): Detailed containerization instructions
- [**API Documentation**](docs/index.md): API reference and component documentation

## Project Structure

```
wound_management_interpreter_LLM/
├── config/                   # Configuration files
│   ├── .env.example          # Template for environment variables
│   └── environment.yml       # Conda environment specification
├── docker/                   # Docker configuration
│   ├── Dockerfile            # Container definition
│   ├── docker-compose.yml    # Service orchestration
│   └── .dockerignore         # Build exclusions
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
│   ├── run_docker.sh         # Docker management script
│   ├── setup_conda.sh        # Conda environment setup
│   └── setup_env.sh          # Environment configuration
├── tests/                    # Test suite
├── wound_analysis/           # Core application code
│   ├── dashboard.py          # Streamlit interface
│   ├── main.py               # CLI entry point
│   └── utils/                # Utility modules
├── dataset/                  # Data directory (mounted at runtime)
├── requirements.txt          # Python dependencies
└── setup.py                  # Package configuration
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC 4.0), which permits non-commercial use with attribution. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was developed as part of advanced research in AI-assisted healthcare
- Special thanks to the healthcare professionals who provided domain expertise
