from setuptools import setup, find_packages
import os
import pathlib

# Get the current directory (project root directory)
project_root = pathlib.Path(__file__).parent.absolute()

# Read requirements from requirements.txt or use hardcoded list if file not found
try:
    with open(os.path.join(project_root, 'requirements.txt')) as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    # Fallback requirements list if requirements.txt is not available (e.g., during build from sdist)
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "python-docx>=0.8.11",
        "python-dotenv>=0.19.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "pillow>=10.0.0",
        "pydantic>=2.0.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.1",
        "openai>=0.27.0",
        "openpyxl>=3.1.0",
        "protobuf>=3.20.0",
        "sacremoses",
        "watchdog"
    ]

# Read long description from README.md
with open(os.path.join(project_root, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="wound_analysis",
    version="1.5.0",
    description="Wound Care Analysis System using LLMs and sensor data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Artin Majdi",
    author_email="msm2024@gmail.com",
    url="https://github.com/artinmajdi/wound_EHR_analyzer_private",
    project_urls={
        "GitHub": "https://github.com/artinmajdi/wound_EHR_analyzer_private",
    },
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: Other/Proprietary License",  # CC BY-NC 4.0 isn't a standard classifier
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        'console_scripts': [
            'wound-analysis=wound_analysis.main:main',
            'wound-dashboard=wound_analysis.cli:run_dashboard',
        ],
    },
    keywords="wound care, healthcare, LLM, AI, medical analysis",
)
