from setuptools import setup, find_packages
import os
import pathlib

# Get the current directory (project root directory)
project_root = pathlib.Path(__file__).parent.absolute()

# Read requirements from requirements.txt
with open(os.path.join(project_root, 'setup_config', 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# Read long description from README.md
with open(os.path.join(project_root, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="wound_analysis",
    version="1.0.0",
    description="Wound Care Analysis System using LLMs and sensor data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Artin Majdi",
    author_email="artinmajdi@example.com",  # Replace with your actual email
    url="https://github.com/artinmajdi/Wound_management_interpreter_LLM",
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
