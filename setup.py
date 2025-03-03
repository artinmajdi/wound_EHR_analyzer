from setuptools import setup, find_packages

setup(
    name="wound_analysis",
    version="1.0.0",
    description="Wound Care Analysis System using LLMs and sensor data",
    author="Artin Majdi",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
