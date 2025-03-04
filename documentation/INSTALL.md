# Installation Guide for Wound Management Interpreter LLM

This document provides instructions for installing the Wound Management Interpreter LLM package using pip.

## Installation Methods

### 1. Install from PyPI (Recommended for Users)

Once the package is published to PyPI, you can install it directly using pip:

```bash
pip install wound-analysis
```

### 2. Install from GitHub (Latest Development Version)

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/artinmajdi/Wound_management_interpreter_LLM.git
```

### 3. Install in Development Mode (For Contributors)

If you're contributing to the project, you can install it in development mode:

```bash
# Clone the repository
git clone https://github.com/artinmajdi/Wound_management_interpreter_LLM.git
cd Wound_management_interpreter_LLM

# Install in development mode
pip install -e .
```

## Verifying Installation

After installation, you can verify that the package is correctly installed by running:

```bash
# Run the CLI tool
wound-analysis --help

# Launch the dashboard
wound-dashboard
```

## Dependencies

The package automatically installs all required dependencies listed in `config/requirements.txt`.

## Configuration

For API keys and other configuration options, please refer to the [Configuration Guide](docs/configuration.md).

## Troubleshooting

If you encounter any issues during installation:

1. Ensure you have Python 3.10 or higher installed
2. Try upgrading pip: `pip install --upgrade pip`
3. For GPU support with PyTorch, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
4. Check that all dependencies are correctly installed: `pip list`

## Publishing to PyPI (For Maintainers)

To publish the package to PyPI:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI
twine upload dist/*
```

## Build and Publish Process

Make sure to update the version number in `config/setup.py` before publishing a new release.
