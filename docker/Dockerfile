FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file 
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application 
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for data and output if they don't exist
RUN mkdir -p dataset/impedance_frequency_sweep
RUN mkdir -p wound_analysis/utils/logs

# Make sure the directories are writable
RUN chmod -R 777 dataset
RUN chmod -R 777 wound_analysis/utils/logs

# Debug: Create a test file to ensure proper permissions
RUN echo "Test file content" > /app/dataset/test_file.txt
RUN echo "Debug log init" > /app/debug.log
RUN chmod 666 /app/debug.log

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the application
CMD ["streamlit", "run", "wound_analysis/dashboard.py"]
