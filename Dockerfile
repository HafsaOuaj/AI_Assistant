# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /AI_Assistant

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock* /AI_Assistant/

# Install the dependencies
RUN poetry install --only main

# Add debugging steps to check if Streamlit is installed
RUN poetry run streamlit --version || echo "Streamlit is not installed"

# Copy the rest of the application code to the working directory
COPY . /app/

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["poetry", "run", "streamlit", "run", "main.py"]
