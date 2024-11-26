# Use a lightweight Python base image
FROM python:3.12.7-slim

# Install system dependencies for Pipenv and Python build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Set the working directory inside the container
WORKDIR /app

# Copy Pipenv files into the container
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install project dependencies using Pipenv
RUN pipenv install --system --deploy

# Copy the application code and model file into the container
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Expose the Flask application's port
EXPOSE 9696

# Define the entry point for the application
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
