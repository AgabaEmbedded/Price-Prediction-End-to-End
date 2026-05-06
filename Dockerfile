# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
# Prevents Python from writing .pyc files and ensures output is sent to terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some ML libraries or git if using MLflow)
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    curl \
#    git \
#    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY app_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r app_requirements.txt

# Copy the rest of the application code
# This includes app.py, configs/, fetch/, features/, and utils/
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["python", "app.py"]