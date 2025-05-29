# Dockerfile generated with help from ChatGPT (OpenAI)
# Use an official Python image with minimal footprint
FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency declaration file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase into the container
COPY . .

ENV PYTHONPATH=/app

# Set default command (can be overridden by `docker run`)
CMD ["python", "examples/simple_example.py"]