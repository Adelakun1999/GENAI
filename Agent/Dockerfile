# Use a lightweight Python base image
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Install system dependencies for building and PDFs
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
