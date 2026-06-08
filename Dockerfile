# Use slim Python base
FROM python:3.10-slim

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run expects the app to listen on this port
ENV PORT=8080

# Your app defaults
ENV DOCUMENT_CLEANER_BASE_DIR=/tmp
ENV MODEL_WEIGHTS_DIR=/app/model_weights
ENV DEFAULT_WEIGHT_FILE=sigma=20.mat

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Explicit sanity check: fail the image build if the model file is missing
RUN test -f "/app/model_weights/sigma=20.mat" \
    || (echo "ERROR: Missing /app/model_weights/sigma=20.mat" && ls -lah /app && ls -lah /app/model_weights || true && exit 1)

# Expose Cloud Run port
EXPOSE 8080

# Use shell form so ${PORT} expands correctly
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
