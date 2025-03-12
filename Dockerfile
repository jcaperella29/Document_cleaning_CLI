FROM python:3.10-slim

# Set env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ghostscript \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8080

# Run API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
