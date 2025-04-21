# Base Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install fastapi uvicorn boto3

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "scripts.api_server:app", "--host", "0.0.0.0", "--port", "8000"]

