FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application file
COPY correlation_analyzer.py .

# Copy utils and models from parent directory (mounted as volumes)
# These will be mounted at runtime via docker-compose

# Run the analyzer
CMD ["python", "correlation_analyzer.py"]