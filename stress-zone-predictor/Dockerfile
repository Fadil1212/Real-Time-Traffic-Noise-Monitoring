FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY stress_zone_predictor.py .

# Run prediction service
CMD ["python", "-u", "stress_zone_predictor.py"]