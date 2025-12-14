FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy producer script
COPY noise_producer.py .
COPY test_consumer.py .

# Make scripts executable
RUN chmod +x noise_producer.py test_consumer.py

# Default command (can be overridden)
CMD ["python", "noise_producer.py"]