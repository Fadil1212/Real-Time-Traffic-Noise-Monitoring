FROM python:3.9-slim

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY route_recommender.py .
COPY road_network.py .

# Create OSM cache directory
RUN mkdir -p /root/.osmnx

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Run application
CMD ["python", "route_recommender.py"]