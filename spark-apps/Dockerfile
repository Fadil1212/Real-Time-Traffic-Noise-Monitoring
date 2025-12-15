FROM apache/spark:3.5.0

USER root

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    python3-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_LIB=/usr/share/proj

# Set Python path for Spark
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip:${PYTHONPATH}"

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY noise_consumer.py /app/
COPY road_network.py /app/

# Create cache directory for OSMnx
RUN mkdir -p /root/.osmnx && chmod 777 /root/.osmnx

# Use spark-submit instead of python3 directly
CMD ["/opt/spark/bin/spark-submit", \
    "--master", "local[2]", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.1", \
    "/app/noise_consumer.py"]