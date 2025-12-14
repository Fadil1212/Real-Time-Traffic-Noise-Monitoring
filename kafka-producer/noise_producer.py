#!/usr/bin/env python3
"""
Traffic Noise Monitoring - Kafka Producer
Simulates noise sensors across Milwaukee, USA sending real-time data
bound to actual asphalt road geometries.
"""

import json
import time
import random
import logging
import os
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import osmnx as ox
from shapely.geometry import Point

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache the graph so we don't download it every time
_MILWAUKEE_GRAPH = None

def get_milwaukee_roads():
    """
    Downloads the drivable road network for Milwaukee from OpenStreetMap.
    """
    global _MILWAUKEE_GRAPH
    
    if _MILWAUKEE_GRAPH is None:
        logger.info("🌍 Downloading Milwaukee road network from OpenStreetMap (this may take a moment)...")
        # 1. Download raw data (in Lat/Lon degrees)
        # 'network_type="drive"' ensures we only get asphalt roads where cars go
        raw_graph = ox.graph_from_place("Milwaukee, Wisconsin, USA", network_type="drive")
        
        # 2. Project to meters (UTM)
        # This converts coordinates to meters so 'length' is actually in meters
        _MILWAUKEE_GRAPH = ox.project_graph(raw_graph)
    
    # 3. Convert to GeoDataFrame
    gdf_edges = ox.graph_to_gdfs(_MILWAUKEE_GRAPH, nodes=False, edges=True)
    
    # 4. Filter safely
    # Now that we are in meters, this properly filters out tiny segments < 50m
    filtered_edges = gdf_edges[gdf_edges.length > 50]
    
    # Safety check: If filter removed everything (unlikely with projection), revert
    if filtered_edges.empty:
        logger.warning("⚠️ Filter removed all roads! Reverting to full dataset.")
        return gdf_edges
    
    return filtered_edges

def generate_sensors(num_sensors):
    """
    Generate sensors strictly bound to real Milwaukee asphalt roads.
    """
    sensors = []
    
    # 1. Get real road data
    try:
        roads = get_milwaukee_roads()
    except Exception as e:
        logger.error(f"❌ Failed to download map data: {e}")
        return []

    logger.info(f"📍 Generating {num_sensors} sensors on real road geometries...")

    # 2. Generate sensors
    for i in range(num_sensors):
        sensor_num = i + 1
        
        # Randomly select a road segment from the dataframe
        # weights='length' ensures we pick longer main roads more often
        random_road = roads.sample(n=1, weights='length').iloc[0]
        
        # Get road metadata
        street_name = random_road.get('name', 'Unnamed Road')
        if isinstance(street_name, list): 
            street_name = street_name[0]
            
        # 3. CRITICAL: Place sensor EXACTLY on the road line
        # Project the road back to Lat/Lon for the sensor output
        line_geometry = random_road.geometry
        random_position = random.random() * line_geometry.length
        point_meters = line_geometry.interpolate(random_position)
        
        # Convert point back to Lat/Lon for output
        import geopandas as gpd
        point_gpd = gpd.GeoSeries([point_meters], crs=_MILWAUKEE_GRAPH.graph['crs'])
        point_latlon = point_gpd.to_crs(epsg=4326).iloc[0]
        
        lat = point_latlon.y
        lon = point_latlon.x

        # Determine noise profile based on highway type
        highway_type = random_road.get('highway', 'residential')
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        if highway_type in ['motorway', 'trunk', 'primary']:
            base_noise = random.randint(85, 95)  # Loud
            neighborhood = "Highway / Major Arterial"
        elif highway_type in ['secondary', 'tertiary']:
            base_noise = random.randint(70, 85)  # Moderate
            neighborhood = "City Street"
        else:
            base_noise = random.randint(50, 65)  # Quiet
            neighborhood = "Residential Area"

        sensor = {
            'sensor_id': f'SENSOR_{sensor_num:03d}',
            'street_name': street_name,
            'neighborhood': neighborhood,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'base_noise': base_noise,
            'variance': random.randint(5, 12),
            'road_type': highway_type
        }
        
        sensors.append(sensor)

    logger.info(f"✅ Generated {len(sensors)} sensors bound to OSM road network.")
    return sensors

def generate_sentiment_text(noise_level, street_name, neighborhood, hour):
    """
    Generate synthetic sentiment text based on noise level.
    Includes ~25 templates covering various intensity levels.
    """
    # Critical noise (90+ dB) - Very negative
    if noise_level >= 90:
        templates = [
            f"Unbearable noise pollution on {street_name}. Traffic is extremely loud and stressful.",
            f"Critical noise levels in {neighborhood}. Can't stand the constant honking and engine noise.",
            f"Terrible conditions at {street_name}. Heavy traffic making it impossible to think clearly.",
            f"Avoid {street_name} at all costs! Noise is deafening and hazardous to health.",
            f"Extremely loud traffic in {neighborhood}. This level of noise pollution is unacceptable.",
        ]
    
    # High noise (80-89 dB) - Negative
    elif noise_level >= 80:
        templates = [
            f"Heavy traffic on {street_name}. Very noisy and uncomfortable conditions.",
            f"High noise levels in {neighborhood} making the commute unpleasant.",
            f"Too much traffic noise at {street_name}. Getting worse during rush hours.",
            f"Loud and congested conditions in {neighborhood}. Traffic is overwhelming.",
            f"Significant noise pollution on {street_name}. Difficult to handle daily.",
        ]
    
    # Moderate noise (70-79 dB) - Neutral/Slightly negative
    elif noise_level >= 70:
        templates = [
            f"Moderate traffic on {street_name}. Noise levels are acceptable but noticeable.",
            f"Typical urban noise in {neighborhood}. Nothing unusual for this area.",
            f"Normal traffic conditions at {street_name}. Manageable noise levels.",
            f"Average noise in {neighborhood} today. Expected for this time of day.",
            f"Regular traffic flow on {street_name}. Noise is within normal range.",
        ]
    
    # Moderate-low noise (60-69 dB) - Neutral/Slightly positive  
    elif noise_level >= 60:
        templates = [
            f"Calm conditions on {street_name}. Traffic noise is reasonable and bearable.",
            f"Pleasant drive through {neighborhood}. Noise levels are comfortable.",
            f"Good traffic flow at {street_name}. Noise is not a concern here.",
            f"Smooth commute in {neighborhood}. Much quieter than usual.",
            f"Acceptable noise levels on {street_name}. No complaints today.",
        ]
    
    # Low noise (<60 dB) - Very positive
    else:
        templates = [
            f"Quiet and peaceful route through {street_name}. Very low traffic noise.",
            f"Excellent conditions in {neighborhood}! Barely any noise pollution.",
            f"Highly recommended route via {street_name}. Calm and relaxing drive.",
            f"Wonderful! {neighborhood} is so peaceful and quiet today.",
            f"Perfect conditions on {street_name}. Minimal traffic and very low noise.",
        ]
    
    # Add time-specific context during rush hours
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        if noise_level >= 85:
            templates.append(f"Rush hour chaos on {street_name}! Noise is intolerable right now.")
        elif noise_level >= 70:
            templates.append(f"Rush hour traffic in {neighborhood}. Expect higher noise levels.")
    
    return random.choice(templates)

class NoiseProducer:
    """Kafka producer that simulates noise sensors"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='noise-readings', sensors=None):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.message_count = 0
        self.sensors = sensors or []
        
    def connect(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            logger.info(f"✅ Connected to Kafka at {self.bootstrap_servers}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    def generate_noise_reading(self, sensor):
        current_hour = datetime.now().hour
        
        # Rush hour adjustment (Milwaukee / Generic US City)
        # Morning: 7-9 AM, Evening: 4-6 PM
        rush_hour_boost = 0
        if (7 <= current_hour <= 9) or (16 <= current_hour <= 18):
            rush_hour_boost = random.randint(3, 8)
        
        # Night time reduction
        night_reduction = 0
        if current_hour >= 22 or current_hour <= 5:
            night_reduction = random.randint(5, 15)
        
        base = sensor['base_noise']
        variance = random.uniform(-sensor['variance'], sensor['variance'])
        noise_level = base + variance + rush_hour_boost - night_reduction
        
        # Ensure realistic range
        noise_level = max(40, min(120, noise_level))
        
        sentiment_text = generate_sentiment_text(
            noise_level,
            sensor['street_name'],
            sensor['neighborhood'],
            current_hour
        )
        
        reading = {
            'timestamp': datetime.now().isoformat(),
            'sensor_id': sensor['sensor_id'],
            'street_name': sensor['street_name'],
            'neighborhood': sensor['neighborhood'],
            'latitude': sensor['latitude'],
            'longitude': sensor['longitude'],
            'noise_level': round(noise_level, 1),
            'sentiment_text': sentiment_text,
            'unit': 'dB',
            'metadata': {
                'hour': current_hour,
                'city': 'Milwaukee',
                'country': 'USA'
            }
        }
        return reading
    
    def send_reading(self, reading):
        try:
            # Send message
            future = self.producer.send(
                self.topic,
                key=reading['sensor_id'],
                value=reading
            )
            
            # Wait for metadata (needed for logging partition/offset)
            record_metadata = future.get(timeout=10)
            
            self.message_count += 1
            
            # Restored Detailed Logging
            logger.info(
                f"📤 Sent: {reading['sensor_id']} | "
                f"{reading['street_name']} | "
                f"{reading['noise_level']} dB | "
                f"Partition: {record_metadata.partition} | "
                f"Offset: {record_metadata.offset}"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send: {e}")
            return False
    
    def run(self, interval=5):
        if not self.connect(): return
        
        logger.info(f"🚀 Starting Milwaukee noise simulation...")
        logger.info(f"📊 Monitoring {len(self.sensors)} sensors")
        
        try:
            while True:
                for sensor in self.sensors:
                    reading = self.generate_noise_reading(sensor)
                    self.send_reading(reading)
                
                logger.info(f"✅ Batch complete. Total messages: {self.message_count}")
                logger.info("-" * 80)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping producer")
        finally:
            if self.producer: self.producer.close()

def main():
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
    TOPIC_NAME = os.getenv('TOPIC_NAME', 'noise-readings')
    SEND_INTERVAL = int(os.getenv('SEND_INTERVAL', '5'))
    NUM_SENSORS = int(os.getenv('NUM_SENSORS', '10'))
    
    logger.info("🔧 Initializing Milwaukee Sensor Network...")
    sensors = generate_sensors(NUM_SENSORS)
    
    if not sensors:
        logger.error("❌ No sensors generated. Exiting.")
        return

    producer = NoiseProducer(
        bootstrap_servers=KAFKA_BROKER,
        topic=TOPIC_NAME,
        sensors=sensors
    )
    producer.run(interval=SEND_INTERVAL)

if __name__ == "__main__":
    main()