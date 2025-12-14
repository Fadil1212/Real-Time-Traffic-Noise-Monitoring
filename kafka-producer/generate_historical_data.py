#!/usr/bin/env python3
"""
Historical Data Generator - Fast Mode
Generates 24 hours of noise data in 30 minutes
With 50+ sensors across Dire Dawa
"""

import json
import time
import random
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dire Dawa Neighborhoods
NEIGHBORHOODS = [
    'Kezira', 'Railway District', 'Commercial District', 'Dechatu',
    'Old City', 'Sabian', 'Melka Jebdu', 'Legehare', 'Gendakore',
    'Ganda Harla', 'Ganda Hassandana', 'Ganda Gora', 'Ganda Teklehaimanot',
    'Kafira', 'Tsehay Kebele', 'Adde Harege', 'Ashewa', 'Ganda Erer',
    'Ganda Sanbate', 'Ganda Jaliyos', 'University Area', 'Hospital Area',
    'Stadium Area', 'Market Area', 'Industrial Area'
]

# Streets
STREETS = [
    'Mewlid Road', 'Addis Ketema', 'Railway Station Road', 'Market Street',
    'Industrial Road', 'University Avenue', 'Dechatu Road', 'Sabian Street',
    'Legehare Road', 'Melka Jebdu Avenue', 'Gendakore Street', 'Kafira Road',
    'Old City Road', 'Commercial Street', 'Kezira Avenue', 'Stadium Road',
    'Hospital Road', 'Ashewa Street', 'Airport Road', 'Harar Road',
    'Addis Ababa Road', 'City Center', 'Bus Terminal Road', 'Freedom Square',
    'Unity Street', 'Peace Avenue', 'Independence Road', 'Victory Street',
    'Liberty Road', 'Progress Avenue', 'Development Street', 'Main Boulevard',
    'Park Road', 'Garden Street', 'River Road', 'Hill Avenue',
    'Valley Street', 'Mountain Road', 'Forest Edge', 'Lake Road',
    'Desert Avenue', 'Oasis Street', 'Spring Road', 'Summer Avenue',
    'Autumn Street', 'Winter Road', 'Morning Avenue', 'Evening Street',
    'Sunrise Road', 'Sunset Avenue', 'Noon Street', 'Midnight Road'
]

def generate_sensors(num_sensors):
    """Generate sensor configurations"""
    sensors = []
    
    # Dire Dawa bounding box
    lat_min, lat_max = 9.550, 9.630
    lon_min, lon_max = 41.840, 41.900
    
    logger.info(f"🏗️  Generating {num_sensors} sensors...")
    
    for i in range(num_sensors):
        sensor_num = i + 1
        
        # Random location
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        
        # Random neighborhood and street
        neighborhood = random.choice(NEIGHBORHOODS)
        street = random.choice(STREETS)
        
        # Base noise varies by area type
        if 'Commercial' in neighborhood or 'Market' in neighborhood or 'Industrial' in neighborhood:
            base_noise = random.randint(80, 95)  # Busy areas
        elif 'Railway' in neighborhood or 'Station' in neighborhood or 'Terminal' in neighborhood:
            base_noise = random.randint(85, 100)  # Very busy
        elif 'Dechatu' in neighborhood or 'Forest' in neighborhood or 'Park' in neighborhood:
            base_noise = random.randint(50, 65)  # Quiet areas
        else:
            base_noise = random.randint(65, 80)  # Residential
        
        variance = random.randint(5, 10)
        
        sensor = {
            'sensor_id': f'SENSOR_{sensor_num:03d}',
            'street_name': street,
            'neighborhood': neighborhood,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'base_noise': base_noise,
            'variance': variance
        }
        
        sensors.append(sensor)
    
    logger.info(f"✅ Generated {len(sensors)} sensors")
    return sensors

class HistoricalDataGenerator:
    """Generate historical data fast"""
    
    def __init__(self, bootstrap_servers, topic, sensors):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.sensors = sensors
        self.producer = None
        self.message_count = 0
    
    def connect(self):
        """Connect to Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks=1,  # Faster - only wait for leader
                retries=1,
                compression_type='gzip',  # Compress for speed
                batch_size=32768,  # Larger batches
                linger_ms=10  # Wait 10ms to batch messages
            )
            logger.info(f"✅ Connected to Kafka")
            return True
        except Exception as e:
            logger.error(f"❌ Kafka connection failed: {e}")
            return False
    
    def generate_noise_reading(self, sensor, timestamp):
        """Generate realistic noise reading for specific timestamp"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Time-based patterns
        rush_hour_boost = 0
        if (6 <= hour <= 8) or (12 <= hour <= 14) or (17 <= hour <= 19):
            rush_hour_boost = random.randint(3, 10)
        
        night_reduction = 0
        if hour >= 22 or hour <= 5:
            night_reduction = random.randint(8, 18)
        
        # Weekend reduction
        weekend_reduction = 0
        if day_of_week in [5, 6]:  # Saturday, Sunday
            weekend_reduction = random.randint(5, 12)
        
        # Calculate noise
        base = sensor['base_noise']
        variance = random.uniform(-sensor['variance'], sensor['variance'])
        noise_level = base + variance + rush_hour_boost - night_reduction - weekend_reduction
        
        # Realistic range
        noise_level = max(40, min(120, noise_level))
        
        reading = {
            'timestamp': timestamp.isoformat(),
            'sensor_id': sensor['sensor_id'],
            'street_name': sensor['street_name'],
            'neighborhood': sensor['neighborhood'],
            'latitude': sensor['latitude'],
            'longitude': sensor['longitude'],
            'noise_level': round(noise_level, 1),
            'unit': 'dB',
            'metadata': {
                'hour': hour,
                'is_rush_hour': rush_hour_boost > 0,
                'is_night': night_reduction > 0,
                'city': 'Dire Dawa',
                'country': 'Ethiopia'
            }
        }
        
        return reading
    
    def send_batch(self, readings):
        """Send batch of readings"""
        try:
            for reading in readings:
                self.producer.send(
                    self.topic,
                    key=reading['sensor_id'],
                    value=reading
                )
                self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"❌ Send failed: {e}")
            return False
    
    def generate_24h_data(self, target_duration_minutes=30):
        """
        Generate 24 hours of data in target_duration_minutes
        
        Args:
            target_duration_minutes: How long to take (default 30 minutes)
        """
        if not self.connect():
            return
        
        # Calculate parameters
        hours_to_generate = 24
        readings_per_sensor = hours_to_generate * 12  # Every 5 minutes = 12 per hour
        total_readings = readings_per_sensor * len(self.sensors)
        
        target_duration_seconds = target_duration_minutes * 60
        delay_between_batches = target_duration_seconds / readings_per_sensor
        
        logger.info("=" * 70)
        logger.info("🚀 HISTORICAL DATA GENERATOR")
        logger.info("=" * 70)
        logger.info(f"📊 Sensors: {len(self.sensors)}")
        logger.info(f"⏱️  Generating: {hours_to_generate} hours of data")
        logger.info(f"⏱️  Target time: {target_duration_minutes} minutes")
        logger.info(f"📈 Readings per sensor: {readings_per_sensor}")
        logger.info(f"📈 Total readings: {total_readings:,}")
        logger.info(f"⚡ Delay per batch: {delay_between_batches:.3f} seconds")
        logger.info("=" * 70)
        
        # Start timestamp (24 hours ago)
        start_time = datetime.now() - timedelta(hours=hours_to_generate)
        current_timestamp = start_time
        
        # Time increment (5 minutes)
        time_increment = timedelta(minutes=5)
        
        batch_count = 0
        start_generation = time.time()
        
        try:
            # Generate readings
            for reading_num in range(readings_per_sensor):
                batch = []
                
                # Generate reading for each sensor at this timestamp
                for sensor in self.sensors:
                    reading = self.generate_noise_reading(sensor, current_timestamp)
                    batch.append(reading)
                
                # Send batch
                self.send_batch(batch)
                batch_count += 1
                
                # Progress update
                if batch_count % 10 == 0:
                    progress = (batch_count / readings_per_sensor) * 100
                    elapsed = time.time() - start_generation
                    remaining = (elapsed / batch_count) * (readings_per_sensor - batch_count)
                    
                    logger.info(
                        f"📊 Progress: {progress:.1f}% | "
                        f"Timestamp: {current_timestamp.strftime('%Y-%m-%d %H:%M')} | "
                        f"Sent: {self.message_count:,} | "
                        f"ETA: {remaining/60:.1f} min"
                    )
                
                # Move to next timestamp
                current_timestamp += time_increment
                
                # Delay to spread over target duration
                time.sleep(delay_between_batches)
            
            # Flush remaining messages
            logger.info("🔄 Flushing messages...")
            self.producer.flush()
            
            elapsed_total = time.time() - start_generation
            
            logger.info("=" * 70)
            logger.info("✅ GENERATION COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"📊 Total readings: {self.message_count:,}")
            logger.info(f"⏱️  Time taken: {elapsed_total/60:.1f} minutes")
            logger.info(f"⚡ Rate: {self.message_count/elapsed_total:.0f} msg/sec")
            logger.info(f"📅 Data range: {start_time.strftime('%Y-%m-%d %H:%M')} to {current_timestamp.strftime('%Y-%m-%d %H:%M')}")
            logger.info("=" * 70)
            
        except KeyboardInterrupt:
            logger.info("\n⏹️  Stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.producer:
                self.producer.close()
                logger.info("✅ Producer closed")

def main():
    """Main function"""
    import os
    
    # Configuration
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:29092')
    TOPIC_NAME = os.getenv('TOPIC_NAME', 'noise-readings')
    NUM_SENSORS = int(os.getenv('NUM_SENSORS', '50'))
    TARGET_DURATION = int(os.getenv('TARGET_DURATION', '30'))  # Minutes
    
    logger.info("🔧 Configuration:")
    logger.info(f"   Kafka: {KAFKA_BROKER}")
    logger.info(f"   Topic: {TOPIC_NAME}")
    logger.info(f"   Sensors: {NUM_SENSORS}")
    logger.info(f"   Duration: {TARGET_DURATION} minutes")
    logger.info("")
    
    # Generate sensors
    sensors = generate_sensors(NUM_SENSORS)
    
    # Create generator
    generator = HistoricalDataGenerator(
        bootstrap_servers=KAFKA_BROKER,
        topic=TOPIC_NAME,
        sensors=sensors
    )
    
    # Generate data
    generator.generate_24h_data(target_duration_minutes=TARGET_DURATION)

if __name__ == "__main__":
    main()