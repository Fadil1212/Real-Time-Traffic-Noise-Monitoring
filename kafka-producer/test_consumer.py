#!/usr/bin/env python3
"""
Kafka Consumer Test Script
Reads messages from noise-readings topic to verify producer is working
"""

import json
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_consumer(bootstrap_servers=None, topic='noise-readings', max_messages=10):
    """
    Test Kafka consumer - reads messages from topic
    
    Args:
        bootstrap_servers: Kafka broker address
        topic: Topic to consume from
        max_messages: Maximum messages to read (None = infinite)
    """
    import os
    
    # Read from environment variable if not provided
    if bootstrap_servers is None:
        bootstrap_servers = os.getenv('KAFKA_BROKER', 'localhost:9092')
    
    try:
        logger.info(f"🔍 Connecting to Kafka at {bootstrap_servers}")
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',  # Start from latest message
            enable_auto_commit=True,
            group_id='test-consumer-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        logger.info(f"✅ Connected to Kafka!")
        logger.info(f"📖 Reading from topic: {topic}")
        logger.info(f"⏳ Waiting for messages... (Press Ctrl+C to stop)\n")
        
        message_count = 0
        
        for message in consumer:
            message_count += 1
            
            # Parse message
            data = message.value
            
            # Display message info
            logger.info(f"📩 Message #{message_count}")
            logger.info(f"   Partition: {message.partition} | Offset: {message.offset}")
            logger.info(f"   Sensor: {data.get('sensor_id')}")
            logger.info(f"   Location: {data.get('street_name')}, {data.get('neighborhood')}")
            logger.info(f"   Noise Level: {data.get('noise_level')} {data.get('unit')}")
            logger.info(f"   Timestamp: {data.get('timestamp')}")
            logger.info(f"   Coordinates: ({data.get('latitude')}, {data.get('longitude')})")
            logger.info("-" * 80)
            
            # Check if we should stop
            if max_messages and message_count >= max_messages:
                logger.info(f"✅ Read {max_messages} messages. Stopping.")
                break
        
        consumer.close()
        logger.info(f"✅ Consumer closed. Total messages read: {message_count}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping consumer (Ctrl+C pressed)")
    except KafkaError as e:
        logger.error(f"❌ Kafka error: {e}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Allow custom number of messages
    max_msgs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    test_consumer(max_messages=max_msgs)