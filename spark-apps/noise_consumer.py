"""
Milwaukee Traffic Noise Monitoring - Integrated Spark Streaming Consumer
Combines:
1. Real-time streaming (processes Kafka messages)
2. Periodic route analysis using ACTUAL ROADS (every 5 minutes)

Now generates route segments that follow real Milwaukee asphalt roads!
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, current_timestamp, window, avg, max as spark_max, min as spark_min, count as spark_count,
    round as spark_round, when, lit, to_timestamp, expr, array, concat_ws
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, IntegerType, BooleanType
)
import logging
import time

# Import road network manager
from road_network import road_network

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka message schema
NOISE_SCHEMA = StructType([
    StructField("timestamp", StringType(), True),
    StructField("sensor_id", StringType(), True),
    StructField("street_name", StringType(), True),
    StructField("neighborhood", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("noise_level", DoubleType(), True),
    StructField("sentiment_text", StringType(), True),
    StructField("unit", StringType(), True),
    StructField("metadata", StructType([
        StructField("hour", IntegerType(), True),
        StructField("is_rush_hour", BooleanType(), True),
        StructField("is_night", BooleanType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True)
    ]), True)
])

# Global variable for route analysis timing
last_route_analysis = time.time()
ROUTE_ANALYSIS_INTERVAL = 300  # 5 minutes in seconds

def create_spark_session():
    """Initialize Spark session for both streaming and batch"""
    logger.info("🚀 Creating unified Spark session...")
    
    spark = SparkSession.builder \
        .appName("MilwaukeeIntegratedProcessor") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.postgresql:postgresql:42.7.1") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info("✅ Unified Spark session created")
    return spark

def read_kafka_stream(spark, kafka_broker, topic):
    """Connect to Kafka and read streaming data"""
    logger.info(f"📖 Connecting to Kafka: {kafka_broker}")
    
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_broker) \
        .option("subscribe", topic) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .option("maxOffsetsPerTrigger", "100") \
        .load()
    
    logger.info("✅ Connected to Kafka stream")
    return df

def parse_and_validate(kafka_df):
    """Parse JSON and validate data"""
    logger.info("🔍 Parsing and validating messages...")
    
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), NOISE_SCHEMA).alias("data"),
        col("timestamp").alias("kafka_timestamp")
    ).select("data.*")
    
    parsed_df = parsed_df.withColumn(
        "reading_time",
        to_timestamp(col("timestamp"))
    )
    
    validated_df = parsed_df.filter(
        (col("noise_level").isNotNull()) &
        (col("noise_level") >= 40) &
        (col("noise_level") <= 120) &
        (col("sensor_id").isNotNull()) &
        (col("neighborhood").isNotNull())
    )
    
    logger.info("✅ Parsing and validation complete")
    return validated_df

def enrich_data(df):
    """Enrich data with stress levels and alerts"""
    logger.info("🔧 Enriching data...")
    
    enriched_df = df.withColumn(
        "stress_level",
        when(col("noise_level") < 55, lit(1))
        .when(col("noise_level") < 70, lit(2))
        .when(col("noise_level") < 85, lit(3))
        .when(col("noise_level") < 100, lit(4))
        .otherwise(lit(5))
    )
    
    enriched_df = enriched_df.withColumn(
        "is_critical",
        when(col("noise_level") >= 90, lit(True)).otherwise(lit(False))
    )
    
    enriched_df = enriched_df.withColumn(
        "alert_level",
        when(col("noise_level") < 70, lit("low"))
        .when(col("noise_level") < 85, lit("moderate"))
        .when(col("noise_level") < 95, lit("high"))
        .otherwise(lit("critical"))
    )
    
    enriched_df = enriched_df.withColumn("processed_at", current_timestamp())
    
    logger.info("✅ Data enrichment complete")
    return enriched_df

def calculate_aggregations(df):
    """Calculate real-time aggregations"""
    logger.info("📊 Calculating aggregations...")
    
    sensor_agg = df \
        .withWatermark("reading_time", "15 minutes") \
        .groupBy(
            window(col("reading_time"), "10 minutes", "5 minutes"),
            col("sensor_id"),
            col("street_name"),
            col("neighborhood")
        ) \
        .agg(
            avg("noise_level").alias("avg_noise"),
            spark_max("noise_level").alias("max_noise"),
            spark_min("noise_level").alias("min_noise"),
            spark_count("*").alias("reading_count"),
            avg("stress_level").alias("avg_stress")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("sensor_id"),
            col("street_name"),
            col("neighborhood"),
            spark_round(col("avg_noise"), 1).alias("avg_noise"),
            spark_round(col("max_noise"), 1).alias("max_noise"),
            spark_round(col("min_noise"), 1).alias("min_noise"),
            col("reading_count"),
            spark_round(col("avg_stress"), 1).alias("avg_stress")
        )
    
    logger.info("✅ Aggregations calculated")
    return sensor_agg

def generate_road_bound_segments(sensor_readings):
    """
    Generate route segments that follow ACTUAL Milwaukee roads.
    This is the NEW road-bound implementation!
    
    Args:
        sensor_readings: List of dicts with sensor data
    
    Returns:
        List of route segment dicts ready for database insertion
    """
    if not sensor_readings or len(sensor_readings) < 2:
        logger.warning("Not enough sensor readings to generate segments")
        return []
    
    # Initialize road network (only happens once, then cached)
    if not road_network.initialize():
        logger.error("❌ Failed to initialize Milwaukee road network")
        logger.warning("⚠️ Falling back to simple straight-line segments")
        return generate_segments_fallback(sensor_readings)
    
    route_segments = []
    processed_pairs = set()
    
    try:
        # Import clustering tools
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        # Prepare coordinates for clustering
        coords = np.array([[s['latitude'], s['longitude']] for s in sensor_readings])
        
        # DBSCAN clustering to find nearby sensor groups
        # eps=0.02 ≈ 2km in lat/lon degrees at Milwaukee latitude
        clustering = DBSCAN(eps=0.02, min_samples=2).fit(coords)
        
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        logger.info(f"🗺️ Found {num_clusters} sensor clusters for road routing")
        
        # Process each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip outliers
                continue
            
            # Get sensors in this cluster
            cluster_sensors = [
                sensor_readings[i] 
                for i in range(len(sensor_readings)) 
                if clustering.labels_[i] == cluster_id
            ]
            
            logger.debug(f"   Cluster {cluster_id}: {len(cluster_sensors)} sensors")
            
            # Connect sensors along actual roads
            for i, sensor1 in enumerate(cluster_sensors):
                for sensor2 in cluster_sensors[i+1:]:
                    # Avoid duplicate pairs
                    pair_id = tuple(sorted([sensor1['sensor_id'], sensor2['sensor_id']]))
                    if pair_id in processed_pairs:
                        continue
                    processed_pairs.add(pair_id)
                    
                    # Get actual road route between sensors
                    route_coords = road_network.get_route_between_sensors(
                        sensor1['latitude'], sensor1['longitude'],
                        sensor2['latitude'], sensor2['longitude']
                    )
                    
                    if not route_coords:
                        logger.debug(f"   No route: {sensor1['sensor_id']} → {sensor2['sensor_id']}")
                        continue
                    
                    # Break route into segments
                    segments = road_network.generate_route_segments(route_coords)
                    
                    if not segments:
                        continue
                    
                    # Calculate noise metrics
                    avg_noise = (sensor1['noise_level'] + sensor2['noise_level']) / 2
                    max_noise = max(sensor1['noise_level'], sensor2['noise_level'])
                    min_noise = min(sensor1['noise_level'], sensor2['noise_level'])
                    
                    # Determine noise category
                    if avg_noise >= 90:
                        noise_category = 'critical'
                    elif avg_noise >= 80:
                        noise_category = 'high'
                    elif avg_noise >= 70:
                        noise_category = 'moderate'
                    else:
                        noise_category = 'low'
                    
                    # Get street name
                    street_name = sensor1.get('street_name') or sensor2.get('street_name') or 'Unnamed Road'
                    if street_name == 'Unnamed Road':
                        # Try to get from road network
                        mid_lat = (sensor1['latitude'] + sensor2['latitude']) / 2
                        mid_lon = (sensor1['longitude'] + sensor2['longitude']) / 2
                        street_name = road_network.get_street_name_for_segment(mid_lat, mid_lon)
                    
                    neighborhood = sensor1.get('neighborhood') or sensor2.get('neighborhood') or 'Milwaukee'
                    
                    # Create segment entries
                    for segment in segments:
                        route_segments.append({
                            'timestamp': time.time(),
                            'street_name': street_name,
                            'neighborhood': neighborhood,
                            'segment_start_lat': segment['segment_start_lat'],
                            'segment_start_lon': segment['segment_start_lon'],
                            'segment_end_lat': segment['segment_end_lat'],
                            'segment_end_lon': segment['segment_end_lon'],
                            'center_lat': segment['center_lat'],
                            'center_lon': segment['center_lon'],
                            'avg_noise_level': round(avg_noise, 1),
                            'max_noise_level': round(max_noise, 1),
                            'min_noise_level': round(min_noise, 1),
                            'noise_category': noise_category,
                            'segment_length_meters': segment['segment_length_meters'],
                            'sensors_in_segment': 2,
                            'bearing': 0.0
                        })
        
        logger.info(f"✅ Generated {len(route_segments)} ROAD-BOUND segments from {len(sensor_readings)} sensors")
        return route_segments
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.warning("⚠️ Falling back to simple segments")
        return generate_segments_fallback(sensor_readings)
    except Exception as e:
        logger.error(f"❌ Error in road-bound routing: {e}")
        import traceback
        traceback.print_exc()
        logger.warning("⚠️ Falling back to simple segments")
        return generate_segments_fallback(sensor_readings)

def generate_segments_fallback(sensor_readings):
    """
    Fallback: Generate simple straight-line segments.
    Used if road network initialization fails.
    """
    logger.warning("⚠️ Using FALLBACK method - segments will be straight lines")
    
    route_segments = []
    
    for i in range(len(sensor_readings)):
        for j in range(i + 1, min(i + 3, len(sensor_readings))):
            sensor1 = sensor_readings[i]
            sensor2 = sensor_readings[j]
            
            # Calculate straight-line distance
            lat_diff = abs(sensor2['latitude'] - sensor1['latitude'])
            lon_diff = abs(sensor2['longitude'] - sensor1['longitude'])
            distance = ((lat_diff * 111000) ** 2 + (lon_diff * 111000) ** 2) ** 0.5
            
            # Only connect if within 1km
            if distance > 1000:
                continue
            
            avg_noise = (sensor1['noise_level'] + sensor2['noise_level']) / 2
            
            if avg_noise >= 90:
                noise_category = 'critical'
            elif avg_noise >= 80:
                noise_category = 'high'
            elif avg_noise >= 70:
                noise_category = 'moderate'
            else:
                noise_category = 'low'
            
            route_segments.append({
                'timestamp': time.time(),
                'street_name': sensor1.get('street_name', 'Unknown'),
                'neighborhood': sensor1.get('neighborhood', 'Milwaukee'),
                'segment_start_lat': sensor1['latitude'],
                'segment_start_lon': sensor1['longitude'],
                'segment_end_lat': sensor2['latitude'],
                'segment_end_lon': sensor2['longitude'],
                'center_lat': (sensor1['latitude'] + sensor2['latitude']) / 2,
                'center_lon': (sensor1['longitude'] + sensor2['longitude']) / 2,
                'avg_noise_level': round(avg_noise, 1),
                'max_noise_level': round(max(sensor1['noise_level'], sensor2['noise_level']), 1),
                'min_noise_level': round(min(sensor1['noise_level'], sensor2['noise_level']), 1),
                'noise_category': noise_category,
                'segment_length_meters': round(distance, 2),
                'sensors_in_segment': 2,
                'bearing': 0.0
            })
    
    logger.info(f"Generated {len(route_segments)} fallback segments")
    return route_segments

def run_route_analysis(spark, jdbc_url, db_properties):
    """
    Run route analysis with ROAD-BOUND routing
    Called periodically from streaming context
    """
    global last_route_analysis
    
    logger.info("=" * 80)
    logger.info("🗺️  ROAD-BOUND ROUTE ANALYSIS - Periodic Update")
    logger.info("=" * 80)
    
    try:
        # Read recent noise data (last 1 hour)
        query = """
        (SELECT 
            sensor_id,
            street_name,
            neighborhood,
            latitude,
            longitude,
            noise_level,
            timestamp
         FROM noise_readings
         WHERE timestamp > NOW() - INTERVAL '1 hour'
        ) AS recent_noise
        """
        
        noise_df = spark.read.jdbc(
            url=jdbc_url,
            table=query,
            properties=db_properties
        )
        
        count = noise_df.count()
        logger.info(f"📖 Read {count} noise readings from last hour")
        
        if count < 10:
            logger.warning("⚠️  Not enough data for route analysis (need 10+)")
            return
        
        # Aggregate by sensor
        sensor_agg = noise_df.groupBy("sensor_id", "street_name", "neighborhood", "latitude", "longitude") \
            .agg(
                avg("noise_level").alias("avg_noise"),
                spark_max("noise_level").alias("max_noise"),
                spark_min("noise_level").alias("min_noise"),
                spark_count("*").alias("reading_count")
            ) \
            .withColumn("avg_noise", spark_round(col("avg_noise"), 2)) \
            .withColumn("max_noise", spark_round(col("max_noise"), 2)) \
            .withColumn("min_noise", spark_round(col("min_noise"), 2))
        
        sensor_count = sensor_agg.count()
        logger.info(f"✅ Aggregated {sensor_count} unique sensors")
        
        # Convert to list of dicts for road routing
        sensor_readings = []
        for row in sensor_agg.collect():
            sensor_readings.append({
                'sensor_id': row['sensor_id'],
                'street_name': row['street_name'],
                'neighborhood': row['neighborhood'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'noise_level': row['avg_noise']
            })
        
        # Generate road-bound segments
        route_segments = generate_road_bound_segments(sensor_readings)
        
        if not route_segments:
            logger.warning("⚠️  No route segments generated")
            return
        
        # Convert to Spark DataFrame
        from datetime import datetime
        
        # Add timestamp to all segments
        for seg in route_segments:
            seg['timestamp'] = datetime.now()
        
        segments_df = spark.createDataFrame(route_segments)
        
        segment_count = segments_df.count()
        logger.info(f"✅ Created {segment_count} ROAD-BOUND route segments")
        
        # Write to PostgreSQL
        segments_df.write.jdbc(
            url=jdbc_url,
            table="route_noise_segments",
            mode="append",
            properties=db_properties
        )
        
        logger.info(f"✅ Wrote {segment_count} road-bound segments to database")
        
        # Show summary
        logger.info("📊 Segments by noise category:")
        segments_df.groupBy("noise_category").count().show()
        
        logger.info("=" * 80)
        logger.info("✅ Road-bound route analysis complete!")
        logger.info("=" * 80)
        
        last_route_analysis = time.time()
        
    except Exception as e:
        logger.error(f"❌ Error in route analysis: {e}")
        import traceback
        traceback.print_exc()

def write_to_postgres(df, table_name, jdbc_url, db_properties, spark_session, batch_id_prefix=""):
    """Write streaming data to PostgreSQL with periodic route analysis trigger"""
    global last_route_analysis
    
    def batch_writer(batch_df, batch_id):
        """Write each micro-batch and check if route analysis is due"""
        try:
            record_count = batch_df.count()
            if record_count > 0:
                batch_df.write \
                    .jdbc(url=jdbc_url, table=table_name, mode="append", properties=db_properties)
                logger.info(f"✅ {batch_id_prefix}Batch {batch_id}: Wrote {record_count} records to {table_name}")
            
            # Check if it's time for route analysis
            current_time = time.time()
            if current_time - last_route_analysis >= ROUTE_ANALYSIS_INTERVAL:
                logger.info(f"⏰ 5 minutes elapsed - triggering ROAD-BOUND route analysis...")
                run_route_analysis(spark_session, jdbc_url, db_properties)
                
        except Exception as e:
            logger.error(f"❌ {batch_id_prefix}Batch {batch_id}: Error: {e}")
    
    return batch_writer

def main():
    """Main integrated streaming + periodic batch processing"""
    logger.info("=" * 80)
    logger.info("🇺🇸 Milwaukee Integrated Processor - ROAD-BOUND EDITION")
    logger.info("   - Streaming: Continuous noise processing")
    logger.info("   - Batch: Road-bound route analysis every 5 minutes")
    logger.info("   - Routes follow ACTUAL Milwaukee asphalt roads!")
    logger.info("=" * 80)
    
    # Configuration
    KAFKA_BROKER = "kafka:29092"
    KAFKA_TOPIC = "noise-readings"
    POSTGRES_HOST = "postgres"
    POSTGRES_PORT = "5432"
    POSTGRES_DB = "traffic_noise_db"
    POSTGRES_USER = "traffic_user"
    POSTGRES_PASSWORD = "traffic_pass"
    
    jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    db_properties = {
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "driver": "org.postgresql.Driver"
    }
    
    logger.info(f"📍 Kafka: {KAFKA_BROKER}")
    logger.info(f"📍 PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    logger.info(f"⏱️  Route analysis: Every {ROUTE_ANALYSIS_INTERVAL/60} minutes")
    logger.info(f"🗺️  Road network: Milwaukee, Wisconsin, USA")
    
    # Create unified Spark session
    spark = create_spark_session()
    
    try:
        # Streaming processing
        kafka_df = read_kafka_stream(spark, KAFKA_BROKER, KAFKA_TOPIC)
        validated_df = parse_and_validate(kafka_df)
        enriched_df = enrich_data(validated_df)
        
        # Prepare for database
        readings_df = enriched_df.select(
            col("reading_time").alias("timestamp"),
            col("sensor_id"),
            col("street_name"),
            col("neighborhood"),
            col("latitude"),
            col("longitude"),
            col("noise_level"),
            col("sentiment_text")
        )
        
        # Write with integrated route analysis trigger
        logger.info("🚀 Starting integrated stream processing...")
        readings_query = readings_df.writeStream \
            .foreachBatch(write_to_postgres(readings_df, "noise_readings", jdbc_url, db_properties, spark, "[READINGS] ")) \
            .outputMode("append") \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # Aggregations
        sensor_agg = calculate_aggregations(enriched_df)
        
        sensor_query = sensor_agg.writeStream \
            .foreachBatch(write_to_postgres(sensor_agg, "noise_aggregates", jdbc_url, db_properties, spark, "[AGG] ")) \
            .outputMode("append") \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # Console monitoring
        console_query = enriched_df \
            .select(
                col("reading_time"),
                col("sensor_id"),
                col("street_name"),
                col("noise_level"),
                col("alert_level")
            ) \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", "false") \
            .option("numRows", "5") \
            .trigger(processingTime="30 seconds") \
            .start()
        
        logger.info("=" * 80)
        logger.info("✅ ROAD-BOUND SYSTEM STARTED!")
        logger.info("=" * 80)
        logger.info("📊 Streaming: Processing noise data every 10 seconds")
        logger.info("🗺️  Route Analysis: ROAD-BOUND routing every 5 minutes")
        logger.info("🛣️  Heatmap will display on ACTUAL Milwaukee streets!")
        logger.info("⏸️  Press Ctrl+C to stop gracefully")
        logger.info("=" * 80)
        
        # Wait for termination
        spark.streams.awaitAnyTermination()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Gracefully stopping...")
        spark.stop()
        logger.info("✅ Stopped successfully")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()