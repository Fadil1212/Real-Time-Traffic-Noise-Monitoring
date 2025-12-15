"""
Quick Test: Spark Kafka Consumer
Runs for 60 seconds to verify Spark can read from Kafka
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
)
import time

print("=" * 80)
print("🧪 Testing Spark Consumer - Reading from Kafka")
print("=" * 80)

# Schema
schema = StructType([
    StructField("timestamp", StringType()),
    StructField("sensor_id", StringType()),
    StructField("street_name", StringType()),
    StructField("neighborhood", StringType()),
    StructField("noise_level", DoubleType()),
    StructField("metadata", StructType([
        StructField("city", StringType()),
        StructField("country", StringType())
    ]))
])

# Create Spark session
print("\n🚀 Creating Spark session...")
spark = SparkSession.builder \
    .appName("SparkKafkaTest") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark session created\n")

# Read from Kafka
print("📖 Connecting to Kafka...")
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "noise-readings") \
    .option("startingOffsets", "latest") \
    .load()

print("✅ Connected to Kafka\n")

# Parse messages
print("🔍 Parsing messages...\n")
parsed = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select(
    col("data.sensor_id"),
    col("data.street_name"),
    col("data.neighborhood"),
    col("data.noise_level"),
    col("data.metadata.city"),
    col("data.metadata.country")
)

# Display
print("📊 Displaying messages for 60 seconds...")
print("=" * 80)

query = parsed.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="5 seconds") \
    .start()

# Run for 60 seconds
time.sleep(60)

print("\n" + "=" * 80)
print("✅ Test Complete! Spark can read from Kafka successfully.")
print("=" * 80)

query.stop()
spark.stop()