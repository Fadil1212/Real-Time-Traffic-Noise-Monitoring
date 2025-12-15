#!/bin/bash

# Traffic Noise Monitoring System - Startup Script
# This script helps you start, stop, and manage the entire system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "  Traffic Noise Monitoring System - Control Panel"
echo "=================================================="
echo ""

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running. Please start Docker Desktop.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker is running${NC}"
}

# Function to check system resources
check_resources() {
    echo -e "${YELLOW}Checking system resources...${NC}"
    
    # Check available memory (macOS/Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        total_mem=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        total_mem=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    if [ "$total_mem" -lt 8 ]; then
        echo -e "${RED}Warning: System has less than 8GB RAM. Performance may be degraded.${NC}"
    else
        echo -e "${GREEN}✓ Sufficient memory available ($total_mem GB)${NC}"
    fi
}

# Function to start all services
start_services() {
    echo ""
    echo -e "${YELLOW}Starting all services...${NC}"
    docker compose up -d
    
    echo ""
    echo -e "${GREEN}Services are starting up...${NC}"
    echo "This may take 1-2 minutes for all services to be ready."
    echo ""
    
    # Wait for services to be healthy
    echo "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    docker compose ps
    
    echo ""
    echo -e "${GREEN}=================================================="
    echo "  System Started Successfully!"
    echo "==================================================${NC}"
    echo ""
    echo "Access the following services:"
    echo "  📊 Streamlit Dashboard:  http://localhost:8501"
    echo "  📓 Jupyter Notebook:     http://localhost:8888"
    echo "  ⚡ Spark Master UI:      http://localhost:8080"
    echo "  💾 PostgreSQL:           localhost:5432"
    echo ""
    echo "Background Processing:"
    echo "  🎛️  Kafka Producer:      Sending data (every 5 sec)"
    echo "  🔄 Spark Consumer:      Processing data (every 10 sec)"
    echo "  🗺️  Route Analysis:      Integrated (every 5 min)"
    echo "  📊 Correlation Analyzer: Running (every 5 min)"
    echo ""
    echo "Database credentials:"
    echo "  Username: traffic_user"
    echo "  Password: traffic_pass"
    echo "  Database: traffic_noise_db"
    echo ""
    echo "View logs:"
    echo "  docker compose logs -f kafka-producer"
    echo "  docker compose logs -f spark-consumer"
    echo "  docker compose logs -f correlation-analyzer"
    echo ""
}

# Function to stop all services
stop_services() {
    echo ""
    echo -e "${YELLOW}Stopping all services...${NC}"
    docker compose down
    echo -e "${GREEN}✓ All services stopped${NC}"
}

# Function to start Kafka producer only
start_producer() {
    echo ""
    echo -e "${YELLOW}Starting Kafka Producer container...${NC}"
    
    # Check if Kafka is running
    if ! docker compose ps kafka | grep -q "Up"; then
        echo -e "${RED}Error: Kafka is not running!${NC}"
        echo "Please start all services first (option 1)"
        return 1
    fi
    
    docker compose up -d kafka-producer
    echo -e "${GREEN}✓ Producer started and streaming data${NC}"
    echo ""
    echo "The producer is now sending sensor data to Kafka every 5 seconds"
    echo ""
    echo "View live logs:"
    echo "  docker compose logs -f kafka-producer"
    echo ""
}

# Function to stop Kafka producer only
stop_producer() {
    echo ""
    echo -e "${YELLOW}Stopping Kafka Producer...${NC}"
    docker compose stop kafka-producer
    echo -e "${GREEN}✓ Producer stopped${NC}"
}

# Function to restart Kafka producer
restart_producer() {
    echo ""
    echo -e "${YELLOW}Restarting Kafka Producer...${NC}"
    docker compose restart kafka-producer
    echo -e "${GREEN}✓ Producer restarted${NC}"
}

# Function to start Spark consumer only
start_spark_consumer_service() {
    echo ""
    echo -e "${YELLOW}Starting Spark Consumer container...${NC}"
    
    # Check dependencies
    if ! docker compose ps kafka | grep -q "Up"; then
        echo -e "${RED}Error: Kafka is not running!${NC}"
        echo "Please start all services first (option 1)"
        return 1
    fi
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}Error: PostgreSQL is not running!${NC}"
        echo "Please start all services first (option 1)"
        return 1
    fi
    
    docker compose up -d spark-consumer
    echo -e "${GREEN}✓ Spark consumer started${NC}"
    echo ""
    echo "The consumer is now processing Kafka messages"
    echo ""
    echo "View live logs:"
    echo "  docker compose logs -f spark-consumer"
    echo ""
}

# Function to stop Spark consumer only
stop_spark_consumer_service() {
    echo ""
    echo -e "${YELLOW}Stopping Spark Consumer...${NC}"
    docker compose stop spark-consumer
    echo -e "${GREEN}✓ Spark consumer stopped${NC}"
}

# Function to restart Spark consumer
restart_spark_consumer_service() {
    echo ""
    echo -e "${YELLOW}Restarting Spark Consumer...${NC}"
    docker compose restart spark-consumer
    echo -e "${GREEN}✓ Spark consumer restarted${NC}"
}

# Function to test consumer
test_consumer() {
    echo ""
    echo -e "${YELLOW}Testing Kafka Consumer...${NC}"
    echo "Reading 10 messages from Kafka topic..."
    echo ""
    
    # Check if producer is running
    if ! docker compose ps kafka-producer | grep -q "Up"; then
        echo -e "${RED}Error: Producer is not running!${NC}"
        echo "Please start the producer first (option 3)"
        return 1
    fi
    
    # Run test consumer
    docker compose exec kafka-producer python test_consumer.py 10
    
    echo ""
    echo -e "${GREEN}✓ Consumer test complete${NC}"
}

# Function to view producer logs
view_producer_logs() {
    echo ""
    echo -e "${YELLOW}Showing producer logs (Ctrl+C to exit)...${NC}"
    docker compose logs -f kafka-producer
}

# Function to view Spark consumer logs
view_spark_logs() {
    echo ""
    echo -e "${YELLOW}Spark Consumer Status & Logs${NC}"
    echo ""
    
    if ! docker compose ps spark-consumer | grep -q "Up"; then
        echo -e "${RED}❌ Spark Consumer is not running!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Spark Consumer is running${NC}"
    echo ""
    
    # Check streaming activity
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 STREAMING ACTIVITY (Last 5 minutes)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check for recent batch processing
    recent_batches=$(docker compose logs --tail=100 spark-consumer 2>/dev/null | grep -c "Batch.*Wrote.*records" || echo "0")
    if [ "$recent_batches" -gt 0 ]; then
        echo -e "${GREEN}✅ Streaming is active ($recent_batches recent batches)${NC}"
        echo ""
        echo "Latest batches:"
        docker compose logs --tail=100 spark-consumer 2>/dev/null | grep "Batch.*Wrote.*records" | tail -5
    else
        echo -e "${RED}⚠️  No streaming batches detected in recent logs${NC}"
        echo "   Consumer may be starting up or waiting for data"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🗺️  ROUTE ANALYSIS STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check for route analysis activity
    route_analysis_count=$(docker compose logs spark-consumer 2>/dev/null | grep -c "ROUTE ANALYSIS - Periodic Update" || echo "0")
    
    if [ "$route_analysis_count" -gt 0 ]; then
        echo -e "${GREEN}✅ Route analysis has run $route_analysis_count time(s)${NC}"
        echo ""
        echo "Latest route analysis:"
        docker compose logs spark-consumer 2>/dev/null | grep -A 10 "ROUTE ANALYSIS - Periodic Update" | tail -15
        echo ""
        
        # Check database for route segments
        segment_count=$(docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
            "SELECT COUNT(*) FROM route_noise_segments;" 2>/dev/null | xargs || echo "0")
        
        if [ "$segment_count" -gt 0 ]; then
            echo -e "${GREEN}✅ Database has $segment_count route segments${NC}"
            echo ""
            echo "Route segments by category:"
            docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
                "SELECT noise_category, COUNT(*) as count 
                 FROM route_noise_segments 
                 WHERE timestamp > NOW() - INTERVAL '2 hours'
                 GROUP BY noise_category 
                 ORDER BY count DESC;" 2>/dev/null
        else
            echo -e "${YELLOW}⚠️  No route segments in database yet${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Route analysis has not run yet${NC}"
        echo ""
        echo "Route analysis triggers every 15 minutes."
        echo "If consumer just started, wait up to 15 minutes."
        echo ""
        
        # Calculate time since consumer started
        start_time=$(docker inspect spark-consumer --format='{{.State.StartedAt}}' 2>/dev/null)
        if [ -n "$start_time" ]; then
            echo "Consumer started at: $start_time"
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 DATABASE STATISTICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if docker compose ps postgres | grep -q "Up"; then
        noise_count=$(docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
            "SELECT COUNT(*) FROM noise_readings;" 2>/dev/null | xargs || echo "0")
        agg_count=$(docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
            "SELECT COUNT(*) FROM noise_aggregates;" 2>/dev/null | xargs || echo "0")
        route_count=$(docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
            "SELECT COUNT(*) FROM route_noise_segments;" 2>/dev/null | xargs || echo "0")
        
        echo "Noise Readings:    $noise_count"
        echo "Aggregates:        $agg_count"
        echo "Route Segments:    $route_count"
    else
        echo -e "${RED}❌ PostgreSQL is not running${NC}"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 FULL LOGS (Press Ctrl+C to exit)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${YELLOW}Showing live logs...${NC}"
    echo ""
    
    # Show live logs with both streaming and route analysis
    docker compose logs -f spark-consumer
}

# Function to check Kafka topic
check_kafka_topic() {
    echo ""
    echo -e "${YELLOW}Checking Kafka topic 'noise-readings'...${NC}"
    echo ""
    
    # List topics
    echo "Available topics:"
    docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
    
    echo ""
    echo "Message count in 'noise-readings' topic:"
    docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic noise-readings 2>/dev/null | awk -F':' '{sum+=$3} END {print "Total messages: " sum}'
    
    echo ""
    echo "Latest message:"
    docker exec kafka kafka-console-consumer \
        --bootstrap-server localhost:9092 \
        --topic noise-readings \
        --max-messages 1 \
        --timeout-ms 5000 2>/dev/null || echo "No messages available"
}

# Function to check Kafka topic
check_kafka_topic() {
    echo ""
    echo -e "${YELLOW}Checking Kafka topic 'noise-readings'...${NC}"
    echo ""
    
    # List topics
    echo "Available topics:"
    docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
    
    echo ""
    echo "Message count in 'noise-readings' topic:"
    docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic noise-readings 2>/dev/null | awk -F':' '{sum+=$3} END {print "Total messages: " sum}'
    
    echo ""
    echo "Latest message:"
    docker exec kafka kafka-console-consumer \
        --bootstrap-server localhost:9092 \
        --topic noise-readings \
        --max-messages 1 \
        --timeout-ms 5000 2>/dev/null || echo "No messages available"
}

# Function to test Spark consumer
test_spark_consumer() {
    echo ""
    echo -e "${YELLOW}Testing Spark Consumer...${NC}"
    echo ""
    
    # Check if Kafka has messages
    message_count=$(docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic noise-readings 2>/dev/null | awk -F':' '{sum+=$3} END {print sum}')
    
    if [ "$message_count" -lt 1 ]; then
        echo -e "${RED}❌ No messages in Kafka!${NC}"
        echo "Please start the producer first (option 3)"
        return 1
    fi
    
    echo -e "${GREEN}✅ Found $message_count messages in Kafka${NC}"
    echo ""
    echo "Running Spark consumer test for 60 seconds..."
    echo "You should see sensor data appearing..."
    echo ""
    
    docker exec spark-master /opt/spark/bin/spark-submit \
        --master local[2] \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        /opt/spark/work-dir/test_consumer.py
    
    echo ""
    echo -e "${GREEN}✅ Test complete!${NC}"
}

# Function to start Spark consumer (full)
start_spark_consumer() {
    echo ""
    echo -e "${YELLOW}Starting Spark Streaming Consumer...${NC}"
    echo ""
    
    # Check dependencies
    if ! docker compose ps kafka-producer | grep -q "Up"; then
        echo -e "${RED}❌ Producer is not running!${NC}"
        echo "Please start the producer first (option 3)"
        return 1
    fi
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        echo "Please start all services first (option 1)"
        return 1
    fi
    
    echo -e "${GREEN}✅ Dependencies check passed${NC}"
    echo ""
    echo "Starting Spark consumer..."
    echo "This will:"
    echo "  1. Read from Kafka topic 'noise-readings'"
    echo "  2. Process and enrich sensor data"
    echo "  3. Calculate aggregations"
    echo "  4. Store in PostgreSQL tables"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    docker exec -it spark-master /opt/spark/bin/spark-submit \
        --master local[2] \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.6.0 \
        /opt/spark/work-dir/noise_consumer.py
}

# Function to check Kafka topic
check_kafka_topic() {
    echo ""
    echo -e "${YELLOW}Checking Kafka topic 'noise-readings'...${NC}"
    echo ""
    
    # List topics
    echo "Available topics:"
    docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
    
    echo ""
    echo "Message count in 'noise-readings' topic:"
    docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic noise-readings 2>/dev/null | awk -F':' '{sum+=$3} END {print "Total messages: " sum}'
    
    echo ""
    echo "Latest message:"
    docker exec kafka kafka-console-consumer \
        --bootstrap-server localhost:9092 \
        --topic noise-readings \
        --max-messages 1 \
        --timeout-ms 5000 2>/dev/null || echo "No messages available"
}

# Function to test Spark consumer (Step 3)
test_spark_consumer() {
    echo ""
    echo -e "${YELLOW}Testing Spark Consumer (Step 3)...${NC}"
    echo ""
    
    # Check prerequisites
    if ! docker compose ps kafka-producer | grep -q "Up"; then
        echo -e "${RED}❌ Producer is not running!${NC}"
        echo "Start producer first: option 3"
        return 1
    fi
    
    # Check if Kafka has messages
    message_count=$(docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic noise-readings 2>/dev/null | awk -F':' '{sum+=$3} END {print sum}')
    
    if [ "$message_count" -lt 1 ]; then
        echo -e "${RED}❌ No messages in Kafka!${NC}"
        echo "Wait for producer to send messages"
        return 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites met${NC}"
    echo "   - Kafka: Running"
    echo "   - Messages: $message_count"
    echo ""
    echo "Running Spark test for 60 seconds..."
    echo "You should see sensor data from Dire Dawa appearing..."
    echo ""
    
    docker exec spark-master /opt/spark/bin/spark-submit \
        --master local[2] \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        /opt/spark/work-dir/test_spark.py
    
    echo ""
    echo -e "${GREEN}✅ Spark test complete!${NC}"
    echo ""
    echo "What this test did:"
    echo "  ✅ Connected Spark to Kafka"
    echo "  ✅ Read messages from 'noise-readings' topic"
    echo "  ✅ Parsed JSON data"
    echo "  ✅ Displayed sensor readings"
    echo ""
    echo "Next: Run full Spark consumer (option 14)"
}

# Function to start full Spark consumer
start_spark_consumer() {
    echo ""
    echo -e "${YELLOW}Starting Full Spark Streaming Consumer (Step 3)...${NC}"
    echo ""
    
    # Check dependencies
    if ! docker compose ps kafka-producer | grep -q "Up"; then
        echo -e "${RED}❌ Producer is not running!${NC}"
        echo "Start producer first: option 3"
        return 1
    fi
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        echo "Start all services first: option 1"
        return 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
    echo ""
    echo "Starting Spark Streaming Consumer..."
    echo ""
    echo "This will:"
    echo "  1. Read from Kafka topic 'noise-readings'"
    echo "  2. Validate & clean sensor data"
    echo "  3. Enrich with stress levels & alerts"
    echo "  4. Calculate real-time aggregations"
    echo "  5. Store in PostgreSQL tables"
    echo ""
    echo -e "${YELLOW}⏸️  Press Ctrl+C to stop gracefully${NC}"
    echo ""
    
    docker exec -it spark-master /opt/spark/bin/spark-submit \
        --master local[2] \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.1 \
        /opt/spark/work-dir/noise_consumer.py
}

# Function to check database records
check_database_records() {
    echo ""
    echo -e "${YELLOW}Checking PostgreSQL Database Records...${NC}"
    echo ""
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        return 1
    fi
    
    echo "Noise Readings Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) as total_readings FROM noise_readings;" | xargs echo "  Total:"
    
    echo ""
    echo "Latest 5 Readings:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT reading_time, sensor_id, location, neighborhood, noise_level, stress_level 
         FROM noise_readings 
         ORDER BY reading_time DESC 
         LIMIT 5;"
    
    echo ""
    echo "Aggregates Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) FROM noise_aggregates;" | xargs echo "  Total:"
    
    echo ""
    echo "Neighborhood Summary:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT neighborhood, COUNT(*) as reading_count, 
         ROUND(AVG(noise_level)::numeric, 1) as avg_noise
         FROM noise_readings 
         GROUP BY neighborhood 
         ORDER BY avg_noise DESC;"
}

# Function to check database records
check_database_records() {
    echo ""
    echo -e "${YELLOW}Checking PostgreSQL Database Records...${NC}"
    echo ""
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        return 1
    fi
    
    echo "Noise Readings Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) as total_readings FROM noise_readings;" | xargs echo "  Total:"
    
    echo ""
    echo "Latest 5 Readings:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT timestamp::timestamp(0), sensor_id, street_name, neighborhood, noise_level 
         FROM noise_readings 
         ORDER BY timestamp DESC 
         LIMIT 5;"
    
    echo ""
    echo "Aggregates Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) FROM noise_aggregates;" | xargs echo "  Total:"
    
    echo ""
    echo "Neighborhood Summary:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT neighborhood, COUNT(*) as reading_count, 
         ROUND(AVG(noise_level)::numeric, 1) as avg_noise
         FROM noise_readings 
         GROUP BY neighborhood 
         ORDER BY avg_noise DESC;"
}

# Function to clean dummy NYC data
clean_dummy_data() {
    echo ""
    echo -e "${YELLOW}Clean Dummy NYC Data${NC}"
    echo ""
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        return 1
    fi
    
    echo "This will remove sample NYC data but keep:"
    echo "  ✅ All Dire Dawa sensor data"
    echo "  ✅ Sentiment scores (for future use)"
    echo ""
    echo -e "${RED}⚠️  WARNING: This action cannot be undone!${NC}"
    echo ""
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        return 0
    fi
    
    echo ""
    echo "🧹 Cleaning NYC sample data..."
    
    # Remove NYC noise readings
    deleted=$(docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c "
    WITH deleted AS (
        DELETE FROM noise_readings 
        WHERE street_name IN (
            'Wall Street', 'Broadway', 'Park Avenue', 'Canal Street', 
            '5th Avenue', 'Brooklyn Bridge', 'Times Square', 'Washington Square'
        )
        AND neighborhood IN (
            'Financial District', 'Midtown', 'Upper East Side', 
            'Chinatown', 'Central Park', 'DUMBO', 'Greenwich Village'
        )
        RETURNING id
    )
    SELECT COUNT(*) FROM deleted;
    " 2>/dev/null | xargs)
    
    echo "  ✅ Removed $deleted NYC noise readings"
    
    # Remove NYC stress zones
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c "
    DELETE FROM current_stress_zones 
    WHERE neighborhood IN ('Midtown', 'Financial District', 'Chinatown');
    " 2>/dev/null
    
    echo "  ✅ Removed NYC stress zones"
    
    # Remove NYC routes
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c "
    DELETE FROM alternative_routes 
    WHERE route_name LIKE '%Midtown%' OR route_name LIKE '%Brooklyn%';
    " 2>/dev/null
    
    echo "  ✅ Removed NYC alternative routes"
    
    echo ""
    echo -e "${GREEN}✅ Cleanup complete!${NC}"
    echo ""
    echo "Your database now has only:"
    echo "  🇪🇹 Dire Dawa sensor data"
    echo "  💬 Sentiment scores (preserved)"
}

# Function to check database records
check_database_records() {
    echo ""
    echo -e "${YELLOW}Checking PostgreSQL Database Records...${NC}"
    echo ""
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        return 1
    fi
    
    echo "Noise Readings Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) as total_readings FROM noise_readings;" | xargs echo "  Total:"
    
    echo ""
    echo "Latest 5 Readings:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT reading_time, sensor_id, location, neighborhood, noise_level, stress_level 
         FROM noise_readings 
         ORDER BY reading_time DESC 
         LIMIT 5;"
    
    echo ""
    echo "Aggregates Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) FROM noise_aggregates;" | xargs echo "  Total:"
    
    echo ""
    echo "Neighborhood Summary:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT neighborhood, COUNT(*) as reading_count, 
         ROUND(AVG(noise_level)::numeric, 1) as avg_noise
         FROM noise_readings 
         GROUP BY neighborhood 
         ORDER BY avg_noise DESC;"
}

# Function to cleanup dummy data
cleanup_dummy_data() {
    echo ""
    echo -e "${YELLOW}Cleaning Up Dummy/Sample Data...${NC}"
    echo ""
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        return 1
    fi
    
    echo "This will remove:"
    echo "  - Dummy NYC noise readings (Wall Street, Times Square, etc.)"
    echo "  - Dummy aggregates"
    echo "  - Dummy stress zones"
    echo "  - Dummy alternative routes"
    echo ""
    echo "This will KEEP:"
    echo "  - All REAL Dire Dawa sensor data (from Kafka/Spark)"
    echo "  - Sentiment data (for future work)"
    echo ""
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" == "yes" ]; then
        echo ""
        echo "Running cleanup script..."
        docker exec postgres psql -U traffic_user -d traffic_noise_db < cleanup_dummy_data.sql
        echo ""
        echo -e "${GREEN}✅ Cleanup complete!${NC}"
        echo ""
        echo "Check data:"
        echo "  ./start.sh → Choose 15"
    else
        echo "Cancelled."
    fi
}

# Function to check database records
check_database_records() {
    echo ""
    echo -e "${YELLOW}Checking PostgreSQL Database Records...${NC}"
    echo ""
    
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        return 1
    fi
    
    echo "Noise Readings Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) as total_readings FROM noise_readings;" | xargs echo "  Total:"
    
    echo ""
    echo "Latest 5 Readings:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT timestamp::timestamp(0), sensor_id, street_name, neighborhood, noise_level 
         FROM noise_readings 
         ORDER BY timestamp DESC 
         LIMIT 5;" 2>/dev/null
    
    echo ""
    echo "Aggregates Count:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) FROM noise_aggregates;" | xargs echo "  Total:"
    
    echo ""
    echo "Route Segments Count (Heat Map):"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -t -c \
        "SELECT COUNT(*) FROM route_noise_segments;" | xargs echo "  Total:" 2>/dev/null || echo "  Total: 0 (table not initialized yet)"
    
    echo ""
    echo "Route Segments by Category:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT noise_category, COUNT(*) as count 
         FROM route_noise_segments 
         WHERE timestamp > NOW() - INTERVAL '2 hours'
         GROUP BY noise_category 
         ORDER BY count DESC;" 2>/dev/null || echo "  (No route segments yet - wait for 15 min)"
    
    echo ""
    echo "Neighborhood Summary:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT neighborhood, COUNT(*) as reading_count, 
         ROUND(AVG(noise_level)::numeric, 1) as avg_noise
         FROM noise_readings 
         GROUP BY neighborhood 
         ORDER BY avg_noise DESC;"
    
    echo ""
    echo -e "${YELLOW}Note: Route analysis runs automatically every 15 minutes in Spark Consumer${NC}"
}

# Function to clean dummy data
clean_dummy_data() {
    echo ""
    echo -e "${YELLOW}Cleaning Dummy Noise Data...${NC}"
    echo -e "${RED}This will remove NYC sample data but keep Dire Dawa real-time data${NC}"
    echo -e "${GREEN}Sentiment data will be preserved${NC}"
    echo ""
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        return
    fi
    
    echo ""
    echo "Running cleanup script..."
    bash ./clean_dummy_data.sh 2>/dev/null || \
    docker exec postgres psql -U traffic_user -d traffic_noise_db <<-EOSQL
        -- Delete NYC sample data
        DELETE FROM noise_readings WHERE sensor_id LIKE 'SENSOR_00%' AND street_name IN (
            'Wall Street', 'Broadway', 'Park Avenue', 'Canal Street', 
            '5th Avenue', 'Brooklyn Bridge', 'Times Square', 'Washington Square'
        );
        
        -- Clear aggregates (will be recalculated from real data)
        TRUNCATE TABLE noise_aggregates;
        
        -- Delete NYC stress zones
        DELETE FROM current_stress_zones WHERE street_name IN (
            'Wall Street', 'Broadway', 'Park Avenue', 'Canal Street',
            'Times Square', 'Washington Square'
        );
        
        -- Clear predictions
        TRUNCATE TABLE noise_predictions;
        TRUNCATE TABLE predicted_stress_zones;
        
        -- Clear correlations
        TRUNCATE TABLE noise_sentiment_correlation;
        
        -- Clear peak patterns
        TRUNCATE TABLE peak_noise_patterns;
        
        -- Clear route recommendations
        TRUNCATE TABLE route_recommendations;
        
        SELECT 'Cleanup complete!' as status;
EOSQL
    
    echo ""
    echo -e "${GREEN}✓ Dummy data removed${NC}"
    echo ""
    echo "Remaining data:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT COUNT(*) as noise_readings FROM noise_readings;"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT COUNT(*) as sentiment_scores FROM sentiment_scores;"
}

# Function to clean dummy data
clean_dummy_data() {
    echo ""
    echo -e "${YELLOW}Cleaning Dummy Noise Data...${NC}"
    echo -e "${RED}This will remove NYC sample data but keep Dire Dawa real-time data${NC}"
    echo -e "${GREEN}Sentiment data will be preserved${NC}"
    echo ""
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        return
    fi
    
    echo ""
    echo "Running cleanup script..."
    bash ./clean_dummy_data.sh 2>/dev/null || \
    docker exec postgres psql -U traffic_user -d traffic_noise_db <<-EOSQL
        -- Delete NYC sample data
        DELETE FROM noise_readings WHERE sensor_id LIKE 'SENSOR_00%' AND street_name IN (
            'Wall Street', 'Broadway', 'Park Avenue', 'Canal Street', 
            '5th Avenue', 'Brooklyn Bridge', 'Times Square', 'Washington Square'
        );
        
        -- Clear aggregates (will be recalculated from real data)
        TRUNCATE TABLE noise_aggregates;
        
        -- Delete NYC stress zones
        DELETE FROM current_stress_zones WHERE street_name IN (
            'Wall Street', 'Broadway', 'Park Avenue', 'Canal Street',
            'Times Square', 'Washington Square'
        );
        
        -- Clear predictions
        TRUNCATE TABLE noise_predictions;
        TRUNCATE TABLE predicted_stress_zones;
        
        -- Clear correlations
        TRUNCATE TABLE noise_sentiment_correlation;
        
        -- Clear peak patterns
        TRUNCATE TABLE peak_noise_patterns;
        
        -- Clear route recommendations
        TRUNCATE TABLE route_recommendations;
        
        SELECT 'Cleanup complete!' as status;
EOSQL
    
    echo ""
    echo -e "${GREEN}✓ Dummy data removed${NC}"
    echo ""
    echo "Remaining data:"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT COUNT(*) as noise_readings FROM noise_readings;"
    docker exec postgres psql -U traffic_user -d traffic_noise_db -c \
        "SELECT COUNT(*) as sentiment_scores FROM sentiment_scores;"
}

# Function to view logs
view_logs() {
    echo ""
    echo -e "${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
    docker compose logs -f
}

# Function to restart a specific service
restart_service() {
    echo ""
    echo "Available services:"
    echo "  1. kafka"
    echo "  2. postgres"
    echo "  3. spark-master"
    echo "  4. spark-worker"
    echo "  5. streamlit"
    echo "  6. jupyter"
    echo "  7. kafka-producer"
    echo ""
    read -p "Enter service name to restart: " service
    
    echo -e "${YELLOW}Restarting $service...${NC}"
    docker compose restart "$service"
    echo -e "${GREEN}✓ $service restarted${NC}"
}

# Function to start only PostgreSQL + Jupyter for ML development
start_postgres_jupyter_only() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Starting PostgreSQL + Jupyter for ML Work${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    # Stop all services first
    echo -e "${YELLOW}Stopping other services...${NC}"
    docker compose stop
    
    # Start only PostgreSQL and Jupyter
    echo -e "${YELLOW}Starting PostgreSQL...${NC}"
    docker compose up -d postgres
    
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    sleep 5
    
    echo -e "${YELLOW}Starting Jupyter Notebook...${NC}"
    docker compose up -d jupyter
    
    echo ""
    echo -e "${GREEN}✓ PostgreSQL and Jupyter started!${NC}"
    echo ""
    echo -e "${BLUE}📊 Access Jupyter Notebook:${NC}"
    echo "   http://localhost:8888"
    echo ""
    echo -e "${BLUE}🔌 PostgreSQL Connection (from Jupyter):${NC}"
    echo "   Host: postgres"
    echo "   Port: 5432"
    echo "   Database: traffic_noise_db"
    echo "   User: traffic_user"
    echo "   Password: traffic_pass"
    echo ""
    echo -e "${BLUE}📝 Sample Connection Code:${NC}"
    echo "   import pandas as pd"
    echo "   import psycopg2"
    echo "   "
    echo "   conn = psycopg2.connect("
    echo "       host='postgres',"
    echo "       port=5432,"
    echo "       database='traffic_noise_db',"
    echo "       user='traffic_user',"
    echo "       password='traffic_pass'"
    echo "   )"
    echo "   df = pd.read_sql('SELECT * FROM noise_readings LIMIT 10', conn)"
    echo ""
    echo -e "${YELLOW}Note: Only PostgreSQL and Jupyter are running.${NC}"
    echo -e "${YELLOW}      Use option 1 to start all services again.${NC}"
    echo ""
}

# Function to download sentiment model
download_sentiment_model() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Downloading Sentiment Analysis Model${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    # Check if utils/download_model.py exists
    if [ ! -f "utils/download_model.py" ]; then
        echo -e "${RED}❌ Error: utils/download_model.py not found!${NC}"
        echo ""
        echo "Please make sure the file exists in your project directory."
        return 1
    fi
    
    echo -e "${YELLOW}📦 Model: DistilBERT (Sentiment Analysis)${NC}"
    echo -e "${YELLOW}📏 Size: ~255 MB${NC}"
    echo -e "${YELLOW}⏱️  Time: 1-2 minutes${NC}"
    echo ""
    
    # Check if transformers is installed
    if ! python3 -c "import transformers" 2>/dev/null; then
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install -q transformers torch 2>/dev/null || pip3 install -q transformers torch
    fi
    
    # Run download script
    echo -e "${YELLOW}Downloading model...${NC}"
    echo ""
    
    python3 utils/download_model.py || python utils/download_model.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}================================================${NC}"
        echo -e "${GREEN}  ✅ Model Downloaded Successfully!${NC}"
        echo -e "${GREEN}================================================${NC}"
        echo ""
        echo -e "${BLUE}📂 Location: models/sentiment/${NC}"
        echo ""
        echo "You can now use sentiment analysis in your code:"
        echo ""
        echo "  from utils.sentiment_analyzer import SentimentAnalyzer"
        echo "  analyzer = SentimentAnalyzer()"
        echo "  result = analyzer.analyze('Heavy traffic, very loud')"
        echo ""
    else
        echo ""
        echo -e "${RED}❌ Download failed!${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check internet connection"
        echo "  2. Install dependencies: pip install transformers torch"
        echo "  3. Run manually: python utils/download_model.py"
    fi
}

# Function to test correlation analyzer
test_correlation_analyzer() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Testing Correlation Analyzer${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    # Check if test script exists
    if [ ! -f "correlation-analyzer/test_correlation.py" ]; then
        echo -e "${RED}❌ Error: test_correlation.py not found!${NC}"
        echo ""
        echo "Please make sure the file exists in your project directory."
        return 1
    fi
    
    # Check if pandas and tabulate are installed
    echo -e "${YELLOW}Checking dependencies...${NC}"
    if ! python3 -c "import pandas, tabulate" 2>/dev/null; then
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install -q pandas tabulate psycopg2-binary 2>/dev/null || pip3 install -q pandas tabulate psycopg2-binary
    fi
    
    # Run test
    echo -e "${YELLOW}Running correlation analyzer tests...${NC}"
    echo ""
    
    python3 correlation-analyzer/test_correlation.py || python correlation-analyzer/test_correlation.py
    
    echo ""
    echo -e "${BLUE}Tip: To view correlation analyzer logs:${NC}"
    echo "  docker logs -f correlation-analyzer"
    echo ""
}

# Function to test noise predictor
test_noise_predictor() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Testing Noise Predictor${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    # Check if service is running
    if ! docker compose ps noise-predictor | grep -q "Up"; then
        echo -e "${RED}❌ Noise predictor is not running!${NC}"
        echo "Start all services first: option 1"
        return 1
    fi
    
    # Check if test script exists
    if [ ! -f "noise-predictor/test_predictor.py" ]; then
        echo -e "${RED}❌ Test script not found!${NC}"
        return 1
    fi
    
    # Install dependencies
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -q pandas tabulate psycopg2-binary 2>/dev/null || pip3 install -q pandas tabulate psycopg2-binary
    
    # Run test
    echo -e "${YELLOW}Running noise predictor tests...${NC}"
    echo ""
    
    python3 noise-predictor/test_predictor.py || python noise-predictor/test_predictor.py
    
    echo ""
    echo -e "${BLUE}Tip: To view noise predictor logs:${NC}"
    echo "  docker logs -f noise-predictor"
    echo ""
}

# Function to start stress zone predictor
start_stress_zone_predictor() {
    echo ""
    echo -e "${YELLOW}Starting Stress Zone Predictor...${NC}"
    
    # Check if PostgreSQL is running
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        echo "Start all services first: option 1"
        return 1
    fi
    
    # Check if noise predictor is running
    if ! docker compose ps noise-predictor | grep -q "Up"; then
        echo -e "${YELLOW}⚠️  Noise predictor is not running${NC}"
        echo "Stress zone predictor needs noise predictions to work"
        echo "Start noise predictor first: option 25"
        read -p "Continue anyway? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            return 0
        fi
    fi
    
    docker compose up -d stress-zone-predictor
    echo -e "${GREEN}✅ Stress zone predictor started${NC}"
    echo ""
    echo "The predictor will:"
    echo "  1. Read noise predictions"
    echo "  2. Analyze sentiment for each location"
    echo "  3. Calculate stress scores (0-100)"
    echo "  4. Identify high-stress zones"
    echo "  5. Generate recommended actions"
    echo "  6. Run every 15 minutes (1 min after noise predictor)"
    echo ""
    echo "View logs:"
    echo "  docker logs -f stress-zone-predictor"
}

# Function to stop stress zone predictor
stop_stress_zone_predictor() {
    echo ""
    echo -e "${YELLOW}Stopping Stress Zone Predictor...${NC}"
    docker compose stop stress-zone-predictor
    echo -e "${GREEN}✅ Stress zone predictor stopped${NC}"
}

# Function to view stress zone predictor logs
view_stress_zone_logs() {
    echo ""
    echo -e "${YELLOW}Stress Zone Predictor Logs (Ctrl+C to exit)...${NC}"
    echo ""
    docker logs -f stress-zone-predictor
}

# Function to test stress zone predictor
test_stress_zone_predictor() {
    echo ""
    echo -e "${BLUE}Testing Stress Zone Predictor...${NC}"
    echo ""
    
    # Check if service is running
    if ! docker compose ps stress-zone-predictor | grep -q "Up"; then
        echo -e "${RED}❌ Stress zone predictor is not running!${NC}"
        echo "Start it first: option 28"
        return 1
    fi
    
    # Check if test script exists
    if [ ! -f "stress-zone-predictor/test_stress_zones.py" ]; then
        echo -e "${RED}❌ Test script not found!${NC}"
        return 1
    fi
    
    # Check dependencies
    echo "Installing test dependencies..."
    pip install -q pandas tabulate psycopg2-binary 2>/dev/null || pip3 install -q pandas tabulate psycopg2-binary
    
    # Run test
    python3 stress-zone-predictor/test_stress_zones.py || python stress-zone-predictor/test_stress_zones.py
}

# Function to test alternative routes API
test_alternative_routes_api() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Testing Alternative Routes API${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    # Check if service is running
    if ! docker compose ps alternative-routes-api | grep -q "Up"; then
        echo -e "${RED}❌ Alternative Routes API is not running!${NC}"
        echo "Start all services first: option 1"
        return 1
    fi
    
    # Check if test script exists
    if [ ! -f "route-recommendation/test_api.py" ]; then
        echo -e "${RED}❌ Test script not found!${NC}"
        echo "Please make sure route-recommendation/test_api.py exists"
        return 1
    fi
    
    # Install dependencies
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -q requests 2>/dev/null || pip3 install -q requests
    
    echo ""
    echo -e "${YELLOW}Running API tests...${NC}"
    echo ""
    
    # Run test
    python3 route-recommendation/test_api.py || python route-recommendation/test_api.py
    
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  API Endpoints Available:${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo "Health Check:"
    echo "  curl http://localhost:5000/health"
    echo ""
    echo "Get Available Locations:"
    echo "  curl http://localhost:5000/api/locations"
    echo ""
    echo "Get Route Recommendations:"
    echo "  curl -X POST http://localhost:5000/api/route-recommendations \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"origin\": \"Mewlid Road\", \"destination\": \"Railway Station Road\"}'"
    echo ""
    echo -e "${BLUE}Tip: To view API logs:${NC}"
    echo "  docker logs -f alternative-routes-api"
    echo ""
}

# Function to start alternative routes API
start_alternative_routes_api() {
    echo ""
    echo -e "${YELLOW}Starting Alternative Routes API...${NC}"
    
    # Check if PostgreSQL is running
    if ! docker compose ps postgres | grep -q "Up"; then
        echo -e "${RED}❌ PostgreSQL is not running!${NC}"
        echo "Start all services first: option 1"
        return 1
    fi
    
    docker compose up -d alternative-routes-api
    echo -e "${GREEN}✅ Alternative Routes API started${NC}"
    echo ""
    echo "The API is now available at: http://localhost:5000"
    echo ""
    echo "Test it:"
    echo "  ./start.sh → Choose option 29 (Test Routes API)"
    echo ""
    echo "View logs:"
    echo "  docker logs -f alternative-routes-api"
}

# Function to stop alternative routes API
stop_alternative_routes_api() {
    echo ""
    echo -e "${YELLOW}Stopping Alternative Routes API...${NC}"
    docker compose stop alternative-routes-api
    echo -e "${GREEN}✅ Alternative Routes API stopped${NC}"
}

# Function to view alternative routes API logs
view_alternative_routes_logs() {
    echo ""
    echo -e "${YELLOW}Alternative Routes API Logs (Ctrl+C to exit)...${NC}"
    echo ""
    docker logs -f alternative-routes-api
}

# Function to clean everything (including volumes)
clean_all() {
    echo ""
    echo -e "${RED}WARNING: This will delete all data including database contents!${NC}"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" == "yes" ]; then
        echo -e "${YELLOW}Cleaning all containers and volumes...${NC}"
        docker compose down -v
        echo -e "${GREEN}✓ All data cleaned${NC}"
    else
        echo "Cancelled."
    fi
}

# Function to show system status
show_status() {
    echo ""
    echo -e "${YELLOW}System Status:${NC}"
    echo ""
    docker compose ps
    echo ""
    echo "Resource usage:"
    docker stats --no-stream
}

# Main menu
main_menu() {
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "  === Step 1-2: Data Generation & Kafka ==="
    echo "  1. Start all services (producer + consumer)"
    echo "  2. Stop all services"
    echo "  3. Start Kafka Producer only"
    echo "  4. Stop Kafka Producer only"
    echo "  5. Restart Kafka Producer"
    echo "  6. Test Kafka Consumer (read 10 messages)"
    echo "  7. View Producer logs"
    echo "  8. Check Kafka topic status"
    echo ""
    echo "  === Step 3: Spark Streaming (Service) ==="
    echo "  9. Start Spark Consumer service"
    echo " 10. Stop Spark Consumer service"
    echo " 11. Restart Spark Consumer service"
    echo " 12. View Spark Consumer logs"
    echo ""
    echo "  === Step 3: Spark Streaming (Manual) ==="
    echo " 13. Test Spark Consumer (60 sec test)"
    echo " 14. Start Spark Consumer (interactive)"
    echo " 15. Check Database Records"
    echo " 16. Clean Dummy NYC Data (keep Dire Dawa)"
    echo ""
    echo "  === ML Development ==="
    echo " 17. Start PostgreSQL + Jupyter Only"
    echo " 18. Download Sentiment Model"
    echo " 19. Test Correlation Analyzer"
    echo " 20. Test Noise Predictor"
    echo ""
    echo "  === System Management ==="
    echo " 21. View all logs"
    echo " 22. Show system status"
    echo " 23. Restart a service"
    echo " 24. Clean all data (WARNING: destructive)"
    echo " 25. Exit"
    echo ""
    echo "  Note: Route analysis runs automatically in Spark Consumer (every 5 min)"
    echo ""
    read -p "Enter your choice (1-25): " choice
    
    case $choice in
        1)
            check_docker
            check_resources
            start_services
            ;;
        2)
            stop_services
            ;;
        3)
            start_producer
            ;;
        4)
            stop_producer
            ;;
        5)
            restart_producer
            ;;
        6)
            test_consumer
            ;;
        7)
            view_producer_logs
            ;;
        8)
            check_kafka_topic
            ;;
        9)
            start_spark_consumer_service
            ;;
        10)
            stop_spark_consumer_service
            ;;
        11)
            restart_spark_consumer_service
            ;;
        12)
            view_spark_logs
            ;;
        13)
            test_spark_consumer
            ;;
        14)
            start_spark_consumer
            ;;
        15)
            check_database_records
            ;;
        16)
            clean_dummy_data
            ;;
        17)
            start_postgres_jupyter_only
            ;;
        18)
            download_sentiment_model
            ;;
        19)
            test_correlation_analyzer
            ;;
        20)
            test_noise_predictor
            ;;
        21)
            start_stress_zone_predictor
            ;;
        22)
            stop_stress_zone_predictor
            ;;
        23)
            view_stress_zone_logs
            ;;
        24)
            test_stress_zone_predictor
            ;;
        25)
            view_logs
            ;;
        26)
            show_status
            ;;
        27)
            restart_service
            ;;
        28)
            clean_all
            ;;
        29)
            test_alternative_routes_api
            ;;
        30)
            start_alternative_routes_api
            ;;
        31)
            stop_alternative_routes_api
            ;;
        32)
            view_alternative_routes_logs
            ;;
        33)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
    
    # Loop back to menu
    main_menu
}

# Run main menu
main_menu