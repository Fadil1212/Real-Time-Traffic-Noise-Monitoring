#!/bin/bash

# Quick Test Script for Kafka Producer
# Tests if producer can send messages to Kafka

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=================================================="
echo "  Kafka Producer Test Script"
echo -e "==================================================${NC}"
echo ""

# Step 1: Check if Kafka is running
echo -e "${YELLOW}Step 1: Checking if Kafka is running...${NC}"
if docker compose ps kafka | grep -q "Up"; then
    echo -e "${GREEN}✅ Kafka is running${NC}"
else
    echo -e "${RED}❌ Kafka is not running${NC}"
    echo "Starting Kafka..."
    docker compose up -d kafka zookeeper
    echo "Waiting 30 seconds for Kafka to be ready..."
    sleep 30
fi

echo ""

# Step 2: Check if Zookeeper is running
echo -e "${YELLOW}Step 2: Checking if Zookeeper is running...${NC}"
if docker compose ps zookeeper | grep -q "Up"; then
    echo -e "${GREEN}✅ Zookeeper is running${NC}"
else
    echo -e "${RED}❌ Zookeeper is not running${NC}"
    echo "Starting Zookeeper..."
    docker compose up -d zookeeper
    sleep 10
fi

echo ""

# Step 3: Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
if command -v pip3 &> /dev/null; then
    pip3 install -q kafka-python==2.0.2
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${RED}❌ pip3 not found. Please install Python 3 and pip${NC}"
    exit 1
fi

echo ""

# Step 4: Check if topic exists, create if not
echo -e "${YELLOW}Step 4: Checking Kafka topic...${NC}"
if docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 2>/dev/null | grep -q "noise-readings"; then
    echo -e "${GREEN}✅ Topic 'noise-readings' exists${NC}"
else
    echo -e "${YELLOW}⚠️  Topic doesn't exist, creating...${NC}"
    docker exec kafka kafka-topics \
        --create \
        --bootstrap-server localhost:9092 \
        --topic noise-readings \
        --partitions 3 \
        --replication-factor 1 2>/dev/null
    echo -e "${GREEN}✅ Topic created${NC}"
fi

echo ""

# Step 5: Run producer for 30 seconds
echo -e "${YELLOW}Step 5: Running producer for 30 seconds...${NC}"
echo -e "${BLUE}(You should see sensor data being sent)${NC}"
echo ""

cd kafka-producer

# Run producer in background
timeout 30 python3 noise_producer.py &
PRODUCER_PID=$!

# Wait a bit for producer to start
sleep 3

# Run consumer to verify messages
echo ""
echo -e "${YELLOW}Verifying messages with consumer...${NC}"
timeout 10 python3 test_consumer.py 5 &
CONSUMER_PID=$!

# Wait for both to complete
wait $PRODUCER_PID 2>/dev/null
wait $CONSUMER_PID 2>/dev/null

echo ""
echo ""

# Step 6: Check message count
echo -e "${YELLOW}Step 6: Checking message count...${NC}"
MESSAGE_COUNT=$(docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 \
    --topic noise-readings 2>/dev/null | awk -F':' '{sum+=$3} END {print sum}')

if [ "$MESSAGE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Success! $MESSAGE_COUNT messages in Kafka${NC}"
else
    echo -e "${RED}❌ No messages found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=================================================="
echo "  ✅ All Tests Passed!"
echo -e "==================================================${NC}"
echo ""
echo "What happened:"
echo "  1. ✅ Producer sent sensor data to Kafka"
echo "  2. ✅ Consumer verified messages are readable"
echo "  3. ✅ Topic has $MESSAGE_COUNT messages"
echo ""
echo "Next steps:"
echo "  - Run producer continuously: python3 kafka-producer/noise_producer.py"
echo "  - Build Spark consumer to process messages"
echo "  - Store data in PostgreSQL"
echo ""