#!/bin/bash

# Quick start script for Social Services Experience Analytics Platform
# This script sets up the virtual environment and runs everything

set -e

echo "=========================================="
echo "Social Services Experience Analytics Platform"
echo "Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Step 1: Setup virtual environment
echo "Step 1: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    ./setup.sh
else
    echo "✓ Virtual environment already exists"
    source venv/bin/activate
fi

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Step 2: Starting Weaviate with CLIP module..."
docker-compose up -d

echo ""
echo "Waiting for Weaviate to be ready..."
sleep 10

# Check if Weaviate is ready
for i in {1..30}; do
    if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
        echo "✓ Weaviate is ready!"
        break
    fi
    echo "Waiting for Weaviate... ($i/30)"
    sleep 2
done

echo ""
echo "Step 3: Ingesting data..."
# Use -B flag to disable bytecode caching (ensures fresh code execution)
python -B ingest_data.py

echo ""
echo "Step 4: Running query demonstrations..."
python -B query_demo.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now:"
echo "  - Query the database: source venv/bin/activate && python query_demo.py"
echo "  - View Weaviate console: http://localhost:8080/v1/schema"
echo "  - Stop services: docker-compose down"
echo ""
echo "Remember to activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""

