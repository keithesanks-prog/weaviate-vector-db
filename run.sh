#!/bin/bash

# Run script that activates venv and runs commands
# Usage: ./run.sh [ingest|query|all]

set -e

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup.sh
fi

# Activate virtual environment
source venv/bin/activate

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Weaviate is running
if ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo "Weaviate is not running. Starting Weaviate..."
    docker-compose up -d
    echo "Waiting for Weaviate to be ready..."
    sleep 10
    for i in {1..30}; do
        if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
            echo "✓ Weaviate is ready!"
            break
        fi
        echo "Waiting for Weaviate... ($i/30)"
        sleep 2
    done
fi

# Parse command
COMMAND=${1:-all}

case $COMMAND in
    ingest)
        echo "Ingesting data..."
        python -B ingest_data.py
        ;;
    query)
        echo "Running query demonstrations..."
        python -B query_demo.py
        ;;
    all)
        echo "Ingesting data..."
        python -B ingest_data.py
        echo ""
        echo "Running query demonstrations..."
        python -B query_demo.py
        ;;
    *)
        echo "Usage: $0 [ingest|query|all]"
        echo "  ingest - Run data ingestion only"
        echo "  query  - Run query demonstrations only"
        echo "  all    - Run both (default)"
        exit 1
        ;;
esac

