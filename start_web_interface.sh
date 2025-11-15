#!/bin/bash

# Start the web query interface

echo "=========================================="
echo "Starting Web Query Interface"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask and Flask-CORS..."
    pip install flask flask-cors
fi

# Check if Weaviate is running
if ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo "Warning: Weaviate doesn't appear to be running."
    echo "Start it with: docker-compose up -d"
    echo ""
fi

echo "Starting Flask web server..."
echo "Open your browser to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Flask server
python -B web_query_server.py

