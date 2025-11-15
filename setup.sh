#!/bin/bash

# Setup script for Social Services Experience Analytics Platform
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Social Services Experience Analytics Platform"
echo "Setup Script"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if venv module is available
if ! python3 -m venv --help &> /dev/null; then
    echo "ERROR: python3-venv is not installed."
    echo "Install it with: sudo apt install python3-venv python3-full"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  python ingest_data.py"
echo "  python query_demo.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""

