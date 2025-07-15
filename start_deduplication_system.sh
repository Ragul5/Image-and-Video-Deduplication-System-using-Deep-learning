#!/bin/bash

echo "============================================================"
echo "               DeduplicationSystem Launcher"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in your PATH."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "For macOS, install Python from https://www.python.org/downloads/"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "For Linux, use: sudo apt-get install python3 python3-pip"
    else
        echo "Please install Python 3 for your platform."
    fi
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Running DeduplicationSystem..."
echo "This will automatically install all needed dependencies"
echo "and start the application."
echo
echo "(This might take a minute on first run)"
echo

# Run the launcher script
python3 run_app.py

echo
if [ $? -ne 0 ]; then
    echo "Application failed to start. Please check the error messages above."
    echo
    read -p "Press Enter to exit..."
else
    echo "Application closed. You can restart it by running this script again."
    sleep 5
fi