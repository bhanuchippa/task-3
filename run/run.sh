#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=============================================================="
echo "    AI Image Classifier - Launcher"
echo "=============================================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 was not found on your system!"
    echo "Please download and install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Run the Python launcher script
python3 "$SCRIPT_DIR/run.py"
