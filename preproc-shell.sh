#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting bootstrap script for preprocessing dependencies..."

# Install only the additional dependencies required for preprocessing.py
sudo python3 -m pip install numpy opencv-python-headless matplotlib seaborn boto3 pandas

echo "Preprocessing dependencies installed successfully!"
