#!/bin/bash
echo "Starting Model Training Bootstrap..."
sudo yum update -y
sudo yum install -y python3 python3-pip

# Install required Python libraries
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install torch torchvision boto3 Pillow

echo "Model Training Bootstrap completed!"
