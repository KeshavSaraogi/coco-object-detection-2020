#!/bin/bash
echo "Starting EDA Bootstrap..."
sudo yum update -y
sudo yum install -y python3 python3-pip

# Install required Python libraries
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install boto3 pandas matplotlib seaborn pyarrow

echo "EDA Bootstrap completed!"
