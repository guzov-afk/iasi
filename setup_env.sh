#!/bin/bash

ENV_NAME="kan_env"
PYTHON_VERSION="3.8"  


sudo apt update
sudo apt upgrade -y


sudo apt install -y python3-dev python3-venv python3-pip libopenblas-dev libjpeg-dev zlib1g-dev


python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate


pip install --upgrade pip


pip install numpy matplotlib torch==1.10.0 torchvision==0.11.1 scikit-learn


pip install pykan  


echo "Environment setup complete. Activate it with: source $ENV_NAME/bin/activate"
