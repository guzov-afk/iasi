#!/bin/bash

ENV_NAME="kan_env"

sudo apt install python3.9.7-venv


python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate


pip install --upgrade pip

pip install numpy==1.24.4 matplotlib==3.6.2 torch==2.2.2 scikit-learn==1.1.3 setuptools==65.5.0 sympy==1.11.1 tqdm==4.66.2 pyyaml seaborn pandas tensorflow
pip install pykan

echo "Environment setup complete. Activate it with: source $ENV_NAME/bin/activate"
