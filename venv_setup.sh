#!/usr/bin/env python3
# Install venv on UCloud:
sudo apt-get update
sudo apt-get install python3-venv

# Create virtual environment
python3 -m venv spacy_env

# Activate virtual environment (only if running the script directly with Python)
source ./spacy_env/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_md

# Run your Python script
python3 src/logi_reg_cifar10.py